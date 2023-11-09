import logging
from typing import Sequence

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import numpy as np

from config import Config
from models.hfmodel import HFModel
from ds.hfdataset import HFDataset
from metrics.metric import Metric
from utils.hfrepo import SaveCallback, get_model_id
from utils.files import mkdir
from exec.execute import Execute


class Finetune(Execute):
    DEFAULT_EXEC_KWARGS = {
        # "enc_max_len": 100,
        "gen_max_len": 100,
        "num_beams": 1,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_epochs": 1,
        "optimizer": "adamw_hf",
        "learning_rate": 0.00005,
        "fp16": False,
        "bf16": True,
    }

    TRAIN_SPLIT = "train"
    VAL_SPLIT = "validation"

    model: HFModel = None
    train_dataset: Dataset = None
    val_dataset: Dataset = None
    kwargs: dict = {}

    config: Config = None
    hf_path: str = ""

    def __init__(
        self,
        model: HFModel,
        dataset: HFDataset,
        metrics: Sequence[Metric],
        config: Config,
        kwargs: dict,
    ) -> None:
        mkdir(config.ckpt_log_path())
        self.model = model

        self.train_dataset = dataset.get_split(self.TRAIN_SPLIT)
        self.val_dataset = dataset.get_split(self.VAL_SPLIT)
        self.kwargs = kwargs if kwargs else self.DEFAULT_EXEC_KWARGS

        self.config = config
        self.hf_model_path = self.model.hf_upload_name(
            enc_max_len=self.kwargs.get("enc_max_len"), dataset=dataset.ds_name
        )
        self.hf_path = get_model_id(config.hf_user, self.hf_model_path)

        model_key = (
            self.model.finetune_key(
                hf_path=self.hf_path,
                available_models=config.available_models,
            )
            if not config.ckpt
            else config.ckpt
        )
        self.model.load(model_key)

        # FIXME set this file as summ and define rouge by itself
        assert (
            metrics[0].metric_name == "rouge"
        ), "Rouge metric not supplied for summarization"
        self.rouge = metrics[0]

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.model.decode(batch=predictions)
        labels = np.where(labels != -100, labels, self.model.tokenizer.pad_token_id)
        decoded_labels = self.model.decode(batch=labels)
        return self.rouge.compute(predictions=decoded_preds, references=decoded_labels)

    def train(
        self,
        enc_max_len: int = 8192,
        gen_max_len: int = 512,
        num_beams: int = 1,
        num_epochs: int = 1,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        fp16: bool = False,
        bf16: bool = False,
        optimizer: str = "adamw_hf",
        lr_scheduler_type: str = "linear",
        warmup_steps: int = 0,
        learning_rate: float = 0.00005,
        truncation: bool = True,
        padding: str = "max_length",
        logging_steps: int = 50,
        save_strategy: str = "steps",
        save_num: int = 1,
        evaluation_strategy: str = "steps",
        evaluation_steps: int = 400,
        metric_for_best_model="eval_rouge1",
        load_best_model_at_end: bool = True,
        **kwargs,
    ) -> None:
        # dinamically get max allowed input size from model if not set
        enc_max_len = enc_max_len if enc_max_len else self.model.max_enc_len

        # use previous log folder if resuming training
        log_path = (
            self.config.prev_log_path()
            if self.config.ckpt
            else self.config.ckpt_log_path()
        )

        save_steps = evaluation_steps

        logging.info("Tokenizing dataset")
        train_ds = self.model.tokenize_dataset(
            self.train_dataset,
            padding=padding,
            truncation=truncation,
            max_len=enc_max_len,
        )

        eval_ds = self.model.tokenize_dataset(
            self.val_dataset,
            padding=padding,
            truncation=truncation,
            max_len=enc_max_len,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.model.tokenizer,
            model=self.model.model,
            padding=padding,
            max_length=enc_max_len,
        )

        args = Seq2SeqTrainingArguments(
            output_dir=log_path,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            fp16=fp16,
            bf16=bf16,
            optim=optimizer,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            deepspeed=self.config.dsconfig,
            predict_with_generate=True,
            warmup_steps=warmup_steps,
            generation_num_beams=num_beams,
            generation_max_length=gen_max_len,
            save_strategy=save_strategy,
            save_total_limit=save_num,
            save_steps=save_steps,
            logging_strategy=evaluation_strategy,
            logging_steps=logging_steps,
            evaluation_strategy=evaluation_strategy,
            eval_steps=evaluation_steps,
            metric_for_best_model=metric_for_best_model,
            load_best_model_at_end=load_best_model_at_end,
            run_name=self.hf_model_path,
        )

        trainer = Seq2SeqTrainer(
            model=self.model.model,
            tokenizer=self.model.tokenizer,
            data_collator=data_collator,
            args=args,
            compute_metrics=self.compute_metrics,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            callbacks=[SaveCallback(self.config.hf_token, self.hf_path)],
        )

        logging.info("starting training")
        trainer.train(resume_from_checkpoint=self.config.ckpt)
        logging.info(f"training finished; logging in {log_path}")

        # save at the end of epoch
        trainer.save_model(output_dir=log_path)

    def execute(self) -> None:
        self.train(**self.kwargs)
