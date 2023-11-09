import logging
import os
from typing import Sequence

from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig,
)
from torch.cuda import current_device, is_available

from metrics.metric import Metric
from ds.hfdataset import HFDataset
from exec.execute import Execute
from models.model import Model
from config import Config
from utils.files import log_json, log_output, mkdir, push_files_git
from utils.hfrepo import get_model_id


class Evaluation(Execute):
    OUTPUT_FILE_KEY = "predictions.out"
    METRICS_FILE_KEY = "metrics.json"

    DEFAULT_EXEC_KWARGS = {
        "enc_max_len": 4096,
        "gen_max_len": 512,
        "num_beams": 1,  # greedy decoding
        "batch_size": 2,
    }

    ds_split = "test"

    model: Model = None
    dataset: Dataset = None
    metrics: Sequence[Metric] = []
    kwargs: dict = {}

    config: Config = None
    log_path: str = ""

    def __init__(
        self,
        model: Model,
        dataset: HFDataset,
        metrics: Sequence[Metric],
        config: Config,
        kwargs: dict,
    ) -> None:
        self.config = config
        self.kwargs = kwargs if kwargs else self.DEFAULT_EXEC_KWARGS

        self.model = model
        hf_model_path = self.model.hf_upload_name(
            enc_max_len=self.kwargs.get("enc_max_len"), dataset=dataset.ds_name
        )
        hf_path = get_model_id(config.hf_user, hf_model_path)
        key = self.model.eval_key(
            hf_path=hf_path, available_models=config.available_models
        )
        self.model.load(key)

        self.metrics = metrics
        self.dataset = dataset.get_split(self.ds_split)
        self.labels = self.dataset["summary"]

        self.log_path = config.log_path()
        mkdir(self.log_path)

    def predict(
        self,
        enc_max_len: int = 4096,
        gen_max_len: int = 256,
        batch_size: int = 32,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.2,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> tuple:
        logging.info("Running evaluation")

        tok_ds = self.model.tokenize_dataset(
            data=self.dataset,
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
            output_dir=self.config.log_path(),
            per_device_eval_batch_size=batch_size,
            deepspeed=self.config.dsconfig,
            predict_with_generate=True,
            report_to="all",
            bf16=True,
        )

        trainer = Seq2SeqTrainer(
            model=self.model.model,
            tokenizer=self.model.tokenizer,
            data_collator=data_collator,
            args=args,
        )

        out = trainer.predict(
            test_dataset=tok_ds,
            metric_key_prefix="",
            num_beams=num_beams,
            max_length=gen_max_len,
            do_sample=False,
        )

        return self.model.decode(out.predictions)

    def eval(
        self,
        predictions: list,
    ) -> dict:
        logging.info(f"Evaluating metrics {[m.metric_name for m in self.metrics]}")
        result = {}
        for metric in self.metrics:
            result[metric.metric_name] = metric.compute(
                predictions=predictions, references=self.labels
            )
        return result

    def log_sentences(self, output: list) -> str:
        path = os.path.join(self.log_path, self.OUTPUT_FILE_KEY)
        logging.info(f"Logging output in {path}")
        log_output(path=path, output=output)
        return path

    def log_metrics(self, metrics: dict) -> str:
        path = os.path.join(self.log_path, self.METRICS_FILE_KEY)
        logging.info(f"Logging metrics in {path}")
        log_json(path=path, metrics=metrics)
        return path

    def log(
        self,
        output_path: list,
        metrics_path: dict,
    ) -> None:
        commit_msg = f"results eval: {self.config.exec_string}"
        ssh_key_path = self.config.ssh_key_path()
        ssh_repo_path = self.config.working_dir

        push_files_git(
            commit_msg=commit_msg,
            file_paths=[output_path, metrics_path],
            repo_path=ssh_repo_path,
            ssh_key_path=ssh_key_path,
        )

    def execute(self) -> None:
        output = self.predict(**self.kwargs)
        if not is_available() or (is_available() and current_device()) == 0:
            spath = self.log_sentences(output)
            mpath = self.log_metrics(self.eval(output))
            self.log(spath, mpath)
