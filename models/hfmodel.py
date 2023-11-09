import logging
from typing import Sequence

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from huggingface_hub.hf_api import ModelInfo

from models.model import Model
from utils.hfrepo import model_is_available


class HFModel(Model):

    name: str
    hf_upload_prefix: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    default_key: str

    def generate(self, batch: list, max_len: int, num_beams: int) -> list:

        return self.model.generate(batch, max_length=max_len, num_beams=num_beams)

    def encode(self, batch: list, padding: str, truncation: bool, max_len: int) -> list:

        return self.tokenizer(
            batch,
            padding=padding,
            truncation=truncation,
            max_length=max_len,
            return_tensors="pt",
        )

    def decode(
        self,
        batch: list,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> list:

        return self.tokenizer.batch_decode(
            batch,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def tokenize_dataset(
        self, data: Dataset, padding: str, truncation: bool, max_len: int
    ) -> Dataset:
        def tokenize_fn(data: dict):
            tok_text = self.encode(
                data["text"], padding=padding, truncation=truncation, max_len=max_len
            )
            tok_labels = self.encode(
                data["summary"], padding=padding, truncation=truncation, max_len=max_len
            )["input_ids"]
            return {"labels": tok_labels, **tok_text}

        return data.map(tokenize_fn, remove_columns=["text", "summary"], batched=True)

    def to(self, device: str) -> None:
        self.model = self.model.to(device)

    def eval_key(
        self,
        hf_path: str,
        available_models: Sequence[ModelInfo],
    ) -> str:

        assert model_is_available(
            available_models=available_models, path=hf_path
        ), "finetuned model is not available"
        return hf_path

    def finetune_key(self, hf_path: str, available_models: Sequence[ModelInfo]) -> str:

        if model_is_available(available_models=available_models, path=hf_path):
            logging.info(f"found available hf model {hf_path}, resuming training")
            return hf_path

        logging.info("no pretrained model found, training with default version")
        return self.default_key

    def load(self, name: str) -> None:
        pass

    def hf_upload_name(
        self,
        enc_max_len: str,
        dataset: str,
    ) -> str:
        return f"{self.hf_upload_prefix}-{enc_max_len}-{dataset}"

    @property
    def max_enc_len(self) -> int:
        return self.model.config.max_position_embeddings
