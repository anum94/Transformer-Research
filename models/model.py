from typing import Sequence

from datasets import Dataset
from huggingface_hub.hf_api import ModelInfo


class Model:

    name: str

    def __init__(self) -> None:
        pass

    def generate(self, batch: list, max_len: int, num_beams: int) -> list:
        pass

    def encode(
        self, batch: list, padding: bool, truncation: bool, max_len: int
    ) -> list:
        pass

    def decode(
        self, batch: list, skip_special_tokens: bool, clean_up_tokenization_spaces: bool
    ) -> list:
        pass

    def tokenize_dataset(
        self, data: Dataset, padding: str, truncation: bool, max_len: int
    ) -> Dataset:
        pass

    def to(self, device: str) -> None:
        pass

    def eval_key(
        self,
        hf_path: str,
        available_models: Sequence[ModelInfo],
    ) -> str:
        pass

    def finetune_key(self, hf_path: str, available_models: Sequence[ModelInfo]) -> str:
        pass

    def load(self, key: str) -> None:
        pass

    def hf_upload_name(
        self,
        enc_max_len: str,
        dataset: str,
    ) -> str:
        pass

    @property
    def max_enc_len(self) -> int:
        pass
