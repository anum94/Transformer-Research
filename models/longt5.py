from transformers import LongT5ForConditionalGeneration, T5TokenizerFast
from models.hfmodel import HFModel


class LongT5(HFModel):

    name = "longt5"
    hf_upload_prefix = "long-t5-tglobal-large"
    model: LongT5ForConditionalGeneration = None
    tokenizer: T5TokenizerFast = None
    default_key = "google/long-t5-tglobal-large"

    # relevant models
    # "pubmed": "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps",

    def load(self, key: str) -> None:
        self.model = LongT5ForConditionalGeneration.from_pretrained(key)
        self.tokenizer = T5TokenizerFast.from_pretrained(key)
