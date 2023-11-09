from transformers import BartTokenizerFast, BartForConditionalGeneration
from models.hfmodel import HFModel


class Bart(HFModel):

    name = "bart"
    hf_upload_prefix = "facebook-bart-large"
    model: BartForConditionalGeneration = None
    tokenizer: BartTokenizerFast = None
    default_key = "facebook/bart-large"

    def load(self, key: str) -> None:
        self.model = BartForConditionalGeneration.from_pretrained(key)
        self.tokenizer = BartTokenizerFast.from_pretrained(key)
