from transformers import PegasusTokenizerFast, PegasusXForConditionalGeneration
from models.hfmodel import HFModel


class PegasusX(HFModel):

    name: str = "pegasusx"
    hf_upload_prefix: str = "pegasus-x-large"
    model: PegasusXForConditionalGeneration = None
    tokenizer: PegasusTokenizerFast = None
    default_key: str = "google/pegasus-x-large"

    def load(self, key: str) -> None:
        self.model = PegasusXForConditionalGeneration.from_pretrained(key)
        self.tokenizer = PegasusTokenizerFast.from_pretrained(key)
