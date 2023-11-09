from transformers import PegasusTokenizerFast, BigBirdPegasusForConditionalGeneration
from models.hfmodel import HFModel


class BigBirdPegasus(HFModel):
    name = "bigbirdpegasus"  # framework name
    hf_upload_prefix = (
        "bigbird-pegasus-large"  # prefix according to the other models on hf
    )
    model: BigBirdPegasusForConditionalGeneration = None  # model itself
    tokenizer: PegasusTokenizerFast = None  # tokenizer
    default_key = "twigs/bigbird-pegasus-large"  # name of the pretrained model on hf

    # relevant models
    # "arxiv": "google/bigbird-pegasus-large-arxiv",
    # "pubmed": "google/bigbird-pegasus-large-pubmed",
    # "bigpatent": "google/bigbird-pegasus-large-bigpatent",

    def load(self, key: str) -> None:
        self.model = BigBirdPegasusForConditionalGeneration.from_pretrained(key)
        self.tokenizer = PegasusTokenizerFast.from_pretrained(key)
