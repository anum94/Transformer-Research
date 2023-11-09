import os
import json
import argparse
import logging
from tqdm import tqdm


# create argument parser
parser = argparse.ArgumentParser()

# add --config argument
parser.add_argument(
    "--config", type=str, required=True, help="Path to the configuration JSON file"
)

# parse arguments
args = parser.parse_args()

# read the json file
with open(args.config, "r") as f:
    config = json.load(f)

# parse args
dataset = config["dataset"]
preview = config["preview"]
samples = config["samples"]
min_input_size = config["min_input_size"]
batch_size = config["batch_size"]
enc_max_len = config["max_length"]
gen_max_len = config["max_new_tokens"]
model_key = config["model_hf_key"]


repetition_penalty = 1.0
length_penalty = 1.0
padding = "max_length"
truncation = True

model_hf_key = {
    "bigbirdpegasus": "twigs/bigbird-pegasus-large",
    "pegasusx": "google/pegasus-x-large",
    "bart": "facebook/bart-large"
}


# for some models the tokenizer may not be available so we use a different one.

working_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(working_dir, "results")
exec_path = f"{model_key}-{dataset}"
log_path = os.path.join(log_dir, exec_path)

if not os.path.exists(log_path):
    os.makedirs(log_path)

outputs_path = os.path.join(log_path, "log.out")
metrics_path = os.path.join(log_path, "metrics.json")

logging.info(f"logging in {log_path}")

hf_hub_cache_key: str = "TRANSFORMERS_CACHE"
hf_ds_cache_key: str = "HF_DATASETS_CACHE"
container_prefix: str = "/mnt/container"
hf_hub_cache_path: str = os.path.join(container_prefix, ".cache/huggingface/hub")
hf_ds_cache_path: str = os.path.join(container_prefix, ".cache/huggingface/datasets")

os.environ[hf_ds_cache_key] = hf_ds_cache_path
os.environ[hf_hub_cache_key] = hf_hub_cache_path
# we need to set the cache keys before loading transformers

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from ds.supported import load_dataset
from metrics.rouge import Rouge


# load/process ds
dataset = load_dataset(
    dataset=dataset,
    preview=preview,
    samples=samples,
    min_input_size=min_input_size,
)
ds = dataset.get_split("test")

model = AutoModelForSeq2SeqLM.from_pretrained(model_hf_key[model_key])
tokenizer = AutoTokenizer.from_pretrained(model_hf_key[model_key])


def tokenize_dataset(data, padding=padding, truncation=truncation, max_len=enc_max_len):
    tok = tokenizer(
        data["text"],
        truncation=truncation,
        padding=padding,
        max_length=max_len,
        return_tensors="pt",
    )

    return {"input_ids": tok.input_ids.squeeze()}


tok_ds = ds.map(tokenize_dataset)


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=padding,
    max_length=enc_max_len,
)

args = Seq2SeqTrainingArguments(
    output_dir=log_path,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    report_to="all",
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    args=args,
)


out = trainer.predict(
    test_dataset=tok_ds,
    metric_key_prefix="",
    max_length=gen_max_len,
    num_beams=1,
)

dec_out = tokenizer.batch_decode(
    out.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

with open(os.path.join(log_path, "sentences.out"), "w") as f:
    json.dump(dec_out, f)

rouge = Rouge()
res = rouge.compute(predictions=dec_out, references=ds["summary"])

with open(os.path.join(log_path, "metrics.json"), "w") as f:
    json.dump(res, f)
