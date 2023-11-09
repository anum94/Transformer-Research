import os
import json
import torch
import argparse
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

# for sanity
torch.cuda.empty_cache()
print("CUDA available: ", torch.cuda.is_available())

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
max_length = config["max_length"]
batch_size = config["batch_size"]
max_new_tokens = config["max_new_tokens"]

model_hf_key = config["model_hf_key"]
# for some models the tokenizer may not be available so we use a different one.
tokenizer_hf_key = {
    "decapoda-research/llama-13b-hf": "elinas/llama-7b-hf-transformers-4.29",
    "elinas/llama-7b-hf-transformers-4.29": "elinas/llama-7b-hf-transformers-4.29",
    "TheBloke/vicuna-13B-1.1-HF": "TheBloke/vicuna-13B-1.1-HF",
    "gpt2": "gpt2",
    "tiiuae/falcon-40b-instruct": "tiiuae/falcon-40b-instruct",
    "tiiuae/falcon-7b-instruct": "tiiuae/falcon-7b-instruct",
    "baichuan-inc/baichuan-7B": "baichuan-inc/baichuan-7B",
    "meta-llama/Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
}

model_key = {
    "decapoda-research/llama-13b-hf": "llama13b",
    "TheBloke/vicuna-13B-1.1-HF": "vicuna13b",
    "elinas/llama-7b-hf-transformers-4.29": "llama7b",
    "gpt2": "gpt2",
    "tiiuae/falcon-40b-instruct": "falcon40b",
    "tiiuae/falcon-7b-instruct": "falcon7b",
    "baichuan-inc/baichuan-7B": "baichuan7b",
    "meta-llama/Llama-2-7b-chat-hf": "llama2chat7b",
}


working_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(working_dir, "results")
exec_path = f"{model_key[model_hf_key]}-{dataset}"

if not os.path.exists(os.path.join(log_dir, exec_path)):
    os.makedirs(os.path.join(log_dir, exec_path))

outputs_path = os.path.join(log_dir, exec_path, "log.out")
metrics_path = os.path.join(log_dir, exec_path, "metrics.json")

logging.info(f"logging in {os.path.join(log_dir, exec_path)}")

hf_hub_cache_key: str = "TRANSFORMERS_CACHE"
hf_ds_cache_key: str = "HF_DATASETS_CACHE"
container_prefix: str = "/mnt/container"
hf_hub_cache_path: str = os.path.join(container_prefix, ".cache/huggingface/hub")
hf_ds_cache_path: str = os.path.join(container_prefix, ".cache/huggingface/datasets")

os.environ[hf_ds_cache_key] = hf_ds_cache_path
os.environ[hf_hub_cache_key] = hf_hub_cache_path


# we need to set the cache keys before loading transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from ds.supported import load_dataset
from metrics.rouge import Rouge

logging.info("tokenizing dataset")

# load/process ds
dataset = load_dataset(
    dataset=dataset,
    preview=preview,
    samples=samples,
    min_input_size=min_input_size,
)
dtest = dataset.get_split("test")


prefix = lambda batch: {
    "text": "You are an expert at summarization. Proceed to summarize the following text. TEXT: "
    + batch["text"],
    "summary": batch["summary"],
}
prefix_dtest = dtest.map(prefix)

# load model

model = AutoModelForCausalLM.from_pretrained(
    model_hf_key,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_hf_key[model_hf_key], trust_remote_code=True,
)

# tok_ds
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = model.generation_config.eos_token_id


def tokenize_dataset(data, padding="max_length", truncation=True, max_len=1024):
    # get tokens for SUMMARY: in shape torch.Size([len])
    suffix = tokenizer(" SUMMARY:", return_tensors="pt").input_ids.view(-1)
    l = suffix.shape[0]

    def fn(sample, _):
        tok = tokenizer(
            sample["text"],
            truncation=truncation,
            padding=padding,
            max_length=max_len,
            return_tensors="pt",
        ).input_ids

        # set last l tokens to suffix
        tok[:, -l:] = suffix

        return {"input_ids": tok}

    return data.map(fn, data)


tok_dtest = tokenize_dataset(
    prefix_dtest,
    padding="max_length",
    truncation=True,
    max_len=max_length - max_new_tokens,
)


logging.info("running inference")


# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

gc = GenerationConfig(
    num_beams=1,
    max_new_tokens=max_new_tokens,
    do_sample=True,
)

# create dataloader
data_loader = DataLoader(tok_dtest, batch_size=batch_size)

# initialize lists to hold model outputs and failures
outputs = []

for batch in tqdm(data_loader):
    # move batch to device after removing sample outer dimension
    input_ids = batch["input_ids"].squeeze(1).to(device)
    # generate summary
    generated_ids = model.generate(input_ids=input_ids, generation_config=gc)

    # move generated ids back to cpu and decode
    generated_ids = generated_ids.to("cpu")
    dec_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # break after SUMMARY: and remove leading/trailing whitespace
    for o in dec_outputs:
        split = o.split("SUMMARY:")
        outputs.append(split[1].strip())

rouge = Rouge()
metrics = rouge.compute(predictions=outputs, references=dtest["summary"])

# logs

with open(metrics_path, "w") as fp:
    json.dump(metrics, fp)

with open(outputs_path, "w") as fp:
    json.dump(outputs, fp)

print(metrics_path)
print(outputs_path)
print (len(outputs))
