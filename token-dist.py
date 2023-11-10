import json
import logging
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

# read the json file
with open("llm.json", "r") as f:
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

# we need to set the cache keys before loading transformers
from transformers import AutoTokenizer
from ds.supported import load_dataset

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


tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_hf_key[model_hf_key], trust_remote_code=True,
)

# tok_ds
tokenizer.pad_token = tokenizer.eos_token


def get_ds_distribution(data, mode = "text"):

    # get a list of indices that contain samples with text tokens

    def fn(sample):
        tok = tokenizer(
            sample[mode],
            return_tensors="pt",
        ).input_ids
        return {"num_tok" : tok.shape[1]}

    data = data.map(fn)
    tok_dist = [int(num_tok) for num_tok in list(data["num_tok"])]

    # get a subset of the data using those indices and return it

    return tok_dist


text_token_dist = get_ds_distribution(
        prefix_dtest)


a = np.array(text_token_dist)

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
bins = np.arange(0,np.max(a),10)
ax.hist(a, bins = bins)

# Show plot
plt.xlabel("# tokens" )
plt.ylabel("# samples")
file_name = "{}_{}_text.png".format(dataset.ds_name, model_hf_key.split("/")[1])
plt.savefig(file_name, bbox_inches='tight')
plt.show()

logging.FileHandler("logs/{}.txt".format(dataset.ds_name))

counter = Counter(a)
#print (counter.most_common(20))
print ("Text Distribution")
print ("Longest: ", max(a))
print ("Shortest: ", min (a))
print ("Average: ", a.mean())


filter = 18000
text_token_dist = [tok for tok in text_token_dist if tok < filter]
a = np.array(text_token_dist)

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
bins = np.arange(0,np.max(a),10)
ax.hist(a, bins = bins)

# Show plot
plt.xlabel("# tokens" )
plt.ylabel("# samples")
file_name = "{}_{}_{}_dist.png".format(dataset.ds_name, filter, model_hf_key.split("/")[1])
plt.savefig(file_name, bbox_inches='tight')
plt.show()



# -----------------------------------------------------------------------------------------------------------------------------#
#plot distribution of summaries.
print ("Summary distribution")

summ_token_dist = get_ds_distribution(
        prefix_dtest, "summary")


a = np.array(summ_token_dist)

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
bins = np.arange(0,np.max(a),10)
ax.hist(a, bins = bins)

# Show plot
plt.xlabel("# tokens" )
plt.ylabel("# samples")
file_name = "{}_{}_summ.png".format(dataset.ds_name, model_hf_key.split("/")[1])
plt.savefig(file_name, bbox_inches='tight')
plt.show()

counter = Counter(a)
#print (counter.most_common(20))
print ("Longest: ", max(a))
print ("Shortest: ", min (a))
print ("Average: ", a.mean())

