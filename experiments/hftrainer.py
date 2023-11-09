from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from metrics.rouge import Rouge
from models.bigbirdpegasus import BigBirdPegasus
from ds.arxiv import Arxiv

ds = Arxiv(True)
train_ds = ds.get_dataset('train')
eval_ds = ds.get_dataset('validation')


model = BigBirdPegasus()
model.load(model.models['arxiv'])
tok = model.tokenizer


metric = Rouge()

enc_max_len = 512
num_train_epochs = 1


def tok_ds(data):

    def preprocess_fn(examples):
        inputs = examples['text']
        targets = examples['summary']

        model_inputs = tok(
            inputs, max_length=enc_max_len, padding="max_length", truncation=True, return_tensors="pt")
        model_inputs['labels'] = tok(
            targets, max_length=enc_max_len, padding="max_length", truncation=True, return_tensors="pt")['input_ids']

        return model_inputs

    return data.map(preprocess_fn, remove_columns=['text', 'summary'], batched=True)


train_ds = tok_ds(train_ds)
eval_ds = tok_ds(eval_ds)


def compute_metrics(r):
    preds, labels = r[0], r[1]
    return metric.compute(preds, labels)


eval_steps = max(len(train_ds) * 0.2 // 1, 1) 


args = Seq2SeqTrainingArguments(
    output_dir="/home/twigs/tum/transformer-research/testing",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=num_train_epochs,
    logging_strategy="steps",
    logging_steps=eval_steps,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=eval_steps,
    save_total_limit=2,
    metric_for_best_model="rouge",
    load_best_model_at_end=True
)

trainer = Seq2SeqTrainer(
    model=model.model,
    tokenizer=model.tokenizer,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds)


trainer.train()
