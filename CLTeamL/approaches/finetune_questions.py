import ast

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq, \
    AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Trainer


def prepare_for_model(data):
    # Preprocess
    knowledge = ["SUMMARIZE: " + "\n\t".join(ast.literal_eval(sample)) for sample in data["ref_knowledge"]]
    model_inputs = tokenizer(knowledge, max_length=512, truncation=True)

    summaries = [sample if sample else 'This was fun' for sample in data["ref_response_summary"]]
    labels = tokenizer(summaries, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = bleu.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


# parameters
model_name = "t5-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classes needed (tokenizer, model, metrics)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_config = AutoConfig.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=model_config).to(device)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)
bleu = evaluate.load("bleu")

# Load dataset
dataset = load_dataset("csv", data_files=f"./../data_analysis/output/analysis_val.csv")
dataset = dataset["train"].train_test_split(test_size=0.2)

# Preprocess
tokenized_dataset = dataset.map(prepare_for_model, batched=True)

# Configure Parameters for training
batch_size = 8
epochs = 4
logging_steps = len(dataset["train"]) // batch_size
training_args = Seq2SeqTrainingArguments(output_dir=f"./analysis/models/{model_name}",
                                         evaluation_strategy="epoch",
                                         learning_rate=2e-5,
                                         per_device_train_batch_size=batch_size,
                                         per_device_eval_batch_size=batch_size,
                                         weight_decay=0.01,
                                         save_total_limit=3,
                                         num_train_epochs=epochs,
                                         predict_with_generate=True,
                                         fp16=True,
                                         push_to_hub=False,
                                         disable_tqdm=False,
                                         logging_steps=logging_steps,
                                         log_level="error")
# Train Model
trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=tokenized_dataset["train"],
                  eval_dataset=tokenized_dataset["test"],
                  tokenizer=tokenizer,
                  data_collator=data_collator,
                  compute_metrics=compute_metrics)

trainer.train()

# Save Trained Model
trainer.save_model(f"./analysis/models/{model_name}")
