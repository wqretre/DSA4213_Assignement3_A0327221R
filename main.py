import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from peft import LoraConfig, get_peft_model, TaskType

device = "cuda" if torch.cuda.is_available() else "cpu"

# load dataset
dataset = load_dataset("go_emotions", "simplified")

label_names = dataset['train'].features['labels'].feature.names
num_labels = len(label_names)

# convert labels to one-hot encode
def encode_labels(examples):
    labels = np.zeros((len(examples['labels']), num_labels), dtype=np.float32)
    for i, lab_list in enumerate(examples['labels']):
        for lab in lab_list:
            labels[i, lab] = 1.0
    examples['labels'] = labels.tolist()
    return examples


encoded_dataset = dataset.map(encode_labels, batched=True)

# encode text
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def preprocess(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)


tokenizer_dataset = encoded_dataset.map(preprocess, batched=True, remove_columns=['text'])

train_dataset = tokenizer_dataset['train']
val_dataset = tokenizer_dataset['validation']
test_dataset = tokenizer_dataset['test']

# load evaluate metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.35).numpy().astype(int)
    labels = labels.astype(int)

    f1 = f1_score(labels, predictions, average='micro')

    per_label_acc = np.mean(np.sum(predictions == labels, axis=0) / labels.shape[0])

    subset_acc = np.mean(np.all(predictions == labels, axis=1))

    ham_loss = hamming_loss(labels, predictions)

    return {
        "f1": f1,
        "per_label_accuracy": per_label_acc,
        "subset_accuracy": subset_acc,
        "hamming_loss": ham_loss
    }

def data_collator(batch):
    return {
        "input_ids": torch.tensor([f["input_ids"] for f in batch], dtype=torch.long),
        "attention_mask": torch.tensor([f["attention_mask"] for f in batch], dtype=torch.long),
        "labels": torch.tensor([f["labels"] for f in batch], dtype=torch.float),
    }

# full fine_tuning
training_args = TrainingArguments(
    output_dir="./full_finetune_gomotions",
    eval_strategy="steps",
    eval_steps=400,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=20,
    save_strategy="steps",
    save_steps=2000,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    fp16=torch.cuda.is_available(),   
    dataloader_pin_memory=True,
    dataloader_num_workers=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# train
train_result = trainer.train()
metrics = train_result.metrics
trainer.save_model()

# plot the loss curve
logs = trainer.state.log_history
train_loss = [x["loss"] for x in logs if "loss" in x]
eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]

# train loss
plt.figure(figsize=(8, 5))
plt.plot(range(len(train_loss)), train_loss, color='blue')
plt.xlabel("Step")
plt.ylabel("Train Loss")
plt.title("Training Loss Curve-Full fine-tuning")

#validation loss
plt.figure(figsize=(8, 5))
plt.plot(range(len(eval_loss)), eval_loss, color='orange')
plt.xlabel("Step")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Curve-Full fine-tuning")

test_metrics = trainer.evaluate(eval_dataset=test_dataset)
print(f"F1 score:{test_metrics['eval_f1']}\nper label accuracy:{test_metrics['eval_per_label_accuracy']}\nsubset accuracy:{test_metrics['eval_subset_accuracy']}\nhamming loss:{test_metrics['eval_hamming_loss']}")

# test sample
num_samples_to_show = 10
model.eval()
for i in range(num_samples_to_show):
    sample = test_dataset[i]
    input_ids = torch.tensor([sample['input_ids']], dtype=torch.long).to(device)
    attention_mask = torch.tensor([sample['attention_mask']], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        pred_labels = [label_names[j] for j, p in enumerate(probs) if p > 0.35]
    print(f"Sample {i + 1}:")
    print("Text", tokenizer.decode(sample['input_ids'], skip_special_tokens=True))
    print("True labels:", [label_names[j] for j, l in enumerate(sample['labels']) if l == 1])
    print("predicted labels:", pred_labels)
    print("-" * 50)

# load model
model_lora = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                           num_labels=num_labels,
                                                           problem_type="multi_label_classification").to(device)

# configure lora
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,  # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]  
)

model_lora = get_peft_model(model_lora, lora_config)
model_lora.print_trainable_parameters() 

# LoRA
training_args_lora = TrainingArguments(
    output_dir="./lora_gomotions",
    eval_strategy="steps",
    eval_steps=400,
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=20,
    save_strategy="steps",
    save_steps=2000,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=True,
    dataloader_num_workers=2
)

trainer_lora = Trainer(
    model=model_lora,
    args=training_args_lora,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

train_result_lora = trainer_lora.train()
metrics_lora = train_result_lora.metrics
trainer_lora.save_model()

# plot the loss curve
logs_lora = trainer_lora.state.log_history
train_loss_lora = [x["loss"] for x in logs_lora if "loss" in x]
eval_loss_lora = [x["eval_loss"] for x in logs_lora if "eval_loss" in x]

# train loss
plt.figure(figsize=(8, 5))
plt.plot(range(len(train_loss_lora)), train_loss_lora, color='blue')
plt.xlabel("Step")
plt.ylabel("Train Loss")
plt.title("Training Loss Curve-LoRA")

#validation loss
plt.figure(figsize=(8, 5))
plt.plot(range(len(eval_loss)), eval_loss, color='orange')
plt.xlabel("Step")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Curve-LoRA")

test_metrics_lora = trainer_lora.evaluate(eval_dataset=test_dataset)
print(f"F1 score:{test_metrics_lora['eval_f1']}\nper label accuracy:{test_metrics_lora['eval_per_label_accuracy']}\nsubset accuracy:{test_metrics_lora['eval_subset_accuracy']}\nhamming loss:{test_metrics_lora['eval_hamming_loss']}")

# test sample
num_samples_to_show = 10
model_lora.eval()
for i in range(num_samples_to_show):
    sample = test_dataset[i]
    input_ids = torch.tensor([sample['input_ids']], dtype=torch.long).to(device)
    attention_mask = torch.tensor([sample['attention_mask']], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model_lora(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        pred_labels = [label_names[j] for j, p in enumerate(probs) if p > 0.35]
    print(f"Sample {i + 1}:")
    print("Text", tokenizer.decode(sample['input_ids'], skip_special_tokens=True))
    print("True labels:", [label_names[j] for j, l in enumerate(sample['labels']) if l == 1])
    print("predicted labels:", pred_labels)
    print("-" * 50)
