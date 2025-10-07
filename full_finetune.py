import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, hamming_loss

device = "cuda" if torch.cuda.is_available() else "cpu"


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


def encode_labels(examples, num_labels):
    labels = np.zeros((len(examples['labels']), num_labels), dtype=np.float32)
    for i, lab_list in enumerate(examples['labels']):
        for lab in lab_list:
            labels[i, lab] = 1.0
    examples['labels'] = labels.tolist()
    return examples


def run_full_finetuning():
    dataset = load_dataset("go_emotions", "simplified")
    label_names = dataset['train'].features['labels'].feature.names
    num_labels = len(label_names)

    encoded_dataset = dataset.map(lambda e: encode_labels(e, num_labels), batched=True)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    tokenized_dataset = encoded_dataset.map(preprocess, batched=True, remove_columns=['text'])
    train_dataset = tokenized_dataset['train']
    val_dataset = tokenized_dataset['validation']
    test_dataset = tokenized_dataset['test']

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        problem_type="multi_label_classification"
    ).to(device)

    training_args = TrainingArguments(
        output_dir="./results/full_finetune",
        evaluation_strategy="steps",
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
        compute_metrics=compute_metrics
    )

    print("Training full fine-tuning model...")
    trainer.train()
    model.save_pretrained("./results/full_finetune_model")

    print("Evaluating full fine-tuning model...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(test_metrics)

    # Plot train loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, color='blue')
    plt.xlabel("Step")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Curve - Full Fine-tuning")
    os.makedirs("./results", exist_ok=True)
    plt.savefig("./results/full_finetune_train_loss.png")
    plt.close()
    
    # Plot validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(eval_loss, color='orange')
    plt.xlabel("Step")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Curve - Full Fine-tuning")
    plt.savefig("./results/full_finetune_val_loss.png")
    plt.close()

    # Test sample predictions
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
        print(f"Sample {i+1}:")
        print("Text:", tokenizer.decode(sample['input_ids'], skip_special_tokens=True))
        print("True labels:", [label_names[j] for j, l in enumerate(sample['labels']) if l==1])
        print("Predicted labels:", pred_labels)
        print("-"*50)
