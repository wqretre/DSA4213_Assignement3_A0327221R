# DSA4213_Assignement3_A0327221R

This repository contains code for fine-tuning and parameter-efficient tuning (LoRA) of **BERT-base-uncased** on the **GoEmotions** dataset for multi-label emotion classification.  
It compares **Full Fine-tuning** and **LoRA** in terms of training efficiency and performance.

---

## Project Structure
```goemotions-finetune/
├── main.py           # One-click entry point for both experiments
├── full_finetune.py  # Full fine-tuning logic
├── lora_finetune.py  # LoRA fine-tuning logic
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── results/          # Saved models, logs, and loss curves
```

## Overview

| Item | Description |
|------|--------------|
| **Dataset** | [GoEmotions (simplified)](https://huggingface.co/datasets/go_emotions) |
| **Model** | `bert-base-uncased` |
| **Task** | Multi-label emotion classification |
| **Methods** | Full Fine-tuning and LoRA |
| **Metrics** | F1 score, Per-label Accuracy, Subset Accuracy, Hamming Loss |

---

## Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/wqretre/DSA4213_Assignement3_A0327221R.git
```
### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## Reproduce Experiments

### A simple entry point 
Run both full fine-tuning and LoRA experiments to reproduce result
```bash
python main.py
```
This will show the processes of training, plot training and validation loss curve, show the metrics of the test dataset and some labels of examples predicted by model.
