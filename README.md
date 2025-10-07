# DSA4213_Assignement3_A0327221R
# BERT Fine-tuning and LoRA on GoEmotions

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

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/goemotions-finetune.git
cd goemotions-finetune
