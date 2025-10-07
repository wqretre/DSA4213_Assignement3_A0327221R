from full_finetune import run_full_finetuning
from lora_finetune import run_lora_finetuning

if __name__ == "__main__":
    print("====== Running Full Fine-tuning on GoEmotions ======")
    run_full_finetuning()

    print("\n====== Running LoRA Fine-tuning on GoEmotions ======")
    run_lora_finetuning()

    print("\nAll experiments completed! Results and plots saved in ./results/")
