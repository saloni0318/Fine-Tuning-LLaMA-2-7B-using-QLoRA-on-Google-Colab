# FORCE NO MIXED PRECISION
import os
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# ================= CONFIG ================= #

model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "databricks/databricks-dolly-15k"

# -------- QLoRA -------- #
use_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = "float16"  
use_nested_quant = True

# -------- LoRA -------- #
lora_r = 8
lora_alpha = 32
lora_dropout = 0.05

# -------- Training -------- #
output_dir = "./results"
num_train_epochs = 1
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
learning_rate = 2e-4

logging_steps = 10
save_steps = 100

# ---------------------------------------- #

print("Loading dataset...")
dataset = load_dataset(dataset_name, split="train")

# Format dataset
def format_dolly(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Context:
{example['context']}

### Response:
{example['response']}"""
    }

dataset = dataset.map(format_dolly)

# BitsAndBytes config
compute_dtype = torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.use_cache = False

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA config
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)


training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,

    fp16=False,   
    bf16=False,   

    logging_steps=logging_steps,
    save_steps=save_steps,
    report_to="none",
)

def formatting_func(example):
    return example["text"]

print("Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_func,
    args=training_arguments,
)

print("Starting training")
trainer.train()

print("TRAINING FINISHED SUCCESSFULLY ")
