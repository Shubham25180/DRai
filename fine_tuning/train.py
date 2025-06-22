import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# --- Configuration ---
# Model
model_name = "mistralai/BioMistral-7B"
# Dataset
dataset_path = "../data/training_dataset.jsonl"
# Output directory for fine-tuned model
output_dir = "./output"
# LoRA configuration
lora_r = 8
lora_alpha = 16
lora_target_modules = ["q_proj", "v_proj"]
lora_bias = "none"
lora_task_type = "CAUSAL_LM"
# Training arguments
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
warmup_steps = 10
max_steps = 200 # Adjust as needed for your dataset size
learning_rate = 2e-4
fp16 = True # Use mixed precision training
save_steps = 50

# --- 1. Load Dataset ---
print(f"Loading dataset from: {dataset_path}")
data = load_dataset("json", data_files=dataset_path)
print(f"Dataset loaded: {data}")

# --- 2. Load Tokenizer and Model ---
print(f"Loading model: {model_name}")

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config, 
    device_map="auto"
)
print("Model loaded successfully.")

# --- 3. Prepare Model for LoRA ---
print("Preparing model for LoRA training...")
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=lora_r, 
    lora_alpha=lora_alpha, 
    target_modules=lora_target_modules, 
    bias=lora_bias, 
    task_type=lora_task_type
)

model = get_peft_model(model, peft_config)
print("LoRA model prepared.")
model.print_trainable_parameters()

# --- 4. Tokenize Data ---
print("Tokenizing dataset...")
def format_prompt(entry):
    # This function creates a structured prompt for the model
    # It follows a standard instruction-following format
    prompt = f"### Instruction:\n{entry['instruction']}\n\n"
    if entry.get('input'):
        prompt += f"### Input:\n{entry['input']}\n\n"
    prompt += f"### Response:\n{entry['output']}"
    return prompt

def tokenize(entry):
    # Get the formatted prompt
    prompt = format_prompt(entry)
    # Add the end-of-sentence token
    prompt_with_eos = prompt + tokenizer.eos_token
    # Tokenize the prompt
    result = tokenizer(
        prompt_with_eos,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    # Set the labels to be the same as the input IDs
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_data = data["train"].map(tokenize)
print("Dataset tokenized.")
print(f"Sample tokenized entry: {tokenized_data[0]}")

# --- 5. Set Up Training Arguments ---
print("Setting up training arguments...")
training_args = TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=warmup_steps,
    max_steps=max_steps,
    learning_rate=learning_rate,
    fp16=fp16,
    output_dir=output_dir,
    save_strategy="steps",
    save_steps=save_steps,
    logging_steps=10,
    report_to="tensorboard",
    optim="paged_adamw_8bit",
)

# --- 6. Initialize Trainer ---
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)

# --- 7. Start Training ---
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# --- 8. Save Final Model ---
final_model_path = os.path.join(output_dir, "final")
print(f"Saving final model to: {final_model_path}")
trainer.save_model(final_model_path)
print("Model saved.") 