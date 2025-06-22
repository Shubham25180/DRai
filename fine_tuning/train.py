"""
Fine-tuning Script for DrAI Medical Tutor
Trains LoRA adapters on BioMistral-7B for enhanced medical knowledge.
Uses 4-bit quantization and LoRA for efficient fine-tuning.
"""

# Import required libraries
import os  # For file operations
import torch  # For PyTorch operations
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer  # For model training
from datasets import load_dataset  # For dataset loading
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # For LoRA fine-tuning
from transformers import BitsAndBytesConfig  # For quantization

# --- Configuration ---
# Model configuration
model_name = "mistralai/BioMistral-7B"
# Dataset configuration
dataset_path = "../data/training_dataset.jsonl"
# Output directory for fine-tuned model
output_dir = "./output"
# LoRA configuration parameters
lora_r = 8  # Rank of LoRA matrices
lora_alpha = 16  # Scaling factor for LoRA
lora_target_modules = ["q_proj", "v_proj"]  # Target modules for LoRA
lora_bias = "none"  # Bias handling in LoRA
lora_task_type = "CAUSAL_LM"  # Task type for LoRA
# Training arguments
per_device_train_batch_size = 2  # Batch size per device
gradient_accumulation_steps = 4  # Gradient accumulation steps
warmup_steps = 10  # Warmup steps for learning rate
max_steps = 200  # Maximum training steps (adjust for dataset size)
learning_rate = 2e-4  # Learning rate for training
fp16 = True  # Use mixed precision training
save_steps = 50  # Save model every N steps

# --- 1. Load Dataset ---
print(f"Loading dataset from: {dataset_path}")
# Load dataset from JSONL file
data = load_dataset("json", data_files=dataset_path)
print(f"Dataset loaded: {data}")

# --- 2. Load Tokenizer and Model ---
print(f"Loading model: {model_name}")

# Configure 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config, 
    device_map="auto"
)
print("Model loaded successfully.")

# --- 3. Prepare Model for LoRA ---
print("Preparing model for LoRA training...")
# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA parameters
peft_config = LoraConfig(
    r=lora_r,  # LoRA rank
    lora_alpha=lora_alpha,  # LoRA scaling factor
    target_modules=lora_target_modules,  # Target modules for LoRA
    bias=lora_bias,  # Bias handling
    task_type=lora_task_type  # Task type
)

# Apply LoRA configuration to model
model = get_peft_model(model, peft_config)
print("LoRA model prepared.")
# Print trainable parameters information
model.print_trainable_parameters()

# --- 4. Tokenize Data ---
print("Tokenizing dataset...")

def format_prompt(entry):
    """
    Create a structured prompt for the model.
    Follows a standard instruction-following format for training.
    """
    # Create structured prompt with instruction, input, and response
    prompt = f"### Instruction:\n{entry['instruction']}\n\n"
    if entry.get('input'):
        prompt += f"### Input:\n{entry['input']}\n\n"
    prompt += f"### Response:\n{entry['output']}"
    return prompt

def tokenize(entry):
    """
    Tokenize a single entry for training.
    Handles prompt formatting and tokenization with proper labels.
    """
    # Get the formatted prompt
    prompt = format_prompt(entry)
    # Add the end-of-sentence token
    prompt_with_eos = prompt + tokenizer.eos_token
    # Tokenize the prompt with truncation and padding
    result = tokenizer(
        prompt_with_eos,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    # Set the labels to be the same as the input IDs for causal language modeling
    result["labels"] = result["input_ids"].copy()
    return result

# Apply tokenization to the entire dataset
tokenized_data = data["train"].map(tokenize)
print("Dataset tokenized.")
print(f"Sample tokenized entry: {tokenized_data[0]}")

# --- 5. Set Up Training Arguments ---
print("Setting up training arguments...")
# Configure training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=warmup_steps,
    max_steps=max_steps,
    learning_rate=learning_rate,
    fp16=fp16,  # Use mixed precision training
    output_dir=output_dir,
    save_strategy="steps",
    save_steps=save_steps,
    logging_steps=10,  # Log every 10 steps
    report_to="tensorboard",  # Use TensorBoard for logging
    optim="paged_adamw_8bit",  # Use 8-bit optimizer for memory efficiency
)

# --- 6. Initialize Trainer ---
print("Initializing Trainer...")
# Create trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)

# --- 7. Start Training ---
print("Starting fine-tuning...")
# Begin the training process
trainer.train()
print("Fine-tuning complete.")

# --- 8. Save Final Model ---
final_model_path = os.path.join(output_dir, "final")
print(f"Saving final model to: {final_model_path}")
# Save the trained model
trainer.save_model(final_model_path)
print("Model saved.") 