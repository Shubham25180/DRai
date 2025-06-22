"""
Fine-tuned Model Inference Script for DrAI Medical Tutor
Loads and runs inference with fine-tuned LoRA adapters for enhanced medical responses.
"""

# Import required libraries
import torch  # For PyTorch operations
from transformers import AutoTokenizer, AutoModelForCausalLM  # For Hugging Face models
from peft import PeftModel  # For LoRA adapter loading

# --- Configuration ---
# Base model configuration
base_model_name = "mistralai/BioMistral-7B"
# LoRA adapter path (from fine-tuning output)
adapter_path = "./output/final" 

# --- 1. Load Tokenizer and Base Model ---
print(f"Loading base model: {base_model_name}")
# Load tokenizer for text processing
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# Load base model with 16-bit precision for memory efficiency
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Base model loaded.")

# --- 2. Load and Merge LoRA Adapter ---
print(f"Loading LoRA adapter from: {adapter_path}")
# This merges the LoRA weights into the base model for inference
model = PeftModel.from_pretrained(base_model, adapter_path)
print("LoRA adapter loaded and merged.")

# --- 3. Prepare for Inference ---
model.eval()  # Set the model to evaluation mode for inference

def format_prompt(instruction, input_text=None):
    """
    Format the prompt for inference in the same way as training.
    Creates structured prompts with instruction, input, and response sections.
    """
    prompt = f"### Instruction:\n{instruction}\n\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n\n"
    prompt += f"### Response:\n"
    return prompt

def generate_response(instruction, input_text=None, max_new_tokens=256):
    """
    Generate a response from the fine-tuned model.
    Handles prompt formatting, tokenization, and response generation.
    """
    # Format the prompt for the model
    prompt = format_prompt(instruction, input_text)
    
    # Tokenize the input text
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response without computing gradients
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.7,  # Controls randomness in generation
            do_sample=True,  # Enable sampling instead of greedy decoding
        )
    
    # Decode generated tokens to text
    response_ids = outputs[0]
    full_response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    # Extract only the response part (after "### Response:")
    response_only = full_response.split("### Response:")[1].strip()
    
    return response_only

# --- 4. Run Inference Examples ---
if __name__ == "__main__":
    print("\n--- Running Inference ---")
    
    # Example 1: Q&A - Test medical question answering
    instruction1 = "Q: What are the key differences between Type 1 and Type 2 Diabetes?"
    print(f"\nInstruction: {instruction1}")
    response1 = generate_response(instruction1)
    print(f"Response:\n{response1}")

    # Example 2: Summarization - Test medical concept summarization
    instruction2 = "Summarize the mechanism of action for Penicillin."
    print(f"\nInstruction: {instruction2}")
    response2 = generate_response(instruction2)
    print(f"Response:\n{response2}")

    # Example 3: Clinical Case - Test clinical reasoning
    instruction3 = "Clinical Case: A 30-year-old patient has a blood pressure of 150/95 mmHg on three separate occasions. What is the diagnosis and initial management?"
    print(f"\nInstruction: {instruction3}")
    response3 = generate_response(instruction3)
    print(f"Response:\n{response3}")

    print("\n--- Inference Complete ---") 