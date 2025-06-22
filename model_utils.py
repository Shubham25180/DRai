"""
Model Utilities for DrAI Medical Tutor
Handles AI model operations, prompts, and mock test functionality.
Provides comprehensive model management for both base and fine-tuned models.
"""

import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from typing import List, Dict, Any, Optional
import logging

# 1. Configure a logger for this module before any other code runs.
logger = logging.getLogger(__name__)

# 2. Check for PEFT library for fine-tuning support. This uses the logger.
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
    logger.info("PEFT library found. Fine-tuning features are enabled.")
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not found. Fine-tuning features will be disabled.")

class ModelManager:
    """
    Manages AI model operations including loading, inference, and mock test functionality.
    Supports both base BioMistral model and fine-tuned LoRA adapters.
    """
    def __init__(self):
        # Initialize model components
        self.tokenizer = None  # Tokenizer for text processing
        self.model = None  # Base model
        self.fine_tuned_model = None  # Fine-tuned model with LoRA
        # Load mock questions for testing
        self.mock_questions = self.load_mock_questions()
        logger.info("ModelManager initialized. Model and tokenizer are not yet loaded.")
        
    def load_base_model(self, model_name: str = "BioMistral/BioMistral-7B"):
        """
        Load the base BioMistral model, adapting to the available hardware.
        Uses 4-bit quantization on compatible GPUs, otherwise falls back to CPU.
        """
        logger.info(f"--- Starting Base Model Loading: '{model_name}' ---")
        
        # Check for CUDA GPU
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info("✅ CUDA-enabled GPU detected. Loading model with 4-bit quantization.")
            # Configure quantization for GPU
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": "auto"
            }
        else:
            logger.warning("⚠️ No CUDA-enabled GPU found. Loading model on CPU.")
            logger.warning("Performance will be significantly slower. AI features will have a delay.")
            # CPU configuration
            model_kwargs = {
                "device_map": "cpu",
                "torch_dtype": torch.float32  # Use full precision for CPU
            }

        try:
            logger.info(f"Step 1/3: Downloading and caching tokenizer for '{model_name}'.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("✅ Tokenizer loaded successfully.")

            if self.tokenizer.pad_token is None:
                logger.warning("Tokenizer does not have a pad token. Setting it to eos_token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Step 2/3: Downloading and caching model '{model_name}'. This may take time.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            device = "GPU" if use_gpu else "CPU"
            logger.info(f"✅ Model loaded successfully onto {device}.")

            logger.info("Step 3/3: Setting model to evaluation mode.")
            self.model.eval()
            logger.info("--- Base Model Loading Complete ---")
            return True
            
        except Exception as e:
            logger.critical(f"❌ FATAL ERROR during model loading: {e}")
            logger.critical("The application's AI features will not work.")
            return False
    
    def load_fine_tuned_model(self, adapter_path: str = "fine_tuning/output/final"):
        """
        Load fine-tuned LoRA adapter for enhanced medical knowledge.
        Requires PEFT library and a trained LoRA adapter.
        """
        if not PEFT_AVAILABLE:
            print("❌ PEFT not available. Cannot load fine-tuned model.")
            return False
            
        try:
            # Ensure base model is loaded first
            if self.model is None:
                print("⚠️ Base model not loaded. Loading base model first...")
                if not self.load_base_model():
                    print("❌ Failed to load base model. Cannot load fine-tuned model.")
                    return False

            # This check is for the linter, confirming self.model is not None.
            if self.model is None:
                print("❌ Unexpected error: Base model is still None after loading attempt.")
                return False

            print(f"🔄 Loading fine-tuned adapter from: {adapter_path}")
            # Load and merge LoRA adapter with base model
            self.fine_tuned_model = PeftModel.from_pretrained(self.model, adapter_path)
            print("✅ Fine-tuned model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading fine-tuned model: {e}")
            return False
    
    def generate_response(self, prompt: str, max_new_tokens: int = 200, use_fine_tuned: bool = True) -> str:
        """
        Generate response using the appropriate model (base or fine-tuned).
        Formats prompts for instruction-following and handles response extraction.
        """
        # Choose which model to use
        model_to_use = self.fine_tuned_model if use_fine_tuned and self.fine_tuned_model else self.model
        
        # Check if model is loaded
        if model_to_use is None or self.tokenizer is None:
            return "⚠️ Model not loaded. Please load the model first."
        
        try:
            # Format prompt for instruction-following models
            formatted_prompt = self.format_prompt(prompt)
            
            # Tokenize input with truncation
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
            # Move inputs to model device
            inputs = {k: v.to(model_to_use.device) for k, v in inputs.items()}
            
            # Generate response without computing gradients
            with torch.no_grad():
                outputs = model_to_use.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=0.7,  # Controls randomness
                    do_sample=True,  # Enable sampling
                )
            
            # Decode generated tokens to text
            response_ids = outputs[0]
            full_response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Extract only the response part (after "### Response:")
            if "### Response:" in full_response:
                response_only = full_response.split("### Response:")[1].strip()
            else:
                # Fallback: remove the original prompt
                response_only = full_response.replace(formatted_prompt, "").strip()
            
            # Return response or default message
            return response_only if response_only else "I apologize, but I couldn't generate a response. Please try again."
            
        except Exception as e:
            return f"❌ Error generating response: {e}"
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """
        Format prompt for instruction-following models.
        Creates structured prompts with instruction, input, and response sections.
        """
        prompt = f"### Instruction:\n{instruction}\n\n"
        if input_text:
            prompt += f"### Input:\n{input_text}\n\n"
        prompt += f"### Response:\n"
        return prompt
    
    def ask_doubt(self, question: str) -> str:
        """
        Handle doubt clearance queries.
        Formats medical questions for the AI model.
        """
        # Check if question is provided
        if not question.strip():
            return "Please enter your medical question."
        
        # Format question as instruction
        instruction = f"Q: {question}"
        return self.generate_response(instruction, max_new_tokens=300)
    
    def generate_notes(self, topic: str) -> str:
        """
        Generate notes for a given medical topic.
        Creates comprehensive, structured notes for NEET-PG preparation.
        """
        # Check if topic is provided
        if not topic.strip():
            return "Please enter a medical topic for notes generation."
        
        # Format topic as instruction for summarization
        instruction = f"Summarize: {topic}"
        return self.generate_response(instruction, max_new_tokens=400)
    
    def load_mock_questions(self) -> Dict[str, List[Dict]]:
        """
        Load mock questions from JSON file.
        Returns a dictionary of topics with their associated questions.
        """
        try:
            # Load questions from JSON file
            with open('mock_questions.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Warning: mock_questions.json not found. Using empty database.")
            return {}
    
    def get_available_topics(self) -> List[str]:
        """
        Get list of available topics for mock tests.
        Returns keys from the mock questions dictionary.
        """
        return list(self.mock_questions.keys())
    
    def get_mock_questions(self, topic: str, num_questions: int = 5) -> List[Dict]:
        """
        Get random questions for a specific topic.
        Ensures the number of questions doesn't exceed available questions.
        """
        # Check if topic exists
        if topic not in self.mock_questions:
            return []
        
        # Get available questions for the topic
        available_questions = self.mock_questions[topic]
        # Adjust number if not enough questions available
        if len(available_questions) < num_questions:
            num_questions = len(available_questions)
        
        # Return random sample of questions
        return random.sample(available_questions, num_questions)
    
    def evaluate_answers(self, questions: List[Dict], user_answers: List[str]) -> Dict[str, Any]:
        """
        Evaluate mock test answers and provide detailed feedback.
        Returns comprehensive results with explanations and score feedback.
        """
        # Check if number of answers matches number of questions
        if len(user_answers) != len(questions):
            return {"error": f"Expected {len(questions)} answers, got {len(user_answers)}."}
        
        correct = 0  # Counter for correct answers
        detailed_results = []  # List to store detailed results
        
        # Evaluate each question
        for i, (question, user_answer) in enumerate(zip(questions, user_answers), 1):
            correct_answer = question['correct_answer']
            is_correct = user_answer.upper() == correct_answer.upper()
            
            # Increment correct counter if answer is right
            if is_correct:
                correct += 1
            
            # Store detailed result for this question
            detailed_results.append({
                "question_number": i,
                "question": question['question'],
                "user_answer": user_answer.upper(),
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "explanation": question.get('explanation', '')
            })
        
        # Calculate score percentage
        score_percentage = (correct / len(questions)) * 100
        
        # Return comprehensive results
        return {
            "total_questions": len(questions),
            "correct_answers": correct,
            "score_percentage": score_percentage,
            "detailed_results": detailed_results,
            "feedback": self.get_score_feedback(score_percentage)
        }
    
    def get_score_feedback(self, score_percentage: float) -> str:
        """
        Get motivational feedback based on test score.
        Provides encouraging messages for different performance levels.
        """
        if score_percentage >= 90:
            return "🌟 Outstanding! You have excellent knowledge of this topic!"
        elif score_percentage >= 80:
            return "🎯 Great job! You have a solid understanding of this topic."
        elif score_percentage >= 70:
            return "👍 Good work! You're on the right track, keep studying!"
        elif score_percentage >= 60:
            return "📚 Not bad! Review the incorrect answers and try again."
        else:
            return "💪 Keep studying! Focus on the fundamentals and practice more."

# Create global model manager instance
model_manager = ModelManager()

# Convenience functions for external use
def ask_doubt(question: str) -> str:
    """
    Convenience function to ask a medical question.
    Uses the global model manager instance.
    """
    return model_manager.ask_doubt(question)

def generate_notes(topic: str) -> str:
    """
    Convenience function to generate notes for a topic.
    Uses the global model manager instance.
    """
    return model_manager.generate_notes(topic)

def get_available_topics() -> List[str]:
    """
    Convenience function to get available topics.
    Uses the global model manager instance.
    """
    return model_manager.get_available_topics()

def evaluate_answers(questions: List[Dict], user_answers: List[str]) -> Dict[str, Any]:
    """
    Convenience function to evaluate test answers.
    Uses the global model manager instance.
    """
    return model_manager.evaluate_answers(questions, user_answers)

def load_model_and_tokenizer(model_name="gpt2"):
    """
    Loads a pre-trained model and tokenizer from Hugging Face.

    This function handles the loading of both the model and its corresponding
    tokenizer. It includes detailed logging for each step of the process
    and robust error handling for common issues like network problems or
    missing model files.

    Args:
        model_name (str): The name of the pre-trained model to load,
                          as listed on the Hugging Face Model Hub.
                          Defaults to "gpt2".

    Returns:
        A tuple containing:
        - model (GPT2LMHeadModel): The loaded pre-trained model.
        - tokenizer (GPT2Tokenizer): The loaded tokenizer.
        Returns (None, None) if loading fails.
    """
    logger.info(f"Attempting to load model and tokenizer for '{model_name}'...")
    try:
        # Load the tokenizer
        logger.info(f"Downloading and caching tokenizer: '{model_name}'")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        logger.info("✅ Tokenizer loaded successfully.")

        # Load the pre-trained model
        logger.info(f"Downloading and caching model: '{model_name}'")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        logger.info("✅ Model loaded successfully.")

        # Set the model to evaluation mode
        model.eval()
        logger.info(f"Model '{model_name}' set to evaluation mode.")

        return model, tokenizer

    except OSError as e:
        logger.error(
            f"Model or tokenizer not found for '{model_name}'. "
            f"Please check the model name and your internet connection. Error: {e}"
        )
        return None, None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading the model: {e}"
        )
        return None, None


def generate_text(model, tokenizer, prompt, max_length=150, temperature=0.7):
    """
    Generates text using the provided model, tokenizer, and prompt.

    Args:
        model: The pre-trained language model.
        tokenizer: The tokenizer corresponding to the model.
        prompt (str): The text prompt to start generation from.
        max_length (int): The maximum length of the generated text.
        temperature (float): Controls the randomness of the generation.
                             Lower values are more deterministic.

    Returns:
        str: The generated text.
    """
    logger.info(f"Generating text for prompt (first 50 chars): '{prompt[:50]}...'")
    try:
        # Encode the prompt text into token IDs
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # Generate text using the model
        logger.info(f"Generating response with max_length={max_length} and temperature={temperature}")
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id  # Avoids warning
        )

        # Decode the generated token IDs back to a string
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("✅ Text generation successful.")
        return generated_text
    except Exception as e:
        logger.error(f"An error occurred during text generation: {e}")
        return "Error: Could not generate text." 