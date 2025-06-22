"""
DrAI Medical Tutor - Main Application
A comprehensive AI-powered medical education platform for NEET-PG preparation
"""

# Import the Gradio library for creating web interfaces
import gradio as gr
# Import JSON library for handling JSON data files
import json
# Import random library for generating random selections
import random
# Import typing hints for better code documentation
from typing import List, Dict, Any
# Import PyTorch for deep learning operations
import torch
# Import Hugging Face transformers for AI model operations
from transformers import AutoTokenizer, AutoModelForCausalLM
# Import utility functions from our custom utils module
from utils import (
    create_doubt_prompt, create_notes_prompt, get_motivational_quote, 
    create_mock_test_prompt, setup_global_logger
)
from model_utils import ModelManager
import logging
import sys

# ======================================================================================
# GLOBAL SETUP
# ======================================================================================

# 1. Configure the global logger. This is the first and most important step.
setup_global_logger()
logger = logging.getLogger(__name__)

# 2. Define application constants.
logger.info("Defining application constants.")
MOCK_QUESTIONS_FILE = "mock_questions.json"

# 3. Initialize the AI ModelManager. This will trigger the model loading.
logger.info("--- Initializing ModelManager. This will begin the AI model loading process. ---")
model_manager = ModelManager()
model_loaded = model_manager.load_base_model() # This loads BioMistral-7B by default.

if not model_loaded:
    logger.critical("Failed to load the AI model. The application's AI features will be unavailable.")
else:
    logger.info("--- ModelManager initialization complete. ---")

# ======================================================================================
# DATA LOADING FUNCTION
# ======================================================================================

def load_mock_questions() -> list:
    """Loads and validates mock test questions from the JSON file."""
    logger.info(f"Attempting to load mock questions from '{MOCK_QUESTIONS_FILE}'.")
    try:
        with open(MOCK_QUESTIONS_FILE, 'r', encoding="utf-8") as f:
            data = json.load(f)
            
        # Flatten the categorized questions into a single list
        questions = []
        for category, category_questions in data.items():
            for question in category_questions:
                # Ensure the question has the expected structure
                if all(key in question for key in ["question", "options", "correct_answer"]):
                    questions.append(question)
                else:
                    logger.warning(f"Skipping malformed question in {category}: {question}")
            
        logger.info(f"✅ Successfully loaded {len(questions)} questions from {len(data)} categories.")
        return questions
    except FileNotFoundError:
        logger.error(f"The file '{MOCK_QUESTIONS_FILE}' was not found.")
        return []
    except json.JSONDecodeError:
        logger.error(f"The file '{MOCK_QUESTIONS_FILE}' is corrupted and not valid JSON.")
        return []

mock_questions_db = load_mock_questions()

# ======================================================================================
# CORE LOGIC FUNCTIONS (tied to UI components)
# ======================================================================================

def handle_doubt_clearance(question: str, context: str) -> str:
    """Logic for the 'AI-Powered Doubt Clearance' feature."""
    logger.info("--- 'Doubt Clearance' button clicked. ---")
    if not model_loaded:
        logger.error("Doubt clearance aborted: The AI model is not loaded.")
        return "⚠️ **Model Error**: AI features are unavailable. Please check the console logs."
    if not question or not question.strip():
        logger.warning("Request is empty. Asking user for a question.")
        return "Please enter a medical question to get an AI-powered explanation."

    prompt = create_doubt_prompt(question, context)
    return model_manager.generate_response(prompt)

def handle_notes_generation(topic: str) -> str:
    """Logic for the 'Smart Notes Generator' feature."""
    logger.info(f"--- 'Generate Notes' button clicked for topic: '{topic}'. ---")
    if not model_loaded:
        logger.error("Notes generation aborted: The AI model is not loaded.")
        return "⚠️ **Model Error**: AI features are unavailable. Please check the console logs."
    if not topic or not topic.strip():
        logger.warning("Request is empty. Asking user for a topic.")
        return "Please enter a topic to generate your smart study notes."

    prompt = create_notes_prompt(topic)
    return model_manager.generate_response(prompt, max_new_tokens=600)

def handle_start_mock_test(num_questions_str: str):
    """Prepares and displays the UI for a new mock test."""
    logger.info(f"--- 'Start Mock Test' button clicked with setting: '{num_questions_str}' questions. ---")
    
    if not mock_questions_db:
        logger.error("Cannot start test: The question database is empty or failed to load.")
        raise gr.Error("Mock Test Unavailable: Could not load the question database. Check logs.")

    num_questions = int(num_questions_str)
    logger.info(f"Preparing a test with {num_questions} questions.")
    selected_questions = random.sample(mock_questions_db, num_questions)
    
    test_state = {"questions": selected_questions}
    
    ui_updates = []
    # Un-hide and populate the question components
    for q in selected_questions:
        ui_updates.append(gr.update(label=q["question"], visible=True))
        # Convert options dictionary to formatted list for radio buttons
        formatted_options = [f"{key}: {value}" for key, value in q["options"].items()]
        ui_updates.append(gr.update(choices=formatted_options, value=None, visible=True))

    # Hide any unused question components
    for _ in range(5 - num_questions):
        ui_updates.append(gr.update(visible=False))
        ui_updates.append(gr.update(visible=False))
        
    # Show the submit button
    ui_updates.append(gr.update(visible=True))
    
    logger.info("✅ UI updated for the new mock test.")
    return ui_updates + [test_state]

def handle_submit_mock_test(test_state: dict, *answers: str) -> str:
    """Evaluates the submitted mock test and returns the score."""
    logger.info("--- 'Submit Test' button clicked. ---")
    if not test_state or "questions" not in test_state:
        logger.error("Evaluation failed: The test state is invalid or missing.")
        return "Error: Could not grade the test. Please start a new one."

    total_questions = len(test_state["questions"])
    correct_answers = 0
    unanswered_questions = 0
    logger.info(f"Evaluating {total_questions} answers...")
    
    result_details = []
    for i, q_data in enumerate(test_state["questions"]):
        user_answer = answers[i]
        
        # Check if answer was provided
        if not user_answer:
            unanswered_questions += 1
            result_details.append(f"Q{i+1}: ❌ Not answered")
            continue
            
        # Extract the answer key from the formatted string (e.g., "A: Coronary artery spasm" -> "A")
        if ":" in user_answer:
            user_answer = user_answer.split(":")[0].strip()
        
        correct_answer = q_data["correct_answer"]
        is_correct = user_answer == correct_answer
        
        if is_correct:
            correct_answers += 1
            result_details.append(f"Q{i+1}: ✅ Correct ({user_answer})")
        else:
            result_details.append(f"Q{i+1}: ❌ Wrong ({user_answer}) - Correct: {correct_answer}")
            
        logger.info(f"Q{i+1}: User answered '{user_answer}', Correct is '{correct_answer}'. ({'Correct' if is_correct else 'Incorrect'})")
    
    score = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
    # Build result text
    result_text = f"**Final Score: {correct_answers} out of {total_questions} ({score:.2f}%)**\n\n"
    if unanswered_questions > 0:
        result_text += f"⚠️ {unanswered_questions} question(s) not answered\n\n"
    
    result_text += "**Detailed Results:**\n" + "\n".join(result_details)
    
    logger.info(f"✅ Test evaluation complete. Final Score: {score:.2f}%")
    return result_text

# ======================================================================================
# GRADIO UI DEFINITION AND LAUNCH
# ======================================================================================

def create_gradio_interface():
    """Builds the full Gradio interface and connects logic to components."""
    logger.info("--- Building the Gradio User Interface. ---")
    
    with gr.Blocks(title="Dr. AI Medical Tutor") as demo:
        gr.Markdown("# 🧠 Dr. AI Medical Tutor for NEET-PG")

        with gr.Tabs():
            # Tab 1: Doubt Clearance
            with gr.TabItem("❓ AI-Powered Doubt Clearance"):
                # UI Components defined here...
                doubt_question = gr.Textbox(lines=4, label="Your Medical Question")
                doubt_context = gr.Textbox(lines=4, label="Optional Context")
                doubt_button = gr.Button("💬 Get AI Explanation", variant="primary")
                doubt_output = gr.Markdown(label="AI Response")

            # Tab 2: Smart Notes
            with gr.TabItem("📝 Smart Notes Generator"):
                # UI Components defined here...
                notes_topic = gr.Textbox(label="Enter Topic")
                notes_button = gr.Button("📄 Generate Notes", variant="primary")
                notes_output = gr.Markdown(label="Your Smart Notes")

            # Tab 3: Mock Test
            with gr.TabItem("📊 Interactive Mock Test"):
                # UI Components defined here...
                num_questions_slider = gr.Slider(minimum=1, maximum=5, value=5, step=1, label="Number of Questions")
                start_test_button = gr.Button("🚀 Start Mock Test", variant="primary")
                
                q_components = []
                for i in range(5):
                    with gr.Group(visible=False):
                        q_components.append(gr.Label(f"Question {i+1}"))
                        q_components.append(gr.Radio(label="", choices=[]))
                
                submit_button = gr.Button("🏁 Submit Test", variant="stop", visible=False)
                score_output = gr.Markdown(label="Your Result")

            # Tab 4: Motivation
            with gr.TabItem("💪 Motivation Boost"):
                # UI Components defined here...
                quote_output = gr.Markdown(label="Quote of the Day")
                quote_button = gr.Button("✨ Get New Quote")

            # Tab 5: System Logs (NEW)
            with gr.TabItem("📋 System Logs"):
                gr.Markdown("### 🔍 Real-time Application Logs")
                gr.Markdown("**Model Status:** Loading BioMistral-7B AI Model (15.9GB)")
                gr.Markdown("**Progress:** Currently downloading model files...")
                gr.Markdown("**Expected Time:** ~10-15 minutes depending on internet speed")
                
                log_display = gr.Textbox(
                    lines=15, 
                    label="📊 Live Logs", 
                    value="Application starting...\nModel loading in progress...\nCheck terminal for detailed logs.",
                    interactive=False
                )
                refresh_logs_button = gr.Button("🔄 Refresh Logs", variant="secondary")

        logger.info("Connecting UI components to backend functions.")
        
        state = gr.State({}) 

        # Connect event handlers
        doubt_button.click(handle_doubt_clearance, inputs=[doubt_question, doubt_context], outputs=doubt_output)
        notes_button.click(handle_notes_generation, inputs=notes_topic, outputs=notes_output)
        quote_button.click(get_motivational_quote, outputs=quote_output)

        radio_buttons = [comp for i, comp in enumerate(q_components) if i % 2 != 0]
        start_test_button.click(
            handle_start_mock_test,
            inputs=num_questions_slider,
            outputs=q_components + [submit_button, state]
        )
        submit_button.click(
            handle_submit_mock_test,
            inputs=[state] + radio_buttons,
            outputs=score_output
        )
    
    logger.info("--- Gradio interface built successfully. Ready to launch. ---")
    return demo

if __name__ == "__main__":
    logger.info("==========================================================")
    logger.info("      Starting the Dr. AI Tutor Application v1.0          ")
    logger.info("==========================================================")
    
    app_interface = create_gradio_interface()
    
    # Launch the Gradio app
    logger.info("Launching Gradio web server...")
    app_interface.launch(share=True, server_name="0.0.0.0", server_port=7860)
    
    logger.info("--- Application has been shut down. ---") 