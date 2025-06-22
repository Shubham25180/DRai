"""
Utility functions for DrAI Medical Tutor
"""

import webbrowser
import sys
import logging
import random


def create_doubt_prompt(question, context=""):
    """Creates a standardized prompt for the AI to answer a medical question."""
    logging.info(f"Creating doubt prompt for question: '{question[:50]}...'")
    # This structure helps the AI understand it needs to act as an expert tutor.
    return f"You are an expert medical tutor. A student has the following question: '{question}'. Optional context provided: '{context}'. Provide a clear, concise, and accurate answer suitable for a medical student preparing for the NEET-PG exam."

def create_notes_prompt(topic):
    """Creates a standardized prompt for the AI to generate study notes."""
    logging.info(f"Creating notes prompt for topic: '{topic}'")
    # This prompt guides the AI to create structured, high-yield notes.
    return f"Generate comprehensive, well-structured study notes on the topic of '{topic}'. The notes should be easy to understand, cover the key concepts, and be tailored for a student preparing for the NEET-PG medical entrance exam. Use headings, bullet points, and bold text to organize the information."

def create_mcq_prompt(topic, num_questions=5):
    """Create prompt for MCQ generation"""
    # Create a formatted prompt string for AI model to generate multiple choice questions
    return f"""You are DrAI-Tutor, an expert NEET-PG AI assistant. Generate {num_questions} high-quality multiple choice questions (MCQs) for NEET-PG preparation.

Topic: {topic}

Format each question as:
Q1. [Question text]
A) [Option A]
B) [Option B] 
C) [Option C]
D) [Option D]
Correct Answer: [A/B/C/D]

Questions:"""

def create_mock_test_prompt(topic, num_questions=5):
    """Create prompt for mock test generation"""
    # Create a formatted prompt string for AI model to generate mock test questions
    return f"""You are DrAI-Tutor, an expert NEET-PG AI assistant. Generate {num_questions} high-quality multiple choice questions (MCQs) for a mock test.

Topic: {topic}

Format each question as:
Q1. [Question text]
A) [Option A]
B) [Option B] 
C) [Option C]
D) [Option D]
Correct Answer: [A/B/C/D]

Questions:"""

def get_motivational_quote():
    """Returns a random motivational quote for students."""
    logging.info("Fetching a random motivational quote.")
    quotes = [
        "The expert in anything was once a beginner. Keep going.",
        "Success is the sum of small efforts, repeated day in and day out.",
        "Believe you can and you're halfway there.",
        "The only way to do great work is to love what you do.",
        "Strive for progress, not perfection."
    ]
    quote = random.choice(quotes)
    logging.info(f"Selected quote: '{quote}'")
    return quote

def setup_global_logger():
    """
    Sets up a highly descriptive global logger that prints to the console.
    This is designed to be verbose so every step of the application's execution is visible.
    """
    # Get the root logger. All other loggers will inherit from this.
    logger = logging.getLogger()

    # Clear any existing handlers to avoid duplicate messages.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a handler to stream logs to the console (stdout).
    handler = logging.StreamHandler(sys.stdout)
    
    # Define a very descriptive format for the log messages.
    # Includes timestamp, log level, the file and line number, and the message.
    log_format = '%(asctime)s - %(levelname)-8s - [%(filename)s:%(lineno)d] - %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    
    # Add the configured handler to the root logger.
    logger.addHandler(handler)
    
    # Set the minimum logging level to INFO to capture all important steps.
    logger.setLevel(logging.INFO)
    
    logging.info("--- CONSOLE LOGGER INITIALIZED ---")

def open_docs():
    """Opens the project's documentation in a web browser."""
    docs_url = "https://github.com/Shubham25180/DRai"
    logging.info(f"Opening documentation in web browser at: {docs_url}")
    webbrowser.open(docs_url) 