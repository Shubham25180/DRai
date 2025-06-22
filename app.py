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
from utils import create_doubt_prompt, create_notes_prompt, create_mcq_prompt, get_motivational_quotes

class DrAITutor:
    def __init__(self):
        # Initialize the AI model as None (will be loaded later)
        self.model = None
        # Initialize the tokenizer as None (will be loaded later)
        self.tokenizer = None
        # Load mock questions from JSON file for practice tests
        self.mock_questions = self.load_mock_questions()
        # Initialize current test as None (no active test)
        self.current_test = None
        # Initialize dictionary to store test answers
        self.test_answers = {}
        
    def load_mock_questions(self) -> Dict[str, List[Dict]]:
        """Load mock questions from JSON file"""
        try:
            # Open and read the mock questions JSON file
            with open('mock_questions.json', 'r') as f:
                # Parse JSON content and return as dictionary
                return json.load(f)
        except FileNotFoundError:
            # If file not found, print warning and return empty dictionary
            print("Warning: mock_questions.json not found. Using empty database.")
            return {}
    
    def load_model(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Load the AI model (using DialoGPT as fallback for demo)"""
        try:
            # Print loading message to console
            print("🔄 Loading AI model...")
            # Load the tokenizer from Hugging Face model hub
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Load the model from Hugging Face model hub
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if not present (required for some models)
            if self.tokenizer.pad_token is None:
                # Use end-of-sequence token as padding token
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Print success message
            print("✅ Model loaded successfully!")
            # Return success message for UI
            return "✅ Model loaded successfully! You can now use all features."
        except Exception as e:
            # If error occurs, print error and return error message
            print(f"❌ Error loading model: {e}")
            return f"❌ Error loading model: {e}\n\nFor demo purposes, you can still use the mock test feature!"
    
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate response using the loaded model"""
        # Check if model and tokenizer are loaded
        if self.model is None or self.tokenizer is None:
            return "⚠️ Model not loaded. Please load the model first or use the mock test feature."
        
        try:
            # Encode the prompt text into token IDs
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response without computing gradients (for inference)
            with torch.no_grad():
                # Generate text using the model
                outputs = self.model.generate(
                    inputs,  # Input token IDs
                    max_length=max_length,  # Maximum length of generated text
                    num_return_sequences=1,  # Number of sequences to generate
                    temperature=0.7,  # Controls randomness (0.7 = balanced)
                    do_sample=True,  # Enable sampling (not greedy decoding)
                    pad_token_id=self.tokenizer.eos_token_id  # Padding token ID
                )
            
            # Decode the generated token IDs back to text
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from response to get clean answer
            response = response.replace(prompt, "").strip()
            # Return response or default message if empty
            return response if response else "I apologize, but I couldn't generate a response. Please try again."
            
        except Exception as e:
            # Return error message if generation fails
            return f"❌ Error generating response: {e}"
    
    def doubt_clearance(self, question: str) -> str:
        """Handle doubt clearance queries"""
        # Check if question is not empty
        if not question.strip():
            return "Please enter your medical question."
        
        # Create prompt using utility function
        prompt = create_doubt_prompt(question)
        # Generate and return response
        return self.generate_response(prompt, max_length=300)
    
    def generate_notes(self, topic: str) -> str:
        """Generate notes for a given topic"""
        # Check if topic is not empty
        if not topic.strip():
            return "Please enter a medical topic for notes generation."
        
        # Create prompt using utility function
        prompt = create_notes_prompt(topic)
        # Generate and return response
        return self.generate_response(prompt, max_length=400)
    
    def get_available_topics(self) -> List[str]:
        """Get list of available topics for mock tests"""
        # Return list of topic keys from mock questions dictionary
        return list(self.mock_questions.keys())
    
    def start_mock_test(self, topic: str, num_questions: int = 5) -> tuple:
        """Start a mock test for the selected topic"""
        # Check if topic exists in mock questions
        if topic not in self.mock_questions:
            return "❌ Topic not found in database.", None, None
        
        # Get available questions for the topic
        available_questions = self.mock_questions[topic]
        # Adjust number of questions if not enough available
        if len(available_questions) < num_questions:
            num_questions = len(available_questions)
        
        # Randomly select questions from available pool
        selected_questions = random.sample(available_questions, num_questions)
        # Store current test information
        self.current_test = {
            'topic': topic,  # Test topic
            'questions': selected_questions,  # Selected questions
            'answers': {},  # User answers (empty initially)
            'started': True  # Test status flag
        }
        
        # Format questions for display
        questions_text = f"📝 Mock Test: {topic.upper()}\n"
        questions_text += f"Total Questions: {num_questions}\n"
        questions_text += "="*50 + "\n\n"
        
        # Loop through each question and format it
        for i, q in enumerate(selected_questions, 1):
            # Add question number and text
            questions_text += f"Q{i}. {q['question']}\n"
            # Add each option (A, B, C, D)
            for option, text in q['options'].items():
                questions_text += f"   {option}) {text}\n"
            questions_text += "\n"
        
        # Return formatted questions and update UI components
        return questions_text, gr.update(visible=True), gr.update(visible=True)
    
    def submit_test_answers(self, answers: str) -> str:
        """Submit and grade mock test answers"""
        # Check if there's an active test
        if not self.current_test or not self.current_test['started']:
            return "❌ No active test. Please start a test first."
        
        # Check if answers were provided
        if not answers.strip():
            return "❌ Please provide your answers."
        
        # Parse answers from comma-separated string (format: A,B,C,D,A)
        try:
            # Split by comma and convert to uppercase
            user_answers = [ans.strip().upper() for ans in answers.split(',')]
        except:
            return "❌ Invalid answer format. Please use format: A,B,C,D,A"
        
        # Get questions from current test
        questions = self.current_test['questions']
        # Check if number of answers matches number of questions
        if len(user_answers) != len(questions):
            return f"❌ Expected {len(questions)} answers, got {len(user_answers)}."
        
        # Grade the test
        correct = 0  # Counter for correct answers
        results = f"📊 Test Results: {self.current_test['topic'].upper()}\n"
        results += "="*50 + "\n\n"
        
        # Loop through each question and user answer
        for i, (question, user_answer) in enumerate(zip(questions, user_answers), 1):
            # Get correct answer from question data
            correct_answer = question['correct_answer']
            # Check if user answer is correct
            is_correct = user_answer == correct_answer
            
            if is_correct:
                # Increment correct counter
                correct += 1
                # Add correct answer to results
                results += f"✅ Q{i}: Correct ({user_answer})\n"
            else:
                # Add incorrect answer with explanation
                results += f"❌ Q{i}: Wrong ({user_answer}) - Correct: {correct_answer}\n"
                results += f"   💡 {question['explanation']}\n"
            results += "\n"
        
        # Calculate percentage score
        score = (correct / len(questions)) * 100
        results += f"🎯 Final Score: {correct}/{len(questions)} ({score:.1f}%)\n"
        
        # Add motivational message based on score
        if score >= 80:
            results += "🌟 Excellent! Keep up the great work!"
        elif score >= 60:
            results += "👍 Good job! Review the incorrect answers."
        else:
            results += "📚 Keep studying! Review the topic thoroughly."
        
        # Reset test state
        self.current_test = None
        
        # Return formatted results
        return results
    
    def get_motivation(self) -> str:
        """Get a random motivational quote"""
        # Get list of motivational quotes from utils
        quotes = get_motivational_quotes()
        # Return a random quote
        return random.choice(quotes)

def create_interface():
    """Create the Gradio interface"""
    # Create instance of DrAITutor class
    tutor = DrAITutor()
    
    # Create Gradio interface with custom CSS
    with gr.Blocks(
        title="DrAI Medical Tutor",  # Browser tab title
        css="""
        .gradio-container {
            max-width: 1200px !important;  /* Set maximum width for container */
        }
        .main-header {
            text-align: center;  /* Center align header text */
            padding: 20px;  /* Add padding around header */
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  /* Gradient background */
            color: white;  /* White text color */
            border-radius: 10px;  /* Rounded corners */
            margin-bottom: 20px;  /* Bottom margin */
        }
        """
    ) as interface:
        
        # Create header section with HTML
        gr.HTML("""
        <div class="main-header">
            <h1>🧠 DrAI Medical Tutor</h1>
            <p>Your AI-powered companion for NEET-PG preparation</p>
        </div>
        """)
        
        # Create tabbed interface
        with gr.Tabs():
            
            # Tab 1: Model Loading
            with gr.Tab("🚀 Setup"):
                # Add markdown description
                gr.Markdown("### Load AI Model")
                gr.Markdown("Start by loading the AI model to enable doubt clearance and notes generation.")
                
                # Create load button and status display
                load_btn = gr.Button("Load AI Model", variant="primary")
                load_status = gr.Textbox(label="Status", interactive=False)
                # Connect button click to load_model function
                load_btn.click(tutor.load_model, outputs=load_status)
            
            # Tab 2: Doubt Clearance
            with gr.Tab("❓ Doubt Clearance"):
                # Add markdown description
                gr.Markdown("### Ask any medical question and get expert answers")
                
                # Create input textbox for questions
                doubt_input = gr.Textbox(
                    label="Your Medical Question",
                    placeholder="e.g., What are the symptoms of myocardial infarction?",
                    lines=3  # Set number of lines for textbox
                )
                # Create button and output textbox
                doubt_btn = gr.Button("Get Answer", variant="primary")
                doubt_output = gr.Textbox(label="DrAI's Answer", lines=8, interactive=False)
                
                # Connect button click to doubt_clearance function
                doubt_btn.click(tutor.doubt_clearance, inputs=doubt_input, outputs=doubt_output)
            
            # Tab 3: Notes Generator
            with gr.Tab("📝 Notes Generator"):
                # Add markdown description
                gr.Markdown("### Generate comprehensive notes for any medical topic")
                
                # Create input textbox for topics
                notes_input = gr.Textbox(
                    label="Medical Topic",
                    placeholder="e.g., Types of Shock, ECG Interpretation, Diabetes Management",
                    lines=2
                )
                # Create button and output textbox
                notes_btn = gr.Button("Generate Notes", variant="primary")
                notes_output = gr.Textbox(label="Generated Notes", lines=12, interactive=False)
                
                # Connect button click to generate_notes function
                notes_btn.click(tutor.generate_notes, inputs=notes_input, outputs=notes_output)
            
            # Tab 4: Mock Test
            with gr.Tab("📊 Mock Test"):
                # Add markdown description
                gr.Markdown("### Practice with topic-specific MCQs")
                
                # Create row with topic dropdown and question count slider
                with gr.Row():
                    # Create dropdown for topic selection
                    topic_dropdown = gr.Dropdown(
                        choices=tutor.get_available_topics(),  # Get available topics
                        label="Select Topic",
                        value=tutor.get_available_topics()[0] if tutor.get_available_topics() else None
                    )
                    # Create slider for number of questions
                    num_questions = gr.Slider(
                        minimum=1,  # Minimum questions
                        maximum=10,  # Maximum questions
                        value=5,  # Default value
                        step=1,  # Step size
                        label="Number of Questions"
                    )
                
                # Create start test button and questions display
                start_test_btn = gr.Button("Start Test", variant="primary")
                test_questions = gr.Textbox(label="Test Questions", lines=15, interactive=False)
                
                # Create row for answer input and submit button (initially hidden)
                with gr.Row():
                    # Create textbox for answers
                    answers_input = gr.Textbox(
                        label="Your Answers (A,B,C,D,A)",
                        placeholder="Enter answers separated by commas",
                        visible=False  # Initially hidden
                    )
                    # Create submit button (initially hidden)
                    submit_btn = gr.Button("Submit Test", variant="secondary", visible=False)
                
                # Create results display textbox
                test_results = gr.Textbox(label="Test Results", lines=10, interactive=False)
                
                # Connect start test button to start_mock_test function
                start_test_btn.click(
                    tutor.start_mock_test,
                    inputs=[topic_dropdown, num_questions],
                    outputs=[test_questions, answers_input, submit_btn]
                )
                
                # Connect submit button to submit_test_answers function
                submit_btn.click(
                    tutor.submit_test_answers,
                    inputs=answers_input,
                    outputs=test_results
                )
            
            # Tab 5: Motivation
            with gr.Tab("💪 Motivation"):
                # Add markdown description
                gr.Markdown("### Get motivated for your medical journey!")
                
                # Create button and output textbox
                motivation_btn = gr.Button("Get Motivation", variant="primary")
                motivation_output = gr.Textbox(label="Today's Motivation", lines=3, interactive=False)
                
                # Connect button click to get_motivation function
                motivation_btn.click(tutor.get_motivation, outputs=motivation_output)
        
        # Create footer with HTML
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>🧠 DrAI Medical Tutor - Empowering medical students with AI</p>
            <p>Built with ❤️ for NEET-PG aspirants</p>
        </div>
        """)
    
    # Return the created interface
    return interface

# Main execution block
if __name__ == "__main__":
    # Create the interface
    interface = create_interface()
    # Launch the interface with specific settings
    interface.launch(
        server_name="localhost",  # Use localhost for local access
        server_port=7860,  # Port number
        share=True,  # Enable sharing (creates public link)
        show_error=True  # Show detailed error messages
    ) 