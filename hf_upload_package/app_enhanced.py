# -*- coding: utf-8 -*-
"""
DrAI Medical Tutor - Enhanced Frontend Application
A comprehensive AI-powered medical education platform for NEET-PG preparation.
Features enhanced UI, multiple model support, and advanced functionality.
"""

# Import required libraries
import gradio as gr  # For creating web interface
import json  # For JSON operations
import random  # For random selections
from typing import List, Dict, Any  # For type hints
# Import custom modules
from model_utils import model_manager, ask_doubt, generate_notes, get_available_topics, evaluate_answers
from utils import get_motivational_quotes

class DrAITutorEnhanced:
    """
    Enhanced DrAI Tutor class with advanced features.
    Supports multiple model types, enhanced UI, and improved functionality.
    """
    def __init__(self):
        # Initialize current test state
        self.current_test = None
        # Initialize current questions list
        self.current_questions = []
        
    def load_model(self, model_type: str = "base"):
        """
        Load the appropriate model based on user choice.
        Supports both base and fine-tuned models.
        """
        try:
            # Check if user wants fine-tuned model
            if model_type == "fine_tuned":
                success = model_manager.load_fine_tuned_model()
                if success:
                    return "✅ Fine-tuned model loaded successfully! You now have enhanced medical knowledge."
                else:
                    return "⚠️ Fine-tuned model not found. Loading base model instead."
            
            # Load base model
            success = model_manager.load_base_model()
            if success:
                return "✅ Base model loaded successfully! You can now use all features."
            else:
                return "❌ Error loading model. You can still use the mock test feature!"
                
        except Exception as e:
            return f"❌ Error: {e}"
    
    def chat_interface(self, message: str, history: list) -> tuple:
        """Enhanced chat interface for doubt clearance - UPDATED for messages format"""
        if not message.strip():
            return "", history
        
        # Append user message to history
        history.append({"role": "user", "content": message})
        
        # Get AI response
        response = ask_doubt(message)
        
        # Append bot response to history
        history.append({"role": "assistant", "content": response})
        
        return "", history
    
    def generate_notes_enhanced(self, topic: str, detail_level: str = "comprehensive") -> str:
        """
        Generate notes with different detail levels.
        Supports basic, comprehensive, and advanced detail levels.
        """
        # Check if topic is provided
        if not topic.strip():
            return "Please enter a medical topic for notes generation."
        
        # Add detail level to the prompt for enhanced generation
        enhanced_topic = f"{topic} ({detail_level} level)"
        return generate_notes(enhanced_topic)
    
    def start_mock_test_enhanced(self, topic: str, num_questions: int = 5) -> str:
        """
        Start a mock test with enhanced UI and functionality.
        Provides better question selection and formatting.
        """
        # Check if topic exists in database
        if topic not in model_manager.mock_questions:
            return "❌ Topic not found in database."
        
        # Get questions from model manager
        self.current_questions = model_manager.get_mock_questions(topic, num_questions)
        # Set up current test state
        self.current_test = {
            'topic': topic,
            'questions': self.current_questions,
            'started': True
        }
        
        # Create formatted question display
        questions_display = f"📝 Mock Test: {topic.upper()}\n"
        questions_display += f"Total Questions: {len(self.current_questions)}\n"
        questions_display += "="*50 + "\n\n"
        
        # Format each question with options
        for i, q in enumerate(self.current_questions, 1):
            questions_display += f"Q{i}. {q['question']}\n"
            for option, text in q['options'].items():
                questions_display += f"   {option}) {text}\n"
            questions_display += "\n"
        
        return questions_display
    
    def submit_test_enhanced(self, answers: str) -> str:
        """
        Submit test with enhanced evaluation and detailed feedback.
        Provides comprehensive results with explanations.
        """
        # Check if there's an active test
        if not self.current_test or not self.current_test['started']:
            return "❌ No active test. Please start a test first."
        
        # Parse answers from comma-separated string
        try:
            user_answers = [ans.strip().upper() for ans in answers.split(',')]
        except:
            return "❌ Invalid answer format. Please use format: A,B,C,D,A"
        
        # Evaluate answers using model manager
        results = model_manager.evaluate_answers(self.current_questions, user_answers)
        
        # Check for evaluation errors
        if "error" in results:
            return results["error"]
        
        # Format comprehensive results
        output = f"📊 Test Results: {self.current_test['topic'].upper()}\n"
        output += "="*50 + "\n\n"
        output += f"🎯 Score: {results['correct_answers']}/{results['total_questions']} ({results['score_percentage']:.1f}%)\n\n"
        output += f"💬 {results['feedback']}\n\n"
        output += "📋 Detailed Results:\n"
        output += "-" * 30 + "\n"
        
        # Add detailed results for each question
        for result in results['detailed_results']:
            status = "✅" if result['is_correct'] else "❌"
            output += f"{status} Q{result['question_number']}: "
            output += f"Your answer: {result['user_answer']}, "
            output += f"Correct: {result['correct_answer']}\n"
            if not result['is_correct'] and result['explanation']:
                output += f"   💡 {result['explanation']}\n"
            output += "\n"
        
        # Reset test state
        self.current_test = None
        self.current_questions = []
        
        return output
    
    def get_motivation(self) -> str:
        """
        Get a random motivational quote for medical students.
        """
        # Get quotes from utils and return a random one
        quotes = get_motivational_quotes()
        return random.choice(quotes)
    
    def get_daily_tip(self) -> str:
        """
        Get a daily study tip for medical students.
        Provides practical advice for NEET-PG preparation.
        """
        # List of study tips for medical students
        tips = [
            "💡 Study Tip: Use the Feynman Technique - explain concepts in simple terms to reinforce learning.",
            "💡 Study Tip: Take regular breaks every 45 minutes to maintain focus and retention.",
            "💡 Study Tip: Practice active recall by testing yourself instead of just re-reading.",
            "💡 Study Tip: Create mind maps to visualize connections between medical concepts.",
            "💡 Study Tip: Review your mistakes in mock tests - they're your best learning opportunities.",
            "💡 Study Tip: Study in chunks of 25 minutes with 5-minute breaks (Pomodoro Technique).",
            "💡 Study Tip: Use mnemonics to remember complex medical terms and processes.",
            "💡 Study Tip: Teach concepts to others - it's the best way to learn yourself."
        ]
        # Return a random tip
        return random.choice(tips)

def create_enhanced_interface():
    """
    Create the enhanced Gradio interface with modern UI.
    Features improved styling, better organization, and enhanced functionality.
    """
    # Create instance of enhanced tutor
    tutor = DrAITutorEnhanced()
    
    # Create Gradio interface with enhanced styling
    with gr.Blocks(
        title="DrAI Medical Tutor - Enhanced",
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        """
    ) as interface:
        
        # Enhanced header with gradient background
        gr.HTML("""
        <div class="main-header">
            <h1>🧠 DrAI Medical Tutor - Enhanced</h1>
            <p>Your AI-powered companion for NEET-PG preparation</p>
            <p style="font-size: 14px; opacity: 0.9;">Powered by BioMistral-7B | Enhanced with LoRA Fine-tuning</p>
        </div>
        """)
        
        # Create tabbed interface for different features
        with gr.Tabs():
            
            # Tab 1: Model Setup
            with gr.Tab("🚀 Setup"):
                gr.Markdown("### Load AI Model")
                gr.Markdown("Choose your preferred model for the best medical assistance.")
                
                # Model selection and loading interface
                with gr.Row():
                    model_type = gr.Dropdown(
                        choices=["base", "fine_tuned"],
                        value="base",
                        label="Model Type",
                        info="Fine-tuned model provides better medical knowledge"
                    )
                    load_btn = gr.Button("Load Model", variant="primary", size="lg")
                
                # Status display
                load_status = gr.Textbox(label="Status", interactive=False, lines=3)
                # Connect load button to model loading function
                load_btn.click(tutor.load_model, inputs=model_type, outputs=load_status)
            
            # Tab 2: Enhanced Chat Interface
            with gr.Tab("💬 Doubt Clearance"):
                gr.Markdown("### Ask any medical question and get expert answers")
                
                # Chat interface components
                chatbot = gr.Chatbot(label="DrAI Chat", height=500, type="messages")
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask any medical question...",
                    lines=2
                )
                clear = gr.Button("Clear Chat", variant="secondary")
                
                # Connect chat interface functions
                msg.submit(tutor.chat_interface, [msg, chatbot], [msg, chatbot])
                clear.click(lambda: None, None, chatbot, queue=False)
            
            # Tab 3: Enhanced Notes Generator
            with gr.Tab("📝 Notes Generator"):
                gr.Markdown("### Generate comprehensive notes for any medical topic")
                
                # Notes generation interface
                with gr.Row():
                    notes_topic = gr.Textbox(
                        label="Medical Topic",
                        placeholder="e.g., Cardiology, Pharmacology, Neurology"
                    )
                    detail_level = gr.Dropdown(
                        choices=["basic", "comprehensive", "advanced"],
                        value="comprehensive",
                        label="Detail Level"
                    )
                
                notes_btn = gr.Button("Generate Notes", variant="primary")
                notes_output = gr.Textbox(label="Generated Notes", lines=15, interactive=False)
                
                # Connect notes generation
                notes_btn.click(tutor.generate_notes_enhanced, [notes_topic, detail_level], notes_output)
            
            # Tab 4: Enhanced Mock Test
            with gr.Tab("📊 Mock Test"):
                gr.Markdown("### Practice with topic-specific MCQs")
                
                # Mock test interface
                with gr.Row():
                    test_topic = gr.Dropdown(
                        choices=get_available_topics(),
                        value=get_available_topics()[0] if get_available_topics() else None,
                        label="Select Topic"
                    )
                    num_questions = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Number of Questions"
                    )
                
                start_test_btn = gr.Button("Start Test", variant="primary")
                test_questions = gr.Textbox(label="Test Questions", lines=15, interactive=False)
                
                # Answer submission interface
                with gr.Row():
                    answers_input = gr.Textbox(
                        label="Your Answers (A,B,C,D,A)",
                        placeholder="Enter answers separated by commas"
                    )
                    submit_btn = gr.Button("Submit Test", variant="secondary")
                
                test_results = gr.Textbox(label="Test Results", lines=12, interactive=False)
                
                # Connect mock test functions
                start_test_btn.click(tutor.start_mock_test_enhanced, [test_topic, num_questions], test_questions)
                submit_btn.click(tutor.submit_test_enhanced, answers_input, test_results)
            
            # Tab 5: Motivation & Tips
            with gr.Tab("💪 Motivation"):
                gr.Markdown("### Stay motivated and get study tips")
                
                # Motivation and tips interface
                with gr.Row():
                    motivation_btn = gr.Button("Get Motivation", variant="primary")
                    tip_btn = gr.Button("Get Study Tip", variant="secondary")
                
                motivation_output = gr.Textbox(label="Today's Motivation", lines=3, interactive=False)
                tip_output = gr.Textbox(label="Study Tip", lines=3, interactive=False)
                
                # Connect motivation functions
                motivation_btn.click(tutor.get_motivation, outputs=motivation_output)
                tip_btn.click(tutor.get_daily_tip, outputs=tip_output)
        
        # Footer with information
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>🧠 DrAI Medical Tutor - Enhanced</p>
            <p>Built with ❤️ for NEET-PG aspirants</p>
        </div>
        """)
    
    # Return the interface
    return interface

# Main execution block
if __name__ == "__main__":
    # Create and launch the enhanced interface
    interface = create_enhanced_interface()
    interface.launch(
        server_name="localhost",
        server_port=7865,
        share=False,
        show_error=True,
        prevent_thread_lock=True,
        show_api=False
    )
