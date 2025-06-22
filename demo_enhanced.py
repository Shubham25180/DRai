# -*- coding: utf-8 -*-
"""
DrAI Medical Tutor - Enhanced Demo
Showcase the enhanced features without loading the full AI model.
Provides a comprehensive demo of all DrAI Tutor capabilities.
"""

# Import required libraries
import gradio as gr  # For creating web interface
import json  # For reading JSON files
import random  # For random selections
from typing import List, Dict, Any  # For type hints

def load_mock_questions() -> Dict[str, List[Dict]]:
    """
    Load mock questions for demo from JSON file.
    Returns a dictionary of topics with their associated questions.
    Falls back to sample questions if file not found.
    """
    try:
        # Try to load questions from JSON file
        with open('mock_questions.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return sample questions if file not found
        return {
            "Cardiology": [
                {
                    "question": "Which of the following is the most common cause of heart failure?",
                    "options": {
                        "A": "Coronary artery disease",
                        "B": "Hypertension",
                        "C": "Valvular heart disease",
                        "D": "Cardiomyopathy"
                    },
                    "correct_answer": "A",
                    "explanation": "Coronary artery disease is the most common cause of heart failure, accounting for about 50% of cases."
                },
                {
                    "question": "What is the first-line treatment for stable angina?",
                    "options": {
                        "A": "Beta blockers",
                        "B": "Calcium channel blockers",
                        "C": "Nitrates",
                        "D": "ACE inhibitors"
                    },
                    "correct_answer": "A",
                    "explanation": "Beta blockers are the first-line treatment for stable angina as they reduce myocardial oxygen demand."
                }
            ],
            "Pharmacology": [
                {
                    "question": "Which drug class is used to treat hypertension and heart failure?",
                    "options": {
                        "A": "ACE inhibitors",
                        "B": "Beta blockers",
                        "C": "Calcium channel blockers",
                        "D": "Diuretics"
                    },
                    "correct_answer": "A",
                    "explanation": "ACE inhibitors are used for both hypertension and heart failure due to their vasodilatory and neurohormonal effects."
                }
            ]
        }

def demo_chat_interface(message: str, history: list) -> tuple:
    """Demo chat interface - UPDATED for messages format"""
    if not message.strip():
        return "", history
    
    # Append user message to history in the new dictionary format
    history.append({"role": "user", "content": message})
    
    # Demo responses
    demo_responses = {
        "heart": "The heart is a muscular organ that pumps blood throughout the body. It has four chambers: two atria and two ventricles.",
        "diabetes": "Diabetes mellitus is a metabolic disorder characterized by high blood glucose levels due to insulin deficiency or resistance.",
        "hypertension": "Hypertension is persistently elevated blood pressure (>140/90 mmHg). Treatment includes lifestyle modifications and medications.",
        "shock": "Shock is a life-threatening condition with inadequate tissue perfusion. Types include hypovolemic, cardiogenic, distributive, and obstructive."
    }
    
    # Find relevant response
    response = "I'm a demo version of DrAI Tutor. For full medical assistance, please load the AI model."
    for keyword, demo_response in demo_responses.items():
        if keyword.lower() in message.lower():
            response = demo_response
            break
    
    # Append bot response to history
    history.append({"role": "assistant", "content": response})
    
    # Return an empty string to clear the textbox and the updated history
    return "", history

def demo_generate_notes(topic: str, detail_level: str = "comprehensive") -> str:
    """
    Demo notes generation for medical topics.
    Provides structured notes with different detail levels.
    """
    # Check if topic is provided
    if not topic.strip():
        return "Please enter a medical topic for notes generation."
    
    # Demo notes for different topics
    demo_notes = {
        "cardiology": f"""
📝 {topic.upper()} - {detail_level.upper()} NOTES

🔍 Overview:
Cardiology deals with disorders of the heart and cardiovascular system.

📋 Key Topics:
• Cardiac anatomy and physiology
• ECG interpretation
• Heart failure management
• Arrhythmias and their treatment
• Hypertension guidelines

💡 Study Tips:
• Practice ECG reading daily
• Understand hemodynamics
• Know the latest guidelines
• Focus on clinical correlations

🎯 NEET-PG Focus Areas:
• ECG patterns and their significance
• Drug mechanisms and side effects
• Clinical scenarios and management
• Recent guidelines and updates
        """,
        "pharmacology": f"""
📝 {topic.upper()} - {detail_level.upper()} NOTES

🔍 Overview:
Pharmacology studies drug actions and their effects on living systems.

📋 Key Topics:
• Drug mechanisms of action
• Pharmacokinetics and dynamics
• Drug interactions
• Adverse effects
• Therapeutic uses

💡 Study Tips:
• Understand mechanisms first
• Group drugs by class
• Learn side effects systematically
• Practice clinical scenarios

🎯 NEET-PG Focus Areas:
• Drug mechanisms and interactions
• Adverse effects and contraindications
• Clinical applications
• Recent drug approvals
        """
    }
    
    # Find relevant notes based on topic
    for keyword, notes in demo_notes.items():
        if keyword.lower() in topic.lower():
            return notes
    
    # Default notes template if topic not found
    return f"""
📝 {topic.upper()} - {detail_level.upper()} NOTES

🔍 Overview:
This is a demo response for {topic}. In the full version, you'll get comprehensive, AI-generated notes tailored to your topic.

📋 Key Features:
• Detailed explanations
• Clinical correlations
• NEET-PG focus areas
• Study tips and mnemonics

💡 Note: This is a demo version. Load the AI model for full functionality.
    """

def demo_mock_test(topic: str, num_questions: int = 5) -> str:
    """
    Demo mock test generation.
    Creates a formatted test with questions from the selected topic.
    """
    # Load mock questions
    mock_questions = load_mock_questions()
    
    # Check if topic exists
    if topic not in mock_questions:
        return "❌ Topic not found in demo database."
    
    # Get questions for the topic
    questions = mock_questions[topic]
    # Adjust number of questions if not enough available
    if len(questions) < num_questions:
        num_questions = len(questions)
    
    # Randomly select questions
    selected_questions = random.sample(questions, num_questions)
    
    # Create formatted question display
    questions_display = f"📝 Demo Mock Test: {topic.upper()}\n"
    questions_display += f"Total Questions: {len(selected_questions)}\n"
    questions_display += "="*50 + "\n\n"
    
    # Format each question with options
    for i, q in enumerate(selected_questions, 1):
        questions_display += f"Q{i}. {q['question']}\n"
        for option, text in q['options'].items():
            questions_display += f"   {option}) {text}\n"
        questions_display += "\n"
    
    return questions_display

def demo_evaluate_answers(answers: str) -> str:
    """
    Demo answer evaluation.
    Evaluates user answers and provides feedback.
    """
    # Check if answers were provided
    if not answers.strip():
        return "❌ Please provide your answers."
    
    try:
        # Parse answers from comma-separated string
        user_answers = [ans.strip().upper() for ans in answers.split(',')]
    except:
        return "❌ Invalid answer format. Please use format: A,B,C,D,A"
    
    # Demo evaluation (in real version, this would compare with actual questions)
    num_answers = len(user_answers)
    correct_answers = random.randint(0, num_answers)  # Random for demo
    
    # Calculate score
    score_percentage = (correct_answers / num_answers) * 100 if num_answers > 0 else 0
    
    # Generate feedback based on score
    if score_percentage >= 80:
        feedback = "🌟 Excellent! You have great knowledge of this topic!"
    elif score_percentage >= 60:
        feedback = "👍 Good job! Keep studying to improve further."
    else:
        feedback = "📚 Keep studying! Review the topic thoroughly."
    
    # Format results
    results = f"📊 Demo Test Results\n"
    results += "="*30 + "\n"
    results += f"🎯 Score: {correct_answers}/{num_answers} ({score_percentage:.1f}%)\n"
    results += f"💬 {feedback}\n\n"
    results += "💡 Note: This is a demo evaluation. Load the AI model for real test evaluation."
    
    return results

def get_demo_motivation() -> str:
    """
    Get a demo motivational quote.
    Returns a random motivational message for medical students.
    """
    # List of motivational quotes
    quotes = [
        "🎯 Every hour of study brings you closer to your dream of becoming a doctor!",
        "🧠 Your brain is your most powerful tool - keep feeding it knowledge!",
        "💪 You're stronger than you think. Keep pushing forward!",
        "📚 Today's effort is tomorrow's success in the medical field!",
        "🌟 You have what it takes to ace NEET-PG! Believe in yourself!"
    ]
    # Return a random quote
    return random.choice(quotes)

def get_demo_tip() -> str:
    """
    Get a demo study tip.
    Returns a random study tip for medical students.
    """
    # List of study tips
    tips = [
        "💡 Study Tip: Use the Feynman Technique - explain concepts in simple terms.",
        "💡 Study Tip: Take regular breaks every 45 minutes to maintain focus.",
        "💡 Study Tip: Practice active recall by testing yourself regularly.",
        "💡 Study Tip: Create mind maps to visualize connections between concepts.",
        "💡 Study Tip: Review your mistakes - they're your best learning opportunities."
    ]
    # Return a random tip
    return random.choice(tips)

def create_demo_interface():
    """
    Create the enhanced demo Gradio interface.
    Sets up all the demo features with a modern UI.
    """
    # Create Gradio interface with custom styling
    with gr.Blocks(
        title="DrAI Medical Tutor - Enhanced Demo",
        theme=gr.themes.Soft(),
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
            <h1>🧠 DrAI Medical Tutor - Enhanced Demo</h1>
            <p>Experience the power of AI-powered medical education</p>
            <p style="font-size: 14px; opacity: 0.9;">Demo Version - Load AI model for full functionality</p>
        </div>
        """)
        
        # Create tabbed interface
        with gr.Tabs():
            
            # Tab 1: Chat Interface
            with gr.Tab("💬 Chat Demo"):
                gr.Markdown("### Interactive Chat Interface")
                gr.Markdown("Ask medical questions and get instant responses (demo mode)")
                
                # Chat interface components
                chatbot = gr.Chatbot(label="DrAI Chat", height=400, show_label=True, type="messages")
                msg = gr.Textbox(label="Your Question", placeholder="Ask any medical question...")
                clear = gr.Button("Clear Chat")
                
                # Connect chat interface
                msg.submit(demo_chat_interface, [msg, chatbot], [msg, chatbot])
                clear.click(lambda: None, None, chatbot, queue=False)
            
            # Tab 2: Notes Generator
            with gr.Tab("📝 Notes Demo"):
                gr.Markdown("### Generate Comprehensive Notes")
                gr.Markdown("Get structured notes for any medical topic")
                
                # Notes generation components
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
                notes_btn.click(demo_generate_notes, [notes_topic, detail_level], notes_output)
            
            # Tab 3: Mock Test
            with gr.Tab("📊 Test Demo"):
                gr.Markdown("### Practice with Mock Tests")
                gr.Markdown("Test your knowledge with topic-specific MCQs")
                
                # Mock test components
                with gr.Row():
                    test_topic = gr.Dropdown(
                        choices=["Cardiology", "Pharmacology"],
                        value="Cardiology",
                        label="Select Topic"
                    )
                    num_questions = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Number of Questions"
                    )
                
                start_test_btn = gr.Button("Start Demo Test", variant="primary")
                test_questions = gr.Textbox(label="Test Questions", lines=12, interactive=False)
                
                # Answer submission
                with gr.Row():
                    answers_input = gr.Textbox(
                        label="Your Answers (A,B,C,A)",
                        placeholder="Enter answers separated by commas"
                    )
                    submit_btn = gr.Button("Submit Answers", variant="secondary")
                
                test_results = gr.Textbox(label="Test Results", lines=8, interactive=False)
                
                # Connect mock test functions
                start_test_btn.click(demo_mock_test, [test_topic, num_questions], test_questions)
                submit_btn.click(demo_evaluate_answers, answers_input, test_results)
            
            # Tab 4: Motivation & Tips
            with gr.Tab("💪 Motivation"):
                gr.Markdown("### Stay Motivated and Get Study Tips")
                gr.Markdown("Get daily motivation and study tips for your medical journey")
                
                # Motivation components
                with gr.Row():
                    motivation_btn = gr.Button("Get Motivation", variant="primary")
                    tip_btn = gr.Button("Get Study Tip", variant="secondary")
                
                motivation_output = gr.Textbox(label="Today's Motivation", lines=3, interactive=False)
                tip_output = gr.Textbox(label="Study Tip", lines=3, interactive=False)
                
                # Connect motivation functions
                motivation_btn.click(get_demo_motivation, outputs=motivation_output)
                tip_btn.click(get_demo_tip, outputs=tip_output)
        
        # Footer with information
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>🧠 DrAI Medical Tutor - Enhanced Demo</p>
            <p>This is a demo version. Load the AI model for full functionality!</p>
        </div>
        """)
    
    # Return the interface
    return interface

# Main execution block
if __name__ == "__main__":
    # Create and launch the demo interface
    interface = create_demo_interface()
    interface.launch(
        server_name="localhost",
        server_port=7862,
        share=False,
        show_error=True,
        prevent_thread_lock=True
    ) 