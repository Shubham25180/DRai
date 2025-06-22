"""
Utility functions for DrAI Medical Tutor
"""

def create_doubt_prompt(question):
    """Create prompt for doubt clearance"""
    # Create a formatted prompt string for AI model to answer medical questions
    return f"""You are DrAI-Tutor, an expert NEET-PG AI assistant. Provide accurate, student-friendly, and comprehensive answers to medical questions.

Question: {question}

Answer:"""

def create_notes_prompt(topic):
    """Create prompt for notes generation"""
    # Create a formatted prompt string for AI model to generate medical notes
    return f"""You are DrAI-Tutor, an expert NEET-PG AI assistant. Generate comprehensive, well-structured notes for NEET-PG preparation.

Topic: {topic}

Notes:
"""

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

def get_motivational_quotes():
    """Return list of motivational quotes for medical students"""
    # Return a list of motivational quotes specifically for medical students
    return [
        # Quote 1: Encourages consistent study habits
        "🎯 Every hour of study brings you closer to your dream of becoming a doctor!",
        # Quote 2: Emphasizes the importance of continuous learning
        "🧠 Your brain is your most powerful tool - keep feeding it knowledge!",
        # Quote 3: Motivates during difficult times
        "💪 You're stronger than you think. Keep pushing forward!",
        # Quote 4: Connects present effort to future success
        "📚 Today's effort is tomorrow's success in the medical field!",
        # Quote 5: Builds confidence for NEET-PG exam
        "🌟 You have what it takes to ace NEET-PG! Believe in yourself!",
        # Quote 6: Reminds of the impact on future patients
        "🔬 Every concept you learn today saves a life tomorrow!",
        # Quote 7: Encourages consistency over perfection
        "⚡ Consistency beats perfection. Keep going!",
        # Quote 8: Motivates by thinking about future patients
        "🎓 Your future patients are waiting for the amazing doctor you'll become!",
        # Quote 9: Encourages clearing doubts actively
        "💡 Every doubt you clear today makes you a better doctor tomorrow!",
        # Quote 10: Emphasizes the need for dedicated professionals
        "🏥 The medical field needs dedicated professionals like you!"
    ] 