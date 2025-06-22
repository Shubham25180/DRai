"""
DrAI Medical Tutor - Demo Script
A lightweight demo to showcase the basic features without loading the full AI model
"""

# Import JSON library for reading mock questions data
import json
# Import random library for random selections
import random
# Import motivational quotes function from utils module
from utils import get_motivational_quotes

def demo_mock_test():
    """Demo the mock test functionality"""
    # Print demo header for mock test feature
    print("🧠 DrAI Medical Tutor - Mock Test Demo")
    # Print separator line for better readability
    print("="*50)
    
    # Load mock questions from JSON file
    try:
        # Open and read the mock questions JSON file
        with open('mock_questions.json', 'r') as f:
            # Parse JSON content into Python dictionary
            questions = json.load(f)
    except FileNotFoundError:
        # If file not found, print error and exit function
        print("❌ mock_questions.json not found!")
        return
    
    # Get list of available topics from the questions dictionary
    topics = list(questions.keys())
    # Display available topics to user
    print(f"📚 Available Topics: {', '.join(topics)}")
    
    # Choose cardiology as demo topic
    topic = "cardiology"
    # Display the selected demo topic
    print(f"\n🎯 Demo Topic: {topic.upper()}")
    
    # Get first 2 questions from cardiology topic for demo
    topic_questions = questions[topic][:2]
    
    # Print section header for sample questions
    print(f"\n📝 Sample Questions:")
    # Print separator line
    print("-" * 30)
    
    # Loop through each question and display it
    for i, q in enumerate(topic_questions, 1):
        # Print question number and text
        print(f"\nQ{i}. {q['question']}")
        # Loop through each option (A, B, C, D)
        for option, text in q['options'].items():
            # Print option letter and text
            print(f"   {option}) {text}")
        # Print the correct answer
        print(f"   Correct Answer: {q['correct_answer']}")
        # Print explanation for the answer
        print(f"   💡 {q['explanation']}")
    
    # Print completion message
    print("\n✅ Mock test demo completed!")

def demo_motivation():
    """Demo the motivation feature"""
    # Print demo header for motivation feature
    print("\n💪 DrAI Medical Tutor - Motivation Demo")
    # Print separator line
    print("="*50)
    
    # Get list of motivational quotes from utils
    quotes = get_motivational_quotes()
    
    # Print section header for sample quotes
    print("🎯 Sample Motivational Quotes:")
    # Print separator line
    print("-" * 30)
    
    # Loop through first 3 quotes and display them
    for i, quote in enumerate(quotes[:3], 1):
        # Print quote number and text
        print(f"{i}. {quote}")
    
    # Display total number of quotes available
    print(f"\n📊 Total quotes available: {len(quotes)}")
    # Print completion message
    print("✅ Motivation demo completed!")

def demo_prompts():
    """Demo the prompt generation"""
    # Print demo header for prompt generation feature
    print("\n📝 DrAI Medical Tutor - Prompt Demo")
    # Print separator line
    print("="*50)
    
    # Import prompt creation functions from utils module
    from utils import create_doubt_prompt, create_notes_prompt
    
    # Create sample question for doubt clearance
    question = "What are the symptoms of myocardial infarction?"
    # Generate doubt clearance prompt using utility function
    doubt_prompt = create_doubt_prompt(question)
    
    # Print section header for doubt prompt
    print("❓ Doubt Clearance Prompt:")
    # Print separator line
    print("-" * 30)
    # Display the generated prompt
    print(doubt_prompt)
    
    # Create sample topic for notes generation
    topic = "Types of Shock"
    # Generate notes prompt using utility function
    notes_prompt = create_notes_prompt(topic)
    
    # Print section header for notes prompt
    print("\n📝 Notes Generation Prompt:")
    # Print separator line
    print("-" * 30)
    # Display the generated prompt
    print(notes_prompt)
    
    # Print completion message
    print("✅ Prompt demo completed!")

def main():
    """Main demo function"""
    # Print main demo header
    print("🚀 DrAI Medical Tutor - Feature Demo")
    # Print separator line
    print("="*60)
    # Print description of what the demo does
    print("This demo showcases the basic features without loading the AI model.")
    # Print separator line
    print("="*60)
    
    # Wrap demo execution in try-catch for error handling
    try:
        # Run mock test demo
        demo_mock_test()
        # Run motivation demo
        demo_motivation()
        # Run prompt generation demo
        demo_prompts()
        
        # Print final separator
        print("\n" + "="*60)
        # Print success message
        print("🎉 Demo completed successfully!")
        # Print next steps for user
        print("\n📋 Next Steps:")
        # Step 1: Install dependencies
        print("1. Install dependencies: pip install -r requirements.txt")
        # Step 2: Run full application
        print("2. Run the full app: python app.py")
        # Step 3: Load AI model
        print("3. Load the AI model in the Setup tab")
        # Step 4: Start using features
        print("4. Start using all features!")
        # Print final separator
        print("="*60)
        
    except Exception as e:
        # If any error occurs, print error message
        print(f"\n❌ Demo failed with error: {e}")
        # Print troubleshooting suggestion
        print("Please check that all files are present and try again.")

# Main execution block
if __name__ == "__main__":
    # Call the main demo function
    main() 