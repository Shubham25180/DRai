"""
Test script for DrAI Medical Tutor Enhanced Features
Comprehensive testing suite for all enhanced functionality and model utilities.
"""

# Import required libraries
import json  # For reading JSON files
import random  # For random selections
# Import custom modules for testing
from model_utils import ModelManager, get_available_topics, evaluate_answers

def test_model_utils():
    """
    Test the model utilities functionality.
    Validates ModelManager initialization and basic operations.
    """
    print("🧪 Testing Model Utilities...")
    
    # Test ModelManager initialization
    manager = ModelManager()
    print("✅ ModelManager initialized successfully")
    
    # Test mock questions loading
    topics = get_available_topics()
    print(f"✅ Available topics: {topics}")
    
    # Test mock questions structure
    if topics:
        # Get first topic for testing
        sample_topic = topics[0]
        # Get sample questions for the topic
        questions = manager.get_mock_questions(sample_topic, 2)
        print(f"✅ Sample questions for {sample_topic}: {len(questions)} questions")
        
        if questions:
            # Test answer evaluation with sample answers
            sample_answers = ["A", "B"]  # Sample answers
            results = evaluate_answers(questions, sample_answers)
            print(f"✅ Answer evaluation test: {results.get('correct_answers', 0)}/{results.get('total_questions', 0)} correct")
    
    print("✅ All model utilities tests passed!")

def test_mock_questions():
    """
    Test mock questions structure and format.
    Validates JSON file format and question structure.
    """
    print("\n🧪 Testing Mock Questions...")
    
    try:
        # Load mock questions from JSON file
        with open('mock_questions.json', 'r') as f:
            data = json.load(f)
        
        print(f"✅ Mock questions loaded: {len(data)} topics")
        
        # Test each topic and its questions
        for topic, questions in data.items():
            print(f"  📚 {topic}: {len(questions)} questions")
            
            if questions:
                # Check structure of first question
                first_q = questions[0]
                required_fields = ['question', 'options', 'correct_answer']
                
                # Validate required fields
                for field in required_fields:
                    if field not in first_q:
                        print(f"  ❌ Missing field: {field}")
                        return False
                
                print(f"  ✅ Question structure valid")
        
        print("✅ Mock questions test passed!")
        return True
        
    except FileNotFoundError:
        print("⚠️ mock_questions.json not found")
        return False
    except json.JSONDecodeError:
        print("❌ Invalid JSON in mock_questions.json")
        return False

def test_utils():
    """
    Test utility functions from utils module.
    Validates motivational quotes and other utility functions.
    """
    print("\n🧪 Testing Utility Functions...")
    
    try:
        # Import and test motivational quotes
        from utils import get_motivational_quotes
        quotes = get_motivational_quotes()
        print(f"✅ Motivational quotes loaded: {len(quotes)} quotes")
        
        # Test random selection functionality
        random_quote = random.choice(quotes)
        print(f"✅ Random quote: {random_quote[:50]}...")
        
        print("✅ Utility functions test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """
    Main function to run all tests.
    Executes comprehensive testing of all enhanced features.
    """
    print("🚀 DrAI Medical Tutor - Enhanced Features Test")
    print("=" * 50)
    
    # Run all test functions
    test_model_utils()
    test_mock_questions()
    test_utils()
    
    # Print completion message and next steps
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\n📝 Next Steps:")
    print("1. Run 'python demo_enhanced.py' to test the enhanced UI")
    print("2. Run 'python app_enhanced.py' for full AI-powered experience")
    print("3. Load the AI model in the Setup tab for complete functionality")

# Main execution block
if __name__ == "__main__":
    # Call the main test function
    main() 