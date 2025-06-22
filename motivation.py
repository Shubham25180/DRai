"""
Daily Motivation Script for Medical Students
"""

# Import random library for selecting random quotes
import random
# Import time library for sleep functionality
import time
# Import JSON library for saving quote history
import json
# Import datetime for timestamp functionality
from datetime import datetime
# Import motivational quotes function from utils module
from utils import get_motivational_quotes

class MotivationPing:
    def __init__(self):
        # Initialize quotes list by getting all motivational quotes
        self.quotes = get_motivational_quotes()
        # Initialize last quote time as None (no quotes shown yet)
        self.last_quote_time = None
        
    def get_motivational_quote(self):
        """Get a random motivational quote"""
        # Select a random quote from the quotes list
        quote = random.choice(self.quotes)
        # Get current timestamp in formatted string
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Return quote data as dictionary with metadata
        return {
            "quote": quote,  # The actual quote text
            "timestamp": timestamp,  # When the quote was generated
            "type": "motivation"  # Type identifier for the quote
        }
    
    def display_quote(self):
        """Display a motivational quote with formatting"""
        # Get quote data with timestamp
        quote_data = self.get_motivational_quote()
        # Print top separator line
        print("\n" + "="*60)
        # Print header for motivation ping
        print("🧠 DrAI-Tutor Motivation Ping")
        # Print bottom separator line
        print("="*60)
        # Print timestamp when quote was generated
        print(f"📅 {quote_data['timestamp']}")
        # Print the actual motivational quote
        print(f"💬 {quote_data['quote']}")
        # Print final separator line
        print("="*60 + "\n")
        # Return quote data for potential further use
        return quote_data
    
    def continuous_motivation(self, interval_hours: float = 1.0):
        """Run continuous motivation pings"""
        # Print startup message
        print("🚀 Starting DrAI-Tutor Motivation Service...")
        # Print information about service frequency
        print("💡 You'll receive motivational quotes every hour!")
        # Print instructions for stopping the service
        print("⏹️  Press Ctrl+C to stop\n")
        
        # Wrap in try-catch to handle user interruption
        try:
            # Infinite loop for continuous motivation
            while True:
                # Display current motivational quote
                self.display_quote()
                # Print message about next motivation timing
                print(f"⏰ Next motivation in {interval_hours} hour(s)...")
                # Sleep for specified interval (convert hours to seconds)
                time.sleep(interval_hours * 3600)  # Convert hours to seconds
        except KeyboardInterrupt:
            # Handle user pressing Ctrl+C to stop
            print("\n🛑 Motivation service stopped by user.")
    
    def save_quote_history(self, quote_data, filename="motivation_history.json"):
        """Save quote to history file"""
        # Try to load existing history file
        try:
            # Open and read existing history file
            with open(filename, 'r') as f:
                # Parse JSON content into Python list
                history = json.load(f)
        except FileNotFoundError:
            # If file doesn't exist, start with empty list
            history = []
        
        # Add new quote data to history list
        history.append(quote_data)
        
        # Save updated history back to file
        with open(filename, 'w') as f:
            # Write history as formatted JSON
            json.dump(history, f, indent=2)

def main():
    """Main function to run motivation service"""
    # Create instance of MotivationPing class
    motivator = MotivationPing()
    
    # Print application header
    print("🎯 DrAI-Tutor Motivation System")
    # Print menu options for user
    print("Choose an option:")
    # Option 1: Single quote
    print("1. Get one motivational quote")
    # Option 2: Continuous service (hourly)
    print("2. Start continuous motivation service (every hour)")
    # Option 3: Continuous service (every 30 minutes)
    print("3. Start continuous motivation service (every 30 minutes)")
    
    # Get user input for choice
    choice = input("\nEnter your choice (1-3): ").strip()
    
    # Handle user choice
    if choice == "1":
        # Display single motivational quote
        motivator.display_quote()
    elif choice == "2":
        # Start continuous motivation with 1-hour intervals
        motivator.continuous_motivation(interval_hours=1.0)
    elif choice == "3":
        # Start continuous motivation with 30-minute intervals
        motivator.continuous_motivation(interval_hours=0.5)
    else:
        # Handle invalid input by showing default (single quote)
        print("❌ Invalid choice. Running default: one quote")
        motivator.display_quote()

# Main execution block
if __name__ == "__main__":
    # Call the main function
    main() 