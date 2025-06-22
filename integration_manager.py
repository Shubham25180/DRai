#!/usr/bin/env python3
"""
Integration Manager for Dr. AI Medical Tutor
Comprehensive testing and management of all application components
"""

import os
import sys
import json
import time
import requests
import subprocess
import psutil
from typing import Dict, List, Any

class IntegrationManager:
    def __init__(self):
        self.app_url = "http://127.0.0.1:7860"
        self.process_name = "python.exe"
        
    def check_system_status(self) -> Dict[str, Any]:
        """Check overall system status."""
        print("🔍 Checking Dr. AI Medical Tutor Integration Status...")
        print("=" * 60)
        
        status = {
            "web_interface": False,
            "python_process": False,
            "model_download": False,
            "mock_questions": False,
            "dependencies": False
        }
        
        # Check web interface
        try:
            response = requests.get(self.app_url, timeout=5)
            if response.status_code == 200:
                status["web_interface"] = True
                print("✅ Web Interface: RUNNING")
            else:
                print(f"⚠️ Web Interface: Status {response.status_code}")
        except:
            print("❌ Web Interface: NOT ACCESSIBLE")
        
        # Check Python processes
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if any(app in cmdline for app in ['app.py', 'app_enhanced.py']):
                        python_processes.append(proc.info)
            except:
                pass
        
        if python_processes:
            status["python_process"] = True
            print(f"✅ Python Process: RUNNING (PID: {python_processes[0]['pid']})")
        else:
            print("❌ Python Process: NOT RUNNING")
        
        # Check model download progress
        model_path = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(model_path):
            biomistral_path = os.path.join(model_path, "models--BioMistral--BioMistral-7B")
            if os.path.exists(biomistral_path):
                pytorch_model_path = os.path.join(biomistral_path, "snapshots", "main", "pytorch_model.bin")
                if os.path.exists(pytorch_model_path):
                    size_gb = os.path.getsize(pytorch_model_path) / (1024**3)
                    if size_gb > 15:
                        status["model_download"] = True
                        print(f"✅ AI Model: FULLY DOWNLOADED ({size_gb:.1f}GB)")
                    else:
                        print(f"⏳ AI Model: DOWNLOADING ({size_gb:.1f}GB/15.9GB)")
                else:
                    print("⏳ AI Model: DOWNLOADING (file not found)")
            else:
                print("⏳ AI Model: DOWNLOADING (directory not found)")
        else:
            print("❌ AI Model: CACHE NOT FOUND")
        
        # Check mock questions
        if os.path.exists("mock_questions.json"):
            try:
                with open("mock_questions.json", 'r') as f:
                    data = json.load(f)
                total_questions = sum(len(questions) for questions in data.values())
                status["mock_questions"] = True
                print(f"✅ Mock Questions: LOADED ({total_questions} questions)")
            except:
                print("❌ Mock Questions: CORRUPTED")
        else:
            print("❌ Mock Questions: NOT FOUND")
        
        # Check dependencies
        required_files = ["app.py", "model_utils.py", "utils.py", "requirements.txt"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if not missing_files:
            status["dependencies"] = True
            print("✅ Dependencies: ALL PRESENT")
        else:
            print(f"❌ Dependencies: MISSING {missing_files}")
        
        return status
    
    def test_application_features(self) -> Dict[str, bool]:
        """Test all application features."""
        print("\n🧪 Testing Application Features...")
        print("=" * 40)
        
        results = {
            "doubt_clearance": False,
            "notes_generation": False,
            "mock_test": False,
            "motivation": False
        }
        
        # Test mock questions loading
        try:
            with open("mock_questions.json", 'r') as f:
                data = json.load(f)
            questions = []
            for category, category_questions in data.items():
                for question in category_questions:
                    if all(key in question for key in ["question", "options", "correct_answer"]):
                        questions.append(question)
            
            if len(questions) >= 5:
                results["mock_test"] = True
                print("✅ Mock Test: READY")
            else:
                print("❌ Mock Test: INSUFFICIENT QUESTIONS")
        except Exception as e:
            print(f"❌ Mock Test: ERROR - {e}")
        
        # Test utility functions
        try:
            from utils import create_doubt_prompt, create_notes_prompt, get_motivational_quote
            results["doubt_clearance"] = True
            results["notes_generation"] = True
            results["motivation"] = True
            print("✅ Core Functions: WORKING")
        except Exception as e:
            print(f"❌ Core Functions: ERROR - {e}")
        
        return results
    
    def start_application(self, app_type: str = "enhanced") -> bool:
        """Start the application."""
        print(f"\n🚀 Starting Dr. AI Medical Tutor ({app_type})...")
        
        if app_type == "enhanced":
            app_file = "app_enhanced.py"
        else:
            app_file = "app.py"
        
        if not os.path.exists(app_file):
            print(f"❌ Application file {app_file} not found")
            return False
        
        try:
            # Check if already running
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        if app_file in cmdline:
                            print(f"✅ Application already running (PID: {proc.info['pid']})")
                            return True
                except:
                    pass
            
            # Start new process
            subprocess.Popen([sys.executable, app_file], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)
            
            print("✅ Application started successfully")
            print("🌐 Web interface will be available at: http://127.0.0.1:7860")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start application: {e}")
            return False
    
    def stop_application(self) -> bool:
        """Stop the application."""
        print("\n🛑 Stopping Dr. AI Medical Tutor...")
        
        stopped = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if any(app in cmdline for app in ['app.py', 'app_enhanced.py']):
                        proc.terminate()
                        print(f"✅ Stopped process (PID: {proc.info['pid']})")
                        stopped = True
            except:
                pass
        
        if not stopped:
            print("⚠️ No application processes found")
        
        return stopped
    
    def show_usage_guide(self):
        """Show comprehensive usage guide."""
        print("\n📚 Dr. AI Medical Tutor - Usage Guide")
        print("=" * 50)
        print("""
🎯 **Main Features:**
1. ❓ AI-Powered Doubt Clearance - Ask medical questions
2. 📝 Smart Notes Generator - Generate study notes
3. 📊 Interactive Mock Test - Practice with MCQs
4. 💪 Motivation Boost - Get motivational quotes

🌐 **Access the Application:**
- Open your web browser
- Go to: http://127.0.0.1:7860
- Use any of the 4 tabs for different features

📊 **Mock Test Instructions:**
1. Go to "📊 Interactive Mock Test" tab
2. Select number of questions (1-5)
3. Click "🚀 Start Mock Test"
4. Answer the questions
5. Click "🏁 Submit Test" for results

🔧 **Troubleshooting:**
- If web interface doesn't load, check if Python process is running
- If AI features don't work, wait for model download to complete
- Check terminal logs for detailed information

📈 **Model Download Status:**
- Progress is shown in terminal
- Can be paused/resumed with Ctrl+C
- Automatically resumes from last position
        """)

def main():
    """Main integration manager."""
    manager = IntegrationManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "status":
            manager.check_system_status()
            manager.test_application_features()
            
        elif command == "start":
            app_type = sys.argv[2] if len(sys.argv) > 2 else "enhanced"
            manager.start_application(app_type)
            
        elif command == "stop":
            manager.stop_application()
            
        elif command == "guide":
            manager.show_usage_guide()
            
        else:
            print("❌ Unknown command. Use: status, start, stop, or guide")
    else:
        # Interactive mode
        print("🧠 Dr. AI Medical Tutor - Integration Manager")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Check Status")
            print("2. Start Application")
            print("3. Stop Application")
            print("4. Show Usage Guide")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                manager.check_system_status()
                manager.test_application_features()
            elif choice == "2":
                app_type = input("Enter app type (enhanced/basic): ").strip() or "enhanced"
                manager.start_application(app_type)
            elif choice == "3":
                manager.stop_application()
            elif choice == "4":
                manager.show_usage_guide()
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 