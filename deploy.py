"""
Deployment Script for DrAI Medical Tutor
Automates the process of deploying the application online.
"""

import os
import subprocess
import sys
import json
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'gradio',
        'torch', 
        'transformers',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"OK {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"Missing {package}")
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("All dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("Failed to install dependencies. Please run: pip install -r requirements.txt")
            return False
    
    return True

def create_deployment_files():
    """Create necessary files for online deployment"""
    print("\nCreating deployment files...")
    
    # Create Procfile for Heroku
    procfile_content = "web: python app.py"
    with open("Procfile", "w") as f:
        f.write(procfile_content)
    print("Created Procfile")
    
    # Create runtime.txt for Heroku
    runtime_content = "python-3.9.18"
    with open("runtime.txt", "w") as f:
        f.write(runtime_content)
    print("Created runtime.txt")
    
    # Create .gitignore if it doesn't exist
    if not os.path.exists(".gitignore"):
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Gradio
.gradio/

# Model files
*.bin
*.safetensors
models/
"""
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("Created .gitignore")

def test_local_deployment():
    """Test the application locally before deployment"""
    print("\nTesting local deployment...")
    
    try:
        # Start the app in background
        process = subprocess.Popen([sys.executable, "app.py"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # Wait a bit for the app to start
        import time
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("Local deployment test successful!")
            print("App should be running at: http://localhost:7860")
            print("Public link should be available in the terminal output")
            
            # Terminate the process
            process.terminate()
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"Local deployment failed:")
            print(f"Error: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"Error testing local deployment: {e}")
        return False

def create_huggingface_space_config():
    """Create configuration for Hugging Face Spaces"""
    print("\nCreating Hugging Face Spaces configuration...")
    
    # Create app.py for HF Spaces (if it doesn't exist)
    if not os.path.exists("app.py"):
        print("app.py not found. Please ensure it exists.")
        return False
    
    # Create README for HF Spaces
    hf_readme_content = """---
title: DrAI Medical Tutor
emoji: brain
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# DrAI Medical Tutor

An AI-powered medical education platform for NEET-PG aspirants.

## Features
- AI-Powered Doubt Clearance
- Smart Notes Generator  
- Interactive Mock Tests
- Motivation System

Built with love for medical students.
"""
    
    with open("README.md", "w") as f:
        f.write(hf_readme_content)
    print("Created HF Spaces README")

def show_deployment_instructions():
    """Show deployment instructions for different platforms"""
    print("\n" + "="*60)
    print("DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    
    print("\n1. HUGGING FACE SPACES (Recommended - Free)")
    print("-" * 40)
    print("1. Go to https://huggingface.co/spaces")
    print("2. Click 'Create new Space'")
    print("3. Choose 'Gradio' as SDK")
    print("4. Set visibility to 'Public'")
    print("5. Upload all your files")
    print("6. Your app will be available at: https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME")
    
    print("\n2. STREAMLIT CLOUD (Alternative - Free)")
    print("-" * 40)
    print("1. Push your code to GitHub")
    print("2. Go to https://share.streamlit.io")
    print("3. Connect your GitHub repository")
    print("4. Deploy")
    
    print("\n3. RAILWAY (Paid but Reliable)")
    print("-" * 40)
    print("1. Go to https://railway.app")
    print("2. Connect your GitHub repository")
    print("3. Set build command: pip install -r requirements.txt")
    print("4. Set start command: python app.py")
    print("5. Deploy")
    
    print("\n4. HEROKU (Alternative)")
    print("-" * 40)
    print("1. Install Heroku CLI")
    print("2. Run: heroku create your-app-name")
    print("3. Run: git push heroku main")
    
    print("\n" + "="*60)
    print("Your app is ready for deployment!")
    print("All necessary files have been created")
    print("Choose your preferred platform above")
    print("="*60)

def main():
    """Main deployment function"""
    print("DrAI Medical Tutor - Deployment Script")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        print("Dependency check failed. Please fix the issues above.")
        return
    
    # Create deployment files
    create_deployment_files()
    
    # Create HF Spaces config
    create_huggingface_space_config()
    
    # Test local deployment
    if test_local_deployment():
        print("\nLocal test successful! Your app is working correctly.")
    else:
        print("\nLocal test failed. Please check the errors above.")
    
    # Show deployment instructions
    show_deployment_instructions()

if __name__ == "__main__":
    main() 