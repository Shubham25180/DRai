"""
Push DrAI Medical Tutor to Hugging Face Spaces
Automatically uploads all necessary files to your HF Space
"""

import os
import subprocess
import sys
from pathlib import Path

def check_git_lfs():
    """Check if Git LFS is installed"""
    try:
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def setup_hf_space():
    """Setup the Hugging Face Space repository"""
    space_name = "Shubham25180/Dr-Ai"
    
    print(f"Setting up Hugging Face Space: {space_name}")
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("Error: app.py not found. Make sure you're in the project directory.")
        return False
    
    # Initialize git if not already done
    if not os.path.exists(".git"):
        print("Initializing git repository...")
        subprocess.run(["git", "init"], check=True)
    
    # Add Hugging Face Space as remote
    try:
        subprocess.run(["git", "remote", "add", "space", f"https://huggingface.co/spaces/{space_name}"], check=True)
        print("Added HF Space as remote")
    except subprocess.CalledProcessError:
        print("Remote already exists or error adding remote")
    
    return True

def create_hf_files():
    """Create necessary files for Hugging Face Spaces"""
    print("Creating Hugging Face Space files...")
    
    # Create README.md for HF Space
    hf_readme = """---
title: DrAI Medical Tutor
emoji: 🧠
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
- 🤖 AI-Powered Doubt Clearance
- 📝 Smart Notes Generator  
- 📊 Interactive Mock Tests
- 💪 Motivation System

Built with ❤️ for medical students.
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(hf_readme)
    print("✅ Created README.md for HF Space")

def get_files_to_push():
    """Get list of files to push to HF Space"""
    files = [
        "app.py",
        "app_enhanced.py", 
        "demo.py",
        "demo_enhanced.py",
        "model_utils.py",
        "utils.py",
        "motivation.py",
        "data_preparation.py",
        "validate_dataset.py",
        "test_enhanced.py",
        "requirements.txt",
        "mock_questions.json",
        "README.md"
    ]
    
    # Add data files
    data_files = [
        "data/dataset_template.csv",
        "data/starter_dataset.jsonl", 
        "data/training_dataset.jsonl"
    ]
    
    # Add fine-tuning files
    fine_tuning_files = [
        "fine_tuning/train.py",
        "fine_tuning/inference.py"
    ]
    
    all_files = files + data_files + fine_tuning_files
    
    # Filter out files that don't exist
    existing_files = [f for f in all_files if os.path.exists(f)]
    
    return existing_files

def push_to_hf():
    """Push all files to Hugging Face Space"""
    print("Pushing files to Hugging Face Space...")
    
    # Get files to push
    files = get_files_to_push()
    print(f"Found {len(files)} files to push:")
    for f in files:
        print(f"  - {f}")
    
    try:
        # Add all files
        subprocess.run(["git", "add", "."], check=True)
        print("✅ Added all files to git")
        
        # Commit changes
        subprocess.run(["git", "commit", "-m", "Update DrAI Medical Tutor with all features"], check=True)
        print("✅ Committed changes")
        
        # Push to HF Space
        print("Pushing to Hugging Face Space...")
        subprocess.run(["git", "push", "space", "main"], check=True)
        print("✅ Successfully pushed to Hugging Face Space!")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error pushing to HF Space: {e}")
        return False

def main():
    """Main function to push to HF Space"""
    print("🧠 DrAI Medical Tutor - Push to Hugging Face Space")
    print("=" * 60)
    
    # Setup HF Space
    if not setup_hf_space():
        print("❌ Failed to setup HF Space")
        return
    
    # Create HF files
    create_hf_files()
    
    # Push to HF Space
    if push_to_hf():
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Your DrAI Medical Tutor is now live!")
        print("🌐 Visit: https://huggingface.co/spaces/Shubham25180/Dr-Ai")
        print("=" * 60)
    else:
        print("\n❌ Failed to push to HF Space. Please check the errors above.")

if __name__ == "__main__":
    main() 