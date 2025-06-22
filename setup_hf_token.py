"""
Setup Hugging Face Authentication Token
Helps you configure authentication for pushing to HF Spaces
"""

import subprocess
import os

def setup_hf_token():
    """Setup Hugging Face authentication token"""
    print("🔐 Hugging Face Authentication Setup")
    print("=" * 50)
    
    print("\nTo push to your Hugging Face Space, you need to:")
    print("1. Create an access token on Hugging Face")
    print("2. Configure git to use the token")
    
    print("\n📋 Steps to create your access token:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Give it a name (e.g., 'DrAI-Tutor-Deploy')")
    print("4. Select 'Write' permissions")
    print("5. Copy the generated token")
    
    print("\n🔧 After getting your token, run these commands:")
    print("git config --global credential.helper store")
    print("git push space main")
    print("(When prompted, use your username and the token as password)")
    
    print("\n💡 Alternative: Use the Hugging Face CLI")
    print("pip install huggingface_hub")
    print("huggingface-cli login")
    
    print("\n🌐 Or upload files directly through the web interface:")
    print("1. Go to https://huggingface.co/spaces/Shubham25180/Dr-Ai")
    print("2. Click 'Files' tab")
    print("3. Click 'Add file' to upload your files")
    print("4. Upload: app.py, requirements.txt, mock_questions.json, etc.")

def create_upload_script():
    """Create a script to help with manual upload"""
    script_content = """# Manual Upload Instructions for Hugging Face Space

# Files to upload to https://huggingface.co/spaces/Shubham25180/Dr-Ai

# Core Application Files:
- app.py (main application)
- app_enhanced.py (enhanced version)
- requirements.txt (dependencies)
- mock_questions.json (test questions)

# Utility Files:
- utils.py (utility functions)
- model_utils.py (model management)
- motivation.py (motivation system)

# Data Files:
- data/training_dataset.jsonl
- data/starter_dataset.jsonl
- data/dataset_template.csv

# Fine-tuning Files:
- fine_tuning/train.py
- fine_tuning/inference.py

# Steps:
# 1. Go to https://huggingface.co/spaces/Shubham25180/Dr-Ai
# 2. Click 'Files' tab
# 3. Click 'Add file' for each file above
# 4. Wait for the Space to rebuild
"""
    
    with open("UPLOAD_INSTRUCTIONS.txt", "w") as f:
        f.write(script_content)
    print("✅ Created UPLOAD_INSTRUCTIONS.txt")

def main():
    """Main function"""
    setup_hf_token()
    create_upload_script()
    
    print("\n" + "=" * 50)
    print("🎯 Quick Solution:")
    print("The easiest way is to upload files directly through the web interface!")
    print("Check UPLOAD_INSTRUCTIONS.txt for the file list.")
    print("=" * 50)

if __name__ == "__main__":
    main() 