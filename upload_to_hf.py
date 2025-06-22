"""
Upload DrAI Medical Tutor to Hugging Face Space
Uses Hugging Face Hub API to upload files directly
"""

import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_to_hf_space():
    """Upload files to Hugging Face Space"""
    print("🧠 DrAI Medical Tutor - Upload to Hugging Face Space")
    print("=" * 60)
    
    # Space details
    space_name = "Shubham25180/Dr-Ai"
    
    print(f"Target Space: {space_name}")
    print("\n📋 Files to upload:")
    
    # List of files to upload
    files_to_upload = [
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
        "README.md",
        "data/training_dataset.jsonl",
        "data/starter_dataset.jsonl", 
        "data/dataset_template.csv",
        "fine_tuning/train.py",
        "fine_tuning/inference.py"
    ]
    
    # Show files that exist
    existing_files = []
    for file_path in files_to_upload:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (not found)")
    
    print(f"\n📊 Found {len(existing_files)} files to upload")
    
    print("\n🔐 Authentication Required!")
    print("You need to login to Hugging Face first:")
    print("1. Run: huggingface-cli login")
    print("2. Enter your token when prompted")
    print("3. Then run this script again")
    
    print("\n💡 Alternative: Manual Upload")
    print("1. Go to https://huggingface.co/spaces/Shubham25180/Dr-Ai")
    print("2. Click 'Files' tab")
    print("3. Click 'Add file' for each file above")
    
    return existing_files

def create_upload_guide():
    """Create a detailed upload guide"""
    guide = """# DrAI Medical Tutor - Upload Guide

## Option 1: Using Hugging Face CLI (Recommended)

1. **Login to Hugging Face:**
   ```bash
   huggingface-cli login
   ```
   Enter your token when prompted.

2. **Upload files:**
   ```bash
   python upload_to_hf.py
   ```

## Option 2: Manual Web Upload

1. **Go to your Space:** https://huggingface.co/spaces/Shubham25180/Dr-Ai
2. **Click 'Files' tab**
3. **Click 'Add file' for each file:**

### Core Files:
- `app.py` (main application)
- `app_enhanced.py` (enhanced version)
- `requirements.txt` (dependencies)
- `mock_questions.json` (test questions)
- `README.md` (space description)

### Utility Files:
- `utils.py` (utility functions)
- `model_utils.py` (model management)
- `motivation.py` (motivation system)
- `data_preparation.py` (data processing)
- `validate_dataset.py` (validation)

### Data Files:
- `data/training_dataset.jsonl`
- `data/starter_dataset.jsonl`
- `data/dataset_template.csv`

### Fine-tuning Files:
- `fine_tuning/train.py`
- `fine_tuning/inference.py`

## Option 3: Git Push (Advanced)

1. **Create access token:** https://huggingface.co/settings/tokens
2. **Configure git:**
   ```bash
   git config --global credential.helper store
   git push space main
   ```
3. **Use token as password when prompted**

## After Upload:
1. Wait for the Space to rebuild (5-10 minutes)
2. Test your app at: https://huggingface.co/spaces/Shubham25180/Dr-Ai
3. Share the link with others!

## Troubleshooting:
- If files don't appear, wait a few minutes for the Space to rebuild
- Check the 'Logs' tab for any errors
- Make sure all dependencies are in requirements.txt
"""
    
    with open("UPLOAD_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(guide)
    print("✅ Created UPLOAD_GUIDE.md")

def main():
    """Main function"""
    upload_to_hf_space()
    create_upload_guide()
    
    print("\n" + "=" * 60)
    print("📚 Check UPLOAD_GUIDE.md for detailed instructions!")
    print("🌐 Your Space: https://huggingface.co/spaces/Shubham25180/Dr-Ai")
    print("=" * 60)

if __name__ == "__main__":
    main() 