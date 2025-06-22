"""
Quick Upload Helper for DrAI Medical Tutor
Prepares files and instructions for manual upload to Hugging Face Space
"""

import os
import shutil
from pathlib import Path

def create_upload_package():
    """Create a package of files ready for upload"""
    print("📦 Creating Upload Package for DrAI Medical Tutor")
    print("=" * 60)
    
    # Create upload directory
    upload_dir = "hf_upload_package"
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    os.makedirs(upload_dir)
    
    # Files to upload
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
        "README.md"
    ]
    
    # Data files
    data_files = [
        "data/training_dataset.jsonl",
        "data/starter_dataset.jsonl",
        "data/dataset_template.csv"
    ]
    
    # Fine-tuning files
    fine_tuning_files = [
        "fine_tuning/train.py",
        "fine_tuning/inference.py"
    ]
    
    all_files = files_to_upload + data_files + fine_tuning_files
    
    # Copy files to upload directory
    copied_files = []
    for file_path in all_files:
        if os.path.exists(file_path):
            # Create subdirectories if needed
            dest_path = os.path.join(upload_dir, file_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Copy file
            shutil.copy2(file_path, dest_path)
            copied_files.append(file_path)
            print(f"✅ Copied: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
    
    # Create upload instructions
    instructions = f"""# DrAI Medical Tutor - Quick Upload Instructions

## 🎯 Your Space: https://huggingface.co/spaces/Shubham25180/Dr-Ai

## 📋 Upload Steps:
1. Go to your Space above
2. Click 'Files' tab
3. Click 'Add file' for each file below
4. Wait 5-10 minutes for rebuild

## 📁 Files to Upload ({len(copied_files)} files):

### Core Application Files:
- app.py (main application)
- app_enhanced.py (enhanced version)
- requirements.txt (dependencies)
- mock_questions.json (test questions)
- README.md (space description)

### Utility Files:
- utils.py (utility functions)
- model_utils.py (model management)
- motivation.py (motivation system)
- data_preparation.py (data processing)
- validate_dataset.py (validation)

### Demo Files:
- demo.py (basic demo)
- demo_enhanced.py (enhanced demo)
- test_enhanced.py (testing)

### Data Files:
- data/training_dataset.jsonl
- data/starter_dataset.jsonl
- data/dataset_template.csv

### Fine-tuning Files:
- fine_tuning/train.py
- fine_tuning/inference.py

## 🚀 After Upload:
1. Wait for Space to rebuild (check 'Logs' tab)
2. Test your app
3. Share the link: https://huggingface.co/spaces/Shubham25180/Dr-Ai

## 🆘 If Issues:
- Check 'Logs' tab for errors
- Make sure all files are uploaded
- Wait longer for rebuild
"""
    
    with open(os.path.join(upload_dir, "UPLOAD_INSTRUCTIONS.txt"), "w", encoding="utf-8") as f:
        f.write(instructions)
    
    print(f"\n📦 Upload package created in: {upload_dir}/")
    print(f"📄 {len(copied_files)} files ready for upload")
    print(f"📋 Instructions: {upload_dir}/UPLOAD_INSTRUCTIONS.txt")
    
    return upload_dir, copied_files

def main():
    """Main function"""
    upload_dir, files = create_upload_package()
    
    print("\n" + "=" * 60)
    print("🎉 READY TO UPLOAD!")
    print("=" * 60)
    print("🌐 Go to: https://huggingface.co/spaces/Shubham25180/Dr-Ai")
    print("📁 Upload files from: " + upload_dir)
    print("📋 Follow instructions in: " + upload_dir + "/UPLOAD_INSTRUCTIONS.txt")
    print("=" * 60)

if __name__ == "__main__":
    main() 