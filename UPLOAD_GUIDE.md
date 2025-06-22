# DrAI Medical Tutor - Upload Guide

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
