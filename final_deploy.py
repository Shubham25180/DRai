"""
Final Deployment Script for DrAI Medical Tutor to Hugging Face Spaces.
This script uses the huggingface_hub library to programmatically upload
the project files to the specified Hugging Face Space.
"""

import os
import logging
import getpass
from huggingface_hub import HfApi, login

# ... (logging configuration remains the same)

def deploy_to_huggingface():
    """
    Prompts for a token, logs in, and uploads files to the Hugging Face Space.
    """
    repo_id = "Shubham25180/Dr-Ai"
    
    logging.info("🚀 Starting deployment to Hugging Face Space...")
    
    try:
        token = getpass.getpass("🔑 Please paste your Hugging Face token and press Enter: ")
        if not token:
            logging.error("❌ No token provided. Aborting.")
            return
    except Exception as e:
        logging.error(f"❌ Could not read token: {e}")
        return

    logging.info("Attempting to log in with provided token...")
    try:
        login(token=token, add_to_git_credential=True)
    except Exception as e:
        logging.error(f"❌ Failed to log in with token: {e}")
        return

    logging.info(f"✅ Login successful. Deploying to '{repo_id}'.")

    try:
        api = HfApi()
        
        # Define file patterns
        allow_patterns = ["*.py", "*.json", "*.csv", "*.md", "requirements.txt", "Procfile", "runtime.txt"]
        ignore_patterns = [".git/*", "__pycache__/*", "*.pyc", "*.spec", ".env", ".gitignore", "final_deploy.py"]

        logging.info("☁️  Uploading project files...")
        
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            commit_message="🚀 Deploy DrAI Medical Tutor",
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

        space_url = f"https://huggingface.co/spaces/{repo_id}"
        logging.info("✅ Deployment to Hugging Face Space was successful!")
        logging.info(f"🎉 Your DrAI Medical Tutor should be live shortly at: {space_url}")

    except Exception as e:
        logging.error(f"❌ An error occurred during deployment: {e}")

if __name__ == "__main__":
    deploy_to_huggingface() 