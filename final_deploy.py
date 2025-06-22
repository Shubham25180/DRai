import os

if __name__ == "__main__":
    # The user's token should be provided securely, for example, via an environment variable.
    # Do NOT hardcode your token in the script.
    user_token = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
    if user_token == "YOUR_HF_TOKEN_HERE":
        print("🚨 Please set your Hugging Face token as an environment variable (HF_TOKEN) or replace 'YOUR_HF_TOKEN_HERE'.")
    else:
        deploy_to_huggingface(user_token) 