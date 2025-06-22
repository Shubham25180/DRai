"""
Git Push for Hugging Face Spaces
Simulates git push workflow for HF Spaces
"""

import subprocess
import os

def git_push_to_hf():
    """Push to Hugging Face Space using git workflow"""
    print("🚀 Git Push to Hugging Face Space")
    print("=" * 50)
    
    try:
        # Add all files
        print("📁 Adding files to git...")
        subprocess.run(["git", "add", "."], check=True)
        print("✅ Files added")
        
        # Commit changes
        print("💾 Committing changes...")
        subprocess.run(["git", "commit", "-m", "Update DrAI Medical Tutor - Complete deployment"], check=True)
        print("✅ Changes committed")
        
        # Push to GitHub (origin)
        print("📤 Pushing to GitHub...")
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("✅ Pushed to GitHub")
        
        # Push to Hugging Face Space
        print("🌐 Pushing to Hugging Face Space...")
        print("⚠️  You may need to authenticate with your HF token")
        subprocess.run(["git", "push", "space", "main"], check=True)
        print("✅ Successfully pushed to Hugging Face Space!")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False

def setup_authentication():
    """Setup authentication for HF Space"""
    print("\n🔐 Authentication Setup")
    print("=" * 30)
    
    print("To push to Hugging Face Space, you need:")
    print("1. Create access token: https://huggingface.co/settings/tokens")
    print("2. Configure git credentials:")
    print("   git config --global credential.helper store")
    print("3. When prompted during push:")
    print("   - Username: your HF username")
    print("   - Password: your HF token (not password)")
    
    print("\n💡 Alternative: Use manual upload")
    print("The upload package is ready in: hf_upload_package/")

def main():
    """Main function"""
    print("🧠 DrAI Medical Tutor - Git Push to HF Space")
    print("=" * 60)
    
    # Check if we're in git repo
    if not os.path.exists(".git"):
        print("❌ Not a git repository. Run 'git init' first.")
        return
    
    # Check remotes
    try:
        result = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True, check=True)
        print("📡 Git remotes:")
        print(result.stdout)
    except:
        print("❌ Error checking remotes")
        return
    
    # Try git push
    if git_push_to_hf():
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Your DrAI Medical Tutor is live!")
        print("🌐 GitHub: https://github.com/Shubham25180/DRai")
        print("🌐 HF Space: https://huggingface.co/spaces/Shubham25180/Dr-Ai")
        print("=" * 60)
    else:
        print("\n❌ Git push failed. Setting up authentication...")
        setup_authentication()

if __name__ == "__main__":
    main() 