#!/usr/bin/env python3
"""
Script to update a specific file in your Hugging Face repository
"""

from huggingface_hub import HfApi
import os

MODEL_NAME = "jorgemunozl/flowchart2mermaid"
LOCAL_MODEL_PATH = "/home/jorge/project/githubProjects/vllm"


def update_file(filename, commit_message=None):
    """Update a specific file in the repository"""
    
    api = HfApi()
    file_path = os.path.join(LOCAL_MODEL_PATH, filename)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    if not commit_message:
        commit_message = f"Update {filename}"
    
    print(f"⏫ Updating {filename}...")
    
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=MODEL_NAME,
            commit_message=commit_message
        )
        print(f"✅ Successfully updated {filename}")
        print(f"🔗 View at: https://huggingface.co/{MODEL_NAME}/blob/main/{filename}")
    except Exception as e:
        print(f"❌ Error updating {filename}: {e}")

if __name__ == "__main__":
    # Example: Update specific files
    
    # Update the model file
    # update_file("adapter_model.safetensors", "Updated model weights")
    
    # Update config
    # update_file("adapter_config.json", "Updated configuration")
    
    # Update README
    update_file("README.md", "Updated documentation")
    
    # To update multiple files:
    # files_to_update = ["adapter_config.json", "README.md"]
    # for file in files_to_update:
    #     update_file(file)
