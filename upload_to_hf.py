from huggingface_hub import HfApi, create_repo
import os

MODEL_NAME = "jorgemunozl/flowchart2mermaid"
LOCAL_MODEL_PATH = "/home/jorge/project/githubProjects/vllm"

def upload_model(files_to_update=None, update_mode=False):
    api = HfApi()
    
    if not update_mode:
        try:
            create_repo(
                MODEL_NAME, 
                exist_ok=True,
                private=True  # Set to False for public repository
            )
            print(f"✅ Private repository created: https://huggingface.co/{MODEL_NAME}")
        except Exception as e:
            print(f"Repository might already exist: {e}")
    
    # Default files or custom list
    files_to_upload = files_to_update or [
        "adapter_model.safetensors",  
        "adapter_config.json",                  
        "README.md"                 
        ]
    print("📤 Starting upload...")
    for file_name in files_to_upload:
        file_path = os.path.join(LOCAL_MODEL_PATH, file_name)
        if os.path.exists(file_path):
            print(f"⏫ Uploading {file_name}...")
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_name,
                    repo_id=MODEL_NAME,
                    commit_message=f"Add {file_name}"
                )
                print(f"✅ Successfully uploaded {file_name}")
            except Exception as e:
                print(f"❌ Error uploading {file_name}: {e}")
        else:
            print(f"⚠️  File not found: {file_path}")
    print(f"🎉 Upload complete! Model available at: https://huggingface.co/{MODEL_NAME}")


if __name__ == "__main__":
    # Full upload (first time)
    # upload_model()
    
    # Update specific files only
    upload_model(files_to_update=["README.md"], update_mode=True)
    
    # Update model files
    # upload_model(files_to_update=["adapter_model.safetensors", "adapter_config.json"], update_mode=True)
