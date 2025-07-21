from datasets import load_dataset
from PIL import Image
import os

dataSet0 = "MananSuri27/Flowchart2Mermaid"
# dataSet1 = "sroecker/mermaid-flowchart-transformer-moondream-caption"
# dataSet2 = "rakitha/mermaid-flowchart-transformer"

ds = load_dataset(dataSet0)

print(f"Loading dataset: {dataSet0}")
print(f"Dataset features: {ds['train'].features}")
print(f"Number of training samples: {len(ds['train'])}")

trainDataSet = ds['train']
testDataSet = ds.get('test', ds.get('validation', []))

first_sample = trainDataSet[0]
print("Dataset structure:")
for key, value in first_sample.items():
    print(f"  {key}: {type(value)}")
    if hasattr(value, 'size'):
        print(f"    Size: {value.size}")

print(f"Training samples: {len(trainDataSet) if trainDataSet else 0}")
print(f"Test samples: {len(testDataSet) if testDataSet else 0}")


def download_first_sample():
    """Download the first sample to local directory"""
    
    # Use testDataSet if available, otherwise use trainDataSet
    dataset_to_use = (testDataSet if testDataSet and len(testDataSet) > 0
                      else trainDataSet)
    dataset_name = ("test" if testDataSet and len(testDataSet) > 0
                    else "train")
    
    if len(dataset_to_use) == 0:
        print("No samples available to download")
        return
    
    first_element = dataset_to_use[0]
    print(f"\nDownloading first element from {dataset_name} set...")
    
    # Create downloads directory
    os.makedirs("downloads", exist_ok=True)
    
    # Look for image field and download it
    for key, value in first_element.items():
        if isinstance(value, Image.Image):
            filename = f"downloads/first_sample_{key}.png"
            value.save(filename)
            print(f"✅ Saved image '{key}' to: {filename}")
            print(f"   Image size: {value.size}")
            print(f"   Image mode: {value.mode}")
        
        elif isinstance(value, str) and len(value) < 500:  # Text/caption
            filename = f"downloads/first_sample_{key}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(value)
            print(f"✅ Saved text '{key}' to: {filename}")
        
        elif isinstance(value, (list, dict)):
            filename = f"downloads/first_sample_{key}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(str(value))
            print(f"✅ Saved data '{key}' to: {filename}")
    
    print("\n📁 All files saved to: ./downloads/")


if __name__ == "__main__":
    download_first_sample()
