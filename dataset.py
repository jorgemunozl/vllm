from datasets import load_dataset

dataSet0 = "MananSuri27/Flowchart2Mermaid"
dataSet1 = "sroecker/mermaid-flowchart-transformer-moondream-caption"
dataSet2 = "rakitha/mermaid-flowchart-transformer"

ds = load_dataset(dataSet0)

print(f"Loading dataset: {dataSet0}")
print(f"Dataset features: {ds['train'].features}")
print(f"Number of training samples: {len(ds['train'])}")

trainDataSet = ds['train']
testDataSet = ds.get('test', ds.get('validation', []))

print(f"Training samples: {len(trainDataSet) if trainDataSet else 0}")
print(f"Test samples: {len(testDataSet) if testDataSet else 0}")

dataSet0 = "MananSuri27/Flowchart2Mermaid"
dataSet1 = "sroecker/mermaid-flowchart-transformer-moondream-caption"
dataSet2 = "rakitha/mermaid-flowchart-transformer"
