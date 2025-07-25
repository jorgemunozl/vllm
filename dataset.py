from datasets import load_dataset

dataSet0 = "MananSuri27/Flowchart2Mermaid"
# dataSet1 = "sroecker/mermaid-flowchart-transformer-moondream-caption" <--- Future Improve
# dataSet2 = "rakitha/mermaid-flowchart-transformer" <--- Future

ds = load_dataset(dataSet0)
trainDataSet = ds['train']
testDataSet = ds['validation']
