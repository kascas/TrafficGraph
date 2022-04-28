# How to use TrafficGraph

## Requirement

- PyTorch
- DGL
- seaborn
- networkx
- scikit-learn

## Overview

The folder structure of this project should be as follow:

```text
|--Data
    |--Dataset
        |--raw
            |--train.json
            |--valid.json
            |--test.json
        |--train_valid_test.graph
    |--Summary
        |--Labelled
        |--...(.json)
    |--Traffic
        |--...(.pcap)
|--Model
    |--MGNN.py
|--Preprocess
    |--Dataset.py
    |--TrafficLabel.py
    |--ZeekProcess.py
|--main.py
|--confusion_matrix.csv
|--confusion_matrix.svg
|--train_test.csv
|--train_test.svg
|--train_test.txt
```

Before running any script in this project, .pcap files should be placed in `./Data/Traffic`.

## Usage

- `python Preprocess/ZeekProcess.py -t -p`
  - Get Zeek logs according to network traffic
  - -p: only proto, -t: delete temp files
  - Zeek logs will be generated in `./Data/Summary`
- `python Preprocess/TrafficLabel.py`
  - Label every flow in logs, default type of labels is int
  - Labelled logs will be generated in `./Data/Summary/Labelled`
- `python main.py -i -r -d cicids2017`
  - Build or load dataset from labelled flows. Then train and test GNN model.
  - -i: initialize raw data, -r: rebuild graph, -d dataset_name: adjust the data size for the specified dataset
  - Dataset will be generated in `./Data/Dataset`.
  - Labelled flows are divided into train.json, valid.json and test.json in `./Data/Dataset/raw`
  - `./Data/Dataset/train_valid_test.graph` stores 3 graphs for model's training, validating and testing.
