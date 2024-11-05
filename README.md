# Virtual bonding enhanced graph neural network for molecular property prediction



## Background

To address challenges mentioned above and improve the
molecular property prediction task performance, we propose
a self-supervised learning framework for molecular property
prediction called Virtual Bonding Enhanced MPP (VIBE-
MPP). VIBE-MPP consists of two key elements, virtual
bonding graph neural network (VBGNN) and dual-level self-
supervised boosted pretraining(DSBP).

![未标题-1modified](https://github.com/user-attachments/assets/698d3271-6fa5-48db-9fc1-c328be9fcc99)



### Prerequisites

- OS support: Linux
- Python version: 3.6, 3.7, 3.8

### Dependencies

| name    | version |
| ------- | ------- |
| numpy   |    1.24.4     |
| pandas  |   1.5.3    |
| rdkit   |   2023.9.5     |
| torch      |    2.1.0     |
| torch-geometric      |  2.4.0       |
| torch-sparse        |    0.6.18     |
| tqdm             |  4.66.1 |
|networkx| 3.0  |



## Usage

### Tranning

You can pretrain the model by

```python
mkdir saved_model
python pretrain.py
```




### Evaluation

To assess the pretrained model's performance, you can fine-tune it on specific downstream tasks.

First, download the downstream task data available at [DeepChem's GitHub repository](https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/load_function). Save each dataset's `.csv` file within `./finetune/dataset/[dataset_name]/raw/`, substituting `[dataset_name]` with the appropriate dataset title. For instance, save `bace.csv` to `./finetune/dataset/bace/raw/bace.csv`.

```
cd finetune
mkdir model_checkpoints
python finetune.py
```



## Citation

If you use the code or data in this package, please cite:
