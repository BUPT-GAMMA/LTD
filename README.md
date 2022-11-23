# This is the supplementary materials for WSDM 2023 submission: Learning to Distill Graph Neural Networks

## Appendix.pdf [[download]](https://github.com/ltd-wsdm/ltd-code/archive/refs/heads/main.zip)

We listed the details for reproducibility and additional experiments analysis here. The directory is as follows:  
- Details for reproducibility:
  -  Experimental environments.
  -  The hyper-parameters used in LTD/FT.
  -  The settings for teacher/student models and other distillation frameworks.
  -  Some brief comments on data preparation.

- Additional experiments analysis:
  -  Balance hyper-parameter analysis. 
  -  Performance under different training ratios.
  -  Generalization gap analysis.
  -  Original results of baselines.

## Source code

### Requirements:

- dgl==0.6.0
- Keras==2.4.3
- numpy==1.19.2
- optuna==2.6.0
- pandas==1.2.4
- python==3.8.8
- scikit-learn==0.19.2
- scipy==1.6.2
- sklearn==0.0
- tabulate==0.8.9
- torch==1.8.1
- torchvision==0.9.1

### Data：
Because the A-Computers dataset is too large, we did not put it in the project file. The dataset can be downloaded from the following link and placed in the directory data/npz:

https://www.dropbox.com/s/26zd460xn4u6gmn/amazon_electronics_computers.npz?dl=0

### Training:

Run student model:

```
python spawn_worker.py --dataset XXXX --teacher XXX --student XXX
```

For example, if you want train GCN on cora dataset:

```
python spawn_worker.py --dataset cora --teacher GCN --student GCN
```

We fix hyper-parameters setting as reported in our paper, and you also can use `--automl` to search hyper-parameters with the help of Optuna. 
