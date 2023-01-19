# AllSet

## Run Hypergraph Sheaf experiments
 --tag argument is used just to append a tag in wandb which might help you select the experiments later.
 I am using  --tag testing for dev experiments and  --tag stable for more stable results
 
For Diagonal Hypergraph Sheaves:

```
CUDA_VISIBLE_DEVICES=0 python train.py --dname cora --method DiagSheafs --MLP_hidden 256 --Classifier_hidden 256 --All_num_layers 2 --heads 5 --init_hedge avg --epochs 100 --sheaf_normtype block_norm --runs 20
```

For Orthogonal Hypergraph Sheaves:

```
CUDA_VISIBLE_DEVICES=0 python train.py --dname cora --method OrthoSheafs --MLP_hidden 256 --Classifier_hidden 256 --All_num_layers 2 --heads 5 --init_hedge avg --epochs 100 --sheaf_normtype degree_norm --runs 20
```

For General Hypergraph Sheaves:

```
CUDA_VISIBLE_DEVICES=0 python train.py --dname cora --method GeneralSheafs --MLP_hidden 256 --Classifier_hidden 256 --All_num_layers 2 --heads 5 --init_hedge avg --epochs 100 --sheaf_normtype degree_norm --runs 20
```

## Enviroment requirement:
This repo is tested with the following enviroment, higher version of torch PyG may also be compatible. 

First let's setup a conda enviroment
```
conda create -n "sheaf" python=3.7
conda activate sheaf
```

Then install pytorch and PyG packages with specific version.
```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-sparse==0.6.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-cluster==1.5.2 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-geometric==1.6.3 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
```
Finally, install some relative packages

```
pip install ipdb
pip install tqdm
pip install scipy
pip install matplotlib
```

## Generate dataset from raw data.

extract data from data.tar.gz 

## Create new model

inside models.py you call all the models

## Run one single experiment with one model with specified lr and wd: 
```
source run_one_model.sh [dataset] [method] [MLP_hidden_dim] [Classifier_hidden_dim] [feature noise level]
```

or:

```
python train.py --dname cora --method HCHA --MLP_hidden 256 --Classifier_hidden 256
```


## Issues
If you have any problem about our code, please open an issue **and** @ us (or send us an email) in case the notification doesn't work. Our email can be found in the paper.



