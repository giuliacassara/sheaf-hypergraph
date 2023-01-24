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

## Hyperparameters to tune:
**—method:** DiagSheafs, OrthoSheafs, GeneralSheafs           # vary the constrans on the dxd block

**—heads:** int (usually 1-6)          # for the sheaf methods this confusingly refers to the dim of the stalk

**—sheaf_pred_block:** MLP_var1, MLP_var2, transformer          # the encoder that pred (node, hedge) → dxd block

**—sheaf_normtype:** degree_norm, block_norm;           #type of normalisation hnn-like or sheaf-like

Obs: GeneralSheafs does not work with block_norm yet

**—sheaf_act**: sigmoid, tanh, none          # activation used on top of the dxd block

**—sheaf_dropout:** any float number 0-1          # dropout p on top of the dxd block

**—sheaf_left_proj:** True or False          # use [(IxW1) X W2] or [X W2] inside the model 

**—dynamic_sheaf:** True or False          # predict the sheaf every layer or just in the first layer

**—sheaf_special_head:** True or False          # append a, extra dimension =1 for each (node, hedge)

**—sheaf_transformer_head:** int (usually 1-8)          # number of heads used in the transformer predictor

Obs: only when sheaf_pred_block==transformer

**—init_hedge:** rand, avg          # how to initialise hyeredge attr when not available

**—lr:**          # learning rate

**—wd:**          # weight decay

**—All_num_layers**          # number of propagation layers

**—MLP_hidden**          # number of hidden units in each layer

**—add_self_loop**          # addor not self loop

are there others?

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



