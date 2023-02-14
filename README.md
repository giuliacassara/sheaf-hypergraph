# AllSet

 --tag argument is used just to append a tag in wandb which might help you select the experiments later.
 I am using  --tag testing for dev experiments and  --tag stable for more stable results

## Examples of scripts to run:
For training HCHA-based sheaves: run_hcha_sheaf.sh

For unit testing the models: run_unit_tests.sh

For training EDNN-based sheaves: run_edgnn_sheaf.sh


## The main classes of the code used for sheaf-based models:

`models.HyperSheafs(args, sheaf_type)` -- contains sheaf-based HCHA model  

`models.SheafHyperGCN(V, num_features, num_layers, num_classses, args, sheaf_type)` -- contains sheaf-based HGCN model  

`sheaf_builder.SheafBuilderDiag(args)`, `sheaf_builder.SheafBuilderOrtho(args)`, `sheaf_builder.SheafBuilderGeneral(args)` -- generate the sheaf reduction map (H \in nd x nd) Diag/General or Ortho 

`sheaf_builder.predict_blocks_*()` -- the model predicting d, d*(d-1)//2 or d^2 parameters used to build the reduction map: Transformer, mlp_var1 or mlp_var2 

`layers.HypergraphDiagSheafConv(...)`, `layers.HypergraphGeneralSheafConv(...)`, `layers.HypergraphOrthoSheafConv(...)` -- convolutional propagation for sheaf. Mainly same as before, with laplacian normalisation slightly change \\

`edgnn.SheafEquivSetGNN(... sheaf_type ..)` -- contains sheaf-based EDGNN model  

`hgcn_sheaf_laplacians.SheafLaplacian*` -- builds sheaf laplacians corresponding to the amin-amax graph extracted from the hyeprgraph (with mediators)


E.g. To create a HCHA-sheaf-based model with diagonal sheaf run:

```
model = HyperSheafs(args, 'DiagSheafs')
```


## Hyperparameters to tune:
🔆 **—method:** DiagSheafs, OrthoSheafs, GeneralSheafs, SheafHyperGCNDiag, SheafHyperGCNOrtho, SheafHyperGCNGeneral, SheafEquivSetGNN_Diag, SheafEquivSetGNN_Ortho, SheafEquivSetGNN_General           # vary the constrans on the dxd block on top of HCHA, HGCN or EquivSetGNN 

**—heads:** int (usually 1-6)          # for the sheaf methods this confusingly refers to the dim of the stalk

🔆 **—sheaf_pred_block:** MLP_var1, MLP_var2, transformer          # the encoder that pred (node, hedge) → dxd block

🔆 **—sheaf_normtype:** degree_norm, block_norm;           #type of normalisation hnn-like or sheaf-like

Obs: GeneralSheafs does not work with block_norm yet

**—sheaf_act**: sigmoid, tanh, none          # activation used on top of the dxd block; sigmoid tends to work way better

**—sheaf_dropout:** True or Dalse         # dropout p on top of the dxd block. if True inherit p value from —dropout

**—sheaf_left_proj:** True or False          # use [(IxW1) X W2] or [X W2] inside the model 

🔆 **—dynamic_sheaf:** True or False          # predict the sheaf every layer or just in the first layer

**—sheaf_special_head:** True or False          # append a, extra dimension =1 for each (node, hedge)

**—sheaf_transformer_head:** int (usually 1-8)          # number of heads used in the transformer predictor. MLP_hidden needs to divide this

Obs: only when sheaf_pred_block==transformer

**—init_hedge:** rand, avg          # how to initialise hyeredge attr when not available

**—lr:**          # learning rate

**—wd:**          # weight decay

**—All_num_layers**          # number of propagation layers

**—MLP_hidden**          # number of hidden units in each layer

**—add_self_loop**          # addor not self loop

**—dropout**

🔆 **--AllSet_input_norm**.     # True or False to indicate if we are usinf layernorm before linear projections. Recomand True 

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



