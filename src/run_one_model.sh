#! /bin/sh
#
# Copyright (C) 2021 
#
# Distributed under terms of the MIT license.
#


dname=$1
method=$2
lr=0.001
wd=0
MLP_hidden=$3
Classifier_hidden=$4
cuda=0

runs=10
epochs=500


    
if [ "$method" = "HCHA" ]; then
    echo =============
    echo ">>>>  Model:HCHA (asym deg norm), Dataset: ${dname}"
    python train.py \
        --method HCHA \
        --dname $dname \
        --All_num_layers 1 \
        --MLP_num_layers 2 \
        --Classifier_num_layers 1 \
        --MLP_hidden $MLP_hidden \
        --Classifier_hidden $Classifier_hidden \
        --wd $wd \
        --epochs $epochs \
        --runs $runs \
        --cuda $cuda \
        --lr $lr

fi
echo "Finished training ${method} on ${dname}"
