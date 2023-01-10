#! /bin/sh
#
# Copyright (C) 2021 
#
# Distributed under terms of the MIT license.
#
# dname=$1
# method=$2
lr=0.001
wd=0
# MLP_hidden=$3
# Classifier_hidden=$4
feature_noise=0.6

dataset_list= ( cora citeseer pubmed )
method_list=( HCHA )

for MLP_hidden in 64 128 256 512
do
    for Classifier_hidden in 64 128 256
    do
        for dname in ${dataset_list[*]} 
        do
            for method in ${method_list[*]}
            do
                source run_one_model.sh $dname $method $MLP_hidden $Classifier_hidden $feature_noise
            done
        done
    done   
done
