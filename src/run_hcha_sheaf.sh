
#run HCHA with DIAG  sheaves, DEGREE_NORM and MLP_VAR1 predictor

CUDA_VISIBLE_DEVICES=0 python train.py --dname cora  --method DiagSheafs --MLP_hidden 256 --Classifier_hidden 256 --All_num_layers 2 --MLP_num_layers 0 --MLP2_num_layers 0 --Classifier_num_layers 1 --heads 3 --init_hedge avg --epochs 100 --sheaf_normtype degree_norm --runs 1 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --sheaf_left_proj False --sheaf_dropout False --dynamic_sheaf False --tag tmp3 --sheaf_special_head False --sheaf_pred_block MLP_var1 --sheaf_transformer_head 4

#run HCHA with ORTHOGONAL sheaves, DEGREE_NORM and MLP_VAR1 predictor

CUDA_VISIBLE_DEVICES=0 python train.py --dname cora  --method OrthoSheafs --MLP_hidden 256 --Classifier_hidden 256 --All_num_layers 2 --MLP_num_layers 0 --MLP2_num_layers 0 --Classifier_num_layers 1 --heads 3 --init_hedge avg --epochs 100 --sheaf_normtype degree_norm --runs 1 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --sheaf_left_proj False --sheaf_dropout False --dynamic_sheaf False --tag tmp3 --sheaf_special_head False --sheaf_pred_block MLP_var1 --sheaf_transformer_head 4


#run HCHA with GENERAL sheaves, DEGREE_NORM and MLP_VAR1 predictor

CUDA_VISIBLE_DEVICES=0 python train.py --dname cora  --method GeneralSheafs --MLP_hidden 256 --Classifier_hidden 256 --All_num_layers 2 --MLP_num_layers 0 --MLP2_num_layers 0 --Classifier_num_layers 1 --heads 3 --init_hedge avg --epochs 100 --sheaf_normtype degree_norm --runs 1 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --sheaf_left_proj False --sheaf_dropout False --dynamic_sheaf False --tag tmp3 --sheaf_special_head False --sheaf_pred_block MLP_var1 --sheaf_transformer_head 4

#run HCHA with DIAG  sheaves, DEGREE_NORM and TRANSFORMER predictor

CUDA_VISIBLE_DEVICES=0 python train.py --dname cora  --method DiagSheafs --MLP_hidden 256 --Classifier_hidden 256 --All_num_layers 2 --MLP_num_layers 0 --MLP2_num_layers 0 --Classifier_num_layers 1 --heads 3 --init_hedge avg --epochs 100 --sheaf_normtype degree_norm --runs 1 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --sheaf_left_proj False --sheaf_dropout False --dynamic_sheaf False --tag tmp3 --sheaf_special_head False --sheaf_pred_block transformer --sheaf_transformer_head 4
