# run equivariance test for DiagHNNSheaf
CUDA_VISIBLE_DEVICES="" python test_sheaf_conv.py HCHA_equivariance  diag

# run equivariance test for OrthoHNNSheaf
CUDA_VISIBLE_DEVICES="" python test_sheaf_conv.py HCHA_equivariance  ortho

# run equivariance test for GeneralHNNSheaf
CUDA_VISIBLE_DEVICES="" python test_sheaf_conv.py HCHA_equivariance  general

# run visualisation test for DiagHNNSheaf
CUDA_VISIBLE_DEVICES="" python test_sheaf_conv.py visual_test  diag

# run visualisation test for OrthoHNNSheaf
CUDA_VISIBLE_DEVICES="" python test_sheaf_conv.py visual_test  ortho

# run visualisation test for GeneralHNNSheaf
CUDA_VISIBLE_DEVICES="" python test_sheaf_conv.py visual_test  general

# run equivariance test for DiagEDHDDSheaf
CUDA_VISIBLE_DEVICES="" python test_sheaf_conv.py EDHDD_equivariance diag

# run equivariance test for OrthoEDHDDSheaf
CUDA_VISIBLE_DEVICES="" python test_sheaf_conv.py EDHDD_equivariance  ortho

# run equivariance test for GeneralEDHDDSheaf
CUDA_VISIBLE_DEVICES="" python test_sheaf_conv.py EDHDD_equivariance  general
