import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import pdb
from torch_scatter import scatter, scatter_mean, scatter_add
#these are some utilitary function. I will move them from here soon
