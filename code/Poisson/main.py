from Net import Poisson_DNN
import torch
from testProblem import Poisson_data,plot_3D
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm  
from torch.autograd.functional import hessian
np.random.seed(123)
torch.manual_seed(123)

# 系统参数
dim = 2
region = [(-1,1),(-1,1)]

# 网络结构参数
width = 20
blocks = 3

# 训练参数
bc = 'train'
beta = 20
epochs = 200
lr = 0.01
sample_num_int = 1000
sample_num_partial = 100
show_interval = 100



model = Poisson_DNN(width=width,blocks=blocks,data=Poisson_data,bc=bc)
optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        history_size=50,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe",
        )
optimizer_adam = torch.optim.Adam(model.parameters(),lr=lr)
model.train(optimizer=optimizer_lbfgs,epochs=epochs,beta=beta,loss_term='mix',show_interval=show_interval,sample_num_int=sample_num_int,sample_num_partial=sample_num_partial)