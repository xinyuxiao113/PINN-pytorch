from Net import Burgers_DNN
import torch
from testProblem import Burgers_data
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm  
np.random.seed(123)
torch.manual_seed(123)


# 系统参数
region = [(-1,1),(0,1)]

# 网络结构参数
width = 30
blocks = 4

# 训练参数
bc = 'train'       # 'fix' or 'train'
loss_term = 'mix'  # 'mix' 'bc' or 'int'
beta = 1           # parameter balance loss_int and loss_partial
epochs = 301
sample_num_int = 1000
sample_num_partial = 100
show_interval = 50


model = Burgers_DNN(width=width,blocks=blocks,data=Burgers_data,bc=bc)
if torch.cuda.is_available():
    model = model.cuda()
optimizer = torch.optim.LBFGS(
        model.parameters(),
        history_size=50,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe",
    )
model.train(optimizer=optimizer,epochs=epochs,beta=beta,loss_term=loss_term,show_interval=show_interval,sample_num_int=sample_num_int,sample_num_partial=sample_num_partial)