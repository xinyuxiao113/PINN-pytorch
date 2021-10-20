from Net import Schrodinger_DNN
import torch
from testProblem import schrodinger_data
import numpy as np
import time 
np.random.seed(123)
torch.manual_seed(123)


# 系统参数
region = [(-5,5),(0,np.pi/2)]

# 网络结构参数
width = 30
blocks = 4

# 训练参数
bc = 'fix'   # 'fix' or 'train'
beta = 1     # parameter balance loss_int and loss_partial
epochs = 301
sample_num_int = 2000
sample_num_partial = 100
show_interval = 50


model = Schrodinger_DNN(width=width,blocks=blocks,data=schrodinger_data,bc=bc)
if torch.cuda.is_available():
    model.cuda()
optimizer = torch.optim.LBFGS(
        model.parameters(),
        history_size=50,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe",
    )
model.train(optimizer=optimizer,epochs=epochs,beta=beta,show_interval=1,sample_num_int=sample_num_int,sample_num_partial=sample_num_partial)