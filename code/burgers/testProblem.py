import torch
import numpy as np
from torch.autograd import grad

region = [(-1,1),(0,1)]

def d(x,t):
    '''
    到边界的距离函数
    '''
    return (x.pow(2)-1)*t

def g(x,t):
    '''
    边界条件
    '''
    return -torch.sin(np.pi*x)


u_real= 0

Burgers_data = {'d':d,'g':g,'u_real':u_real,'region':region}