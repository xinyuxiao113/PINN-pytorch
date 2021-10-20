import torch
import numpy as np
from torch.autograd import grad

region = [(-5,5),(0,np.pi/2)]

def d(x,t):
    '''
    到边界的距离函数
    '''
    return (x.pow(2)-25)*t

def g(x,t):
    '''
    初始条件与边界条件
    '''
    return 4/(torch.exp(x) + torch.exp(-x))

h_real= 0

schrodinger_data = {'d':d,'g':g,'h_real':h_real,'region':region}