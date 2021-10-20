import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import grad
import matplotlib.pyplot as plt
import time
from matplotlib import cm

# 求导算子
def D(u,wrt):
    return grad(u.sum(),wrt,create_graph=True)[0]

class Poisson_DNN(nn.Module):
    def __init__(self,dim=2,width=20,blocks=3,data=None,bc='train'):
        super().__init__()
        self.dim = dim
        # 网络结构参数
        self.width = width
        self.blocks = blocks
        # 第一层
        self.pre_layer = nn.Linear(2,self.width)
        # 最后一层
        self.last_layer = nn.Linear(self.width,1)
        # 中间层
        self.layerOp1 = nn.ModuleList()
        self.layerOp2 = nn.ModuleList()
        # 激活函数
        self.activation = nn.Tanh()
        for j in range(self.blocks):
            self.layerOp1.append(nn.Linear(self.width, self.width))
            self.layerOp2.append(nn.Linear(self.width, self.width))
        
        # 求解区域
        self.region = data['region']
        # 源函数
        self.f = data['f']
        # 边界距离函数
        self.d = data['d']
        # 边界函数
        self.g = data['g']
        # 真实函数
        self.u_real = data['u_real']
        # 边界是否固定
        self.bc = bc
    

    def u_net(self, x, y):
        '''
        PDE解的近似函数
        '''
        u = self.last_layer(self.Phi(x,y))
        if self.bc=='fix':
            u = u*self.d(x,y) + self.g(x,y)
        elif self.bc=='train':
            pass
        else:
            raise ValueError('please set bc to:fix or train')
        return u
    
    def Phi(self,x,y):
        '''
        特征映射
        '''
        x.requires_grad_(True)
        y.requires_grad_(True)
        u = torch.cat([x,y],axis=1)

        u = self.pre_layer(u)
        u = self.activation(u)
        for i in range(self.blocks):
            u = self.activation(self.layerOp2[i](self.activation(self.layerOp1[i](u)))) + u
        return u
    
    def forward(self, x,y):
        return self.u_net(x,y)
    
    
    def int_loss(self,x,y):
        '''
        内部loss
        '''
        u = self.u_net(x,y)
        u_x = D(u,x)
        u_y = D(u,y)
        u_xx = D(u_x,x)
        u_yy = D(u_y,y)
        loss = (u_xx + u_yy + self.f(x,y)).pow(2).sum()
        return loss/len(x)
    
    def partial_loss(self,x,y):
        '''
        边界loss
        '''
        u = self.u_net(x,y)
        loss = (u-self.g(x,y)).pow(2).sum()
        return loss/len(x)

    def sample_int(self,number):
        '''
        内部sample
        '''
        X = torch.rand(number,self.dim)
        for i in range(self.dim):
            X[:,i] *= self.region[i][1] - self.region[i][0]
            X[:,i] += self.region[i][0]
        return X[:,0:1],X[:,1:2]
    
    def sample_partial(self,number):
        '''
        边界sample
        '''
        idx = np.random.randint(self.dim,size=number)
        coin = np.random.randint(self.dim,size=number)
        X = torch.rand(number,self.dim)
        for i in range(self.dim):
            X[:,i] *= self.region[i][1] - self.region[i][0]
            X[:,i] += self.region[i][0]
        for j in range(number):
            d = idx[j]
            X[j][d] = self.region[d][coin[j]]
        return X[:,0:1],X[:,1:2]
    
    def loss_fun(self,x_int,y_int,x_partial,y_partial,loss_term,beta):
        '''
        加权loss
        '''
        if loss_term=='int':
            loss = self.int_loss(x_int,y_int)
        elif loss_term=='bc': 
            loss = self.partial_loss(x_partial,y_partial)
        elif loss_term=='mix':
            loss1 = self.int_loss(x_int,y_int)
            loss2 = self.partial_loss(x_partial,y_partial)
            loss = loss1 + beta*loss2
        else:
            raise ValueError('No such loss term: ' + loss_term + 'please use int, bc or mix')
        return loss

    def test_loss(self):
        '''
        格点上的测试误差
        '''
        x = torch.linspace(-1,1,10)
        y = torch.linspace(-1,1,10)
        [X,Y] = torch.meshgrid(x,y)
        X = X.reshape(-1,1)
        Y = Y.reshape(-1,1)
        u = self.u_net(X,Y)
        u_real = self.u_real(X,Y)
        return torch.sqrt((u-u_real).pow(2).mean())
    
    def train(self,optimizer=None,epochs=200,beta=200,sample_num_int = 100,sample_num_partial = 40,loss_term='mix',show_interval=100):
        print('Trainning Start!')
        # 计时
        t_start = time.time()
        # 采样（固定采样不变）
        x_int,y_int = self.sample_int(sample_num_int)
        x_partial,y_partial = self.sample_partial(sample_num_partial)
        
        # 优化
        loss_path = {'int':[],'bc':[],'sum':[],'test':[]}
        for epoch in range(epochs):
            loss_int = self.int_loss(x_int,y_int)
            loss_partial = self.partial_loss(x_partial,y_partial)
            loss = loss_int + beta*loss_partial
            test_loss = self.test_loss()
            loss_path['int'].append(loss_int)
            loss_path['bc'].append(loss_partial)
            loss_path['sum'].append(loss)
            loss_path['test'].append(test_loss)

            if (epoch % show_interval)==0:
                print('%'*25 +f' epoch {epoch} '+'%'*25)
                print('beta = %.2g loss=%.4g   loss_int= %.4g   loss_partial=%.4g ' % (beta,loss.item(), loss_int.item(), loss_partial.item()))
                print('Test loss=%.4g' % test_loss)
            
            def closure():
                optimizer.zero_grad()
                loss = self.loss_fun(x_int,y_int,x_partial,y_partial,loss_term,beta)
                loss.backward()
                return loss
            optimizer.step(closure)
        t_stop = time.time()
        print('Training done! Time cost: %g' % (t_stop - t_start))
        return loss_path