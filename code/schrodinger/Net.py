import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import grad
import time

# 求导算子
def D(u,wrt):
    return grad(u.sum(),wrt,create_graph=True)[0]

class Schrodinger_DNN(nn.Module):
    def __init__(self,width=20,blocks=3,data=None,bc='train'):
        super().__init__()
        # 网络结构参数
        self.width = width
        self.blocks = blocks
        # 第一层
        self.pre_layer_u = nn.Linear(2,self.width)
        self.pre_layer_v = nn.Linear(2,self.width)
        # 最后一层
        self.last_layer_u = nn.Linear(self.width,1)
        self.last_layer_v = nn.Linear(self.width,1)
        # 中间层
        self.layerOp1 = nn.ModuleList()
        self.layerOp2 = nn.ModuleList()
        self.layerOp3 = nn.ModuleList()
        self.layerOp4 = nn.ModuleList()
        # 激活函数
        self.activation = nn.Tanh()
        for j in range(self.blocks):
            self.layerOp1.append(nn.Linear(self.width, self.width))
            self.layerOp2.append(nn.Linear(self.width, self.width))
            self.layerOp3.append(nn.Linear(self.width, self.width))
            self.layerOp4.append(nn.Linear(self.width, self.width))
        
        # 求解区域
        self.region = data['region']
        # 边界距离函数
        self.d = data['d']
        # 边界函数
        self.g = data['g']
        # 真实函数
        self.h_real = data['h_real']
        # 边界是否固定
        self.bc = bc
    
    def Phi(self,x,t):
        '''
        特征映射
        '''
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = torch.cat([x,t],axis=1)
        u = self.pre_layer_u(u)
        u = self.activation(u)
        for i in range(self.blocks):
            u = self.activation(self.layerOp2[i](self.activation(self.layerOp1[i](u)))) + u
        
        v= torch.cat([x,t],axis=1)
        v = self.pre_layer_v(v)
        v = self.activation(v)
        for i in range(self.blocks):
            v = self.activation(self.layerOp4[i](self.activation(self.layerOp3[i](v)))) + v
        return u,v

    def h_net(self,x,t):
        '''
        PDE近似解函数
        '''
        u,v = self.Phi(x,t)
        u = self.last_layer_u(u)
        v = self.last_layer_v(v)
        h = u + 1j*v
        return h
    
    
    def forward(self,x,t):
        return self.h_net(x,t)
    
    def MSE(self,x):
        return x.abs().pow(2).mean()
    
    def int_loss(self,x,t):
        '''
        内部点的 Galerkin loss
        i h_t + 0.5 h_xx + |h|^2 h = 0
        '''
        h = self.h_net(x,t)
        h_t = D(h,t)
        h_x = D(h,x)
        h_xx = D(h_x,x)
        loss = self.MSE(1j*h_t + 0.5*h_xx + torch.abs(h).pow(2) * h)
        return loss
    
    def partial_loss(self,t):
        '''
        边界上的 MSE loss
        '''
        xl = torch.ones_like(t)*self.region[0][0]
        xr = torch.ones_like(t)*self.region[0][1]
        if torch.cuda.is_available():
            xl = xl.cuda()
            xr = xr.cuda()
        hl = self.h_net(xl,t)
        hr = self.h_net(xr,t)
        hl_x = D(hl,xl)
        hr_x = D(hr,xr)
        loss = self.MSE(hl-hr) + self.MSE(hl_x - hr_x)
        return loss
    
    def init_loss(self,x,t):
        h = self.h_net(x,t)
        loss = self.MSE(h - self.g(x,t))
        return loss

    def loss_fun(self,x_int,t_int,t_partial,x_init,t_init):
        '''
        加权loss
        '''
        return self.int_loss(x_int,t_int) + self.partial_loss(t_partial) + self.init_loss(x_init,t_init)
    

    def sample_int(self,N):
        '''
        区域内部的sample
        '''
        X = torch.rand(N,2)
        for i in range(2):
            X[:,i] *= self.region[i][1] - self.region[i][0]
            X[:,i] += self.region[i][0]
        if torch.cuda.is_available():
            X = X.cuda()
        return X[:,0:1],X[:,1:2]
    
    def sample_partial(self,N):
        '''
        边界上的sample
        '''
        t = torch.rand(N,1)
        if torch.cuda.is_available():
            t = t.cuda()
        return t
    
    def sample_init(self,N):
        x = torch.rand(N,1)
        t = torch.rand(N,1)
        # 初始条件
        x *= (self.region[0][1] - self.region[0][0])
        x += self.region[0][0]
        t *= 0
        if torch.cuda.is_available():
            x = x.cuda()
            t = t.cuda()
        return x,t

    def test_loss(self):
        '''
        网格点上的 test loss
        '''
        x = torch.linspace(self.region[0][0],self.region[0][1],100)
        t = torch.ones_like(x)*self.region[1][1]
        x = x.reshape(-1,1)
        t = t.reshape(-1,1)
        if torch.cuda.is_available():
            x = x.cuda()
            t = t.cuda()
        u = self.u_net(x,t)
        return torch.sqrt((u-self.u_real).pow(2).mean())
    
    def train(self,optimizer=None,epochs=200,beta=200,sample_num_int = 100,sample_num_partial = 40,loss_term='mix',show_interval=100):
        print('Trainning Start!')
        # 计时
        t_start = time.time()
        # 采样（固定采样不变）
        x_int,t_int = self.sample_int(sample_num_int)
        t_partial = self.sample_partial(sample_num_partial)
        x_init,t_init = self.sample_init(sample_num_partial)
        def closure():
            optimizer.zero_grad()
            loss = self.loss_fun(x_int,t_int,t_partial,x_init,t_init)
            loss.backward()
            return loss
        # 优化
        loss0 = 1e8
        for epoch in range(epochs):
            if (epoch % show_interval)==0:
                loss_int = self.int_loss(x_int,t_int)
                loss_partial = self.partial_loss(t_partial)
                loss_init = self.init_loss(x_init,t_init)
                loss = loss_int + loss_partial + loss_init
                if np.abs(loss.item() - loss0) < 1e-6:
                    break
                loss0 = loss.item()
                print('%'*25 +f' epoch {epoch} '+'%'*25)
                print('loss=%.4g   loss_int= %.4g   loss_partial=%.4g loss_init=%.4g' % (loss.item(), loss_int.item(), loss_partial.item(),loss_init.item()))
            optimizer.step(closure)
        t_stop = time.time()
        print('Training done! Time cost: %g' % (t_stop - t_start))


