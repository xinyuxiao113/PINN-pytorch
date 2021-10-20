import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import grad
import time
from matplotlib import cm

# 求导算子
def D(u,wrt):
    return grad(u.sum(),wrt,create_graph=True)[0]

class Burgers_DNN(nn.Module):
    def __init__(self,width=20,blocks=3,data=None,bc='train'):
        super().__init__()
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
        # 边界距离函数
        self.d = data['d']
        # 边界函数
        self.g = data['g']
        # 真实函数
        self.u_real = data['u_real']
        # 边界是否固定
        self.bc = bc
    
    def Phi(self,x,t):
        '''
        特征映射
        '''
        x.requires_grad_(True)
        t.requires_grad_(True)
        u = torch.cat([x,t],axis=1)

        u = self.pre_layer(u)
        u = self.activation(u)
        for i in range(self.blocks):
            u = self.activation(self.layerOp2[i](self.activation(self.layerOp1[i](u)))) + u
        return u

    def u_net(self,x,t):
        '''
        PDE近似解函数
        '''
        u = self.last_layer(self.Phi(x,t))
        if self.bc=='fix':
            u = u*self.d(x,t) + self.g(x,t)
        elif self.bc=='train':
            pass
        else:
            raise ValueError('please set bc to:fix or train')
        return u
    
    def forward(self,x,t):
        return self.u_net(x,t)
    
    
    def int_loss(self,x,t):
        '''
        内部点的 Galerkin loss
        '''
        u = self.u_net(x,t)
        u_t = D(u,t)
        u_x = D(u,x)
        u_xx = D(u_x,x)
        loss = (u_t + u*u_x - 0.01/np.pi*u_xx).pow(2).mean()
        return loss
    
    def partial_loss(self,x,t):
        '''
        边界上的 MSE loss
        '''
        u = self.u_net(x,t)
        loss = (u-self.g(x,t)).pow(2).mean()
        return loss

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
        x = torch.rand(3*N,1)
        t = torch.rand(3*N,1)
        # 初始条件
        x[0:N,0] *= (self.region[0][1] - self.region[0][0])
        x[0:N,0] += self.region[0][0]
        t[0:N,0] = 0
        # 下边界
        x[N:2*N,0] = self.region[0][0]
        t[N:2*N,0] *= self.region[1][1]
        # 上边界
        x[2*N:3*N,0] = self.region[0][1]
        t[2*N:3*N,0] *= self.region[1][1]
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
        x_int,y_int = self.sample_int(sample_num_int)
        x_partial,y_partial = self.sample_partial(sample_num_partial)
        def closure():
            optimizer.zero_grad()
            loss = self.loss_fun(x_int,y_int,x_partial,y_partial,loss_term,beta)
            loss.backward()
            return loss
        # 优化
        for epoch in range(epochs):
            if (epoch % show_interval)==0:
                loss_int = self.int_loss(x_int,y_int)
                loss_partial = self.partial_loss(x_partial,y_partial)
                loss = loss_int + beta*loss_partial
                print('%'*25 +f' epoch {epoch} '+'%'*25)
                print('loss=%.4g   loss_int= %.4g   loss_partial=%.4g ' % (loss.item(), loss_int.item(), loss_partial.item()))
            optimizer.step(closure)
        t_stop = time.time()
        print('Training done! Time cost: %g' % (t_stop - t_start))


