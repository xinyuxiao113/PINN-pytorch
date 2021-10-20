import torch
import numpy as np
from torch.autograd import grad

def D(u,wrt=None):
    return grad(u.sum(),wrt,create_graph=True)[0]

def laplacian(h,x,y):
    u = h(x,y)
    u_x = D(u,x)
    u_y = D(u,y)
    u_xx = D(u_x,x)
    u_yy = D(u_y,y)
    return u_xx + u_yy

# 求解区域
region = [(-1,1),(-1,1)]


def d(x,y):
    return (x.pow(2)-1)*(y.pow(2)-1)


def u_test(x,y):
    return torch.exp(x**2 + y**2)*torch.sin(5*x-2*y)+x*y

def f(x,y):
    return -laplacian(u_test,x,y)
    #return 1

def g(x,y):
    return u_test(x,y)
    #return torch.tensor(0)

Poisson_data = {'f':f,'g':g,'d':d,'u_real':u_test,'region':region}


def plot_3D(X1,X2,U):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X1, X2, U, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)