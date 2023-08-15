import matplotlib.pyplot as plt
import numpy as np


N = 100
T = 5
theta = np.linspace(0, np.pi*T,N)
x = np.cos(theta)
y = np.sin(theta)
z = np.linspace(0,5,N)
X = np.vstack((x,y,z))
print(X.shape)

def create_figure(i, q, ax= None, only_ax=False):

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([0,5])
    ax.plot(*X[:,i-q:i])
    ax.scatter(*X[:,i-1], marker= "o", s=50)
    
    if only_ax: 
        return ax
    else:
        return fig, ax

fig, ax = plt.subplots(subplot_kw= {'projection':'3d'})

q = 10
for i in range(q,N):
    ax.clear()
    ax = create_figure(i, q, ax=ax, only_ax= True)
    plt.pause(0.1)
plt.show()

# instantiate figure, clear ax,  but then pass old ax reference and modify it inside function 