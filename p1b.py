# APC 523: Problem Set 3
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],  # default LaTeX font
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

})
# Problem 1b

# Second order
x = np.linspace(-4,8,200)
y = np.linspace(-5,5,200)
X,Y = np.meshgrid(x,y)
nx, ny = Y.shape
p1 = X*0
p2 = X*0
p3 = X*0

omega = np.zeros([nx,ny],dtype=complex)
poly = np.zeros([nx,ny],dtype=complex)
for i in range(nx):
    for j in range(ny):
        z = complex(X[i,j],Y[i,j])
        p = [1-z,-1]
        root = np.roots(p)
        p1[i,j] = np.max(abs(root))
        p = [3-2*z,-4,1]
        root = np.roots(p)
        p2[i,j] = np.max(abs(root))
        p = [11-6*z,-18,9,-2]
        root = np.roots(p)
        p3[i,j] = np.max(abs(root))

levels = np.linspace(0,10,1000)
fig, ax = plt.subplots(1,2,figsize=(10,6),layout="constrained")
ax[0].contour(X,Y,p2,levels,cmap="viridis")
ax[0].contour(X,Y,p2,[1],colors=['red'])
ax[0].set_xlabel(r"$\lambda_{Re}\Delta t$")
ax[0].set_ylabel(r"$\lambda_{Im}\Delta t$")
ax[0].set_title("2nd-order")


pl = ax[1].contour(X,Y,p3,levels,cmap="viridis")
ax[1].contour(X,Y,p3,[1],colors=['red'])
ax[1].set_xlabel(r"$\lambda_{Re}\Delta t$")
ax[1].set_ylabel(r"$\lambda_{Im}\Delta t$")
ax[1].set_title("2nd-order")
c = plt.colorbar(pl)
c.set_label("Levels")
fig.savefig('p1b.png',dpi=600)
plt.show()