# APC 523: Problem Set 3
import numpy as np
from matplotlib import pyplot as plt
from p2_functions import fe,se,rk4
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],  # default LaTeX font
})

# Problem 2
X0 = [0,0]
omega = 5
Fm = 1
omegaF = 0.1
tend = 100

n = 10**6
fe_t = np.linspace(0,tend,n+1)
se_t = np.linspace(0,tend,n+1)
rk4_t = np.linspace(0,tend,n+1)


def f(t,x):
    d2xdt2 = Fm * np.cos(omegaF*t) - omega**2*x
    return d2xdt2

colors = ["#9766CB","#E297E8","#E13379"]
labels = ["Forward Euler", "Symplectic Euler", "RK4"]
          

# Execute integration schemes
fesolx, fesolv = fe(fe_t,f,X0)
sesolx, sesolv = se(se_t,f,X0)
rk4solx, rk4solv = rk4(rk4_t,f,X0)
soln = [n, n, n]
solt = [fe_t, se_t, rk4_t]
solx = [fesolx, sesolx, rk4solx]
solv = [fesolv, sesolv, rk4solv]


# Calculate error
A = -Fm/(omega**2 - omegaF**2)
B = 0
errors = [None, None, None]
for i in range(3):
    t = solt[i]
    n = soln[i]
    analyticalx = Fm/(omega**2 - omegaF**2)*np.cos(omegaF*t) + A*np.cos(omega*t) + B*np.sin(omega*t)
    errors[i] = np.sqrt(np.cumsum((analyticalx - solx[i])**2) * tend/n)


# Plot the results
fig, ax = plt.subplots(3,3,figsize=(9,8),layout='constrained')
for i in range(3):
    t = solt[i]
    x = solx[i]
    v = solv[i]
    error = errors[i]
    ax[0,i].plot(t,x,color=colors[i],linewidth=0.8)
    ax[1,i].plot(x,v,color=colors[i],linewidth=0.8)
    ax[2,i].semilogy(t,error,color=colors[i],linewidth=1)
    ax[0,i].set_title(labels[i])
    ax[0,i].set_xlabel(r"$t$")
    ax[0,i].set_ylabel(r"$x(t)$")
    ax[1,i].set_xlabel(r"$x(t)$")
    ax[1,i].set_ylabel(r"$v(t)$")
    ax[2,i].set_ylabel("L2 Error")
    ax[2,i].set_xlabel(r"$t$")
    ax[0,i].set_ylim(-0.1, 0.1)
    ax[1,i].set_ylim(-0.4, 0.4)
    ax[1,i].set_xlim(-0.1,0.1)
    ax[2,i].set_ylim(10**(-11),10**(-1))
    ax[0,i].set_box_aspect(1) 
    ax[1,i].set_box_aspect(1) 
    ax[2,i].set_box_aspect(1)
fig.savefig('p2.png',dpi=600)

