# APC 523: Problem Set 3
import numpy as np
from matplotlib import pyplot as plt
from p4_functions import rk4, dirk2, bdf2, dirk2adaptive, bdf2adaptive
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],  # default LaTeX font
})
# Problem 4b

# Initial conditions
X0 = np.array([0.01,0.01,0.75,0.23,0.00])
kpos1 = 2*10**3
kneg1 = 3*10**(-12) 
kpos2 = 2*10

def f(v):
    N, O , N2, O2, NO = v
    return np.array([kpos1*N2*O - kneg1*NO*N - kpos2*N*O2,
                    -kpos1*N2*O + kneg1*NO*N + kpos2*N*O2,
                    -kpos1*N2*O + kneg1*NO*N,
                    -kpos2*N*O2,
                    kpos1*N2*O - kneg1*NO*N + kpos2*N*O2])

# 4bi) - 4biii)
tend = 40
n = 100000
t = np.linspace(0,tend,n+1)
rk4sol = rk4(t,f,X0)
dirk2sol = dirk2(t,f,X0)
bdf2sol = bdf2(t,f,X0)
sols = [rk4sol,dirk2sol,bdf2sol]

# 4biv) DIRK2 (adaptive)
n = 2
adaptivet = np.linspace(0,tend,n+1)
dirk2tsol, dirk2adaptivesol = dirk2adaptive(adaptivet,f,X0)
bdf2tsol, bdf2adaptivesol = bdf2adaptive(adaptivet,f,X0)



colors = ["#9766CB","#E297E8","#F0B588","#E13379","#523FA8"]
labels = ["RK4", "DIRK2", "BDF2"]
molefrac = [r"$\mathbf{X}_N$",r"$\mathbf{X}_O$",r"$\mathbf{X}_{N_2}$",r"$\mathbf{X}_{O_2}$",r"$\mathbf{X}_{NO}$"]

fig, ax = plt.subplots(1,3,figsize=(12,4),sharey=True)
for i in range(5):
    for j in range(3):
        sol = sols[j]
        ax[j].loglog(t,sol[:,i],color=colors[i],linewidth=1,label=molefrac[i])
        ax[j].set_title(labels[j], fontsize=14)
        ax[j].set_xlabel(r"Time, $t$", fontsize=12)
        ax[1].loglog(dirk2tsol,dirk2adaptivesol[:,i],'.',markersize=3,color=colors[i])
        ax[2].loglog(bdf2tsol,bdf2adaptivesol[:,i],'.',markersize=3,color=colors[i])
plt.legend(bbox_to_anchor=(1.05,0.5),loc="center left")
ax[0].set_ylabel(r"Mole Fraction, $\mathbf{X}_\alpha$", fontsize=12)
plt.tight_layout()
fig.savefig('p4b.png',dpi=600)