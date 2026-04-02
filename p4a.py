# APC 523: Problem Set 3
import numpy as np
from scipy.differentiate import jacobian



# Problem 4a

# Initial conditions

X0 = np.array([0.01,0.01,0.75,0.23,0.00])
kpos1 = 2*10**3
kneg1 = 3*10**(-12) 
kpos2 = 2*10

J = np.array([[-4.6,1500,20,-0.2,0],
              [4.6,-1500,-20,0.2,0],
              [0,-1500,-20,0,0],
              [-4.6,0,0,-0.2,0],
              [4.6,1500,20,0.2,0]])
print("Eigenvalues of Jacobian =", np.linalg.eigvals(J))
