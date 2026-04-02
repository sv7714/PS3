import numpy as np
import sympy as sym
from scipy.optimize import fsolve

def rk4(t,f,IC):
    dt = t[-1]/(t.size-1)
    sol = np.zeros((t.size,5))
    sol[0] = IC

    for i in range(1,t.size):
        sol0 = sol[i-1]
        k1 = f(sol0)
        k2 = f(sol0 + k1*dt/2)
        k3 = f(sol0 + k2*dt/2)
        k4 = f(sol0 + k3*dt)
        
        sol[i] = sol0 + (k1+2*k2+2*k3+k4)*dt/6

    return sol

def dirk2(t,f,IC):
    dt = t[-1]/(t.size-1)
    sol = np.zeros((t.size,5))
    sol[0] = IC

    for i in range(1, t.size):
        sol0 = sol[i-1]
        
        def eqn1(sol1):
            return sol1 - sol0 - (1/3)*f(sol1)*dt
        sol1 = fsolve(eqn1,sol0)

        def eqn2(sol2):
            return sol2 - sol0 - (3/4)*f(sol1)*dt - (1/4)*f(sol2)*dt
        sol2 = fsolve(eqn2,sol1)
        
        sol[i] = sol2
    return sol



def bdf2(t,f,IC):
    dt = t[-1]/(t.size-1)
    sol = np.zeros((t.size,5))
    sol[0] = IC

    # Use DIRK2 to get the first step
    sol[1] = dirk2(np.array([0, dt]), f, IC)[-1]

    for i in range(2, t.size):
        solneg1 = sol[i-2]
        sol0 = sol[i-1]

        def eqn1(sol1):
            return (3/2)*sol1 - 2*sol0 + (1/2)*solneg1 - dt*f(sol1)
        sol[i] = fsolve(eqn1, sol0)

    return sol



def dirk2adaptive(t,f,IC):
    # Set up the start and end times, and create array to store current time
    t0 = t[0]
    tend = t[-1]
    tspan = [t0]
    tcurrent = t0

    # Initialize solution vector
    sol = [IC]
    
    # First time step from 1/max(eigval(J))
    dtinit = 1/(1524*2)
    dt = dtinit

    # DIRK2 one step
    def dirk2_1step (dt,f,sol0):      
        def eqn1(sol1):
            return sol1 - sol0 - (1/3)*f(sol1)*dt
        sol1 = fsolve(eqn1,sol0)

        def eqn2(sol2):
            return sol2 - sol0 - (3/4)*f(sol1)*dt - (1/4)*f(sol2)*dt
        sol2 = fsolve(eqn2,sol1)
        
        return sol2
    
    while (tcurrent + dt < tend):
        sol0 = sol[-1]
        sol1 = dirk2_1step(dt,f,sol0)
        
        # 10% rule
        epsilon = np.max(np.abs(sol1 - sol0)/(np.maximum(np.abs(sol0), 1e-12)))

        if (epsilon < 0.1):
            dt = 2*dt
            tcurrent = tcurrent + dt
            tspan.append(tcurrent)
            sol.append(sol1)
        else:
            dt = dt
            tcurrent = tcurrent + dt
            tspan.append(tcurrent)
            sol.append(sol1)
    return np.array(tspan), np.array(sol)



def bdf2adaptive(t,f,IC):
    # Set up the start and end times, and create array to store current time
    t0 = t[0]
    tend = t[-1]
    tspan = [t0]
    tcurrent = t0

    # Initialize solution vector
    sol = [IC]
    
    # First time step from 1/max(eigval(J))
    dtinit = 1/(1524*2)
    dt = dtinit


    # DIRK2 first step 
    def dirk2_1step (dt,f,sol0):      
        def eqn1(sol1):
            return sol1 - sol0 - (1/3)*f(sol1)*dt
        sol1 = fsolve(eqn1,sol0)

        def eqn2(sol2):
            return sol2 - sol0 - (3/4)*f(sol1)*dt - (1/4)*f(sol2)*dt
        sol2 = fsolve(eqn2,sol1)
        
        return sol2

    # BDF2 one step
    def bdf2_1step(dt,f,solneg1,sol0):      
        def eqn1(sol1):
            return sol1 - f(sol1)*(2/3)*dt - (4/3)*sol0 +  (1/3)*solneg1 
        sol1 = fsolve(eqn1, sol0)
        return sol1
    


    sol0 = sol[0]
    sol.append(dirk2_1step(dt,f,sol0))
    tcurrent = tcurrent + dt
    tspan.append(tcurrent)

    while (tcurrent + dt < tend):
        solneg1 = sol[-2]
        sol0 = sol[-1]
        sol1 = bdf2_1step(dt,f,solneg1,sol0)
        
        # 10% rule
        epsilon = np.max(np.abs(sol1 - sol0)/(np.maximum(np.abs(sol0), 1e-12)))

        if (epsilon < 0.1):
            dt = 2*dt
            tcurrent = tcurrent + dt
            tspan.append(tcurrent)
            sol.append(sol1)
        else:
            dt = dt
            tcurrent = tcurrent + dt
            tspan.append(tcurrent)
            sol.append(sol1)

    return np.array(tspan), np.array(sol)