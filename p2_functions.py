import numpy as np

def fe(t,f,IC):
    dt = t[-1]/(t.size-1)
    x = np.zeros(t.size)
    v = np.zeros(t.size)
    x[0] = IC[0]
    v[0] = IC[1]

    for i in range(1,t.size):
        d2xdt2 = f(t[i-1],x[i-1])      
        x[i] = x[i-1] + dt*v[i-1]
        v[i] = v[i-1] + dt*d2xdt2

    return x,v

def se(t,f,IC):
    dt = t[-1]/(t.size-1)
    x = np.zeros(t.size)
    v = np.zeros(t.size)
    x[0] = IC[0]
    v[0] = IC[1]

    for i in range(1,t.size):
        d2xdt2 = f(t[i-1],x[i-1],)      
        v[i] = v[i-1] + dt*d2xdt2
        x[i] = x[i-1] + dt*v[i]
    return x,v

def rk4(t,f,IC):
    dt = t[-1]/(t.size-1)
    x = np.zeros(t.size)
    v = np.zeros(t.size)
    x[0] = IC[0]
    v[0] = IC[1]

    for i in range(1,t.size):
        x0 = x[i-1]
        v0 = v[i-1]

        k1x = v0
        k1v = f(t[i-1],x0)
        k2x = v0 + k1v*dt/2 
        k2v = f(t[i-1] + dt/2, x0 + k1x*dt/2)
        k3x = v0 + k2v*dt/2 
        k3v = f(t[i-1] + dt/2, x0 + k2x*dt/2)
        k4x = v0 + k3v*dt
        k4v = f(t[i-1] + dt/2, x0 + k3x*dt)
        
        x[i] = x0 + (k1x+2*k2x+2*k3x+k4x)*dt/6
        v[i] = v0 + (k1v+2*k2v+2*k3v+k4v)*dt/6

    return x, v
