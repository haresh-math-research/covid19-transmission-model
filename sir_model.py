import numpy as np 
from scipy.integrate import odeint 
def sir_derivatives(y, t, N, beta, gamma): 
    S, I, R = y 
    dSdt = -beta * S * I / N 
    dIdt = (beta * S * I / N) - (gamma * I) 
    dRdt = gamma * I 
    return dSdt, dIdt, dRdt 
N, beta, gamma = 1000, 0.3, 0.1 
y0 = (N - 1, 1, 0) 
t = np.linspace(0, 160, 160) 
