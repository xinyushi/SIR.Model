import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def SIR_continuous_mask(b,k,time,ii,q):
    """
    Simulates continuous SIR model
    ii = initial percentage of infected
    time = Days of simulation
    b = probability that people getting infectious 
    k = probability that people getting recovered
    q = people wearing masks to slow down the parameter b (0 < q < 1)
    
    returns sol from solve_ivp
    """
    def SIR(t, X):

        #The main set of equations
        Y = np.zeros((3))
        Y[0] = -b * q * X[0] * X[2]
        Y[1] = k * X[2] 
        Y[2] = b * q * X[0] * X[2] - (k * X[2]) 
        return Y
    t_eval = np.linspace(0, time, 1000)
    sol1 = solve_ivp(SIR, [0, time], [1-ii, 0, ii], method='RK45', t_eval=t_eval)    # solve the equation
    return sol1
