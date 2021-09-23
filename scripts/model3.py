import sys
sys.path.append("../")
from sir import *
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sir import *
from sir.SIR_continuous_mask import *









sol1 = SIR_continuous_mask(1/2,1/3,80,0.01,1)     # solve the equation
sol2 = SIR_continuous_mask(1/2,1/3,80,0.01,0.83)
sol3 = SIR_continuous_mask(1/2,1/3,80,0.01,0.74)



plt.plot(sol1.t, sol1.y[0]*1000, c='b',label='s')    # generate the plot
plt.plot(sol1.t, sol1.y[1]*1000, c='g',label='r')
plt.plot(sol1.t, sol1.y[2]*1000, c='r',label='i')
plt.title("SIR Model:People not wearing masks")
plt.ylabel("y")
plt.xlabel("x")
plt.legend(['s','r','i'])
plt.show()

plt.plot(sol2.t, sol2.y[0]*1000, c='b',label='s')    # generate the plot
plt.plot(sol2.t, sol2.y[1]*1000, c='g',label='r')
plt.plot(sol2.t, sol2.y[2]*1000, c='r',label='i')
plt.title("SIR Model:People wearing surgical masks")
plt.ylabel("y")
plt.xlabel("x")
plt.legend(['s','r','i'])
plt.show()

plt.plot(sol3.t, sol3.y[0]*1000, c='b',label='s')    # generate the plot
plt.plot(sol3.t, sol3.y[1]*1000, c='g',label='r')
plt.plot(sol3.t, sol3.y[2]*1000, c='r',label='i')
plt.title("SIR Model:People wearing P2 masks")
plt.ylabel("y")
plt.xlabel("x")
plt.legend(['s','r','i'])
plt.show()
