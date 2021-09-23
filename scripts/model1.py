import sys
sys.path.append("../")

from sir import *

## CONTINUOUS 

NumPop = 473 # https://nces.ed.gov/programs/digest/d07/tables/dt07_095.asp
b = 1       # number of contact people 
k = 0.074
time= 40
ii = 0.01

sol1=SIR_continuous(b,k,time,ii)
plt.plot(sol1.t, sol1.y[0]*NumPop, c='b',label='s')    # generate the plot
plt.plot(sol1.t, sol1.y[1]*NumPop, c='g',label='r')
plt.plot(sol1.t, sol1.y[2]*NumPop, c='r',label='i')

plt.title("Continuous SIR Model(ii=0.01, b=1, k=0.074 )")
plt.ylabel("Number of People")
plt.xlabel("Time Period")
plt.legend(['s','r','i'])
plt.show()

from sir import *

## CONTINUOUS 
k1 = 0.074
time1= 40
ii1=0.01
NumPop1 = 473
bs = np.arange(1, 11, dtype=np.int64)
cts = np.zeros(len(bs))
for i, b in enumerate(bs):
    cts[i] = np.max(SIR_continuous(b,k1,time1,ii1).y[2,:])
    
plt.scatter(bs,cts*NumPop1)    
plt.title("Maximum Infected People (ii=0.01, b=varying, k=0.074)")
plt.ylabel("Max Number of People Infected")
plt.xlabel("b (number of contacts people)")
plt.show()