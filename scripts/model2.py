import sys
sys.path.append("../")
from sir import *
from sir.SIR_continuous_reinfected import *
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
t=pd.read_csv("../data/case_daily_trends__united_states.csv",header=2).iloc[:,2] # read in daily case in US
t=pd.read_csv("../data/case_total_and_rate_per_100000__united_states.csv",header=2).iloc[:,1] # read in total case in US
total_case=np.array(t)
total_case[:10]

totalpopulation=328200000 # total population in us
S=totalpopulation-total_case
S=S/totalpopulation # in percentage
IandR=total_case/totalpopulation
time=[*range(len(total_case), 0, -1)] 
plt.plot(time,S,label="S")
plt.plot(time,IandR,label="I+R")
plt.legend()
plt.xlabel("number of days")
plt.ylabel("probabillity")
plt.show()

b=0.1
k=0.1
time=319
ii=0.01
r=0
q=1


sol1=SIR_continuous_reinfected(b,k,time,ii,r)
plt.plot(sol1.t, sol1.y[0], c='b',label='s')    # generate the plot
plt.plot(sol1.t, (sol1.y[1]+sol1.y[2]), c='g',label='r+i')
#plt.plot(sol1.t, sol1.y[2]*1000, c='r',label='i')
plt.show()

def loss(b,k,r):
    """
    loss function with parameter b, k,r
    """
    ii=0.01
    time=len(S)
    q=1
    sol1=SIR_continuous_reinfected(b,k,time,ii,r)
    return np.linalg.norm(sol1.y[0]-S)+np.linalg.norm((sol1.y[1]+sol1.y[2])-IandR)

def loss2(x):
    """
    loss function with vector x that contains b,k,r
    """
    b=x[0]
    k=x[1]
    r=x[2]
    ii=0.01
    time=len(S)
    q=1
    sol1=SIR_continuous_reinfected(b,k,time,ii,r)
    return np.linalg.norm(sol1.y[0]-S)+np.linalg.norm((sol1.y[1]+sol1.y[2])-IandR)

def loss3(x):
    """
    loss function for r
    """
    b=0.0001
    k=0.074
    ii=0.01
    time=len(S)
    q=1
    sol1=SIR_continuous_reinfected(b,k,time,ii,x)
    return np.linalg.norm(sol1.y[0]-S)+np.linalg.norm((sol1.y[1]+sol1.y[2])-IandR)

sol = opt.minimize(loss2, np.array([-9.66889007e-06, -9.85879953e-01,  9.77224552e-01]))
sol

losslist=[]
rlist=[]
n=1000
for i in range(n):
    r=i/n
    losslist.append(loss3(r))
    rlist.append(r)
plt.plot(rlist,losslist)
plt.xlabel('r')
plt.ylabel('loss')
plt.title('The relationship of reinfection rate and loss function')
plt.savefig('rvsloss')


b=0.0001
k=0.074
time=319
ii=0.01
r=0.01
q=1
times=[*range(len(total_case), 0, -1)] 
sol1=SIR_continuous_reinfected(b,k,time,ii,r)
plt.plot(sol1.t, sol1.y[0], c='b',label='S(simulation)')    # generate the plot
plt.plot(sol1.t, (sol1.y[1]+sol1.y[2]), c='g',label='R+I(Simulation)')
plt.plot(times,S,label="S(data)")
plt.plot(times,IandR,label="I+R(data)")
plt.legend()
plt.title("COVID 19: SIR model vs data")
plt.savefig('modelvsdata')
