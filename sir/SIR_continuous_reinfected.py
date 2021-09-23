import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.random import randint, rand
from sir import *


def SIR_continuous_reinfected(b,k,time,ii,r):
    """
    Simulates continuous SIR model
    ii = initial percentage of infected
    time = Days of simulation
    b = probability that people getting infectious 
    k = probability that people getting recovered
    r = reinfected probability
    
    returns sol from solve_ivp
    """
    def SIR(t, X):

        #The main set of equations
        Y = np.zeros((3))
        Y[0] = -b * X[0] * X[2]
        Y[1] = k * X[2] - r * X[1]
        Y[2] = b * X[0] * X[2] - (k * X[2]) + r * X[1]
        return Y
    t_eval = np.linspace(0, time, time)
    sol1 = solve_ivp(SIR, [0, time], [1-ii, 0, ii], method='RK45', t_eval=t_eval)    # solve the equation
    return sol1

## Discrete

class Person_reinfection(Person):
    """
    An agent representing a person.
    
    By default, a person is susceptible but not infectious. They can become infectious by exposing with disease method.
    
    Status: 0 = susceptible        1 = infected          2 = removed
    """  
    
    def __init__(self,startpos=None):
        self.status = 0
        if startpos==None:
            self.pos = np.random.rand(2)
        else:
            self.pos = np.array(startpos)
        self.reinfection=1

   
    def reinfectionrate(self):
        return self.reinfection
    def immunization(self,p):
        q=self.reinfection-p
        if q<0:
            q=0
        self.reinfection=q
    

def count_susceptible(pop):
    """
    counts number of susceptible
    """
    return sum(p.is_susceptible() for p in pop)

def count_infected(pop):
    """
    counts number of infected
    """
    return sum(p.is_infected() for p in pop)

def count_removed(pop):
    """
    counts number of removed
    """
    return sum(p.is_removed() for p in pop)

def SIR_discrete_reinfection(N,ii,b,T,k):
    """
    Simulates discrete SIR model
    N = Total number of people
    ii = initial percentage of infected
    b = number of contacts per day
    T = Days of simulation
    k = probability that people getting recovered
    
    returns list of s,i,r
    """
    pop = [Person_reinfection() for i in range(N)]
    initial_infection = randint(N,size=np.int(N*ii))
    for i in initial_infection:
        pop[i].infection()

    counts_susceptible = [count_susceptible(pop)]
    counts_infected = [count_infected(pop)]
    counts_removed = [count_removed(pop)]

    for t in range(T):
        # update the population
        for i in range(N):
            if pop[i].is_infected():
                # person i infected all their contacts
                contacts = randint(N, size=b)
                for j in contacts:
                    if not pop[j].is_removed():
                        pop[j].infection()
                        #if rand() < p:
                        #    pop[j].infection()
                    if pop[j].is_removed():
                        if rand()<pop[j].reinfectionrate():
                            pop[j].infection()
                if rand()< k:
                    pop[i].remove()
                    pop[i].immunization(rand())

        # add to our counts
        counts_susceptible.append(count_susceptible(pop))
        counts_infected.append(count_infected(pop))
        counts_removed.append(count_removed(pop))
    return np.array([counts_susceptible,counts_infected,counts_removed])


