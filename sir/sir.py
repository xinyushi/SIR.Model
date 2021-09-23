# discrete
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.random import randint, rand
from scipy.spatial import KDTree

class Person():
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


    
    def is_susceptible(self):
        """
        returns true if the person is susceptible
        """
        return self.status==0

    def susceptible(self):
        """
        once the person
        """
        self.status = 0
        
    def is_infected(self):
        """
        returns true if the person is infectious
        """
        return self.status==1

    def infection(self):
        """
        once the person
        """
        self.status=1
      
    def is_removed(self):
        """
        returns true if the person is recovered
        """
        return self.status==2

    def remove(self):
        """
        once the person
        """
        self.status=2
        
    def newpos(self,p):
        """
        new position
        """
        dpos = np.random.randn(2)
        dpos = dpos / np.linalg.norm(dpos)
        newpos = self.pos + dpos * p
        if newpos[0]>=0 and newpos[0]<=1:
            if newpos[1]>=0 and newpos[1]<=1:
                self.pos = newpos

    

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

def SIR_discrete(N,ii,b,T,k):
    """
    Simulates discrete SIR model
    N = Total number of people
    ii = initial percentage of infected
    b = number of contacts per day
    T = Days of simulation
    k = probability that people getting recovered
    
    returns list of s,i,r
    """
    pop = [Person() for i in range(N)]
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
                if rand()< k:
                    pop[i].remove()

        # add to our counts
        counts_susceptible.append(count_susceptible(pop))
        counts_infected.append(count_infected(pop))
        counts_removed.append(count_removed(pop))
    return np.array([counts_susceptible,counts_infected,counts_removed])
    

    
    
def SIR_discrete_spatial(N,ii,p,q,T,k,startpos):
    """
    Simulates discrete SIR model
    N = Total number of people
    ii = initial percentage of infected
    p = step of length p
    q = individual radius of interact people
    T = Days of simulation
    k = probability that people getting recovered
    startpos = starting position for each individual. Default is uniform random in 1 by 1 grid.
    
    returns list of s,i,r
    """
    pop = [Person(startpos) for i in range(N)]
    initial_infection = randint(N,size=np.int(N*ii))
    for i in initial_infection:
        pop[i].infection()

    counts_susceptible = [count_susceptible(pop)]
    counts_infected = [count_infected(pop)]
    counts_removed = [count_removed(pop)]
    
    def matrixX(pop,N):
        """
        create matrix X, which stores the position of the population, for KDtree
        """
        X=np.random.rand(N,2)
        for i in range(N):
            X[i]=pop[i].pos
        return X
    

    for t in range(T):
        # update the population
        for i in range(N):
            pop[i].newpos(p)
            if pop[i].is_infected():
                # person i infected all their contacts
                X = matrixX(pop,N) # a 2d spatial matrix 
                tree = KDTree(X)
                inds = tree.query_ball_point([pop[i].pos], q) # finds neighbors in ball of radius q        
                contacts = inds[0] # From my understand, inds is the total contact people, But not sure????
                for j in contacts:
                    if not pop[j].is_removed():
                        pop[j].infection()
                        #if rand() < p:
                        #    pop[j].infection()
                if rand()< k:
                    pop[i].remove()

        # add to our counts
        counts_susceptible.append(count_susceptible(pop))
        counts_infected.append(count_infected(pop))
        counts_removed.append(count_removed(pop))
    return np.array([counts_susceptible,counts_infected,counts_removed])    


#continuous

def SIR_continuous(b,k,time,ii):
    """
    Simulates continuous SIR model
    ii = initial percentage of infected
    time = Days of simulation
    b = probability that people getting infectious 
    k = probability that people getting recovered
    
    returns sol from solve_ivp
    """
    def SIR(t, X):

        #The main set of equations
        Y = np.zeros((3))
        Y[0] = -b * X[0] * X[2]
        Y[1] = k * X[2] 
        Y[2] = b * X[0] * X[2] - (k * X[2]) 
        return Y
    t_eval = np.linspace(0, time-1, time)
    sol1 = solve_ivp(SIR, [0, time], [1-ii, 0, ii], method='RK45', t_eval=t_eval)    # solve the equation
    return sol1

import scipy.sparse as sparse

# create matrix A to apply forward difference scheme
def forward_diff_matrix(n):
    data = []
    i = []
    j = []
    for k in range(n - 1):
        i.append(k)
        j.append(k)
        data.append(-1)

        i.append(k)
        j.append(k+1)
        data.append(1)
        
    # incidence matrix of the 1-d mesh
    return sparse.coo_matrix((data, (i,j)), shape=(n-1, n)).tocsr()

def Laplacian(n):
    """
    Create Laplacian on 2-dimensional grid with n*n nodes
    """
    B = forward_diff_matrix(n)
    D = -B.T @ B
    Dx = sparse.kron(sparse.eye(n), D).tocsr()
    Dy = sparse.kron(D, sparse.eye(n)).tocsr()
    return Dx + Dy




def SIR_continuous2(b,p,k,time,ii,startpos,M,N,L):
    """
    Simulates continuous SIR model
    ii = initial percentage of infected
    time = Days of simulation
    b = probability that people getting infectious 
    k = probability that people getting recovered
    M = number of grid in each side
    N = initial population to estimate S,I,R in grid
    L = M*M finite difference Laplacian matrix
    
    returns sol from solve_ivp
    """
    pop = [Person(startpos) for i in range(N)]
    initial_infection = randint(N,size=np.int(N*ii))
    for i in initial_infection:
        pop[i].infection()
    S = np.zeros((M,M))
    I = np.zeros((M,M))
    R = np.zeros((M,M))
    l=1/M
    for i in range(N):
        index_x=np.floor(pop[i].pos/np.array([l,l]))[0]
        index_y=np.floor(pop[i].pos/np.array([l,l]))[1]
        if pop[i].is_susceptible:
            S[int(index_x),int(index_y)]+=1
        if pop[i].is_infected():
            I[int(index_x),int(index_y)]+=1
        if pop[i].is_removed():
            R[int(index_x),int(index_y)]+=1
    Sflat=S.flatten()/N
    Iflat=I.flatten()/N
    Rflat=R.flatten()/N
            
            
    def SIR(t, X):
        z=np.zeros((M*M))
        Y=np.append(np.append(z,z),z)
        Y[0:M*M] = -b * X[0:M*M] * X[2*M*M:] + p * L @ X[0:M*M]
        Y[M*M:2*M*M] = k * X[2*M*M:] + p * L @ X[M*M:2*M*M]
        Y[2*M*M:] = b * X[0:M*M] * X[2*M*M:] - (k * X[2*M*M:]) + p * L @ X[2*M*M:]
        return Y
    t_eval = np.linspace(0, time, 1000)
    y0=np.append(np.append(Sflat,Rflat),Iflat)
    sol1 = solve_ivp(SIR, [0, time], y0, method='RK45', t_eval=t_eval)    # solve the equation
    return sol1




