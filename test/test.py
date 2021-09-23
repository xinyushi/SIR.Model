import sys

sys.path.append("../sir/")

#from sir.sir import *
from sir import *

import unittest
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.random import randint, rand
from scipy.special import comb


class TestDiscreteSIR(unittest.TestCase):
    def test_count_infected(self):
        '''
        Test if count_infected is correct
        ''' 
        N = 1000
        k = 5
        T = 100
        n = 1000
        ii = 0.01
        pop = [Person() for i in range(n)]
        initial_infection = randint(n,size=np.int(n*ii))
        for i in initial_infection:
            pop[i].infection()
            
        countInf = count_infected(pop)

        # Hand calcualation for getting infected people
        Count_inf = N*ii        # getting infectious at the end of the day




        self.assertEqual(countInf,Count_inf)
    
    def test_count_susceptible(self):
        '''
        Test if count_susceptible is correct
        '''
        N = 100   # Total number of people
        pop = [Person() for i in range(N)]
        countSus = sum(p.is_susceptible() for p in pop)
        
        self.assertEqual(countSus,N)
        
class TestContinuousSIR(unittest.TestCase):
    def test_continuous(self):
        """
        Check the sum of s,i,r=1 across t
        """
        b=1
        k=1/3
        time=150
        ii=0.01


        sol1=SIR_continuous(b,k,time,ii)
        sumy=sol1.y[0]+sol1.y[1]+sol1.y[2]
        self.assertAlmostEqual(np.linalg.norm(sumy),np.sqrt(time))
