import cma
import cma.purecma as purecma
from deap import benchmarks
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random
from scipy.optimize import minimize

from plot import *


def ackley(x):
    return benchmarks.ackley(x)[0]

ma_func=ackley


############## Test CMA-ES ###################

def launch_cmaes(center, sigma, nbeval=100, display=True):
    es = cma.CMAEvolutionStrategy(center, sigma)
    
    for _ in range(nbeval):
        solutions = es.ask()
        
        if display:
            plot_results(ackley, solutions)
        
        es.tell(solutions, [ackley(s) for s in solutions])
        
        
       # es.disp()
    
    ### A completer pour utiliser CMA-ES et tracer les individus générés à chaque étape avec plot_results###

    return es.result

def launch_cmaes_pure(center, sigma, nbeval=100, display=True):
    
    es = purecma.CMAES(center, sigma)
    
    ### A completer pour utiliser CMA-ES et tracer les individus générés à chaque étape avec plot_results###
    
    for _ in range(nbeval):
        solutions = es.ask()
        
        if display:
            plot_results(ackley, solutions)
        
        es.tell(solutions, [ackley(s) for s in solutions])
        
       # es.disp()

    return es.result
