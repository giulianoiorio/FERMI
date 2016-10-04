from fermi.tsintegrator import Tsintegrator1D
import numpy as np
import time


def cvar(rs):
    def cvartmp(func):
        def func_wrapper(r,*args):
            return func(r*rs,*args)
        return func_wrapper
    return cvartmp

#@cvar(1)
def f(x): return (1./(1-x*x))*np.exp(-x/5)

i=Tsintegrator1D(30,hstep=3.17)



t1=time.time()
print(i.integrate(f,3,10.))
print(time.time()-t1)
