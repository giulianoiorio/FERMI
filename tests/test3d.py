from fermi.tsintegrator import Tsintegrator3D, Tsintegrator2D, Tsintegrator1D
from scipy.integrate import nquad
import numpy as np
import time

def f(x,y,z): return (x+y+z)*np.exp(-x/5)/y**2
xlim=(0,3)
ylim=(1,2)
zlim=(5,6)

i=Tsintegrator3D()


t1=time.time()
print(i.integrate(f,xlim,ylim,zlim))
print(time.time()-t1)

t1=time.time()
print(nquad(f,[xlim,ylim,zlim],opts={'epsabs':1e-6,'epsrel':1e-6}))
print(time.time()-t1)
