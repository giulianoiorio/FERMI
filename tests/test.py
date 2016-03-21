from fermi.tsintegrator import Tsintegrator1D
import numpy as np
import time

f=lambda x: x*np.exp(-x/5)

i=Tsintegrator1D(30000000)

t1=time.time()
print(i.integrate(f,0,1,use_c=True))
print(time.time()-t1)

t1=time.time()
print(i.integrate(f,0,1,use_c=False))
print(time.time()-t1)
