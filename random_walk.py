import numpy as np
import random
import math
from analysis import bin_msd

q=1.602E-19
kB=1.38064852E-23
T=300
dec=3.5*8.85E-8
pi=3.14159265
h=(6.602E-34)/(2*pi)
m= 9.31 *10**-31
c= 3* 10**8
qsbym= 1.602**2/9.31 * 10**-7  #in SI units

W=400
L=400
H=400
N= H*(L*W)

d= 3.27
sigma= 0.1
RE= 0.1
"""every point gets an index from 0 to N-1. Neighbours of i^th point are found in nbr[i]. Donor indices are [0:N/2] """

def nbr(i):
    nb=(np.array([i+1, i-1, i+L, i-L, i+1+L,i+1-L, i-L-1, i+L-1, i+(L*W), i+(L*W)+L, i+(L*W)-L, i+(L*W)+1, i+(L*W)-1, i+(L*W)+L+1, i+1-L+(L*W),i-L-1+(L*W),i+L-1+(L*W), i-(L*W), i-(L*W)+L, i-(L*W)-L, i-(L*W)-1, i-(L*W)+1, i-(L*W)+L+1, i+1-L-(L*W),i-L-1-(L*W),i+L-1-(L*W)]).T).astype(int)
    nb[nb>(N-1)]=nb[nb>(N-1)]-N
    nb[nb<0]=nb[nb<0]+N
    return nb

def distance( A1, A2): #   np.array([a,b,c]), np.array([p,q,r])):
    return( np.sqrt(  (A1[0]- A2[0])**2 + (A1[1]- A2[1])**2  + (A1[2]- A2[2])**2 ))

s= 0.10
dx= abs(np.array([np.random.normal(0.0, s,  N ) ]))
dy= abs(np.array([np.random.normal(0.0, s,  N ) ]))
dz= abs(np.array([np.random.normal(0.0, s,  N ) ]))
shift= np.sqrt(dx**2+dy**2+dz**2)
#costheta is the cosine of the angle between molecule and X-axis and cosphi is cosine of the angle between molecule and Z-axis or X-Y plane
costheta= (np.random.rand(N))
cosphi= (np.random.rand(N))

A_J = np.genfromtxt('AQx2.txt')# usecols=1) #, delimiter= ',', skip_header=1 
 
def discor( A1, A2): #   np.array([a,b,c]), np.array([p,q,r])):
    return( np.sqrt(  (A1//(L*W)- A2//(L*W))**2 + ((A1%((L*W)))//L-(A2%((L*W)))//L)**2 +((A1%((L*W)))%L-(A2%((L*W)))%L )**2 ) ) 
def sel(n):
    return(np.random.randint(0,len(A_J), size=n))
def hop( J, dE):
    return( (q**2*J**2/h)*np.sqrt(np.pi/(q*RE*T*kB))*np.exp( -(RE*q-dE*q)**2/(4*kB*T*RE*q))) 
diff= np.empty(int(N/2), dtype=object) #- L*W), dtype=object)  #shape= ( int(N/2- L*W), 26) )

trials=4000
times=3200
time= np.empty(shape=( trials, times))
x= np.empty(shape=( trials, times))

starter = int(L/2 + (W/2-1)*L + (H/2-1)*L*W)

for m in np.arange(trials):
    p= starter
    t=0
    E= (np.random.normal(0.0, sigma, N) )
    for n in np.arange(times):
        idn= nbr(p) 
        inD= sel(len(idn))
        J= A_J[inD]* np.exp(-dx[0][idn])*costheta[idn]**2#*cosphi[inD]**2

        rate= hop(J, E[p]- E[idn] )
        ks=rate/(np.sum(rate,axis=0))
        kc=np.cumsum(ks,axis=0)
        r=(np.random.rand())
        kc[kc<r]=2
        say=np.argmin((kc),axis=0)

        p= idn[say]
        t= t+1/rate[say] 
        time[m,n]= t
        x[m,n]= p

tm, posm = bin_msd(x, time, starter, d)     # `time` is already cumulative
np.savez('msd.npz', tm=tm, posm=posm, d=d, trials=trials, sigma=sigma)
