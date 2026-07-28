import numpy as np
import matplotlib.pyplot as plt

q=1.602E-19
kB=1.38064852E-23
T=300

L=800
W=800
H=800
trials=400
d= 3.27#(np.genfromtxt('intersite_d'))[0]
def discor( A1, A2): #   np.array([a,b,c]), np.array([p,q,r])):
    return( np.sqrt(  (A1//(L*W)- A2//(L*W))**2 + ((A1%((L*W)))//L-(A2%((L*W)))//L)**2 +((A1%((L*W)))%L-(A2%((L*W)))%L )**2 ) ) 

x= np.load('x.npy')
time= np.load('time.npy')
tt= np.cumsum( time, axis=1)
ts= 10**-10
sn= 100
posm=np.zeros(sn)                                               
tm=np.zeros(sn) 
starter= int(L/2+(W/2-1)*L+(H/2-1)*L*W ) 
for i in np.arange(sn):
    t= tt[np.arange(trials),(tt> i*ts).argmax(axis=1)]    
    t[t>(i+1)*ts]=0
    p= x[np.arange(trials),(tt> i*ts).argmax(axis=1)]    
    s= discor( starter, p ) *d *10**-8
    posm[i]=  np.mean((s[t!=0])**2)
    tm[i]= np.mean(t[np.nonzero(t)])
m,c= np.polyfit(tm[15:], posm[15:] ,1)
print('intersite', m*(q/(kB*T)/6))
plt.plot(tm, posm, 'o')
plt.plot(tm, tm*m+c, '-')
plt.show()
