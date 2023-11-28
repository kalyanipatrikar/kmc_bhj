import numpy as np
import random
import math
#import matplotlib.pyplot as plt

q=1.602E-19
kB=1.38064852E-23
T=300
dec=3.5*8.85E-8
pi=3.14159265
h=(6.602E-34)/(2*pi)
m= 9.31 *10**-31
qsbym= 1.602**2/9.31 * 10**-7


#####################################################################################################################################################################################################
 #                                                                                                    GRID                                                     
#####################################################################################################################################################################################################

"""Let's say start with a grid of 100x100x100. For now, everything with H < 50 is a donor."""
W=10
L=10
H=10
N= H*(L*W)

sigma= 0.06
RE= 0.1 
"""every point gets an index from 0 to N-1. Neighbours of i^th point are found in nbr[i]. Donor indices are [0:N/2] """

def nbr(x):
  ip1 = x + 1
  ip2 = x + L
  ip3 = x - 1
  ip4 = x - L
  ip5 = ip1 + L
  ip6 = ip1 - L
  ip7 = ip3 + L
  ip8 = ip3 - L

  ap0 = x + (L*W)
  ap1 = ap0 + 1
  ap2 = ap0 + L
  ap3 = ap0 - 1
  ap4 = ap0 - L
  ap5 = ap1 + L
  ap6 = ap1 - L
  ap7 = ap3 + L
  ap8 = ap3 - L

  bp0 = x - (L*W)
  bp1 = bp0 + 1
  bp2 = bp0 + L
  bp3 = bp0 - 1
  bp4 = bp0 - L
  bp5 = bp1 + L
  bp6 = bp1 - L
  bp7 = bp3 + L
  bp8 = bp3 - L

  front_face = [ip1, ip2, ip3, ip4+(L*W), ip5, ip6+(L*W), ip7, ip8+(L*W), ap0, ap1, ap2, ap3, ap4+(L*W), ap5, ap6+(L*W), ap7, ap8+(L*W), bp0, bp1, bp2, bp3, bp4+(L*W), bp5, bp6+(L*W), bp7, bp8+(L*W)]
  bottom_face = [ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ap0, ap1, ap2, ap3, ap4, ap5, ap6, ap7, ap8]
  top_face = [ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, bp0, bp1, bp2, bp3, bp4, bp5, bp6, bp7, bp8]
  back_face = [ip1, ip2-(L*W), ip3, ip4, ip5-(L*W), ip6, ip7-(L*W), ip8, ap0, ap1, ap2-(L*W), ap3, ap4, ap5-(L*W), ap6, ap7-(L*W), ap8, bp0, bp1, bp2-(L*W), bp3, bp4, bp5-(L*W), bp6, bp7-(L*W), bp8]
  left_face = [ip1, ip2, ip3+L, ip4, ip5, ip6, ip7+L, ip8+L, ap0, ap1, ap2, ap3+L, ap4, ap5, ap6, ap7+L, ap8+L, bp0, bp1, bp2, bp3+L, bp4, bp5, bp6, bp7+L, bp8+L]
  right_face = [ip1-L, ip2, ip3, ip4, ip5-L, ip6-L, ip7, ip8, ap0, ap1-L, ap2, ap3, ap4, ap5-L, ap6-L, ap7, ap8, bp0, bp1-L, bp2, bp3, bp4, bp5-L, bp6-L, bp7, bp8]


  if x == 0:
    return(np.array([ip1, ip2, ip3+L, ip4+(L*W), ip5, ip6+(L*W), ip7+L, ip8+(L*W)+L, ap0, ap1, ap2, ap3+L, ap4+(L*W), ap5, ap6+(L*W), ap7+L, ap8+(L*W)+L])).astype(int)
  elif x == L-1:
    return(np.array([ip1-L, ip2, ip3, ip4+(L*W), ip5-L, ip6+(L*W)-L, ip7, ip8+(L*W), ap0, ap1-L, ap2, ap3, ap4+(L*W), ap5-L, ap6+(L*W)-L, ap7, ap8+(L*W)])).astype(int)
  elif x == L*W-L:
    return(np.array([ip1, ip2-(L*W), ip3+L, ip4, ip5-(L*W), ip6, ip7-(L*W)+L, ip8+L, ap0, ap1, ap2-(L*W), ap3+L, ap4, ap5-(L*W), ap6, ap7-(L*W)+L, ap8+L])).astype(int)
  elif x == L*W-1:
    return(np.array([ip1-L, ip2-(L*W), ip3, ip4, ip5-(L*W)-L, ip6-L, ip7-(L*W), ip8, ap0, ap1-L, ap2-(L*W), ap3, ap4, ap5-(L*W)-L, ap6-L, ap7-(L*W), ap8])).astype(int)
  elif x == L*W*H-L*W:
    return(np.array([ip1, ip2, ip3+L, ip4+(L*W), ip5, ip6+(L*W), ip7+L, ip8+(L*W)+L, bp0, bp1, bp2, bp3+L, bp4+(L*W), bp5, bp6+(L*W), bp7+L, bp8+(L*W)+L])).astype(int)
  elif x == (L*W*H)-(L*W)+L-1:
    return(np.array([ip1-L, ip2, ip3, ip4+(L*W), ip5-L, ip6+(L*W)-L, ip7, ip8+(L*W), bp0, bp1-L, bp2, bp3, bp4+(L*W), bp5-L, bp6+(L*W)-L, bp7, bp8+(L*W)])).astype(int)
  elif x == L*W*H-L:
    return(np.array([ip1, ip2-(L*W), ip3+L, ip4, ip5-(L*W), ip6, ip7-(L*W)+L, ip8+L, bp0, bp1, bp2-(L*W), bp3+L, bp4, bp5-(L*W), bp6, bp7-(L*W)+L, bp8+L])).astype(int)
  elif x == L*W*H-1:
    return(np.array([ip1-L, ip2-(L*W), ip3, ip4, ip5-(L*W)-L, ip6-L, ip7-(L*W), ip8, bp0, bp1-L, bp2-(L*W), bp3, bp4, bp5-(L*W)-L, bp6-L, bp7-(L*W), bp8])).astype(int)



  elif 0 < x%(L*W) < L:
    has_negative = any(i < 0 for i in front_face)
    has_more = any(j >= L*W*H for j in front_face)
    if has_negative:
      return(np.array(list(filter(lambda i : i >= 0, front_face)))).astype(int)
    elif has_more:
      return(np.array(list(filter(lambda j : j < L*W*H, front_face)))).astype(int)
    else:
      return(np.array(front_face)).astype(int)

  elif 0 < (L*W) - x%(L*W) < L:
    has_negative = any(i < 0 for i in back_face)
    has_more = any(j >= L*W*H for j in back_face)
    if has_negative:
      return(np.array(list(filter(lambda i : i >= 0, back_face)))).astype(int)
    elif has_more:
      return(np.array(list(filter(lambda j : j < L*W*H, back_face)))).astype(int)
    else:
      return(np.array(back_face)).astype(int)

  elif x%L == 0:
    has_negative = any(i < 0 for i in left_face)
    has_more = any(j >= L*W*H for j in left_face)
    if has_negative:
      return(np.array(list(filter(lambda i : i >= 0, left_face)))).astype(int)
    elif has_more:
      return(np.array(list(filter(lambda j : j < L*W*H, left_face)))).astype(int)
    else:
      return(np.array(left_face)).astype(int)

  elif x%L == L-1:
    has_negative = any(i < 0 for i in right_face)
    has_more = any(j >= L*W*H for j in right_face)
    if has_negative:
      return(np.array(list(filter(lambda i : i >= 0, right_face)))).astype(int)
    elif has_more:
      return(np.array(list(filter(lambda j : j < L*W*H, right_face)))).astype(int)
    else:
      return(np.array(right_face)).astype(int)

  elif 0 < x < L*W:
    return(np.array(bottom_face)).astype(int)

  elif L*W*H - L*W < x < L*W*H:
    return(np.array(top_face)).astype(int)

  else:
    return(np.array([ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ap0, ap1, ap2, ap3, ap4, ap5, ap6, ap7, ap8, bp0, bp1, bp2, bp3, bp4, bp5, bp6, bp7, bp8])).astype(int)



#def nbr(i):
#    nb=(np.array([i+1, i-1, i+L, i-L, i+1+L,i+1-L, i-L-1, i+L-1, i+(L*W), i+(L*W)+L, i+(L*W)-L, i+(L*W)+1, i+(L*W)-1, i+(L*W)+L+1, i+1-L+(L*W),i-L-1+(L*W),i+L-1+(L*W), i-(L*W), i-(L*W)+L, i-(L*W)-L, i-(L*W)-1, i-(L*W)+1, i-(L*W)+L+1, i+1-L-(L*W),i-L-1-(L*W),i+L-1-(L*W)]).T).astype(int)
#    nb=np.delete(nb, (np.where(nb<0)) )
#    nb=np.delete(nb, (np.where(nb>N-1)) )
#    # np.delete(nb, np.where(nb[nb<0]) )
#    return nb

def distance( A1, A2): #   np.array([a,b,c]), np.array([p,q,r])):
    return( np.sqrt( (A1[0]- A2[0])**2 + (A1[0]- A2[0])**2 + (A1[1]- A2[1])**2  + (A1[2]- A2[2])**2   ) ) 
   

d= 4.0
s= 0.1
dx= abs(np.array([np.random.normal(0.0, s,  N ) ]))
dy= abs(np.array([np.random.normal(0.0, s,  N ) ]))
dz= abs(np.array([np.random.normal(0.0, s,  N ) ]))

#costheta is the cosine of the angle between molecule and X-axis and cosphi is cosine of the angle between molecule and Z-axis or X-Y plane
costheta=[]
cosphi=[]
for i in range(N):
  costheta.append(random.uniform(-1,1))
  cosphi.append(random.uniform(-1,1))


""" while accessing any point, its position is its expected x,y,z adjusted to x+dx, y+dy, z+dz
as ca is between zero and one, it can be assumed to be cos of the angle molecular axis makes with x-axis"""

A_rates= np.genfromtxt('AQ', delimiter= ',', skip_header=1 )
h_rates= np.genfromtxt('PM6', usecols=1) #, delimiter= ',', skip_header=1 )

def sel(n):
    return(np.random.randint(1,10, size=n))

def hop( J, dE):
    return( 10**10*(q**2*J**2/h)*np.sqrt(np.pi/(RE*T*kB))*np.exp( -(RE*q-dE*q)**2/(4*kB*T*RE*q))) 

holediff= np.empty(int(N/2), dtype=object) #- L*W))  #shape= ( int(N/2- L*W), 26) )
elecdiff= np.empty(int(N/2), dtype=object) #- L*W), dtype=object)  #shape= ( int(N/2- L*W), 26) )

    
#    nD= nbr(i)
#    inD= sel(len(nD))
#    JD= h_rates(inD)
#    elecdiff= hop( JD, h_rates)
    

#############################################################################################################################################################
 #                                                                HOP n BREAK FUNCTIONS
##############################################################################################################################################################
def exciton_diffusion( p ):
    idn = nbr(p).T

    rate=np.random.rand(len(idn))*10**10
    ks=rate/(np.sum(rate,axis=0))
    kc=np.cumsum(ks,axis=0)
    r=(np.random.rand()) # ,(26,1)) #, len(idn[0])
    kc[kc<r]=2
    idxnn=np.argmin((kc),axis=0)

    p= idn[idxnn]
    t0= 1/rate[idxnn]
    return(p, t0)


###########################################################
def eh_diffusion (ep, hp):
    ep = int(ep)
    hp = int(hp)
    xx= distance( [ ep//(L*W), (ep%((L*W)))//L, (ep%((L*W)))%L ], [ hp//(L*W), (hp%((L*W)))//L, (hp%((L*W)))%L ]  )
    if  xx <16:
        iddn= np.array( nbr(ep)[np.where( nbr(ep) >N/2)]).T
        rate=  elecdiff[ep -int(N/2) ][sel(len(iddn))]
        #conf= A_rates[sel(1), 1]
        #rate=np.random.rand(len(iddn))*10**12
        #rate= np.apepnd(rate, conf )
        #rate= np.apepnd(rate, np.random.rand()*10**12 )    #Apepnd rate of recombination of e-h+ pair
        coulrate= np.sqrt(qsbym/ (4*pi*dec*(xx*10**-9)**3))
        rate= np.append(rate, coulrate )
        ks=rate/(np.sum(rate,axis=0))
        kc=np.cumsum(ks,axis=0)
        r=(np.random.rand()) # ,(26,1)) #, len(idn[0])
        kc[kc<r]=2
        idxnn=np.argmin((kc),axis=0)
        te= 1/rate[len(rate)-1]

        if idxnn !=(len(iddn)):
            iddnh= np.array( nbr(hp)[np.where( nbr(hp) <N/2-(L*W))]).T
            rate= holediff[hp][sel(len(iddnh))]
            #rate=np.random.rand(len(iddnh))*10**12
            ks=rate/(np.sum(rate,axis=0))
            kc=np.cumsum(ks,axis=0)
            r=(np.random.rand()) # ,(26,1)) #, len(idn[0])
            kc[kc<r]=2
            idxnnh=np.argmin((kc),axis=0)
            hp= iddnh[idxnnh]
            th= 1/rate[idxnnh]
            ep= iddn[idxnn]

            return( ep, hp, te, th)
        else:
            return(hp, te)

    else:
        iddne= np.array( nbr(ep)[np.where( nbr(ep) >N/2)]).T
        rate= elecdiff[ep- int(N/2)][sel(len(iddne))]
        #rate=np.random.rand(len(iddne))*10**12
        ks=rate/(np.sum(rate,axis=0))
        kc=np.cumsum(ks,axis=0)
        r=(np.random.rand()) # ,(26,1)) #, len(idn[0])
        kc[kc<r]=2
        idxnne=np.argmin((kc),axis=0)
        ep= iddne[idxnne]
        te= 1/rate[idxnne]

        iddnh= np.array( nbr(hp)[np.where( nbr(hp) <N/2-(L*W))]).T
        rate= h_rates[sel(len(iddnh))]
        #rate=np.random.rand(len(iddnh))*10**12
        ks=rate/(np.sum(rate,axis=0))
        kc=np.cumsum(ks,axis=0)
        r=(np.random.rand()) # ,(26,1)) #, len(idn[0])
        kc[kc<r]=2
        idxnnh=np.argmin((kc),axis=0)
        hp= iddnh[idxnnh]
        th= 1/rate[idxnnh]

        return(ep, hp, te, th)
###################################################################################################################
def interface_split(p):
    iddn= np.array( nbr(p)[np.where( nbr(p) <=N/2)]).T
    rate= h_rates[sel(len(iddn))]
    #print(rate)
    #rate=np.random.rand(len(iddn))*10**12
    conf= A_rates[sel(1), 0]
    rate= np.append(rate, conf )
   #rate= np.append(rate, np.random.rand()*10**12 )   #Append rate of dissociation of exciton
    ks=rate/(np.sum(rate,axis=0))
    kc=np.cumsum(ks,axis=0)
    r=(np.random.rand()) # ,(26,1)) #, len(idn[0])
    kc[kc<r]=2
    idxnn=np.argmin((kc),axis=0)
    t= 1/rate[idxnn]
    #print(rate[idxnn]/10**12)

    if idxnn ==(len(iddn)):   # Does it split or recombine
        pe= min(nbr(p)[(np.where(nbr(p) >N/2))])
        ph= p
        iddn= np.array( nbr(pe)[np.where( nbr(pe) >N/2)]).T

        rate= A_rates[sel(len(iddn)), 2]
        conf= A_rates[sel(1), 1]
        rate= np.append(rate, conf )
        #rate=np.random.rand(len(iddn))*10**12
        #rate= np.append(rate, np.random.rand()*10**12 )    #Append rate of recombination of e-h+ pair
        ks=rate/(np.sum(rate,axis=0))
        kc=np.cumsum(ks,axis=0)
        r=(np.random.rand()) # ,(26,1)) #, len(idn[0])
        kc[kc<r]=2
        idxnn=np.argmin((kc),axis=0)
        t= 1/rate[len(rate)-1]

        if idxnn !=(len(iddn)):  # Does it continue split movement, ie does not recombine
            # pe= iddn[idxnn]
            # ph= p
            return( pe, ph, t)
        else:
            return(p, t)

    else:
        return(iddn[idxnn], t)
#####################################################################################################################
 #                                                 LOOPs
#####################################################################################################################
trials=10
times= 100
pe= np.zeros(shape=(trials, times))-1
ph= np.zeros(shape=(trials, times)) + N+L
te= np.zeros(shape=(trials, times))
th= np.zeros(shape=(trials, times))
p0= np.zeros(shape=(trials, times))-1
t0= np.zeros(shape=(trials, times))

V=0.4

for m in np.arange(trials):
    p0[m,0]=  np.random.randint(0, N/2-(L*W))
    c=0
    b=0
    E= (np.random.normal(0.0, sigma, N) )
    xx= np.arange(N)//(L*W)
    E= E- xx*V/(L*W)

    for i in np.arange(int(N/2 )):
        nD= nbr(i)
        inD= sel(len(nD))
        JD= h_rates[inD]
        holediff[i]= hop( JD, E[i]- E[nD])
        
    for i in np.arange(int(N/2)):
        nD= nbr(i+ N/2)
        inD= sel(len(nD))
        JD= A_rates[:,2][inD]
        elecdiff[i]= hop( JD, E[i]- E[nD])
    
    for n in np.arange(times-1):
        if c==1:
            jump= eh_diffusion(pe[m,n], ph[m,n])
            if len(jump)==4:
                pe[m,n+1], ph[m,n+1] = jump[0], jump[1]
                te[m,n+1], th[m,n+1] = jump[2], jump[3]
                c=1
            else:
                 p0[m,n+1]= jump[0]
                 t0[m,n+1]= jump[1]
                 b=1

        if c==0:
            jumpe  =  exciton_diffusion(p0[m,n])
            p0[m,n+1]= jumpe[0]
            t0[m,n+1]= jumpe[1]
            c=0

        if(N/2 > p0[m,n] >= N/2-(L*W) ):
            has_split= interface_split(p0[m,n].astype(int))
            if len(has_split)==3:   ##It has split
                pe[m,n+1]= has_split[0]
                ph[m,n+1]= has_split[1]
                t0[m,n+1]= has_split[2]
                p0[m,n+1]=0
                c=1

            else:
                p0[m, n+1]= has_split[0]
                t0[m, n+1]= has_split[1]
                c=0

        if b==1:
            c=0
            b=0
        else:
            pass

############################################################################################################################################################
  #                                                                  TRAPS
#############################################################################################################################################################

traps_donor = list(range(int(N/2-(L*W))))

# selecting 5 random lattice points to add traps
Dtraps = random.sample(traps_donor, 5)

# Range between what we want traps
traps_acceptor = list(range(int(N/2), N))

# selecting 5 random lattice points to add traps
Atraps = random.sample(traps_acceptor, 5)

nh=0
ne=0 
trapsiteD= np.zeros(len(ph))
for i in range(len(ph)):
  for j in range(len(ph[i])):
    if 0<= ph[i][j] <L*W:
        nh = nh+1
        ph[i][j+1:]= 2*N
        break
    elif ph[i][j] in Dtraps:
      trapsiteD[i]= j
      break
    else:
      pass
trapsiteA= np.zeros(len(pe))
for i in range(len(pe)):
  for j in range(len(pe[i])):
    if pe[i][j] >= N- L*W:
        ne = ne+1
        pe[i][j+1:]= -N
        break
    elif pe[i][j] in Atraps:
      trapsiteA[i]= j
      break
    else:
      pass


######################################################################################################################################################
 #                                                                  MOBILITY
######################################################################################################################################################
mu_h= np.zeros(len(ph))
mu_e= np.zeros(len(pe))

v_h= np.zeros(len(ph))
v_e= np.zeros(len(pe))

for i in np.arange(trials):
    if np.size(np.where(ph[i]< L*W))>0:
        pfh= ph[i][np.where(ph[i]<(L*W))[0][0]]
        pih= ph[i][np.where(ph[i]==N+L)[0][-1]+1]
        displacement_h = distance( [ pfh//(L*W), (pfh%((L*W)))//L, (pfh%((L*W)))%L ], [ pih//(L*W), (pih%((L*W)))//L, (pih%((L*W)))%L ]  )

        tth= np.sum(th[i][np.where(ph[i]==N+L)[0][-1]+1:np.where(ph[i]<(L*W))[0][0]])
        mu_h[i]= (q/(kB*T))*(displacement_h*10**-9)**2/(6*tth)
        v_h[i]= displacement_h/tth

for i in np.arange(trials):
    if np.size(np.where(pe[i]== -N))>0:
        pfe= pe[i][np.where(pe[i]== -N)[0][0]-1]
        pie= pe[i][np.where(pe[i]== -1)[0][-1]+1]
        displacement_e = distance( [ pfe//(L*W), (pfe%((L*W)))//L, (pfe%((L*W)))%L ], [ pie//(L*W), (pie%((L*W)))//L, (pie%((L*W)))%L ]  )

        tte= np.sum(te[i][np.where(pe[i]== -1)[0][-1]+1:np.where(pe[i]== -N)[0][0]-1])
        mu_e[i]= (q/(kB*T))*(displacement_e*10**-9)**2/(6*tte)
        v_e[i]= displacement_e/tte
#        fe=np.where(pe[i]==(-N))[0][0] -1
#        ie= np.where(pe[i]==-1)[0][-1]
#        print(ie, fe)
#    
#        pfe= pe[i][fe]
#        pie= pe[i][ie]
#        displacement_h = distance( [ pfe//(L*W), (pfe%((L*W)))//L, (pfe%((L*W)))%L ], [ pie//(L*W), (pie%((L*W)))//L, (pie%((L*W)))%L ]  )
#    
#        tte= np.sum(te[i][ie:fe])
#        mu_e[i]= (q/(kB*T))*(displacement_h*10**-9)**2/(6*tte)

muh= np.mean(mu_h[np.nonzero(mu_h) ])
mue= np.mean(mu_e[np.nonzero(mu_e) ])
vh= np.mean(v_h[np.nonzero(v_h) ])
ve= np.mean(v_e[np.nonzero(v_e) ])
#################################################################################################################################################################
print(muh, nh, q*vh*nh ) 
print(mue, ne, q*ve*ne ) 
