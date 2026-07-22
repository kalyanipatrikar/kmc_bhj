import numpy as np, matplotlib.pyplot as plt

q, kB, T = 1.602e-19, 1.380649e-23, 300
D = np.load('msd.npz')
tm, posm = D['tm'], D['posm']

ok = np.isfinite(tm) & np.isfinite(posm)
m, c = np.polyfit(tm[ok][15:], posm[ok][15:], 1)
print('mu =', m * q/(kB*T) / 6, 'cm2/Vs')

plt.plot(tm[ok], posm[ok], 'o')
plt.plot(tm[ok], m*tm[ok] + c, '-')
plt.xlabel('t (s)'); plt.ylabel(r'$\langle x^2\rangle$ (cm$^2$)')
plt.show()