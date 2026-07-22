import numpy as np

L = W = H = 400

def discor(A1, A2):
    return np.sqrt((A1//(L*W) - A2//(L*W))**2
                 + ((A1%(L*W))//L - (A2%(L*W))//L)**2
                 + ((A1%(L*W))%L  - (A2%(L*W))%L)**2)

def bin_msd(x, tt, starter, d, ts=1e-12, sn=60):
    """Reduce trajectories to MSD(t). Returns tm, posm."""
    trials = x.shape[0]
    posm = np.zeros(sn)
    tm   = np.zeros(sn)
    for i in range(sn):
        reached = tt[:, -1] > i*ts
        idx = (tt > i*ts).argmax(axis=1)
        t = np.where(reached, tt[np.arange(trials), idx], 0.0)
        p = x[np.arange(trials), idx]
        s = discor(starter, p) * d * 1e-7      # nm -> cm
        good = t != 0
        posm[i] = np.mean(s[good]**2) if good.any() else np.nan
        tm[i]   = np.mean(t[good])    if good.any() else np.nan
    return tm, posm