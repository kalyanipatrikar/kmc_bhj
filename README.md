# kmc_bhj
Monte Carlo based multiscale model to obtain device properties of bulk heterojunction organic photovoltaic based on molecular and interface properties computed by ab-initio methods.
Procedure is described in 'From Molecules to Devices: A Multiscale Approach to Evaluating Organic Photovoltaics' (https://pubs.acs.org/doi/10.1021/acs.jctc.4c01029)

Sample code here calculates the mobility of non-fullerene acceptor known as AQx2. 

File 'AQx2.txt' contains charge transfer integral for pairs AQx2 molecules with different conformations. These are randomly sampled to constitute the grid. 

Running 'random_walk.py' simulates trajectories of electron diffusing in AQx2 cells with molecular conformations scattered throughout. At the end, based on functions from 'analysis.py', the displacement and time of trajectories is stored as 'msd.npz'.

Finally, 'calculate_mobility.py' uses 'msd.npz' to calculate mobility based on diffusivity obtained as <x^2>/6t.

As random_walk.py requires many steps and trials to be statistically meaningful, it is recomended to run it on a cluster. Note that it imports both AQx2.txt and analysis.py, these files have to be kept in the same directory.

** codes calculating J_SC and v_OC have been taken down for maintainance **
