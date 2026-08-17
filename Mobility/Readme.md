
File 'AQx2.txt' contains charge transfer integral for pairs AQx2 molecules with different conformations. These are randomly sampled to constitute the grid. 

Running 'random_walk.py' simulates trajectories of electron diffusing in AQx2 cells with molecular conformations scattered throughout. It generates files 'x.npy' and 'time.npy'.

Then running 'calculate_mobility.py' uses 'x.npy' and 'time.npy' to calculate mobility based on diffusivity obtained as <x^2>/6t.

As random_walk.py requires many steps and trials to be statistically meaningful, it is recomended to run it on a cluster. Note that it imports AQx2.txt, these files have to be kept in the same directory.
