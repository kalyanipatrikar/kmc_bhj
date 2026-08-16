# kmc_bhj
Monte Carlo based multiscale model to obtain device properties of bulk heterojunction organic photovoltaic based on molecular and interface properties computed from first principles. 
Procedure is described in 'From Molecules to Devices: A Multiscale Approach to Evaluating Organic Photovoltaics' (https://pubs.acs.org/doi/10.1021/acs.jctc.4c01029)

Sample code here calculates the mobility of non-fullerene acceptor commonly known as AQx2.
 
Directory 'Configurations' contains method to get the charge transfer integral of AQx-2 for different molecular configurations. Directory 'Mobility' has method to calculate mobility based on these charge transfer integral values of different configurations. The values calculated for AQx2 based on methods involving Gaussian16 (DFT) and CREST (xTB) are already included in the file 'AQx2.txt', so that the mobility method can be run independently.


File 'AQx2.txt' contains charge transfer integral for pairs AQx2 molecules with different conformations. These are randomly sampled to constitute the grid. 

Running 'random_walk.py' simulates trajectories of electron diffusing in AQx2 cells with molecular conformations scattered throughout. It generates files 'x.npy' and 'time.npy'.

Then running 'calculate_mobility.py' uses 'x.npy' and 'time.npy' to calculate mobility based on diffusivity obtained as <x^2>/6t.

As random_walk.py requires many steps and trials to be statistically meaningful, it is recomended to run it on a cluster. Note that it imports AQx2.txt, these files have to be kept in the same directory.

** codes calculating J_SC and v_OC have been taken down for maintainance **
