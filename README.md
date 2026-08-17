# kmc_bhj
Monte Carlo based multiscale model to obtain device properties of bulk heterojunction organic photovoltaic based on molecular and interface properties computed from first principles. 
Procedure is described in 'From Molecules to Devices: A Multiscale Approach to Evaluating Organic Photovoltaics' (https://pubs.acs.org/doi/10.1021/acs.jctc.4c01029)

Sample code here calculates the mobility of non-fullerene acceptor commonly known as AQx2.
 
Directory 'Configurations' contains method to get the charge transfer integral of AQx-2 for different molecular configurations. Directory 'Mobility' has method to calculate mobility based on these charge transfer integral values of different configurations. The values calculated for AQx2 based on methods involving Gaussian16 (DFT) and CREST (xTB) are already included in the file 'AQx2.txt', so that the mobility method can be run independently.

** codes calculating J_SC and v_OC have been taken down for maintainance **
