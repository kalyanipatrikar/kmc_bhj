Run the notebook get_pair_structure to get coordinates of a pair of molecules of AQx-2 based on the crystal structure information in 2161927.cif
Coordinates of each monomer and dimer are generated for H and J stacking, saved as .xyz files.

Next run notebook generate_configs to get different conformations for the pairs using MACE-MD. Reads dimer coordinates, and generates 40 configurations for each stacking.More detials of methods and variables are in the cells. 
Coordinates for dimer and monomers in each configurations are stored in directory aqx2_H_backbone_configs along with energy values. 

Finally run notebook transfer_integrals, which randomly samples 10 configurations and calculates the transfer integral for them, written to a text file. These values can be input to the algorithm calculating mobility. 
