# iSAT_Dong_2022
Software tools used in 'Near-Physiological in vitro Assembly of 50S Ribosomes Involves Parallel Pathways' : Xiyu Dong, Kai Sheng, James R. Williamson, The Scripps Research Institute

This project is licensed under the terms of GNU Affero General Public License version 3 (GNU AGPLv3). 

Tools included in this repository include:

*occupancy_matrix **A set of scripts used to calculate protein and rRNA helix occupancy in density maps, consisting of the following:

Occ_Analysis.py - a python script used to calculate volumes occupied as a function of the threshold chose and draw the matrix of occupancy values.

Occ_Binarization.py - a python script used to preliminarily binarize the occupancy values calculated from Occ_Analysis.py

Projection.py - a python script used to further determine the binarization threshold of some tricky structural elements.

pathway_maps_with_pabel/ - mrc files of aligned EM density maps of iSAT LSU intermediate structures used for occupancy analysis.

*ElementCorrelation&Dependency_Analysis **A tool used to analyze the correlation between structural elements
