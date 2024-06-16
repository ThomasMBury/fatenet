# FateNet

This is the repository to accompnay the article:

***FateNet: an integration of dynamical systems and deep learning for cell fate prediction***. *Mehrshad Sadria and Thomas M. Bury* 
https://www.biorxiv.org/content/10.1101/2024.01.16.575913v1

## Directories

- ```/code/training_data```: Generate training data for the neural network classifiers
- ```/code/dl_train```: Train the neural network classifiers
- ```/code/simple_gene_model```: Evaluate FateNet on simulation of simple gene network
- ```/code/sergio```: Evaluate FateNet on SERGIO simulation
- ```/code/pancreas```: Evaluate FateNet on pancreas data
- ```/code/weinreb_klein```: Evaluate FateNet on hematopoiesis data

## Data availability

The simulated scRNA-seq data from SERGIO can be found at: https://github.com/PayamDiba/SERGIO.

The hematopoiesis data from Weinreb et al. can be downloaded from GEO under accession number GSE140802.

The simulated scRNA-seq data generated by the simple gene regulatory network is saved in ```/code/simple_gene_model/output```.