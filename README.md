# Gain neuromodulation mediates task-relevant perceptual switches: evidence from pupillometry, fMRI, and RNN Modelling

This repository contains the code necessary to reproduce the data analysis and modelling described in https://elifesciences.org/reviewed-preprints/93191.

# Modelling 

All code and data necessary to reproduce the RNN simulations and analyses can be found in the RNN_Analysis_Modelling folder. 

Code to train the RNNs is supplied in a Jypter notebook. To run this code on your computer ensure that all the necessary dependencies are installed and add the desired paths to lines 317,354,364,355,366, to save the weights. 

MATLAB code necessary to reproduce the simulations and analysis of the RNN is supplied in scripts:

RNN_Simulation_Analysis.m
RNN_Lesion_Analysis.m
RNN_Simulation_Analysis.m

To run these scripts either add the location of the saved weights from the Jypoter notebook to the MATLAB path, or for those wanting to just reproduce the simulations, unzip the file EffectiveWeights.zip and add its location to the MATLAB path. 
