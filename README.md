# Gain neuromodulation mediates task-relevant perceptual switches: evidence from pupillometry, fMRI, and RNN Modelling

This repository contains the code necessary to reproduce the data analysis and modelling described in https://elifesciences.org/reviewed-preprints/93191.

# Modelling 

All code and data necessary to reproduce the RNN simulations and analyses can be found in the [RNN_Modelling&Analysis](https://github.com/cjwhyte/AmbiguousFigures/tree/patch-1/RNN_Modelling%26Analysis) folder. 

Code to train the RNNs is supplied in a [Jypter notebook](https://github.com/cjwhyte/AmbiguousFigures/blob/patch-1/RNN_Modelling%26Analysis/RNNTraining.ipynb). To run this code on your computer ensure that all the necessary dependencies are installed and add the desired paths to lines 317, 354, 364, 355, and 366, to save the weights. 

MATLAB code necessary to reproduce the simulations and analysis of the RNN is supplied in scripts:

[RNN_Simulation_Analysis.m](https://github.com/cjwhyte/AmbiguousFigures/blob/patch-1/RNN_Modelling%26Analysis/RNN_Simulation_Analysis.m), 
[RNN_Lesion_Analysis.m](https://github.com/cjwhyte/AmbiguousFigures/blob/patch-1/RNN_Modelling%26Analysis/RNN_Lesion_Analysis.m), &
[RNN_Simulation_Analysis.m](https://github.com/cjwhyte/AmbiguousFigures/blob/patch-1/RNN_Modelling%26Analysis/RNN_Attractor_Analysis.m)

To run these scripts either add the location of the saved weights from the Jypoter notebook to the MATLAB path, or for those wanting to just reproduce the simulations, unzip the file [EffectiveWeights.zip](https://github.com/cjwhyte/AmbiguousFigures/blob/patch-1/RNN_Modelling%26Analysis/EffectiveWeights.zip) and add its location to the MATLAB path. 

Tutorial style code implementing the allocentric and egocentric landscape analyses on a simple dynamical system (normal form of a super critical pitchfork bifurcation) is supplied in the script:

[EnergyLandscape_Tutorial.m ](https://github.com/cjwhyte/AmbiguousFigures/blob/patch-1/RNN_Modelling%26Analysis/EnergyLandscape_Tutorial.m)

---

# Pupil Analysis

The pupil analysis pipeline comprises MATLAB scripts and functions that preprocess and analyze pupillometry data.

- **[pupil_preprocessing.m](https://github.com/ShineLabUSYD/AmbiguousFigures/tree/main/PupilAnalysis/pupil_preprocessing.m)**  
  *Script to import and preprocess pupil data from ASC files. It extracts event and trial information, applies blink correction, removes drift artifacts, and segments the pupil signal into blocks.*

- **[Pupil_Preproc.m](https://github.com/ShineLabUSYD/AmbiguousFigures/tree/main/PupilAnalysis/Pupil_Preproc.m)**  
  *Function for further filtering and drift correction of pupil data, used in conjunction with BlinksOut.m.*

- **[BlinksOut.m](https://github.com/ShineLabUSYD/AmbiguousFigures/tree/main/PupilAnalysis/BlinksOut.m)**  
  *Function to detect and remove blink artifacts from raw pupil data.*

---

# fMRI Analysis

The fMRI analysis pipeline includes MATLAB scripts that process BOLD fMRI data, perform GLM analysis, compute PCA, and implement principal component regression (PCR).

- **[bold_glm_analysis.m](https://github.com/ShineLabUSYD/AmbiguousFigures/tree/main/fMRIAnalysis/bold_glm_analysis.m)**  
  *Script to perform GLM analysis on fMRI BOLD data using a behavioral design matrix. It includes HRF convolution of design regressors and outputs beta coefficients and p-values.*

- **[PCA_fMRI_calculation.m](https://github.com/ShineLabUSYD/AmbiguousFigures/tree/main/fMRIAnalysis/PCA_fMRI_calculation.m)**  
  *Script to perform Principal Component Analysis (PCA) on fMRI data, extracting the dominant components of neural activity.*

- **[principal_component_regression.m](https://github.com/ShineLabUSYD/AmbiguousFigures/tree/main/fMRIAnalysis/principal_component_regression.m)**  
  *Script to implement Principal Component Regression (PCR) by regressing each principal component’s time series on HRF-convolved design regressors. Each PC is treated as the dependent variable.*

