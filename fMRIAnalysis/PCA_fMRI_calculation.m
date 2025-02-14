%% PCA Analysis on Raw fMRI Data
% This script processes raw fMRI time series for 17 subjects.
% It computes correlation matrices for each subject and then performs PCA on 
% the concatenated raw data across all subjects.
%
% Assumptions:
% - Each subject’s fMRI data is stored in Data_design(p).fmri with dimensions 
%   [375 x 1400] (i.e., 375 regions and 1400 time points).

%% Preallocate Variables
numSubjects = 17;   % Total number of subjects
numRegions  = 375;  % Number of brain regions
numTimePts  = 1400; % Number of time points per subject

% Initialize fm_tot to accumulate raw data from all subjects.
% It will have dimensions: [numRegions x (numSubjects*numTimePts)]
fm_tot = [];

% Preallocate a 3D array to store the correlation matrix for each subject.
% Dimensions: [numSubjects x numRegions x numRegions]

%% Process Each Subject
for p = 1:numSubjects
    % Display current subject index for tracking progress.
    disp(['Processing subject ' num2str(p)]);
    
    % Retrieve the raw fMRI data for subject p.
    % Expected dimensions: [numRegions x numTimePts]
    fm_data = Data_design(p).fmri;
    
    % Concatenate the raw data across subjects.
    % fm_tot will have dimensions: [numRegions x (numSubjects*numTimePts)]
    fm_tot = [fm_tot, fm_data];
end

%% Perform PCA on the Concatenated Raw Data
% Transpose fm_tot so that each row is a time point and each column is a region.
% The PCA input will have dimensions: [(numSubjects*numTimePts) x numRegions]
[pca_load, pc_value, ~, ~, exp] = pca(fm_tot');
