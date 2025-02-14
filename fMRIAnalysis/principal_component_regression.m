%% 3. PCR (Principal Component Regression) Analysis Using Convolved Design Regressors
% In this section, each principal component (PC) obtained from the PCA (stored in pc_load_all)
% is treated as a dependent variable. We use the convolved design regressors (from the behavioral 
% design matrix) to predict the PC time series using GLM. This is known as Principal Component Regression (PCR).

% Analysis parameters
RT = 2;           % Repetition time (seconds)
nPC = 375;        % Number of principal components (each PC is treated as a "node")
regs = 9;         % Number of regressors in the design matrix

% HRF parameters for spm_hrf
pp = [6; 16; 1; 1; 6; 0; 32];  % Parameters for HRF estimation
T = 16;                       % Duration parameter for the HRF
hrf = spm_hrf(RT, pp, T);       % Compute the HRF using SPM's spm_hrf function

% Preallocate arrays to store beta coefficients and their p-values.
% Dimensions: [subjects x nPC x (regs+1)] (the +1 accounts for the intercept term)
numSubjects = 17;
bold_beta_tot = nan(numSubjects, nPC, regs+1);
p_beta_tot    = nan(numSubjects, nPC, regs+1);

% Loop over subjects for PCR fitting.
for p = 1:numSubjects
    % Retrieve the design matrix for subject p.
    % dsx_subj(p,:,:) is expected to be [1375 x 9]
    dsx_now = squeeze(dsx_subj(p, :, :));  % Size: [1375 x regs]
    
    % Convolve each regressor with the HRF.
    % The convolution increases the length by (length(hrf)-1). We will truncate later.
    dsmtx_temp1 = nan(1375 + length(hrf) - 1, regs);
    for y = 1:regs
        dsmtx_temp1(:, y) = conv(hrf, dsx_now(:, y));
    end
    
    % Truncate the convolved design matrix to the original time series length (1375).
    dsmtx_temp2 = dsmtx_temp1(1:1375, :);  % Size: [1375 x regs]
    
    % Initialize arrays for beta coefficients and p-values for the current subject.
    bold_beta = zeros(nPC, regs+1);  % One row per PC, with (regs+1) coefficients
    p_beta    = zeros(nPC, regs+1);
    
    % Loop over principal components (each PC's time series serves as the dependent variable).
    for pc = 1:nPC
        % Extract the time series for the current PC for subject p.
        % pc_load_all is [1375 x numSubjects x nPC]. For subject p, we extract a 1375 x 1 vector.
        pc_ts = squeeze(pc_load_all(:, p, pc));  % [1375 x 1]
        
        % Fit the GLM using the convolved design matrix as predictors.
        % The function glmfit returns the beta coefficients (including the intercept) and stats.
        [beta, ~, stats] = glmfit(dsmtx_temp2, pc_ts, 'normal');
        bold_beta(pc, :) = beta;
        p_beta(pc, :)    = stats.p;
    end
    
    % Store the PCR (GLM) results for subject p.
    bold_beta_tot(p, :, :) = bold_beta;
    p_beta_tot(p, :, :) = p_beta;
end

%% (Optional) Display or Save the PCR Results
% For example, display the explained p-values for the first subject and first PC:
disp('PCR p-values for subject 1, PC 1:');
disp(squeeze(p_beta_tot(1,1,:)));
