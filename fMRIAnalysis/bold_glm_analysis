%% Setup and Data Loading
% Replace the file path with your actual file location.
% Example: load('C:\Users\YourName\Documents\Data_design.mat')
load('your_path_here/Data_design.mat')

%% Parameter Setup
opt   = 1;       % Option flag: if 1, use fmri data; else, use fmri_band
subj  = 17;      % Number of subjects
nodes = 375;     % Number of nodes in the fMRI data
runs  = 5;       % Number of runs per subject
trls  = 56;      % Number of trials (or time segments for design matrix reshaping)
ttime = 280;     % Total number of time points per run
delta = 3;       % Delta value for selecting pre and post event windows
nreg  = delta*2 + 3;  % Total number of regressors (pre + post windows plus 3 additional regressors)

% Preallocate output arrays for GLM beta coefficients:
bold_beta_tot      = nan(subj, runs, nodes, nreg);
bold_beta_diff_tot = nan(subj, runs, nodes, nreg);

%% Regressor (HRF) Setup
% Define parameters for the Hemodynamic Response Function (HRF)
RT = 2;                % Repetition time in seconds
pp = [6; 16; 1; 1; 6; 0; 32];  % HRF parameters (specific to your model)
T  = 16;               % Duration parameter for the HRF
hrf = spm_hrf(RT, pp, T);  % Generate the HRF using SPM's spm_hrf function

%% Loop Over Subjects
for p = 1:subj
    % Initialize the design matrix for each session
    % Dimensions: [runs x ttime x nreg]
    design_session = zeros(runs, ttime, nreg);
    
    tic;  % Start timing processing for the current subject
    disp(['Processing subject ' num2str(p)]);
    
    %% Load and Reshape BOLD fMRI Data
    % Select fmri or fmri_band data based on the opt flag.
    if opt == 1
        Bold_fmri = reshape(Data_design(p).fmri, [nodes, ttime, runs]);
    else
        Bold_fmri = reshape(Data_design(p).fmri_band, [nodes, ttime, runs]);
    end
    
    % Load additional design matrices for the subject
    mat = Data_design(p).mat;
    dsg = Data_design(p).dgn_fix;
    
    % Process 'mat' using the design indicator 'dsg':
    % Remove rows where the sum is zero and reshape into a 3D matrix:
    % Dimensions: [images x trials x runs] (15 x 5 x 5)
    dsg2 = sum(dsg, 2);
    mat2 = mat;
    mat2(~dsg2) = [];
    mat2 = reshape(mat2, [15, 5, 5]);
    
    % (The following change variables are declared but not used further.)
    change    = zeros(trls, 25);
    change_p1 = zeros(trls, 25);
    change_p2 = zeros(trls, 25);
    change_m1 = zeros(trls, 25);
    change_m2 = zeros(trls, 25);
    
    % Reshape and permute 'dsg' to obtain a 4D design matrix.
    % Final dimensions: [trls x image x trial x run]
    dsg = permute(reshape(dsg, [trls, 5, 5, 15]), [1, 4, 2, 3]);
    
    % Initialize design matrices for different regressors:
    % 'design' will store the main regressor set extracted around a change event.
    % The additional matrices capture sums of early fixations, late fixations, etc.
    design     = zeros(runs, ttime, 1 + delta*2);  % Main design regressor matrix
    designfirst= zeros(runs, ttime);                % Sum of early events (columns 1:2)
    designlast = zeros(runs, ttime);                % Sum of late events (columns 14:15)
    designall  = zeros(runs, ttime);                % Sum of all events in a trial
    designfix  = zeros(runs, ttime);                % Indicator for fixation events
    
    %% Build Design Matrices for Each Run
    % Loop over runs and trials to accumulate regressor values.
    for run = 1:runs
        % Temporary arrays for accumulating regressors for current run
        des  = [];
        des1 = [];
        des2 = [];
        des3 = [];
        des4 = [];
        
        for tr = 1:5  % Loop over trial (or image/trial combination)
            % Extract the design data for the current trial and run.
            % dsgtr: [trls x 5] matrix for the current trial/run.
            dsgtr = squeeze(dsg(:, :, tr, run));
            % mattr: Vector of length 15 representing image labels.
            mattr = squeeze(mat2(:, tr, run));
            
            % Look for a specific change event (image labeled 25)
            ch = find(mattr == 25, 1);
            
            if ~isempty(ch)
                % Check if the change event is not too close to boundaries
                if ch > delta + 1 && ch < 15 - delta
                    % Extract a window around the change event:
                    des = [des; dsgtr(:, ch-delta:ch+delta)];
                else
                    des = [des; zeros(trls, delta*2+1)];
                end
            else
                des = [des; zeros(trls, delta*2+1)];
            end
            
            % Accumulate additional regressors:
            des1 = [des1; sum(dsgtr(:, 1:2), 2)];  % Early events
            des2 = [des2; sum(dsgtr(:, 14:15), 2)];  % Late events
            des3 = [des3; sum(dsgtr, 2)];            % Sum of all events
            des4 = [des4; sum(dsgtr, 2) ~= 1];         % Fixation indicator (logical)
        end
        
        % Store the accumulated regressors for the current run
        design(run, :, :)  = des;
        designfirst(run, :) = des1;
        designlast(run, :)  = des2;
        designall(run, :)   = des3;
        designfix(run, :)   = des4;
    end
    
    %% Convolve Design Regressors with HRF & Prepare GLM Inputs
    for run = 1:runs
        % Extract the design matrix for the current run (ttime x regressors)
        design_run = squeeze(design(run, :, :));
        designfirst_r = designfirst(run, :);
        designlast_r  = designlast(run, :);
        designall_r   = designall(run, :);
        designfix_r   = designfix(run, :);
        
        % Incorporate additional regressors into the main design matrix.
        % Here, we set specific columns (e.g., columns 8 and 9 when delta==3)
        design_run(:, 2+delta*2 : 3+delta*2) = [designfirst_r', designlast_r'];
        
        % Convolve each regressor with the HRF.
        % The convolution extends the time dimension by (length(hrf)-1) samples.
        dsmtx_temp1 = nan(ttime + length(hrf) - 1, size(design_run, 2));
        for y = 1:size(design_run, 2)
            dsmtx_temp1(:, y) = conv(hrf, design_run(:, y));
        end
        
        % Truncate the convolved design matrix to match the original ttime.
        dsmtx_temp2 = dsmtx_temp1(1:ttime, :);
        
        % Save the design matrix for the current run into the session matrix.
        design_session(run, :, :) = design_run;
        
        % Extract the BOLD time series for the current run.
        % ts2: [nodes x ttime] matrix.
        ts2 = squeeze(Bold_fmri(:, :, run));
        % Compute the temporal derivative of the BOLD time series.
        ts3 = diff(ts2, 1, 2);
        
        %% Run the GLM for the Current Run
        % Preallocate arrays for GLM beta coefficients.
        bold_beta      = zeros(nodes, nreg + 1);   % Coefficients for the original time series
        bold_beta_diff = zeros(nodes, nreg + 1);   % Coefficients for the time-derivative
        
        for j = 1:nodes
            % Fit the GLM for the derivative time series.
            bold_beta_diff(j, :) = glmfit(diff(dsmtx_temp2), ts3(j, :)', 'normal');
            % Fit the GLM for the original BOLD time series.
            bold_beta(j, :) = glmfit(dsmtx_temp2, ts2(j, :)', 'normal');
        end
        
        % Store GLM beta coefficients (excluding the intercept term).
        bold_beta_tot(p, run, :, :) = bold_beta(:, 2:end);
        bold_beta_diff_tot(p, run, :, :) = bold_beta_diff(:, 2:end);
    end
    
    toc;  % End timing for subject p
    
    % Save the design matrix for the current subject.
    Data_design(p).dsign_matrix = design_session;
end

%% Final Processing: Reorder and Compute Mean Beta Coefficients
% Create an ordering vector 'ord' to rearrange the regressors.
% Here, the second-to-last regressor is placed first, followed by the others,
% and the last regressor is moved to the end.
ord = [size(bold_beta_tot, 4)-1, 1:size(bold_beta_tot, 4)-2, size(bold_beta_tot, 4)];

% Compute the mean beta coefficients across subjects and runs for the reordered regressors.
bold_mean = squeeze(nanmean(nanmean(bold_beta_tot(:, :, :, ord), 1), 2));

% Save the beta coefficients to different variables based on the 'opt' flag.
if opt == 1
    bold_high = bold_beta_tot(:, :, :, ord);
else
    bold_pass = bold_beta_tot(:, :, :, ord);
end
