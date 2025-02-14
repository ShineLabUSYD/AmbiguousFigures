%% --- Setup Parameters and Preallocate Variables ---

margins = 7500;               % Number of samples before/after event for pupil evoked response
numSubjects = size(Data, 2);  % Total number of subjects in the Data structure

% Preallocate matrices:
samples     = numSubjects;                     % (Not used further in the code, but preallocated as given)
block       = nan(numSubjects, 50000);         % Preallocation for block-related data (if needed later)
pupil_all   = nan(numSubjects, 20, margins*2+1); % To store pupil evoked responses for each subject & block
change_all  = nan(numSubjects, 20);            % To record the change index for each subject & block

%% --- Loop Over Subjects ---
for p = 1:numSubjects
    disp(p);  % Display current subject number
    
    % Extract event information for subject p
    events = Data(p).event;
    
    % Identify 'Block' and 'Trial' events based on the second column of events
    blocks = strcmp(cellstr(events(:,2)), "Block");
    trials = strcmp(cellstr(events(:,2)), "Trial");
    
    % Extract trial start times:
    %   - Column 1: first time value from the trial event.
    %   - Column 2: second time value from the trial event.
    trial_start = [str2double(cellstr(events(trials, 1))), ...
                   str2double(cellstr(events(trials, 4)))];
    
    % Use every second row: assumes trials come in pairs (start and end)
    trial_start = [trial_start(1:2:end, 1), trial_start(2:2:end, 2)];
    
    % Reshape trial start times into a 15x20 matrix (15 trials per block, 20 blocks)
    trial_all = reshape(trial_start(:, 1), [15, 20]);
    
    % Extract block start times (adjusting the second column by +1)
    blocks_start = [str2double(cellstr(events(blocks, 1))), ...
                    str2double(cellstr(events(blocks, 4))) + 1];
    % (Optional: Save blocks_start into another variable if needed)
    % blocks_start_tot(p, 1:20) = blocks;
    
    %% --- Process Behavioral Data for Subject p ---
    
    % Extract response and trial-related data, skipping the first element (often a header)
    response = reshape(Data(p).choice.choice(2:end), [15, 20]);
    trial    = reshape(Data(p).choice.morphSeqRoot(2:end), [15, 20]);
    change   = reshape(Data(p).choice.choice(2:end), [15, 20]);
    
    % Exclude specific trials by setting rows 1-4 and 13-15 to zero in the change matrix
    change([1:4, 13:15], :) = 0;
    
    % Record trial onset times for subject p (first trial in each block)
    trial_tot(p, 1:20) = trial(1, 1:20);
    
    %% --- Compute Pupil Data ---
    
    % Calculate the net pupil signal by removing the slow component from the cleaned signal
    pupil1 = Data(p).pupil_slow;
    pupil2 = Data(p).pupil_clean;
    pupil  = pupil2 - pupil1;
    
    % Load the corresponding time vector for pupil data
    time = Data(p).time_clean;
    
    % Preallocate matrix to store the pupil evoked response for each block of subject p
    pupil_now = nan(20, margins*2+1);
    
    %% --- Loop Over Blocks for Subject p ---
    for i = 1:20
        block_now = i;
        
        % Find the first trial in the block where a change event (coded as 2) occurs
        change_now = find(change(:, i) == 2, 1);
        
        if ~isempty(change_now)
            % Record the change event index for subject p in the change_all matrix
            change_all(p, i) = change_now;
            
            % Find the index in the time vector corresponding to the trial start for the block
            time_trial = find(time > trial_all(1, block_now), 1);
            % (Optional) Extract the pupil data for the entire trial if needed:
            pupil_trial = pupil(time_trial : time_trial + 39000);
            
            % Determine the time index corresponding to the change event within the trial
            time_now = find(time > trial_all(change_now, block_now), 1);
            
            % Extract the pupil data surrounding the event (window defined by margins)
            pupil_evoked = pupil(time_now - margins : time_now + margins);
            
            % Normalize the evoked pupil response:
            %   - Baseline: mean of the first 2500 samples of the evoked response
            %   - Normalize by subtracting the baseline and dividing by the standard deviation
            baseline = nanmean(pupil_evoked(1:2500));
            pupil_std = nanstd(pupil_evoked);
            pupil_now(i, :) = (pupil_evoked - baseline) / pupil_std;
            
            % Alternative approach (e.g., using gradient) can be implemented if needed:
            % pupil_now(i, :) = gradient(pupil_evoked);
        end
    end
    
    % Store the normalized pupil evoked responses for subject p
    pupil_all(p, :, :) = pupil_now;
end
