%% Pupil Bistable Task
% This script imports and preprocesses data for the pupil bistable task.
% It reads ASC files, extracts event and pupil data, performs preprocessing
% (including blink removal and drift outlier elimination), and splits the
% pupil data by block.
%
% IMPORTANT: Replace 'your_path_here' with the actual path to your data files.
% Custom functions required: BlinksOut, Pupil_Preproc.

%% Setup and File Import
Data = [];  % Initialize structure to store subject data

% Change directory to the folder containing ASC files
cd('your_path_here');  % <-- Replace with your actual directory path
list = dir('*.asc');   % List all ASC files in the directory

%% Loop Over Subjects
% For each subject file, import and preprocess the data.
for sub = 1:35
    %% Import Data from ASC File
    % Configure import options for the ASC file
    opts = delimitedTextImportOptions("NumVariables", 5);
    opts.DataLines = [1, Inf];   % Read from the first to the last line
    opts.Delimiter = "\t";       % Tab-delimited file

    % Define variable names and types; we only need four variables.
    opts.VariableNames = ["time", "x_axis", "y_axis", "pupil", "Var5"];
    opts.SelectedVariableNames = ["time", "x_axis", "y_axis", "pupil"];
    opts.VariableTypes = repmat("string", 1, 5);

    % File-level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";

    % Variable-specific options
    opts = setvaropts(opts, "Var5", "WhitespaceRule", "preserve");
    opts = setvaropts(opts, "Var5", "EmptyFieldRule", "auto");

    % Read the data table for the current subject
    subj = readtable(list(sub).name, opts);
    disp(['Processing subject: ' num2str(sub)]);

    %% Extract and Process Event Data
    % Identify event messages. It is assumed that rows where 'time' equals 'MSG'
    % contain event information in the 'x_axis' field.
    eventMsgs = subj.x_axis(strcmp(subj.time, 'MSG'));

    % Split each event message into tokens. The tokens are stored in a cell array.
    events3 = cell(length(eventMsgs), 10); % Preallocate a cell array with an arbitrary number of columns (adjust as needed)
    for i = 1:length(eventMsgs)
        tokens = split(eventMsgs(i)); % Split event message by whitespace
        for r = 1:length(tokens)
            events3{i, r} = tokens{r};  % Save each token into events3
        end
    end

    %% Extract Block and Trial Information from Events
    freq = 1000;  % Sampling frequency (Hz)

    % --- Block Information ---
    % Blocks are identified by the token 'Block' in the second column.
    % Column 4 (converted to number) contains one block parameter, and column 1
    % contains another parameter (e.g., onset time).
    block = [];
    block(:, 1) = str2double(cellstr(events3(strcmp(cellstr(events3(:, 2)), 'Block'), 4)));
    block(:, 2) = str2double(cellstr(events3(strcmp(cellstr(events3(:, 2)), 'Block'), 1)));

    % --- Trial Information ---
    % Trials are identified by the token 'Trial' in the second column.
    trial = [];
    trial(:, 1) = str2double(cellstr(events3(strcmp(cellstr(events3(:, 2)), 'Trial'), 4)));
    trial(:, 2) = str2double(cellstr(events3(strcmp(cellstr(events3(:, 2)), 'Trial'), 1)));
    
    % Adjust trial rows: it is assumed that trials are recorded in pairs.
    trial(1:2:end, 1) = trial(2:2:end, 1);
    trial(1:2:end, :) = [];
    
  
    
    %% Pupil Preprocessing
    % Convert pupil and time strings to numbers
    pupil = str2double(subj.pupil);
    time = str2double(subj.time);
    
    % Store raw data in the Data structure
    Data(sub).id = list(sub).name;
    Data(sub).pupil = pupil;
    Data(sub).time = time;
    Data(sub).event = events3;

    % Remove blinks using a custom function
    pupil2 = BlinksOut(pupil, 150, 25, freq);
    [pupil3, time2] = Pupil_Preproc(time, pupil2, [3], freq);
    
    %% Drift Outlier Removal
    % Identify and eliminate outliers due to drift
    threshold = 2.5;
    dif_pupil1 = diff(pupil3);
    dif_pupil2 = diff(pupil3, 2);
    
    thr_dif2 = nanstd(dif_pupil2) * threshold;
    thr_dif = nanstd(dif_pupil1) * threshold;
    
    % Find indices where the difference exceeds the threshold
    find_out = dif_pupil2 < -thr_dif2 | dif_pupil2 > thr_dif2 | ...
               dif_pupil1(1:end-1) < -thr_dif | dif_pupil1(1:end-1) > thr_dif;
    
    % Remove these outliers from the raw data before further processing
    [~, b] = intersect(time, time2(find_out));
    pupil_second = pupil;
    pupil_second(b) = 0;  % Set outlier values to 0 (will be removed by BlinksOut)
    
    % Apply blink removal and additional preprocessing on the corrected data
    pupil4 = BlinksOut(pupil_second, 100, 25, freq);
    [pupil5, time3] = Pupil_Preproc(time, pupil4, [3], freq);
    [pupil6, ~] = Pupil_Preproc(time, pupil4, [0.025, 3], freq);
    
    % Compute the slow component of the pupil signal
    pupil_slow = pupil5 - pupil6;
    pupil_clean = pupil5;
    
    % Clear temporary variables to free memory and avoid confusion
    clear pupil5 time time2 pupil4 pupil3 pupil2 pupil_second b find_out thr_dif2 thr_dif dif_pupil1 dif_pupil2
    
    % Save preprocessed pupil signals and time in the Data structure
    Data(sub).pupil_slow = pupil_slow;
    Data(sub).pupil_fast = pupil6;
    Data(sub).pupil_clean = pupil_clean;
    Data(sub).time_clean = time3;
    
    %% Split Pupil Data by Block
    % Preallocate a matrix for block-split pupil data (each row for one block)
    pupil_block = nan(size(block, 1), 50000);
    
    % Determine block boundaries based on trial information:
    %   - It is assumed that when trial(:,1)==2 and trial(:,1)==14 denote the
    %     start and end of a block, respectively.
    place_bl_all = [trial(trial(:, 1) == 2, 2), trial(trial(:, 1) == 14, 2)];
    
    % For each block, extract the pupil data and normalize it (z-score normalization)
    for i = 1:size(block, 1)
        % Find the indices corresponding to the start and end of the block
        startIdx = find(time3 == place_bl_all(i, 1));
        endIdx   = find(time3 == place_bl_all(i, 2) + 2000);
        
        if ~isempty(startIdx) && ~isempty(endIdx)
            % Extract and normalize the pupil data for the current block
            pupil_block(i, 1:(endIdx - startIdx + 1)) = ...
                zscore(pupil_clean(startIdx:endIdx));
        end
    end
    
    % Save the block-split pupil data in the Data structure
    Data(sub).pupil_block = pupil_block;
end
