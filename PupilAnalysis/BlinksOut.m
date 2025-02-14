%% Function to eliminate blinks of a pupil diameter timeseries. 
% It finds NANS or zeros, and extract an extract window around it (marg). To avoid eliminating data from other artifacts (that don't need a marg), BLength is used
% as criteria (apply the marg only if the length of the blink is > than BLength. 
% This function can be used to any timeseries in which we need to eliminate a window around missing data. 

function [Diam, BlinkInd] = BlinksOut(Diam,marg,BLength,freq)
    
    Blink=[];
    Blink2=[];
    Blink3=[];
    BlinkInd=[];                                   % Logic vector. 1 when there is no data (Blink)         
    Blink=Diam==0 | isnan(Diam); 
    
    %% Search for Blink onset and offset. The store it in a Matrix of 2D.
    % The first and second column of BlinkInd is the onset and offset
    % detected. The third column is the length of the Blink (criteria of blink
    % exclusion). 
    if Blink(1)==1
        Blink(1)=0;
    end
    Blink2=diff(Blink);                         % In case of beeing (>1) => onset || offset
    if Blink(end)==1
        if Blink(end-1)==1                      % look if the last data of the block is a 0.
            Blink2(end)=-1;                     % Replace the last '0' for a -1 (offset)
        else
            Blink2(end)=nan;
        end
    end
    
    BlinkInd=[find(Blink2==1)+1,find(Blink2==-1)];   
    BlinkInd(:,3)=BlinkInd(:,2)-BlinkInd(:,1);
    %% Supress Blinks shorter than Blengh (input). Default = 50;
    % Correction for fake blinks. In case of artifacts, not losing 50 ms of
    % that periods. 
    frame2=round(BLength/(1000/freq));
    BlinkInd(BlinkInd(:,3)<frame2,1:2)=[BlinkInd(BlinkInd(:,3)<frame2,1)-2,BlinkInd(BlinkInd(:,3)<frame2,2)+2];
    BlinkInd(BlinkInd(:,3)<frame2,3)=0;
    %% Eliminate the time margin stated in the inputs. Default = 100ms.
    % To correspond the frames with the time margin, I looked at how many
    % data points, corresponds to the margin selected (in ms). 
    
    frame=round(marg/(1000/freq));
    BlinkInd(BlinkInd(:,3)>=frame2,1:2)=[BlinkInd(BlinkInd(:,3)>=frame2,1)-frame,BlinkInd(BlinkInd(:,3)>=frame2,2)+frame]; % Indexes of onset/offset of blink, corrected with "frame"
    BlinkInd(BlinkInd(:,3)>=frame2,3)=1;
    if ~isempty(BlinkInd)
    
        if BlinkInd(1,1)<0
            BlinkInd(1,1)=1;
        end
        if find(BlinkInd(:,2)>length(Diam))
            
            BlinkInd(BlinkInd(:,2)>length(Diam),2)=length(Diam);
    
        end
        
        %% Eliminate the Blink data from the pupil diameter data
        
        BlinkInd(BlinkInd(:,2)>length(Diam),2)=length(Diam);    
        BlinkInd(BlinkInd(:,1)<1,:)=1;
        BlinkInd2=BlinkInd;
        
        % Initialize the list of merged intervals
        merged_intervals = [];
        n = size(BlinkInd2, 1);
        i = 1;
        eliminated = false(n, 1);

        while i <= n
            % Start with the current interval
            current_start = BlinkInd2(i, 1);
            current_end = BlinkInd2(i, 2);
            
            % Compare with subsequent intervals to check for overlaps
            j = i + 1;
            while j <= n && BlinkInd2(j, 1) <= current_end
                % Update the end of the current interval if needed
                current_end = max(current_end, BlinkInd2(j, 2));
                eliminated(j,1)=true;
                j = j + 1;
            end
            
            % Append the merged interval
            merged_intervals = [merged_intervals; current_start, current_end];
            
            % Move to the next interval
            i = j;
        end
        
        merged_intervals(:,3)=BlinkInd(~eliminated,3);
        BlinkInd=merged_intervals;
        for i=1:size(BlinkInd,1)
                
            Diam(BlinkInd(i,1):BlinkInd(i,2),:)=nan;
        end
    end


end








