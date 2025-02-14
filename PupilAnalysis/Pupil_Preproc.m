
function [Diam2,Time,g] = Pupil_Preproc(Time,Diam,filt,freq)        
%% Linear Interpolation
if nargin==2
    freq2=30;
else
    freq2=freq/2;
end

    Diam=Diam(~isnan(Time),:);
    Time=Time(~isnan(Time));

    Diam2=nan(size(Diam));
        
  
        %% Interpolation
        g=~isnan(Diam);
        if sum(g)==0
            Diam2=Diam;
            
        else
            
            % DiamInt=interp1(Time(g),Diam(g),Time);
        
            DiamInt = spline(Time(g),Diam(g),Time); 
        %% Butterworth Filter
        
        if length(filt)==2
            vect=[filt(1)/freq2, filt(2)/freq2];
          
            [b,a] = butter(2,vect,'bandpass');

        else
             [b,a] = butter(2,filt/freq2,'low');
        end
        
        dataIn = DiamInt(~isnan(DiamInt));
        Diam2(~isnan(DiamInt),1) = filtfilt(b,a,dataIn);
        end
    
    %% Replace Diam with the interpolated and filtered Diameter data.

end
