%% Energy Landscape Tutorial Code (based on code written by B.R. Munn)
% https://elifesciences.org/reviewed-preprints/93191
% Christopher J. Whyte 2/12/24

set(0,'defaulttextinterpreter','latex')
rng('shuffle')
close all
clear

%% Params

% (pitchfork) bifurcation params
alpha1 = .25; alpha2 = .75;

% simulation params
DT = .1;    
T = 10000;  
simulation_length = length(0:DT:T);
num_ints = 200;

% standard deviation of noise process
sigma = .25;

% initial conditions / storage containers
x1 = zeros(num_ints,simulation_length); 
x2 = zeros(num_ints,simulation_length);

%% simulate pitch fork time series 

for int_idx = 1:num_ints
    for t = 1:simulation_length-1 
        x1(int_idx,t+1) = x1(int_idx,t) + DT*(alpha1*x1(int_idx,t) - x1(int_idx,t)^3) + sigma*sqrt(DT)*randn;
        x2(int_idx,t+1) = x2(int_idx,t) + DT*(alpha2*x2(int_idx,t) - x2(int_idx,t)^3) + sigma*sqrt(DT)*randn;
    end 
end 

x1 = downsample(x1,1/DT);
x2 = downsample(x2,1/DT);

%% (analytic) potentials

xline = -2:.01:2;
v1 = -(1/2)*alpha1*xline.^2 + (1/4)*xline.^4;
v2 = -(1/2)*alpha2*xline.^2 + (1/4)*xline.^4;

%% allocentric landscape

allo_nrg_shallow = allo_nrg_calc(x1,xline);
allo_nrg_deep = allo_nrg_calc(x2,xline);

%% egocentric landscape

dt_interval = 10:10:500; % look ahead 500 ms in steps of 10
ds = 0:.05:1;

ego_nrg_shallow = ego_nrg_calc(x1,ds,dt_interval);
ego_nrg_deep = ego_nrg_calc(x2,ds,dt_interval);


%% figures

x = dt_interval;
y = ds;
[X,Y] = meshgrid(x,y);

time = 0:DT:T;

figure(1)
subplot(5,2,1);
plot(time,x1(1,:),'r', 'linewidth',1)
xlabel('Time (ms)')
ylabel('x(t)')
ax = gca;
ax.FontSize = 15;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');

subplot(5,2,2);
plot(time,x2(1,:),'b', 'linewidth',1)
xlabel('Time (ms)')
ylabel('x(t)')
ax = gca;
ax.FontSize = 15;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');

subplot(5,2,3);
plot(xline,v1,'r', 'linewidth',1)
ylim([-.2,.2])
xlim([-2,2])
hline = refline(0, 0);
hline.Color = 'k';
xlabel('x')
ylabel('V(x)')
ax = gca;
ax.FontSize = 15;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');

subplot(5,2,4);
plot(xline,v2,'b', 'linewidth',1)
ylim([-.2,.2])
xlim([-2,2])
hline = refline(0, 0);
hline.Color = 'k';
xlabel('x')
ylabel('V(x)')
ax = gca;
ax.FontSize = 15;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');

subplot(5,2,5);
plot(xline,allo_nrg_shallow,'r', 'linewidth',1)
ylim([-1,5])
xlim([-2,2])
xlabel('x')
ylabel('E(x)')
ax = gca;
ax.FontSize = 15;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');

subplot(5,2,6);
plot(xline,allo_nrg_deep,'b', 'linewidth',1)
ylim([-1,5])
xlim([-2,2])

xlabel('x')
ylabel('E(x)')
ax = gca;
ax.FontSize = 15;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');

subplot(5,2,7);
mesh(X,Y,ego_nrg_shallow','edgecolor', 'r')
xlabel('tau')
ylabel('MSD')
zlabel('E(MSD)')
zlim([-2,30])
ax = gca;
ax.FontSize = 15;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');

subplot(5,2,8);
mesh(X,Y,ego_nrg_deep','edgecolor', 'b')
xlabel('tau')
ylabel('MSD')
zlabel('E(MSD)')
zlim([-2,30])
ax = gca;
ax.FontSize = 15;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');

subplot(5,2,9);
mesh(X,Y,ego_nrg_shallow'-ego_nrg_deep','edgecolor','k')
xlabel('Tau')
ylabel('MSD')
zlabel('$\Delta$E(MSD)')
ax = gca;
ax.FontSize = 15;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');

subplot(5,2,10); hold on
plot(ds,sum(ego_nrg_shallow,1),'r', 'LineWidth',2)
plot(ds,sum(ego_nrg_deep,1),'b', 'LineWidth',2)
xlabel('MSD')
ylabel('Energy AUC')
ax = gca;
ax.FontSize = 15;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');


%% functions 

% allocentric energy landscape 
function nrg_data = allo_nrg_calc(data,ds_allo)
          pd = fitdist(data(:),'Kernel','Kernel','normal','bandwidth',.1);
          pdfEstimate = pdf(pd,ds_allo);
          nrg_data = -1.*log(pdfEstimate);
          % figure; hold on
          % histogram(data(:),Normalization="pdf")
          % plot(ds_allo,pdfEstimate)
end 

% egocentric energy landscape 
function [nrg_data, MSD] = ego_nrg_calc(data,ds,dt_interval)
    nrg_data = nan(length(dt_interval),numel(ds));
    for dt_idx = 1:length(dt_interval)
        dt = dt_interval(dt_idx);
        % Mean-Square Displacement calculation 
        % with moving window assuring same number of samples across dt's
        Disp = data(:,1:(max(dt_interval))) - data(:,1+dt:(max(dt_interval)+dt));  
        SqrdDisp = Disp.^2;
        MSD(:,dt_idx) = squeeze(mean(SqrdDisp,1));
        % Calculate probability distribution  and energy for each dt
        pd = fitdist(MSD(:,dt_idx),'Kernel','Kernel','normal','bandwidth',.1);
        pdfEstimate = pdf(pd,ds);
        nrg = -1.*log(pdfEstimate); % energy is negative log(pdf)
        % figure; hold on
        % histogram(MSD(:,dt_idx),Normalization="pdf")
        % plot(ds,pdfEstimate)
        % Pool results across time
        nrg_data(dt_idx,:) = nrg;
    end
end 

