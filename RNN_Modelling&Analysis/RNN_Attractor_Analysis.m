%% Analysis of RNN population level dynamical regime
% https://elifesciences.org/reviewed-preprints/93191
% Christopher J. Whyte 26/11/24

set(0,'defaulttextinterpreter','latex')
rng(0) 
close all
clear 

%% set and import params

% number of networks
num_nets = 50;

% network params
prop_e = .8;
hidden_size = 40;
tau = 100;

% network params to be swept
gain_range = 1:.1:2;
input1 = linspace(1,0,40);
input2 = linspace(0,1,40);
U_range = [input1;input2];
input_delta = abs(input1-input2);

% import trained params
W_out_array = nan(2,hidden_size.*prop_e,num_nets);
W_in_array = nan(2,hidden_size,num_nets);
W_res_array = nan(hidden_size,hidden_size,num_nets);

% add location of trained weights to MATLAB path
addpath('pathname')

for net = 1:num_nets
    % change task weights
    W_out_file = sprintf('W_out_EI%d_40',net-1);
    W_out = load(W_out_file); 
    W_out_array(:,:,net) = W_out;
    W_res_file = sprintf('W_res_EI%d_40',net-1);
    W_res = load(W_res_file); 
    W_res_array(:,:,net) = W_res;
    W_in_file = sprintf('W_in_EI%d_40',net-1);
    W_in = load(W_in_file); 
    W_in_array(:,:,net) = W_in';
end 

% simulation params
sim_length = 10000; % 10 seconds of simulation time 
num_inits = 100;
T = 10; DT = 1;
alpha = DT/tau;
ds_factor = 10;
time = linspace(0,T,sim_length/ds_factor);

% initial conditions
exc_sel = []; inh_sel = []; rand_conds = 1;

%% Run simulation with constant gain and input 

S1_converge = zeros(size(U_range,2),size(gain_range,2)); 
S2_converge = zeros(size(U_range,2),size(gain_range,2));
S1_prop = zeros(size(U_range,2),size(gain_range,2)); 
S2_prop = zeros(size(U_range,2),size(gain_range,2));

for net = 1:num_nets

        r_attractor = zeros(hidden_size,sim_length/ds_factor,num_inits,size(U_range,2),length(gain_range));
        for int_idx = 1:num_inits
            for u_idx = 1:size(U_range,2)
                u_static = U_range(:,u_idx).*ones(2,sim_length);
                for gain_idx = 1:length(gain_range)
                    gain = gain_range(gain_idx);
                        W_in = W_in_array(:,:,net);
                        W_out = W_out_array(:,:,net);
                        W_res = W_res_array(:,:,net);
                        [r,~,~] = RNN_sim_static_relaxation(sim_length,alpha,W_res,W_in,W_out,u_static,gain,rand_conds,exc_sel,inh_sel);
                        r_attractor(:,:,int_idx,u_idx,gain_idx) = downsample(r',ds_factor)';
                end 
            end 
        end 
    
    % --- find convergence time, asymptotic (relaxation) dynamics, and detect oscillations
    
    % calculate convergence time 
    
    convergence_threshold = 1e-2;
    
    convergence_time = zeros(num_inits,size(U_range,2),length(gain_range));
    approximate_fp = 2*ones(hidden_size,num_inits,size(U_range,2),length(gain_range));
    fp_var = zeros(hidden_size,size(U_range,2),length(gain_range));
    
    for int_idx = 1:num_inits
        for u_idx = 1:size(U_range,2)
            for gain_idx = 1:length(gain_range)
                % find time point at which difference between successive
                % values < threshold
                delta_r = r_attractor(:,1:end-1,int_idx,u_idx,gain_idx) - r_attractor(:,2:end,int_idx,u_idx,gain_idx);
                convergence_idx = find(sum(abs(delta_r),1)<convergence_threshold,1);
                if ~isempty(convergence_idx)
                    convergence_time(int_idx,u_idx,gain_idx) = time(convergence_idx);
                    approximate_fp(:,int_idx,u_idx,gain_idx) = r_attractor(:,end,int_idx,u_idx,gain_idx);
                end 
            end 
        end 
    end 
    
    % average time to convergence
    convergence_time(convergence_time==0) = max(time);

    % calculate rolling stats
    data = squeeze(mean(convergence_time,1));
    S1_converge = (S1_converge + data);
    S2_converge = (S2_converge + data.^2);
    rolling_convergence_mean = S1_converge/net;
    rolling_convergence_std = sqrt(S2_converge/net - (S1_converge/net).^2);

    % proportion of initialisations that converged
    data = squeeze(mean(convergence_time == max(time),1));
    S1_prop = (S1_prop + data);
    S2_prop = (S2_prop + data.^2);
    rolling_prop_mean = S1_prop/net;
    rolling_prop_std = sqrt(S2_prop/net - (S1_prop/net).^2);
    
end 

%% run simulation with initial conditions sorted by stimulus selectivity.

% --- compute selectivity to set initial conditions

% compute excitatory selectivity based upon output weights
for net = 1:num_nets
    W_out = W_out_array(:,:,net);
    selectivity(:,net) = W_out(1,:) - W_out(2,:);
end 

selectivity = (selectivity + -1*min(selectivity))./(max(selectivity) + -1*min(selectivity));

% compute inhibitory selectivity ratio by the selectivity of the excitatory nodes they inhibit.
cat1 = (selectivity(1:32,:) >= .5);
cat2 = (selectivity(1:32,:) < .5);
for net_idx = 1:num_nets
    for node = 1:hidden_size-32
        W_res = W_res_array(1:32,33:end,net_idx);
        projectionc1 = W_res(cat1(:,net_idx),node);
        projectionc2 = W_res(cat2(:,net_idx),node);
        inhibitory_ratio(node,net_idx) = sum(projectionc1)./sum(projectionc2);
    end 
end 

% initial conditions set by selectivity
exc_sel_idx = reshape([cat1,cat2],[hidden_size-8,num_nets,2]);
inhibitory_ratio = inhibitory_ratio./max(inhibitory_ratio);
inhibitory_ratio_idx(:,:,1) = inhibitory_ratio >= .5;
inhibitory_ratio_idx(:,:,2) = inhibitory_ratio < .5;

% --- run simulation

% simulation params
sim_length = 500; 
gain = 1.5; u_static = [.5,.5]'.*ones(2,sim_length);

r_LS = zeros(hidden_size,sim_length/ds_factor,num_nets);
for net = 1:num_nets
    W_in = W_in_array(:,:,net);
    W_out = W_out_array(:,:,net);
    W_res = W_res_array(:,:,net);
    [r,~,~] = RNN_sim_static_relaxation(sim_length,alpha,W_res,W_in,W_out,u_static,gain,rand_conds,exc_sel_idx(:,net,1),inhibitory_ratio_idx(:,net,1));
    r_LS(:,:,net) = downsample(r',ds_factor)';
end 

% --- find difference in peak activation for stimulus selective populations
for net_idx = 1:num_nets
    for node_idx = 1:round(hidden_size.*prop_e)
        [~, locs] = max(r_LS(node_idx,:,net_idx));
        peaks(node_idx) = locs(1);
    end 
    peak_diff(net_idx) = abs(mean(mean(peaks(exc_sel_idx(:,net_idx,1)==1)) - mean(peaks(exc_sel_idx(:,net_idx,1)==0))))*ds_factor;
end

peak_diff_avg = mean(peak_diff);
peak_diff_std = std(peak_diff);


%% figures

% contour plots
x = U_range(1,:) - U_range(2,:);
y = gain_range;
[X,Y] = meshgrid(x,y);

% -------------------------------------------
% ---------- convergence time 
% -------------------------------------------

Z = rolling_convergence_mean';
figure(); hold on
contourf(X,Y,Z,'Levellist',0:1:10,'EdgeAlpha',0)
colormap(jet)
colorbar
set(gca, 'XDir','reverse')
set(gca, 'FontName', 'Times')
set(gca, 'FontSize', 20);
plot(x,gain_example(1,1:end-1),'w','linewidth',2)
plot(x,gain_example(2,1:end-1),'w','linewidth',3)
xlabel('$\Delta$ Input')
ylabel('Gain')
set(gcf,'color','w');
colormap(copper)

% -------------------------------------------
% ---------- proportion ints that converged 
% -------------------------------------------

Z = 1-rolling_prop_mean';
figure(); hold on
contourf(X,Y,Z,'Levellist',0:.1:1,'EdgeAlpha',0)
colormap(jet)
colorbar
set(gca, 'XDir','reverse')
set(gca, 'FontName', 'Times')
set(gca, 'FontSize', 20);
plot(x,gain_example(1,1:end-1),'w','linewidth',3)
plot(x,gain_example(2,1:end-1),'w','linewidth',4)
xlabel('$\Delta$ Input')
ylabel('Gain')
set(gcf,'color','w');
colormap(copper)

% -------------------------------------------
% ---------- example dynamics
% -------------------------------------------

sim_length = 10000; % 1 minute of simulation time 
T = 10000;
time = linspace(0,T,sim_length/ds_factor);

% high gain (gain = 1.5)
int_idx = 1;

u_idx = 1; gain_idx = 10;
figure(5); hold on
for node = 1:round(hidden_size*prop_e)
    sel = selectivity(node,example_net);
    plot(time,r_attractor(node,:,int_idx,u_idx,gain_idx),'LineWidth',2,'Color',[1-sel,.5,sel])
end 
xlabel('Time')
ylabel('Activity')
set(gcf,'color','w');
set(gca, 'FontName', 'Times')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;

u_idx = 20; gain_idx = 10;
figure(6); hold on
for node = 1:round(hidden_size*prop_e)
    sel = selectivity(node,example_net);
    plot(time,r_attractor(node,:,int_idx,u_idx,gain_idx),'LineWidth',2,'Color',[1-sel,.5,sel])
end 
xlabel('Time (ms)')
ylabel('Activity')
set(gcf,'color','w');
set(gca, 'FontName', 'Times')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;

u_idx = 40; gain_idx = 10;
figure(7); hold on
for node = 1:round(hidden_size*prop_e)
    sel = selectivity(node,example_net);
    plot(time,r_attractor(node,:,int_idx,u_idx,gain_idx),'LineWidth',2,'Color',[1-sel,.5,sel])
end 
xlabel('Time (ms)')
ylabel('Activity')
set(gcf,'color','w');
set(gca, 'FontName', 'Times')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;

% low gain (gain = 0.1)
u_idx = 1; gain_idx = 2;
figure(11); hold on
for node = 1:round(hidden_size*prop_e)
    sel = selectivity(node,example_net);
    plot(time,r_attractor(node,:,int_idx,u_idx,gain_idx),'LineWidth',2,'Color',[1-sel,.5,sel])
end 
xlabel('Time (ms)')
ylabel('Activity')
set(gcf,'color','w');
set(gca, 'FontName', 'Times')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;

u_idx = 20; gain_idx = 2;
figure(12); hold on
for node = 1:round(hidden_size*prop_e)
    sel = selectivity(node,example_net);
    plot(time,r_attractor(node,:,int_idx,u_idx,gain_idx),'LineWidth',2,'Color',[1-sel,.5,sel])
end 
xlabel('Time (ms)')
ylabel('Activity')
set(gcf,'color','w');
set(gca, 'FontName', 'Times')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;

u_idx = 40; gain_idx = 2;
figure(13); hold on
for node = 1:round(hidden_size*prop_e)
    sel = selectivity(node,example_net);
    plot(time,r_attractor(node,:,int_idx,u_idx,gain_idx),'LineWidth',2,'Color',[1-sel,.5,sel])
end 
xlabel('Time (ms)')
ylabel('Activity')
set(gcf,'color','w');
set(gca, 'FontName', 'Times')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;

% --------------------------------------------
% ---------- initialised in oscillatory regime
% --------------------------------------------
sim_length = 500; T = 500; example_net = 2;
time = linspace(0,T,sim_length/ds_factor);

figure(14);  hold on
for node = 1:round(hidden_size*prop_e)
    sel = selectivity(node,example_net);
    plot(time,r_LS(node,:,example_net),'LineWidth',2,'Color',[1-sel,.5,sel])
end 
xlabel('Time (ms)')
ylabel('Activity')
set(gcf,'color','w');
set(gca, 'FontName', 'Times')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;

%% functions

% rnn activation function
function y = sigmoid(x)
    y = 1./(1+exp(-x));
end 

% static gain network with flexible initial conditions
function [r_store,Z_store,action_store] = RNN_sim_static_relaxation(sim_length,alpha,W_res,W_in,W_out,u,gain,rand_conds,exc_sel,inh_sel)
    % initialise storage containers
    r_store = zeros(size(W_res,1),sim_length); Z_store = zeros(2,sim_length); 
    action_store = zeros(sim_length,1);
    % set initial conditions
    X = zeros(size(W_res,1),1); r = zeros(size(W_res,1),1); 
    if rand_conds == 1
        X = rand(size(W_res,1),1); r = rand(size(W_res,1),1); 
    end 
    if ~isempty(exc_sel)
       sel = [exc_sel;inh_sel];
       X = zeros(size(W_res,1),1); r = zeros(size(W_res,1),1); 
       X(sel==1,1) = 1; r(sel==1,1) = 1;
       X(sel==0,1) = -1; r(sel==0,1) = -1;
    end 
    % simulation loop
    for t = 1:sim_length
        gain_vec = ones(size(W_res,1),1);
        gain_vec = gain*gain_vec;
        % ODE components
        DX = -X + W_res*r + W_in'*u(:,t);
        % X + DX + DW
        X = X + alpha.*DX;
        r = sigmoid(gain_vec.*X);
        Z = W_out*r(1:32);
        % get network output
        [~, action_store(t)] = max(Z);
        % record data
        r_store(:,t) = r;
        Z_store(:,t) = Z;
    end 
end 

