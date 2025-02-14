%% Change detection recurrent neural network simulation and analysis
% https://elifesciences.org/reviewed-preprints/93191
% Christopher J. Whyte 26/11/24

set(0,'defaulttextinterpreter','latex')
rng('shuffle') 
close all
clear 

%% set and import params

% number of networks
num_nets = 50;

% network params
hidden_size = 40;
prop_e = .8;
prop_i = 1 - prop_e;
tau = 100;    % ms
sigma = 0.01; % recurrent noise
inv_temp = 1/4;
gamma_range = 0:.1:1;
gain_range = .5:.1:1.5;

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
num_trls = 20;  % number of trials
T = 1000;       % ms
DT = 1;         % ms
alpha = DT/tau; 
burn_in = 500;  % ms
trial_length = length(0:DT:T);
sim_length = burn_in-1 + trial_length;


% simulation indices 
static = 1; dynamic = 2; rand_conds=0;
trl_order = [ones(num_trls/2,1);2*ones(num_trls/2,1)];

% network inputs 
input1 = [ones(1,burn_in-1),linspace(1,0,trial_length)];
input2 = [zeros(1,burn_in-1),linspace(0,1,trial_length)];
U = zeros(2,sim_length,2);
U(:,:,1) = [input1;input2];
U(:,:,2) = [input2;input1];

% initialise arrays
r_store = zeros(hidden_size,trial_length,num_trls,num_nets,length(gamma_range),2);
Z_store = zeros(2,trial_length,num_trls,num_nets,length(gamma_range),2);
action_store = zeros(trial_length,num_trls,num_nets,length(gamma_range),2);
gain_store = zeros(trial_length,num_trls,num_nets,length(gamma_range),2);
uncertainty_store = zeros(trial_length,num_trls,num_nets,length(gamma_range),2);
switchpoint_store = zeros(num_trls,num_nets,length(gamma_range),2);

%% simulate networks

% static network
for gain_idx = 1:length(gain_range)
    gain = gain_range(gain_idx);
    for net = 1:num_nets
        % grab trained weights for network
        W_in = W_in_array(:,:,net);
        W_out = W_out_array(:,:,net);
        W_res = W_res_array(:,:,net);
        for trl = 1:num_trls
            % trial type
            trl_idx = trl_order(trl,1);
            u = U(:,:,trl_idx);
            [r_store(:,:,trl,net,gain_idx,static),...
            Z_store(:,:,trl,net,gain_idx,static),...
            action_store(:,trl,net,gain_idx,static)] = RNN_sim_static(sim_length,alpha,DT,W_res,W_in,W_out,u,gain,sigma,rand_conds,burn_in);
            % record switch point
            switch_vals = find(diff(action_store(:,trl,net,gain_idx,static)));
            [~,switch_idx] = find(diff(action_store(:,trl,net,gain_idx,static))~=0);
            if sum(switch_idx) > 1
               switch_idx = switch_idx(1);
            end 
            if sum(switch_idx) ~= 0
               switchpoint_store(trl,net,gain_idx,static) = switch_vals(switch_idx);
            end 
        end 
    end 
end 

% dynamic gain network
for gam_idx = 1:length(gamma_range)
    gamma = gamma_range(gam_idx);
    for net = 1:num_nets
        % grab trained weights for network
        W_in = W_in_array(:,:,net);
        W_out = W_out_array(:,:,net);
        W_res = W_res_array(:,:,net);
        for trl = 1:num_trls
            % trial type
            trl_idx = trl_order(trl,1);
            u = U(:,:,trl_idx);
            [r_store(:,:,trl,net,gam_idx,dynamic),...
            Z_store(:,:,trl,net,gam_idx,dynamic),...
            gain_store(:,trl,net,gam_idx,dynamic),...
            uncertainty_store(:,trl,net,gam_idx,dynamic),...
            action_store(:,trl,net,gam_idx,dynamic)] = RNN_sim_dynamic(sim_length,alpha,DT,W_res,W_in,W_out,u,gamma,sigma,inv_temp,rand_conds,burn_in);
            % record switch point
            switch_vals = find(diff(action_store(:,trl,net,gam_idx,dynamic)));
            [~,switch_idx] = find(diff(action_store(:,trl,net,gam_idx,dynamic))~=0);
            if sum(switch_idx) > 1
               switch_idx = switch_idx(1);
            end 
            if sum(switch_idx) ~= 0
               switchpoint_store(trl,net,gam_idx,dynamic) = switch_vals(switch_idx);
            end 
        end 
    end 
end 

% all following "neuronal" analyses focus on dynamic gain networks
r_store_dynamic = r_store(:,:,:,:,:,dynamic);

% free up memory
clear r_store

%% calculate switch times

no_switch = zeros(num_nets,length(gamma_range),dynamic);
switch_store = zeros(num_nets,length(gamma_range),dynamic);

% exlcude and count non-zero values
for dyn_idx = 1:dynamic % dynamic and static gain
    for gam_idx = 1:length(gamma_range)
        for net = 1:num_nets
            switch_int = switchpoint_store(:,net,gam_idx,dyn_idx);
            no_switch(net,gam_idx,dyn_idx) = sum(switch_int==0)./num_trls;
            switch_int = switch_int(switch_int~=0);
            switch_store(net,gam_idx,dyn_idx) = mean(switch_int);
        end 
    end 
end 

avg_switch = mean(switch_store,1);
ste_switch = std(switch_store,1)./sqrt(num_nets);
avg_no_switch = mean(no_switch,1);
ste_no_switch = std(no_switch,1)./sqrt(num_nets);

%% compute recurrent unit selectivity

selectivity = zeros(round(hidden_size*prop_e),num_nets);

% selectivity = projection onto output
for net = 1:num_nets
    W_out = W_out_array(:,:,net);
    selectivity(:,net) = W_out(1,:) - W_out(2,:);
end 

% normalise to vary between [0,1] for plotting instead of [-1,1]
selectivity = (selectivity + -1*min(selectivity))./(max(selectivity) + -1*min(selectivity));

% figure()
% histogram(selectivity,'Normalization','percentage','FaceColor',[76 109 172]/255)
% ylabel('Population %')
% xlabel('Excitatory selectivity')
% ax = gca;
% ax.FontSize = 20;
% ax.LineWidth = 2;
% set(gca, 'FontName', 'Times')
% axis padded
% box off

% compute inhibitory selectivity ratio by the selectivity of the excitatory nodes they inhibit.
cat1 = selectivity > .5;
cat2 = selectivity < .5;

inhibitory_ratio = zeros(round(hidden_size*prop_i),num_nets);

for net_idx = 1:num_nets
    for node = 1:round(hidden_size*prop_i)
        W_res = W_res_array(1:32,33:end,net_idx);
        W_res_c1 = W_res(cat1(:,net_idx),node);
        W_res_c2 = W_res(cat2(:,net_idx),node);
        inhibitory_ratio(node,net_idx) = sum(W_res_c1)./sum(W_res_c2);
    end 
end 

inhibitory_ratio = inhibitory_ratio./max(inhibitory_ratio);
inhibitory_ratio_idx(:,:,1) = inhibitory_ratio > .5;
inhibitory_ratio_idx(:,:,2) = inhibitory_ratio < .5;

% figure()
% histogram(inhibitory_ratio,'Normalization','percentage','FaceColor',[103 56 104]/255)
% ylabel('Population %')
% xlabel('Inhibitory selectivity')
% ax = gca;
% ax.FontSize = 20;
% ax.LineWidth = 2;
% set(gca, 'FontName', 'Times')
% axis padded
% box off


%% PCA trajectory analysis

projection_c1 = zeros(trial_length,num_trls/2,length(gamma_range),num_nets);
projection_c2 = zeros(trial_length,num_trls/2,length(gamma_range),num_nets);
activity_c1 = zeros(trial_length,round(num_trls/2),hidden_size,length(gamma_range));
activity_c2 = zeros(trial_length,round(num_trls/2),hidden_size,length(gamma_range));
velocity_c1 = zeros(trial_length,length(gamma_range),num_nets);
velocity_c2 = zeros(trial_length,length(gamma_range),num_nets);

for net = 1:num_nets

    activity = squeeze(r_store_dynamic(:,:,:,net,:));
    activity = permute(activity,[2,3,1,4]); % time x trials x nodes x gain_range

    % concatenate activity for PCA
    concat_activity = reshape(activity, [size(activity,1)*size(activity,2),size(activity,3),size(activity,4)]);
    concat_activity = zscore(concat_activity);
    
    % seperate into conditions before projecting
    for g = 1:length(gamma_range)
        activity_c1(:,:,:,g) = activity(:,1:num_trls/2,:,g);
        activity_c2(:,:,:,g) = activity(:,num_trls/2+1:end,:,g);
    end 
    
    % PCA on gamma = 0
    [vecs,vals,~,~,explained] = pca(concat_activity(:,:,1));

    % project activity each condition (onto gain = 1)
    for g = 1:length(gamma_range)
        for trl = 1:num_trls/2
            for t = 1:size(activity,1)
                projection_c1(t,trl,g,net) = squeeze(activity_c1(t,trl,:,g))'*vecs(:,1);
                projection_c2(t,trl,g,net) = squeeze(activity_c2(t,trl,:,g))'*vecs(:,1);
            end 
        end 
    end 
end 

% calculate velocity of PC trajectory trial by trial
for net = 1:num_nets
    for g = 1:length(gamma_range)
        for trl = 1:num_trls/2
            velocity_c1(:,trl,g,net) = gradient(projection_c1(:,trl,g,net));
            velocity_c2(:,trl,g,net) = gradient(projection_c2(:,trl,g,net));
        end 
    end 
end 

velocity_c1 = mean(abs(mean(velocity_c1(:,:,:,:),2)),4);
velocity_c2 = mean(abs(mean(velocity_c2(:,:,:,:),2)),4);

velocity = (1/2)*[velocity_c1,velocity_c2];

%% allocentic energy landscape and neural work

stepsize = 25;
t_int = 1:stepsize:size(projection_c1,1);

ds_allo = -3:.1:3;

nrg_t_c1 = zeros(length(ds_allo),length(t_int)-1,length(gamma_range),num_nets);
nrg_t_c2 = zeros(length(ds_allo),length(t_int)-1,length(gamma_range),num_nets);
traj_c1_work = zeros(length(t_int)-1,length(gamma_range),num_nets);
traj_c2_work = zeros(length(t_int)-1,length(gamma_range),num_nets);
traj_c1 = zeros(2,length(t_int)-1,length(gamma_range),num_nets);
traj_c2 = zeros(2,length(t_int)-1,length(gamma_range),num_nets);

for net = 1:num_nets
    for g = 1:length(gamma_range)
        for t = 1:length(t_int)-1

            % Window
            start = t_int(t); stop = t_int(t+1); 

            % --- category 1 -> category 2
            projection = projection_c1(:,:,g,net);
            % allocentric landscape
            [nrg_t_c1(:,t,g,net), pd1] = allo_nrg_calc(projection,ds_allo,start,stop);
            % neural work
            [traj_c1_work(t,g,net), traj_c1(:,t,g,net)] = neural_work_calc(pd1,projection,start,stop);

            % --- category 2 -> category 1
            projection = projection_c2(:,:,g,net);
            % allocentric landscape
            [nrg_t_c2(:,t,g,net), pd2] = allo_nrg_calc(projection,ds_allo,start,stop);
            % neural work
            [traj_c2_work(t,g,net), traj_c2(:,t,g,net)] = neural_work_calc(pd2,projection,start,stop);

        end 
    end 
end 

%% egocentric energy landscape analysis

dt_interval = 100:25:(size(r_store_dynamic,2)-1)/2;
ndt = length(dt_interval);
ds_ego = 0:.01:.3;

nrg_ego = zeros(ndt,length(ds_ego),num_nets,length(gamma_range));
for gamma = 1:length(gamma_range)
    for net = 1:num_nets
        nrg_ego(:,:,net,gamma) = ego_nrg_calc(r_store_dynamic(:,:,:,net,gamma),ds_ego,dt_interval);
    end 
end 

% average over networks
nrg_avg = squeeze(mean(nrg_ego,3));

%% figures

% -------------------------------------------
% ---------- network behaviour
% -------------------------------------------

% load('switch_lesion.mat')
% load('switch_lesion_ste.mat')

figure(1); hold on
title('Static gain network')
errorbar(gain_range,avg_switch(:,:,static),ste_switch(:,:,static),'k-','LineWidth',3)
ylabel('Switch time')
xlabel('Gain')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
axis padded

figure(2); hold on
% title('Dynamic gain network')
% errorbar(gamma_range,switch_lesion(:,1),switch_lesion_ste(:,1),'LineWidth',3, 'Color',[0,.5,1])
% errorbar(gamma_range,switch_lesion(:,2),switch_lesion_ste(:,2),'LineWidth',3, 'Color',[1,.5,0])
errorbar(gamma_range,avg_switch(:,:,dynamic),ste_switch(:,:,dynamic),'k','LineWidth',3)
ylabel('Switch time (s)')
xlabel('$\gamma$')
ax = gca;
ax.FontSize = 15;
ax.LineWidth = 1.5;
set(gca, 'FontName', 'Times')
axis padded

figure(3); hold on
errorbar(gamma_range,avg_no_switch(:,:,static),ste_no_switch(:,:,static),'k','LineWidth',3)
ylabel('Switch time')
xlabel('Gain')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
axis padded

% save('avg_switch','avg_switch')

% -------------------------------------------
% ---------- RNN activity 
% -------------------------------------------

example_trl = 19; example_net = 3; example_gam = 1;
time = linspace(0,T-1,trial_length);

% excitatory activity
figure(4); hold on
for node = 1:round(hidden_size*prop_e)
    sel = selectivity(node,example_net);
    plot(time,r_store_dynamic(node,:,example_trl,example_net,example_gam),'LineWidth',2,'Color',[1-sel,.5,sel])
end 
axis padded
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
xticks(0:1:9)
ylabel('Firing rate (a.u.)')
xlabel('Time')

% inhibitory activity
figure(5); hold on
for node = 1:round(hidden_size*prop_i)
    sel_inh = inhibitory_ratio(node,example_net);
    plot(time,r_store_dynamic(hidden_size*prop_e+node,:,example_trl,example_net,example_gam),'LineWidth',2,'Color',[1-sel_inh,.5,sel_inh])
end 
axis padded
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
xticks(0:1:9)
ylabel('Firing rate (a.u.)')
xlabel('Time')


figure(6); hold on
switch_ts = softmax(Z_store(:,:,example_trl,example_net,example_gam,dynamic),inv_temp);
plot(time,switch_ts(1,:),'LineWidth',3, 'Color', [0,.5,1])
plot(time,switch_ts(2,:),'LineWidth',3,'Color', [1,.5,0])
ax = gca;
ax.FontSize = 15;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
axis padded
ylabel('P(stimulus)')
xlabel('Time')
xticks(0:1:9)

figure(7); example_gam = 6;
plot(time,gain_store(:,example_trl,example_net,example_gam,dynamic),'LineWidth',4, 'Color', [224 104 99]/255)
ylabel('Gain')
xlabel('Time')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
axis padded
xticks(0:1:9)
box off

gain_example = zeros(3,trial_length);
gain_idx_1 = 2; gain_idx_2 = 10; 
gain_example(1,:) = gain_store(:,example_trl,example_net,gain_idx_1,dynamic);
gain_example(2,:) = gain_store(:,example_trl,example_net,gain_idx_2,dynamic);

save('gain_example','gain_example')

% -------------------------------------------
% ---------- PC trajectory and velocity plots
% -------------------------------------------

figure(8); hold on
example_net = 3;
plot(time,projection_c1(:,1,2,example_net) - mean(projection_c1(:,1,2,example_net),1),'b','linewidth',3)
plot(time,projection_c1(:,1,6,example_net) - mean(projection_c1(:,1,6,example_net),1),'g--','linewidth',3)
plot(time,projection_c1(:,1,10,example_net) - mean(projection_c1(:,1,10,example_net),1),'r','linewidth',3)
xlabel('Time')
ylabel('PC loadings')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');
axis padded

figure(9); hold on
plot(time,smooth(velocity(:,2)),'b','linewidth',2)
plot(time,smooth(velocity(:,6)),'g--','linewidth',2)
plot(time,smooth(velocity(:,10)),'r','linewidth',2)
ylabel('abs(PC velocity)')
xlabel('Time')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');
axis padded

% -------------------------------------------
% ---------- Allocentric energy landscape 
% -------------------------------------------

example_net = 1;
x = linspace(0,1000,size(nrg_t_c1,2))/100;
y = ds_allo;

nrg_t_c1(nrg_t_c1>6)=6;

[X,Y] = meshgrid(x,y);

figure(10); gain = 2;
mesh(X,Y,nrg_t_c1(:,:,gain,example_net),'EdgeColor', [0,0,255]./255)
xlabel('Time')
ylabel('PC1 loading')
zlabel('Energy')
hold on
plot(x,smooth(traj_c1(1,:,gain,example_net)),'k','linewidth',6)
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'XDir','reverse')
set(gca, 'YDir','reverse')
set(gca, 'FontName', 'Times')
set(gcf,'color','w');
axis padded
zlim([-1,6])
ylim([-2.5,2.5])

figure(11); gain = 6;
mesh(X,Y,nrg_t_c1(:,:,gain,example_net),'EdgeColor', [0,255,100]./255)
xlabel('Time')
ylabel('PC1 loading')
zlabel('Energy')
hold on
plot(x,smooth(traj_c1(1,:,gain,example_net)),'k','linewidth',6)
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'XDir','reverse')
set(gca, 'YDir','reverse')
set(gca, 'FontName', 'Times')
set(gcf,'color','w');
axis padded
zlim([-1,6])
ylim([-2.5,2.5])

figure(12); gain = 10;
mesh(X,Y,nrg_t_c1(:,:,gain,example_net),'EdgeColor', [255,0,0]./255)
xlabel('Time')
ylabel('PC1 loading')
zlabel('Energy')
hold on
plot(x,smooth(traj_c1(1,:,gain,example_net)),'k','linewidth',6)
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'XDir','reverse')
set(gca, 'YDir','reverse')
set(gca, 'FontName', 'Times')
set(gcf,'color','w');
axis padded
zlim([-1,6])
ylim([-2.5,2.5])

% -------------------------------------------
% ---------- Neural work 
% -------------------------------------------

traj_c1_work = mean(traj_c1_work,3);
traj_c2_work = mean(traj_c2_work,3);
traj_work = .5*(traj_c1_work + traj_c2_work);

figure(13); hold on
plot(t_int(1:end-1)/100,squeeze(smooth(traj_work(:,2))),'b','linewidth',2)
plot(t_int(1:end-1)/100,squeeze(smooth(traj_work(:,6))),'g--','linewidth',2)
plot(t_int(1:end-1)/100,squeeze(smooth(traj_work(:,10))),'r','linewidth',2)
legend('\gamma = 0.1','\gamma = 0.5','\gamma = 0.9')
ylabel('Neural work (a.u.)')
xlabel('Time')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');
axis padded

% -------------------------------------------
% ---------- Egocentric energy landscape  
% -------------------------------------------

x = dt_interval*DT/100;
y = ds_ego;

[X,Y] = meshgrid(x,y);

figure(14)
mesh(X,Y,nrg_avg(:,:,2)','EdgeColor', [0,0,255]./255)
xlabel('Tau')
ylabel('MSD')
zlabel('MSD  energy')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');

figure(15)
mesh(X,Y,nrg_avg(:,:,10)','EdgeColor', [255,0,0]./255)
xlabel('Tau')
ylabel('MSD')
zlabel('MSD  energy')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');

% integrate out time
nrg_AUC = squeeze(sum(nrg_avg,1));

figure(16); hold on
plot(ds_ego,nrg_AUC(:,2),'b','Linewidth',2)
plot(ds_ego,nrg_AUC(:,6),'--','color',[0,.75,.1],'Linewidth',2)
plot(ds_ego,nrg_AUC(:,10),'r','Linewidth',2)
xlabel('MSD')
ylabel('Energy AUC')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
set(gcf,'color','w');
legend('\gamma = 0.1','\gamma = 0.5','\gamma = 0.9')
axis padded

%% functions

% rnn activation function
function y = sigmoid(x)
    y = 1./(1+exp(-x));
end 

% softmax function
function y = softmax(x,inv_temp)
    y = exp(inv_temp*x)./sum(exp(inv_temp*x));
end 

% entropy function
function y = Entropy(p,inv_temp)
    y = sum(-softmax(p,inv_temp).*log2(softmax(p,inv_temp)));
end 

% static gain network
function [r_store,Z_store,action_store] = RNN_sim_static(sim_length,alpha,DT,W_res,W_in,W_out,u,gain,sigma,rand_conds,burn_in)
    % storage containers
    r_store = zeros(size(W_res,1),sim_length); Z_store = zeros(2,sim_length); 
    action_store = zeros(sim_length,1);
    % initial conditions
    X = zeros(size(W_res,1),1); r = zeros(size(W_res,1),1); 
    if rand_conds == 1
        X = rand(size(W_res,1),1); r = rand(size(W_res,1),1); 
    end 
    for t = 1:sim_length
        % ODE components
        DX = -X + W_res*r + W_in'*u(:,t);
        DW = sigma*sqrt(DT)*randn([size(W_res,1),1]);
        % X + DX + DW
        X = X + alpha.*DX + DW;
        r = sigmoid(gain.*X);
        Z = W_out*r(1:32);
        % get network output
        [~, action_store(t)] = max(Z);
        % record data
        r_store(:,t) = r;
        Z_store(:,t) = Z;
    end 
    r_store = r_store(:,burn_in:end);
    Z_store = Z_store(:,burn_in:end);
    action_store = action_store(burn_in:end);
end 

% dynamic gain network
function [r_store,Z_store,gain_store,uncertainty_store,action_store] = RNN_sim_dynamic(sim_length,alpha,DT,W_res,W_in,W_out,u,gamma,sigma,inv_temp,rand_conds,burn_in)
    % storage containers
    r_store = zeros(size(W_res,1),sim_length); Z_store = zeros(2,sim_length); 
    uncertainty_store = zeros(sim_length,1); action_store = zeros(sim_length,1); gain_store = zeros(sim_length,1); 
    % initial conditions
    X = ones(size(W_res,1),1); r = zeros(size(W_res,1),1); gain = 1;
    if rand_conds == 1
        X = rand(size(W_res,1),1); r = rand(size(W_res,1),1); 
    end 
    for t = 1:sim_length
        gain_vec = ones(40,1);
        gain_vec = gain*gain_vec;
        % ODE components
        DX = -X + W_res*r + W_in'*u(:,t);
        DW = sigma*sqrt(1)*randn([size(W_res,1),1]);
        % X + DX + DW
        X = X + alpha.*DX + DW;
        r = sigmoid(gain_vec.*X);
        Z = W_out*r(1:32);
        % get network output
        [~, action_store(t)] = max(Z);
        % update gain with uncertainty
        uncertainty = Entropy(Z,inv_temp);
        gain = gain + alpha*((1 - gain) + gamma.*uncertainty);
        % record data
        r_store(:,t) = r;
        Z_store(:,t) = Z;
        gain_store(t) = gain;
        uncertainty_store(t) = uncertainty;
    end 
    r_store = r_store(:,burn_in:end);
    Z_store = Z_store(:,burn_in:end);
    gain_store = gain_store(burn_in:end);
    uncertainty_store = uncertainty_store(burn_in:end);
    action_store = action_store(burn_in:end);
end 

% allocentric energy landscape
function [nrg_data, pd] = allo_nrg_calc(projection,ds_allo,start,stop)
          data = reshape(projection(start:stop,:),[1,length(start:stop)*size(projection,2)]);
          pd = fitdist(data','Kernel','Kernel','normal','bandwidth',.2);
          pdfEstimate = pdf(pd,ds_allo);
          nrg_data = -1.*log(pdfEstimate);
end 

% neural work
function [trajectory_work, trajectory] = neural_work_calc(pd,projection,start,stop)
        point_start = mean(projection(start,:),2);       % starting point of window averaged over trials
        point_stop = mean(projection(stop,:),2);         % end point of window averaged over trials
        trajectory  = mean(mean(projection(start:stop,:),1),2);
        nrg_grad = gradient([-1*log(pdf(pd,point_start)),-1*log(pdf(pd,point_stop))]); 
        trajectory_work = -nrg_grad(1)*abs(point_stop-point_start);
end 

% egocentric energy landscape
function nrg_data = ego_nrg_calc(data,ds,dt_interval)
    nrg_data = nan(length(dt_interval),numel(ds));
    % permute data (nodes,time,trls) -> (time,nodes,trls)
    data = permute(data,[2,1,3]);
    % concatenate trials (time,nodes,trls) -> (time,nodes x trls)
    data = reshape(data,[size(data,1),size(data,2)*size(data,3)]);
    for dt_idx = 1:length(dt_interval)
        dt = dt_interval(dt_idx);
        % Mean-Square Displacement calculation
        Disp = data(1+dt:end,:) - data(1:end-dt,:);
        SqrdDisp = Disp.^2;
        MSD = mean(SqrdDisp,2);
        % Calculate probability distribution  and energy for each dt
        pd = fitdist(MSD,'Kernel','Kernel','normal','bandwidth',.01);
        pdfEstimate = pdf(pd,ds);
        nrg = -1.*log(pdfEstimate); % energy is negative log(pdf)
        % Pool results across time
        nrg_data(dt_idx,:) = nrg;
    end
end 

