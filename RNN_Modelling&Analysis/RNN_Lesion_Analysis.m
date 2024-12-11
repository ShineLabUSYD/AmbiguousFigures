%% Lesion based circuit level analysis of RNN dynamics
% https://elifesciences.org/reviewed-preprints/93191
% Christopher J. Whyte 26/11/24

set(0,'defaulttextinterpreter','latex')
rng('shuffle') 
close all
clear 

%% simulation params 

% number of networks
num_nets = 50;

% network params
hidden_size = 40;
prop_e = .8;
prop_i = 1 - prop_e;
tau = 100;
sigma = 0.01;
inv_temp = 1/4;
gamma_range = 0:.1:1;

% sim params
T = 1000;
DT = 1;
alpha = DT/tau;
burn_in = 500;
num_trls = 100;
trial_length = length(0:DT:T);
sim_length = burn_in-1 + trial_length ;

% network inputs 
trl_order = [ones(num_trls/2,1);2*ones(num_trls/2,1)];
input1 = [ones(1,burn_in-1),linspace(1,0,trial_length)];
input2 = [zeros(1,burn_in-1),linspace(0,1,trial_length)];
U = zeros(2,sim_length,2);
U(:,:,1) = [input1;input2];
U(:,:,2) = [input2;input1];
lesion_type = [1,0];

% import trained params
W_in_array = nan(2,hidden_size,num_nets);
W_out_array = nan(2,hidden_size.*prop_e,num_nets);
W_res_array = nan(hidden_size,hidden_size,num_nets);

% add location of trained weights to MATLAB path
addpath('pathname')

for net = 1:num_nets
    % change task weights
    W_in_file = sprintf('W_in_EI%d_40',net-1);
    W_in = load(W_in_file); 
    W_in_array(:,:,net) = W_in';
    W_out_file = sprintf('W_out_EI%d_40',net-1);
    W_out = load(W_out_file); 
    W_out_array(:,:,net) = W_out;
    W_res_file = sprintf('W_res_EI%d_40',net-1);
    W_res = load(W_res_file); 
    W_res_array(:,:,net) = W_res;
end 

%% compute selectivity

% compute excitatory selectivity based upon output weights
for net = 1:num_nets
    W_out = W_out_array(:,:,net);
    selectivity(:,net) = W_out(1,:) - W_out(2,:);
end 

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
cat1 = (selectivity(1:32,:) > .5);
cat2 = (selectivity(1:32,:) < .5);

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

%% lesion simulations

% set up containers for selectivity simulations
r_store = zeros(hidden_size,trial_length,num_trls,num_nets,length(gamma_range),2);
Z_store = zeros(2,trial_length,num_trls,num_nets,length(gamma_range),2);
action_store = zeros(trial_length,num_trls,num_nets,length(gamma_range),2);
gain_store = zeros(trial_length,num_trls,num_nets,length(gamma_range),2);
uncertainty_store = zeros(trial_length,num_trls,num_nets,length(gamma_range),2);
switchpoint_store = zeros(num_trls,num_nets,length(gamma_range),2);

for lesion_idx = 1:size(lesion_type,2)
    % lesion = lesion_type(lesion_idx);
    for gam_idx = 1:length(gamma_range)
        gamma = gamma_range(gam_idx);
        for net = 1:num_nets
            % grab trained weights for network
            W_in = W_in_array(:,:,net);
            W_out = W_out_array(:,:,net);
            W_res = W_res_array(:,:,net);
            inhibitory_lesion = [ones(32,1);inhibitory_ratio_idx(:,net,lesion_idx)==1];
            W_res(:,inhibitory_lesion==1) = 0;
            for trl = 1:num_trls
                % trial type
                trl_idx = trl_order(trl,1);
                u = U(:,:,trl_idx);
                [r_store(:,:,trl,net,gam_idx,lesion_idx),...
                Z_store(:,:,trl,net,gam_idx,lesion_idx),...
                gain_store(:,trl,net,gam_idx,lesion_idx),...
                uncertainty_store(:,trl,net,gam_idx,lesion_idx),...
                action_store(:,trl,net,gam_idx,lesion_idx)] = RNN_sim_dynamic_lesion(sim_length,alpha,DT,W_res,W_in,W_out,u,gamma,sigma,inv_temp,burn_in);
                % record switch point
                switch_vals = find(diff(action_store(:,trl,net,gam_idx,lesion_idx)));
                [~, switch_idx] = find(diff(action_store(:,trl,net,gam_idx,lesion_idx))~=0);
                if sum(switch_idx) > 1
                   switch_idx = switch_idx(1);
                end 
                if sum(switch_idx) ~= 0
                   switchpoint_store(trl,net,gam_idx,lesion_idx) = switch_vals(switch_idx);
                end 
            end 
        end 
    end 
end 

%% switch time analysis

% exlcude and count non-zero values
for lesion_idx = 1:size(lesion_type,2) % C1 or C2 selective inhibitory nodes
    for gam_idx = 1:length(gamma_range)
        for net = 1:num_nets
            % category 1
            switch_int = switchpoint_store(1:round(num_nets/2),net,gam_idx,lesion_idx);
            no_switch_c1(net,gam_idx,lesion_idx) = sum(switch_int==0)./num_trls;
            switch_int = switch_int(switch_int~=0);
            switch_store_c1(net,gam_idx,lesion_idx) = mean(switch_int,'omitnan');
            % category 2
            switch_int = switchpoint_store(round(num_nets/2)+1:end,net,gam_idx,lesion_idx);
            no_switch_c2(net,gam_idx,lesion_idx) = sum(switch_int==0)./num_trls;
            switch_int = switch_int(switch_int~=0);
            switch_store_c2(net,gam_idx,lesion_idx) = mean(switch_int,'omitnan');
        end 
    end 
end 


switch_store_1 = [switch_store_c1(:,:,1); switch_store_c2(:,:,2)];
mean_switch_1 = mean(switch_store_1,1,'omitnan');
ste_switch_1 = std(switch_store_1,1,'omitnan')./sqrt(num_nets);

switch_store_2 = [switch_store_c1(:,:,2); switch_store_c2(:,:,1)];
mean_switch_2 = mean(switch_store_2,1,'omitnan');
ste_switch_2 = std(switch_store_2,1,'omitnan')./sqrt(num_nets);

avg_switch_c1 = mean(switch_store_c1,1,'omitnan');
ste_switch_c1 = std(switch_store_c1,1,'omitnan')./sqrt(num_nets);
avg_switch_c2 = mean(switch_store_c2,1,'omitnan');
ste_switch_c2 = std(switch_store_c2,1,'omitnan')./sqrt(num_nets);

avg_no_switch_c1 = mean(no_switch_c1,1,'omitnan');
ste_no_switch_c1 = std(no_switch_c1,1,'omitnan')./sqrt(num_nets);
avg_no_switch_c2 = mean(no_switch_c2,1,'omitnan');
ste_no_switch_c2 = std(no_switch_c2,1,'omitnan')./sqrt(num_nets);


%% Figures

% ----------------------------------------------------------
% ---------- switch time under lesions to C1 or C2 inh units
% ----------------------------------------------------------

figure(1); hold on
errorbar(gamma_range,avg_switch_c1(:,:,1),ste_switch_c1(:,:,1),'LineWidth',3,'Color',[0,.5,1])
errorbar(gamma_range,avg_switch_c1(:,:,2),ste_switch_c1(:,:,2),'LineWidth',3,'Color',[1,.5,0])
ylabel('Switch time')
xlabel('$\gamma$')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
axis padded
set(gcf,'color','w');

figure(2); hold on
errorbar(gamma_range,avg_switch_c2(:,:,1),ste_switch_c2(:,:,1),'LineWidth',3,'Color',[0,.5,1])
errorbar(gamma_range,avg_switch_c2(:,:,2),ste_switch_c2(:,:,2),'LineWidth',3,'Color',[1,.5,0])
ylabel('Switch time')
xlabel('$\gamma$')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
axis padded
set(gcf,'color','w');

figure(3); hold on
errorbar(gamma_range,avg_no_switch_c1(:,:,1),ste_no_switch_c1(:,:,1),'LineWidth',3,'Color',[0,.5,1])
errorbar(gamma_range,avg_no_switch_c1(:,:,2),ste_no_switch_c1(:,:,2),'LineWidth',3,'Color',[1,.5,0])
ylabel('proportion no switch')
xlabel('$\gamma$')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
axis padded
set(gcf,'color','w');

figure(4); hold on
errorbar(gamma_range,avg_no_switch_c2(:,:,1),ste_no_switch_c2(:,:,1),'LineWidth',3,'Color',[0,.5,1])
errorbar(gamma_range,avg_no_switch_c2(:,:,2),ste_no_switch_c2(:,:,2),'LineWidth',3,'Color',[1,.5,0])
ylabel('proportion no switch')
xlabel('$\gamma$')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
axis padded
set(gcf,'color','w');

figure(5); hold on
errorbar(gamma_range,mean_switch_1,ste_switch_1,'LineWidth',3, 'Color',[0 .5 1])
errorbar(gamma_range,mean_switch_2,ste_switch_2,'LineWidth',3, 'Color',[1 .5 0])
xlabel('$\gamma$')
ylabel('Switch time')
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
axis padded
set(gcf,'color','w');
ylim([4.5 7])

% switch_lesion(:,1) = mean_switch_1;
% switch_lesion(:,2) = mean_switch_2;
% switch_lesion_ste(:,1) = ste_switch_1;
% switch_lesion_ste(:,2) = ste_switch_2;
% save('switch_lesion','switch_lesion')
% save('switch_lesion_ste','switch_lesion_ste')

% -------------------------------------------
% ---------- example dynamics
% -------------------------------------------

example_net = 2;
time = linspace(0,T,trial_length);

% excitatory activity
figure(6); hold on
for node = 1:round(hidden_size*prop_i)
    sel = selectivity(node,example_net);
    plot(time,r_store(node,:,20,example_net,6,1),'LineWidth',2,'Color',[sel,0,1-sel])
end 

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

% dynamic gain network 
function [r_store,Z_store,gain_store,uncertainty_store,action_store] = RNN_sim_dynamic_lesion(sim_length,alpha,DT,W_res,W_in,W_out,u,gamma,sigma,inv_temp,burn_in)
    % storage containers
    r_store = zeros(size(W_res,1),sim_length); Z_store = zeros(2,sim_length); 
    uncertainty_store = zeros(sim_length,1); action_store = zeros(sim_length,1); gain_store = zeros(sim_length,1); 
    % initial conditions
    X = ones(size(W_res,1),1); r = zeros(size(W_res,1),1); gain = 1;
    for t = 1:sim_length
        gain_vec = ones(40,1);
        gain_vec = gain*gain_vec;
        % ODE components
        DX = -X + W_res*r + W_in'*u(:,t);
        DW = sigma*sqrt(DT)*randn([size(W_res,1),1]);
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
