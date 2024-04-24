%% sim_script.m

% Stochastic simulation of an antithetic integral feedback controller's performance

%% CLEAR all variables

addpath(genpath('..'))

close all
clear

%% DEFINE number of trajectories and how we simulate the trajectories

num_trajs=48;

%% INITIALISE stochastic trajectory arrays and simulators

% with controller
sim={};
ts={};
% deterministic steady state values
steady_psens={};
steady_l={};
steady_D={};
% relative trajectories
psens_trajs={};
l_trajs={};
D_trajs={};

% open loop
sim_openloop={};
ts_openloop={};
% deterministic steady state values
steady_psens_openloop={};
steady_l_openloop={};
steady_D_openloop={};
% relative trajectories
psens_trajs_openloop={};
l_trajs_openloop={};
D_trajs_openloop={};

%% RUN parallel simulations
tic
parfor traj_cntr=1:num_trajs
    % time scale shifts for pretty plots
    dist_time=7.5;
    inter_rad=dist_time;

    %% CLOSED LOOP
    % define the controller    
    sim{traj_cntr}=cell_simulator;
    
    sim{traj_cntr}.init_conditions('s')=0.5;
    
    sim{traj_cntr}=sim{traj_cntr}.load_heterologous_and_external('aif_controller','step_inducer'); % load the het. gene and ext. inp. modules
    sim{traj_cntr}=sim{traj_cntr}.generate_het_stoichiometry_matrix();
    
    % disturbance signal parameters
    sim{traj_cntr}.ext.input_func_parameters('inducer_base_level')=0; % disturbance flicks transcription reg. func. from 0 to 1 
    sim{traj_cntr}.ext.input_func_parameters('inducer_final_level')=1; % disturbance flicks transcription reg. func. from 0 to 1 
    sim{traj_cntr}.ext.input_func_parameters('step_time')=dist_time; % disturbance flicks transcription reg. func. from 0 to 1 
    sim{traj_cntr}.ext.input_func_parameters('slope_duration')=0.1;% disturbance flicks transcription reg. func. from 0 to 1 
    
    sim{traj_cntr}.het.parameters('c_dist')=100; % gene copy number
    sim{traj_cntr}.het.parameters('a_dist')=500; % max. gene transcription rate
    
    % no output protein expression here
    sim{traj_cntr}.het.parameters('c_x')=0; % gene copy number
    sim{traj_cntr}.het.parameters('a_x')=0; % max. gene transcription rate
    
    % integral controller parameters
    sim{traj_cntr}.het.parameters('K_dna(anti)-sens')=7000; % sensor prot.-DNA binding Hill constant
    sim{traj_cntr}.het.parameters('eta_dna(anti)-sens')=1; % sensor prot.-DNA binding Hill coefficient
    
    sim{traj_cntr}.het.parameters('K_dna(amp)-act')=700; % sensor prot.-DNA binding Hill constant
    sim{traj_cntr}.het.parameters('eta_dna(amp)-act')=1; % sensor prot.-DNA binding Hill coefficient
    
    sim{traj_cntr}.het.parameters('kb_anti')=300; % atcuator-annihilator binding rate constant
    sim{traj_cntr}.het.parameters('c_sens')=100;
    sim{traj_cntr}.het.parameters('a_sens')=50; % sensor gene transcription rate
    sim{traj_cntr}.het.parameters('a_anti')=800; % annigilator transcription rate
    sim{traj_cntr}.het.parameters('a_act')=400; % actuator transcription rate
    
    sim{traj_cntr}.het.parameters('c_amp')=100; % integral controller amplifier gene copy number
    sim{traj_cntr}.het.parameters('a_amp')=4000; % integral controller amplifier transcription rate
       
    % push amended parameter values
    sim{traj_cntr}=sim{traj_cntr}.push_het();
    
    % STOCHASTIC simulation parameters
    sim{traj_cntr}.tf=15; % simulation time frame
    sim{traj_cntr}.record_time_step=1e-3;
    sim{traj_cntr}.tau_step=1e-6;
    sim{traj_cntr}.euler_step=1e-7;
    
    % GET the steady state deterministically to avopid waiting for the system to equilibrate in Gillespie simulations
    % deterministic simulation parameters
    sim{traj_cntr}.opt = odeset('reltol',1.e-6,'abstol',1.e-9);
    
    % 24h to get steady state
    tf_original=sim{traj_cntr}.tf;
    sim{traj_cntr}.tf=24;
    
    % no disturbance
    inducer_base_level_original=sim{traj_cntr}.ext.input_func_parameters('inducer_base_level');
    inducer_final_level=sim{traj_cntr}.ext.input_func_parameters('inducer_final_level');
    sim{traj_cntr}.ext.input_func_parameters('inducer_base_level') = 0;
    sim{traj_cntr}.ext.input_func_parameters('inducer_final_level') = 0;
    
    % simulate and get steady state
    sim{traj_cntr}=sim{traj_cntr}.simulate_model();
    
    % restore simulation time frame
    sim{traj_cntr}.tf=tf_original;
    
    % restore distrubance parameters
    sim{traj_cntr}.ext.input_func_parameters('inducer_base_level') = inducer_base_level_original;
    sim{traj_cntr}.ext.input_func_parameters('inducer_final_level') = inducer_final_level;
    
    % get steady state
    det_steady_state = sim{traj_cntr}.x(end,:);
    
    % get steady state l and D
    steady_psens{traj_cntr}=round(det_steady_state(9+sim{traj_cntr}.num_het+1)); % sens is the first gene in the list
    [steady_l{traj_cntr}, steady_D{traj_cntr}] = get_predist(sim{traj_cntr},det_steady_state); % get the pre-disturbance l and D values
    
    % SET the initial condition according to the deterministic steady state
    
    % mRNAs
    sim{traj_cntr}.init_conditions('m_a')=det_steady_state(1);
    sim{traj_cntr}.init_conditions('m_r')=det_steady_state(2);
    % proteins
    sim{traj_cntr}.init_conditions('p_a')=det_steady_state(3);
    sim{traj_cntr}.init_conditions('R')=det_steady_state(4);
    % tRNAs
    sim{traj_cntr}.init_conditions('tc')=det_steady_state(5);
    sim{traj_cntr}.init_conditions('tu')=det_steady_state(6);
    % inactivated ribosomes
    sim{traj_cntr}.init_conditions('Bcm')=det_steady_state(7);
    % nutrient quality and chloramphenicol
    sim{traj_cntr}.init_conditions('s')=det_steady_state(8);
    sim{traj_cntr}.init_conditions('h')=det_steady_state(9);
    
    % heterologous
    x_het=det_steady_state(10 : (9+2*sim{traj_cntr}.num_het+sim{traj_cntr}.num_misc) );
    for j=1:sim{traj_cntr}.num_het
        % mRNA
        sim{traj_cntr}.init_conditions(['m_',sim{traj_cntr}.het.names{j}])=x_het(j);
        % protein
        sim{traj_cntr}.init_conditions(['p_',sim{traj_cntr}.het.names{j}])=x_het(sim{traj_cntr}.num_het+j);
    end
    % miscellaneous
    for j=1:sim{traj_cntr}.num_misc
        sim{traj_cntr}.init_conditions([sim{traj_cntr}.het.misc_names{j}])=x_het(2*sim{traj_cntr}.num_het+j);
    end
    
    % SIMULATE stochastic trajectories
    sim{traj_cntr}=sim{traj_cntr}.generate_het_stoichiometry_matrix();
    sim{traj_cntr} = sim{traj_cntr}.simulate_model_hybrid_tauleap;
    
    dist_time=sim{traj_cntr}.ext.input_func_parameters('step_time'); % time of disturbance
    
    % calculate l and D
    calculated=calc(sim{traj_cntr},sim{traj_cntr}.x,sim{traj_cntr}.t);

    % record trajectories
    ts{traj_cntr}=sim{traj_cntr}.t;
    psens_trajs{traj_cntr}=sim{traj_cntr}.x(:,9+sim{traj_cntr}.num_het+1);
    l_trajs{traj_cntr}=(calculated.ls);
    D_trajs{traj_cntr}=(calculated.Ds);

    %% OPEN LOOP
%     sim_openloop{traj_cntr}=cell_simulator;
% 
%     sim_openloop{traj_cntr}.init_conditions('s')=sim{traj_cntr}.init_conditions('s');
%     
%     sim_openloop{traj_cntr}=sim_openloop{traj_cntr}.load_heterologous_and_external('aif_controller','step_inducer'); % load the het. gene and ext. inp. modules
%     sim_openloop{traj_cntr}=sim_openloop{traj_cntr}.generate_het_stoichiometry_matrix();
%     
%     % disturbance signal parameters
%     sim_openloop{traj_cntr}.ext.input_func_parameters('inducer_base_level')=sim{traj_cntr}.ext.input_func_parameters('inducer_base_level');
%     sim_openloop{traj_cntr}.ext.input_func_parameters('inducer_final_level')=sim{traj_cntr}.ext.input_func_parameters('inducer_final_level');
%     sim_openloop{traj_cntr}.ext.input_func_parameters('step_time')=sim{traj_cntr}.ext.input_func_parameters('step_time');
%     sim_openloop{traj_cntr}.ext.input_func_parameters('slope_duration')=sim{traj_cntr}.ext.input_func_parameters('slope_duration');
%     
%     sim_openloop{traj_cntr}.het.parameters('c_dist')=sim{traj_cntr}.het.parameters('c_dist');
%     sim_openloop{traj_cntr}.het.parameters('a_dist')=sim{traj_cntr}.het.parameters('a_dist');
%     
%     % no output protein expression here
%     sim_openloop{traj_cntr}.het.parameters('c_x')=sim{traj_cntr}.het.parameters('c_x');
%     sim_openloop{traj_cntr}.het.parameters('a_x')=sim{traj_cntr}.het.parameters('a_x');
%     
%     % integral controller parameters
%     sim_openloop{traj_cntr}.het.parameters('K_dna(anti)-sens')=sim{traj_cntr}.het.parameters('K_dna(anti)-sens');
%     sim_openloop{traj_cntr}.het.parameters('eta_dna(anti)-sens')=sim{traj_cntr}.het.parameters('eta_dna(anti)-sens');
%     
%     sim_openloop{traj_cntr}.het.parameters('K_dna(amp)-act')=sim{traj_cntr}.het.parameters('K_dna(amp)-act');
%     sim_openloop{traj_cntr}.het.parameters('eta_dna(amp)-act')=sim{traj_cntr}.het.parameters('eta_dna(amp)-act');
%     
%     sim_openloop{traj_cntr}.het.parameters('kb_anti')=sim{traj_cntr}.het.parameters('kb_anti'); % atcuator-annihilator binding rate constant
%     sim_openloop{traj_cntr}.het.parameters('c_sens')=sim{traj_cntr}.het.parameters('c_sens');
%     sim_openloop{traj_cntr}.het.parameters('a_sens')=sim{traj_cntr}.het.parameters('a_sens'); % sensor gene transcription rate
% 
%     % all other controller genes on zero
%     sim_openloop{traj_cntr}.het.parameters('c_anti')=0; % annigilator transcription rate
%     sim_openloop{traj_cntr}.het.parameters('c_act')=0; % actuator transcription rate
%     sim_openloop{traj_cntr}.het.parameters('c_amp')=0; % integral controller amplifier gene copy number
%        
%     % push amended parameter values
%     sim_openloop{traj_cntr}=sim_openloop{traj_cntr}.push_het();
%     
%     % STOCHASTIC simulation parameters
%     sim_openloop{traj_cntr}.tf=sim{traj_cntr}.tf; % simulation time frame
%     sim_openloop{traj_cntr}.record_time_step=sim{traj_cntr}.record_time_step;
%     sim_openloop{traj_cntr}.tau_step=sim{traj_cntr}.tau_step;
%     
%     % GET the steady state deterministically to avopid waiting for the system to equilibrate in Gillespie simulations
%     % deterministic simulation parameters
%     sim_openloop{traj_cntr}.opt = odeset('reltol',1.e-6,'abstol',1.e-9);
%     
%     % 24h to get steady state
%     tf_original=sim_openloop{traj_cntr}.tf;
%     sim_openloop{traj_cntr}.tf=24;
%     
%     % no disturbance
%     inducer_base_level_original=sim_openloop{traj_cntr}.ext.input_func_parameters('inducer_base_level');
%     inducer_final_level=sim_openloop{traj_cntr}.ext.input_func_parameters('inducer_final_level');
%     sim_openloop{traj_cntr}.ext.input_func_parameters('inducer_base_level') = 0;
%     sim_openloop{traj_cntr}.ext.input_func_parameters('inducer_final_level') = 0;
%     
%     % simulate and get steady state
%     sim_openloop{traj_cntr}=sim_openloop{traj_cntr}.generate_het_stoichiometry_matrix();
%     sim_openloop{traj_cntr}=sim_openloop{traj_cntr}.simulate_model();
%     
%     % restore simulation time frame
%     sim_openloop{traj_cntr}.tf=tf_original;
%     
%     % restore distrubance parameters
%     sim_openloop{traj_cntr}.ext.input_func_parameters('inducer_base_level') = inducer_base_level_original;
%     sim_openloop{traj_cntr}.ext.input_func_parameters('inducer_final_level') = inducer_final_level;
% 
%     % get steady state
%     det_steady_state_openloop = sim_openloop{traj_cntr}.x(end,:);
%     
%     
%     % get steady state l and D
%     steady_psens_openloop{traj_cntr}=round(det_steady_state_openloop(9+sim{traj_cntr}.num_het+1)); % sens is the first gene in the list
%     [steady_l_openloop{traj_cntr}, steady_D_openloop{traj_cntr}] = get_predist(sim_openloop{traj_cntr},det_steady_state_openloop); % get the pre-disturbance l and D values
%     
%     % SET the initial condition according to the deterministic steady state
%     
%     % mRNAs
%     sim_openloop{traj_cntr}.init_conditions('m_a')=det_steady_state_openloop(1);
%     sim_openloop{traj_cntr}.init_conditions('m_r')=det_steady_state_openloop(2);
%     % proteins
%     sim_openloop{traj_cntr}.init_conditions('p_a')=det_steady_state_openloop(3);
%     sim_openloop{traj_cntr}.init_conditions('R')=det_steady_state_openloop(4);
%     % tRNAs
%     sim_openloop{traj_cntr}.init_conditions('tc')=det_steady_state_openloop(5);
%     sim_openloop{traj_cntr}.init_conditions('tu')=det_steady_state_openloop(6);
%     % inactivated ribosomes
%     sim_openloop{traj_cntr}.init_conditions('Bcm')=det_steady_state_openloop(7);
%     % nutrient quality and chloramphenicol
%     sim_openloop{traj_cntr}.init_conditions('s')=det_steady_state_openloop(8);
%     sim_openloop{traj_cntr}.init_conditions('h')=det_steady_state_openloop(9);
%     
%     % heterologous
%     x_het=det_steady_state_openloop(10 : (9+2*sim_openloop{traj_cntr}.num_het+sim_openloop{traj_cntr}.num_misc) );
%     for j=1:sim_openloop{traj_cntr}.num_het
%         % mRNA
%         sim_openloop{traj_cntr}.init_conditions(['m_',sim_openloop{traj_cntr}.het.names{j}])=x_het(j);
%         % protein
%         sim_openloop{traj_cntr}.init_conditions(['p_',sim_openloop{traj_cntr}.het.names{j}])=x_het(sim_openloop{traj_cntr}.num_het+j);
%     end
%     % miscellaneous
%     for j=1:sim_openloop{traj_cntr}.num_misc
%         sim_openloop{traj_cntr}.init_conditions([sim_openloop{traj_cntr}.het.misc_names{j}])=x_het(2*sim_openloop{traj_cntr}.num_het+j);
%     end
%     
%     % SIMULATE stochastic trajectories
%     sim_openloop{traj_cntr} = sim_openloop{traj_cntr}.simulate_model_hybrid_tauleap;
%     
%     dist_time=sim_openloop{traj_cntr}.ext.input_func_parameters('step_time'); % time of disturbance
%     
%     % calculate l and D
%     calculated_openloop=calc(sim_openloop{traj_cntr},sim_openloop{traj_cntr}.x,sim_openloop{traj_cntr}.t);
% 
%     % record trajectories
%     ts_openloop{traj_cntr}=sim_openloop{traj_cntr}.t;
%     psens_trajs_openloop{traj_cntr}=sim_openloop{traj_cntr}.x(:,9+sim_openloop{traj_cntr}.num_het+1);
%     l_trajs_openloop{traj_cntr}=(calculated_openloop.ls);
%     D_trajs_openloop{traj_cntr}=(calculated_openloop.Ds);
end
toc

%% FIND pre-disturbance means to plot relative trajectories
t_in_frame_predist_openloop=(ts_openloop{1}>=5)&(ts_openloop{1}<=7.5);
t_in_frame_predist=(ts{1}>=5)&(ts{1}<=7.5);

% psens - open loop
psens_trajs_openloop_concatenated=cat(2,psens_trajs_openloop{:});
psens_refmean_openloop=mean(psens_trajs_openloop_concatenated(t_in_frame_predist_openloop,:),'all');

% psens - closed loop
psens_trajs_concatenated=cat(2,psens_trajs{:});
psens_refmean=mean(psens_trajs_concatenated(t_in_frame_predist_openloop,:),'all');

% l - open loop
l_trajs_openloop_concatenated=cat(2,l_trajs_openloop{:});
l_refmean_openloop=mean(l_trajs_openloop_concatenated(t_in_frame_predist_openloop,:),'all');

% l - closed loop
l_trajs_concatenated=cat(2,l_trajs{:});
l_refmean=mean(l_trajs_concatenated(t_in_frame_predist,:),'all');

% D - open loop
D_trajs_openloop_concatenated=cat(2,D_trajs_openloop{:});
D_refmean_openloop=mean(D_trajs_openloop_concatenated(t_in_frame_predist_openloop,:),'all');

% D - open loop
D_trajs_concatenated=cat(2,D_trajs{:});
D_refmean=mean(D_trajs_concatenated(t_in_frame_predist,:),'all');


%% FIGURE S7 D - sensor protein conc.

Fd = figure('Position',[0 0 250 195],'Renderer','painters');
set(Fd, 'defaultAxesFontSize', 9)
set(Fd, 'defaultLineLineWidth', 1.25)
hold on
dist_time=0;
% open loop plots
for traj_cntr=1:num_trajs
    t_in_frame=(ts_openloop{traj_cntr}>=5)&(ts_openloop{traj_cntr}<=12.5);
    plot(ts_openloop{traj_cntr}(t_in_frame)-dist_time,psens_trajs_openloop{traj_cntr}(t_in_frame)./psens_refmean_openloop,'Color',[0.6350 0.0780 0.1840 1])
end

% average trajectory
avg_psens_traj_openloop=mean(psens_trajs_openloop_concatenated,2);
plot(ts_openloop{1}(t_in_frame)-dist_time,avg_psens_traj_openloop(t_in_frame)./psens_refmean_openloop,'Color',[0.6350 0.0780 0.1840 1]);

% closed loop plots
for traj_cntr=1:num_trajs
    t_in_frame=(ts{traj_cntr}>=5)&(ts{traj_cntr}<=12.5);
    plot(ts{traj_cntr}(t_in_frame)-dist_time,psens_trajs{traj_cntr}(t_in_frame)./psens_refmean,'Color',[0 0.4470 0.7410 0.05])
end
% average trajectory
avg_psens_traj=mean(psens_trajs_concatenated,2);
plot(ts{1}(t_in_frame)-dist_time,avg_psens_traj(t_in_frame)./psens_refmean,'Color',[0 0.4470 0.7410 1]);

ylim([0.8 1.2])
xlim([ts{1}(find(t_in_frame,1))-dist_time ts{1}(find(t_in_frame,1,'last'))-dist_time])
xticks(-15:2.5:15)

xlabel('t, time since disturbance [h]','FontName','Arial')
ylabel({'p_{sens}:p_{sens}^0, relative', 'sensor prot. conc.'},'FontName','Arial')


grid on
box on
axis square
hold off

%% FIGURE S7 E - growth rate

Fe = figure('Position',[0 0 250 195],'Renderer','painters');
set(Fe, 'defaultAxesFontSize', 9)
set(Fe, 'defaultLineLineWidth', 1.25)
hold on

% open loop plots
for traj_cntr=1:num_trajs
    t_in_frame=(ts_openloop{traj_cntr}>=5)&(ts_openloop{traj_cntr}<=12.5);
    plot(ts_openloop{traj_cntr}(t_in_frame)-dist_time,l_trajs_openloop{traj_cntr}(t_in_frame)./l_refmean_openloop,'Color',[0.6350 0.0780 0.1840 0.01])
end

% average trajectory
avg_l_traj_openloop=mean(l_trajs_openloop_concatenated,2);
plot(ts_openloop{1}(t_in_frame)-dist_time,avg_l_traj_openloop(t_in_frame)./l_refmean_openloop,'Color',[0.6350 0.0780 0.1840 1]);

% closed loop plots
for traj_cntr=1:num_trajs
    t_in_frame=(ts{traj_cntr}>=5)&(ts{traj_cntr}<=12.5);
    plot(ts{traj_cntr}(t_in_frame)-dist_time,l_trajs{traj_cntr}(t_in_frame)./l_refmean,'Color',[0 0.4470 0.7410 0.01])
end

% average trajectory
avg_l_traj=mean(l_trajs_concatenated,2);
plot(ts{1}(t_in_frame)-dist_time,avg_l_traj(t_in_frame)./l_refmean,'Color',[0 0.4470 0.7410 1]);

ylim([0.9 1.1])
xlim([ts{1}(find(t_in_frame,1)-1)-dist_time ts{1}(find(t_in_frame,1,'last')+1)-dist_time])
xticks(-15:2.5:15)

xlabel('t, time since disturbance [h]','FontName','Arial')
ylabel({'\lambda:\lambda^0, relative growth rate'},'FontName','Arial')

grid on
box on
axis square
hold off

%% FIGURE S7 F - RC denominator

Ff = figure('Position',[0 0 250 195],'Renderer','painters');
set(Ff, 'defaultAxesFontSize', 9)
set(Ff, 'defaultLineLineWidth', 1.25)
hold on

% open loop plots
for traj_cntr=1:num_trajs
    t_in_frame=(ts_openloop{traj_cntr}>=5)&(ts_openloop{traj_cntr}<=12.5);
    plot(ts_openloop{traj_cntr}(t_in_frame)-dist_time,D_trajs_openloop{traj_cntr}(t_in_frame)./D_refmean_openloop,'Color',[0.6350 0.0780 0.1840 0.01])
end

% average trajectory
avg_D_traj_openloop=mean(D_trajs_openloop_concatenated,2);
plot(ts_openloop{1}(t_in_frame)-dist_time,avg_D_traj_openloop(t_in_frame)./D_refmean_openloop,'Color',[0.6350 0.0780 0.1840 1]);

% closed loop plots
for traj_cntr=1:num_trajs
    t_in_frame=(ts{traj_cntr}>=5)&(ts{traj_cntr}<=12.5);
    plot(ts{traj_cntr}(t_in_frame)-dist_time,D_trajs{traj_cntr}(t_in_frame)./D_refmean,'Color',[0 0.4470 0.7410 0.01])
end

% average trajectory
avg_D_traj=mean(D_trajs_concatenated,2);
plot(ts{1}(t_in_frame)-dist_time,avg_D_traj(t_in_frame)./D_refmean,'Color',[0 0.4470 0.7410 1]);

ylim([0.9 1.1])
xlim([ts{1}(find(t_in_frame,1)-1)-dist_time ts{1}(find(t_in_frame,1,'last')+1)-dist_time])
xticks(-15:2.5:15)

xlabel('t, time since disturbance [h]','FontName','Arial')
ylabel({'D:D^0, relative RC','denominator value'},'FontName','Arial')

grid on
box on
axis square
hold off

%% CALCULATOR FUNCTION

% finding annihilator transcription regulation func., growth rate, RC
% denominator on the relevant time interval
function calculated=calc(sim,dist,t)
    calculated.ls=zeros(size(t));
    calculated.Fs=zeros(size(t));   
    calculated.Ds=zeros(size(t));
    
    par=sim.parameters;
    for i=1:size(t)
        % STATE VECTOR TO SINGLE VARIABLES
        m_a = dist(i,1);
        m_r = dist(i,2);
        p_a = dist(i,3);
        R = dist(i,4);
        tc = dist(i,5);
        tu = dist(i,6);
        Bcm = dist(i,7);
        s = dist(i,8);
        h = dist(i,9);
        x_het=dist(i,10 : (9+2*sim.num_het) );
    
        % USEFUL PRE-CALCULATIONS
        % translation elongation rate
        e=sim.form.e(par,tc);
    
        % ribosome inactivation rate due to chloramphenicol
        kcmh=par('kcm').*h;
    
        % ribosome dissociation constants
        k_a=sim.form.k(e,par('k+_a'),par('k-_a'),par('n_a'),kcmh);
        k_r=sim.form.k(e,par('k+_r'),par('k-_r'),par('n_r'),kcmh);
        % heterologous genes
        k_het=ones(1,sim.num_het);
        if(sim.num_het>0)
            for j=1:sim.num_het
                k_het(j)=sim.form.k(e,...
                sim.parameters(['k+_',sim.het.names{j}]),...
                sim.parameters(['k-_',sim.het.names{j}]),...
                sim.parameters(['n_',sim.het.names{j}]),...
                kcmh);
            end
        end
    
        T=tc./tu; % ratio of charged to uncharged tRNAs
        D=1+(m_a./k_a+m_r./k_r+sum(x_het(1:sim.num_het)./k_het))./...
            (1-par('phi_q')); % denominator in ribosome competition calculations
        B=R.*(1-1./D); % actively translating ribosomes - INCLUDING Q
    
        % growth rate
        l=sim.form.l(par,e,B);
    
        p_sens=x_het(6); % get sensor conc.
        
        % RECORD ANNIHILATOR REG.
        calculated.Fs(i)=sim.het.regulation(sim.het.names{2},dist(i,:),0); 
    
        % RECORD GROWTH RATE
        calculated.ls(i)=l;
        
        % RECORD RC DENOMINATOR
        calculated.Ds(i)=D;
    end
end

%% PRE-DISTURBANCE GROWTH RATE AND D
%

% calculated from the state of the system right before disturbance
function [l,D]=get_predist(sim,predist_x) 
    par=sim.parameters;
    % STATE VECTOR TO SINGLE VARIABLES
    m_a = predist_x(1);
    m_r = predist_x(2);
    p_a = predist_x(3);
    R = predist_x(4);
    tc = predist_x(5);
    tu = predist_x(6);
    Bcm = predist_x(7);
    s = predist_x(8);
    h = predist_x(9);
    x_het = predist_x(10 : (9+2*sim.num_het) );

    % USEFUL PRE-CALCULATIONS
    % translation elongation rate
    e=sim.form.e(par,tc);

    % ribosome inactivation rate due to chloramphenicol
    kcmh=par('kcm').*h;

    % ribosome dissociation constants
    k_a=sim.form.k(e,par('k+_a'),par('k-_a'),par('n_a'),kcmh);
    k_r=sim.form.k(e,par('k+_r'),par('k-_r'),par('n_r'),kcmh);
    % heterologous genes
    k_het=ones(1,sim.num_het);
    if(sim.num_het>0)
        for j=1:sim.num_het
            k_het(j)=sim.form.k(e,...
            sim.parameters(['k+_',sim.het.names{j}]),...
            sim.parameters(['k-_',sim.het.names{j}]),...
            sim.parameters(['n_',sim.het.names{j}]),...
            kcmh);
        end
    end

    T=tc./tu; % ratio of charged to uncharged tRNAs
    D=1+(m_a./k_a+m_r./k_r+sum(x_het(1:sim.num_het)./k_het))./...
        (1-par('phi_q')); % denominator in ribosome competition calculations
    B=R.*(1-1./D); % actively translating ribosomes - INCLUDING Q

    % growth rate
    l=sim.form.l(par,e,B);    
end

%% NO BURDEN VALUES FUNCTION
%

function [e,Fr,k_a,k_r,k_sens,k_act,k_amp]=get_nb(sim,ss)   
    par=sim.parameters;
    % STATE VECTOR TO SINGLE VARIABLES
    m_a = ss(1);
    m_r = ss(2);
    p_a = ss(3);
    R = ss(4);
    tc = ss(5);
    tu = ss(6);
    Bcm = ss(7);
    s = ss(8);
    h = ss(9);
    x_het=ss(10 : (9+2*sim.num_het) );

    % USEFUL PRE-CALCULATIONS
    % translation elongation rate
    e=sim.form.e(par,tc);

    % ribosome inactivation rate due to chloramphenicol
    kcmh=par('kcm').*h;

    % ribosome dissociation constants
    k_a=sim.form.k(e,par('k+_a'),par('k-_a'),par('n_a'),kcmh);
    k_r=sim.form.k(e,par('k+_r'),par('k-_r'),par('n_r'),kcmh);
    % heterologous genes
    k_het=ones(1,sim.num_het);
    if(sim.num_het>0)
        for j=1:sim.num_het
            k_het(j)=sim.form.k(e,...
            sim.parameters(['k+_',sim.het.names{j}]),...
            sim.parameters(['k-_',sim.het.names{j}]),...
            sim.parameters(['n_',sim.het.names{j}]),...
            kcmh);
        end
    end

    T=tc./tu; % ratio of charged to uncharged tRNAs
    D=1+(m_a./k_a+m_r./k_r+sum(x_het(1:sim.num_het)./k_het))./...
        (1-par('phi_q')); % denominator in ribosome competition calculations
    B=R.*(1-1./D); % actively translating ribosomes - INCLUDING Q

    % growth rate
    l=sim.form.l(par,e,B);

    p_sens=x_het(6); % get sensor conc.
    
    % RECORD RIBOSOMAL GENE TRANSC. REGULATION FUNCTION
    Fr=sim.form.F_r(par,T);

    % RECORD DISSOCIATION CONSTANTS
    k_sens=k_het(1);
    k_act=k_het(3);
    k_amp=k_het(4);
    
end