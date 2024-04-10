%% step_inducer.m
% Describes an external input (e.g. chemical or light stimulus applied to cells)
% Here,  A STEP CHANGE IN CHEM. INDUCER CONCENTRATION

%%

classdef step_inducer
    properties (SetAccess = public)
        module_name='step_inducer'; % name of the external input module
        name; % name of input
        input_func_parameters; % parameters of the input functions
        
        % 'Compatible' heterologous gene expression modules (e.g. if administered 
        % signal is claculated using fluorescent readouts, only modules with F.P. expression can be used with it)
        % If this is left empty, the module is assumed compatible with any set of synthetic genes.
        compatible_hets; 
    end

    % regulatory functions
    methods (Access = public)
        function obj =step_inducer(obj)
            obj.module_name='step_inducer';
            % -------------------------------------------------------------
            % SPECIFY EXTERNAL INPUT INFO----------------------------------
            
            % SPECIFY INPUT NAMES HERE
            obj.name={'inducer'}; % chemical inducer

            % SPECIFY COMPATIBLE HETEROLOGOUS MODULES
            obj.compatible_hets={};


            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------


            % initialise map storing parameterss
            obj.input_func_parameters=containers.Map('KeyType', 'char', ...
                    'ValueType', 'double');

            % -------------------------------------------------------------
            % SPECIFY INPUT INFO--------------------------------------------

            % SPECIFY PARAMETERS HERE
            obj.input_func_parameters('inducer_base_level')=0; % baseline conc of inducer (nM)
            obj.input_func_parameters('inducer_final_level')=100; % baseline conc of inducer (nM)
            obj.input_func_parameters('step_time')=10; % specify start time of the pulse
            obj.input_func_parameters('slope_duration')=0.5; % rather than abrupt step, can have a (brief) slope to avoid integration errors (h)

            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------
        end
        
        % applied input as a function of the system's state x and time t
        function inp = input(obj,x,t)

            % -------------------------------------------------------------
            % SPECIFY EXTERNAL INPUT FUNCTION------------------------------
            ifpar=obj.input_func_parameters;
            if t<ifpar('step_time')
                inp=ifpar('inducer_base_level'); % before the increase, apply baseline concentration
            elseif (t>ifpar('step_time') && t<ifpar('step_time')+ifpar('slope_duration'))
                inp=ifpar('inducer_base_level')+...
                    (ifpar('inducer_final_level')-ifpar('inducer_base_level')).*...
                    (t-ifpar('step_time'))./ifpar('slope_duration'); % linearly increase concentration
            else
                inp=ifpar('inducer_final_level');
            end

            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------
        end

    end
end