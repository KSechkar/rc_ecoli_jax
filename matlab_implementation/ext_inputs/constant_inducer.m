%% constant_inducer.m
% Describes an external input (e.g. chemical or light stimulus applied to cells)
% Here, CONSTANT LEVEL OF A CHEMICAL INDUCER

%%

classdef constant_inducer
    properties (SetAccess = public)
        module_name='constant_inducer'; % name of the external input module
        name; % name of input
        input_func_parameters; % parameters of the input functions
        
        % 'Compatible' heterologous gene expression modules (e.g. if administered 
        % signal is calculated using fluorescent readouts, only modules with F.P. expression can be used with it)
        % If this is left empty, the module is assumed compatible with any set of synthetic genes.
        compatible_hets; 
        
    end

    methods (Access = public)

        % CONSTRUCTOR
        function obj =constant_inducer(obj)
            obj.module_name='constant_inducer';
            % -------------------------------------------------------------
            % SPECIFY INPUT INFO-------------------------------------------
            
            % SPECIFY INPUT NAMES HERE
            obj.name={'inducer'}; % chemical inducer

            % SPECIFY COMPATIBLE HETEROLOGOUS MODULES
            obj.compatible_hets={};


            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------


            % default parameters
            obj.input_func_parameters=containers.Map('KeyType', 'char', ...
                    'ValueType', 'double');

            % -------------------------------------------------------------
            % SPECIFY INPUT INFO--------------------------------------------

            % SPECIFY PARAMETERS HERE
            obj.input_func_parameters('inducer_level')=100; % constant conc of inducer (nM)

            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------
        end
        
        % applied input as a function of the system's state x and time t
        function inp = input(obj,x,t)

            % -------------------------------------------------------------
            % SPECIFY EXTERNAL INPUT FUNCTION------------------------------

            inp=obj.input_func_parameters('inducer_level'); % constant conc of inducer (nM)

            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------
        end

    end
end