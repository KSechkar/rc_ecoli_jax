%% no_ext.m
% Describes an external input (e.g. chemical or light stimulus applied to cells)
% Here, NO EXTERNAL INPUT

% Due to being 'empty', this script can be copied and amended as a template for 
% describing other input fucntions. The 'SPECIFY INPUT INFO'-'END OF USER
% SPEC' brackets mark areas where the case-specific information should be added

%%

classdef no_ext
    properties (SetAccess = public)
        module_name='no_ext'; % name of the external input module
        name; % name of input
        input_func_parameters; % parameters of the input functions
        
        % 'Compatible' heterologous gene expression modules (e.g. if administered 
        % signal is claculated using fluorescent readouts, only modules with F.P. expression can be used with it)
        % If this is left empty, the module is assumed compatible with any set of synthetic genes.
        compatible_hets; 
    end

    methods (Access = public)

        % CONSTRUCTOR
        function obj = no_ext(obj)
            % -------------------------------------------------------------
            % SPECIFY INPUT INFO----------------------------------
            
            % SPECIFY INPUT NAMES HERE
            obj.name={}; % no input

            % SPECIFY COMPATIBLE HETEROLOGOUS MODULES
            obj.compatible_hets={};


            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------


            % initialise map storing parameters
            obj.input_func_parameters=containers.Map('KeyType', 'char', ...
                    'ValueType', 'double'); % none as there is no input

            % -------------------------------------------------------------
            % SPECIFY GENE INFO--------------------------------------------

            % SPECIFY PARAMETERS HERE

            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------
        end
        

        % applied input as a function of the system's state x and time t
        function inp = input(obj,x,t)

            % -------------------------------------------------------------
            % SPECIFY EXTERNAL INPUT FUNCTION------------------------------

            inp=[];

            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------
        end

    end
end