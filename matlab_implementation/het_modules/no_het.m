%% no_het.m
% Describes heterologous genes expressed in the cell
% Here, NO HETEROLOGOUS GENES

% Due to being 'empty', this script can be copied and amended as a template for 
% describing other synthetic gene circuits. The 'SPECIFY GENE INFO'-'END OF USER
% SPEC' brackets mark areas where the case-specific information should be added

%%

classdef no_het
    properties (SetAccess = public)
        module_name='no_het'; % name of the heterologous gene module
        names; % names of all heterologous genes
        misc_names; % names of modelled miscellanous species (e.g. compound of interest that het. proteins synthesise)
        parameters; % parameters of heterologous genes
        init_conditions; % initial conditions for all modelled species

        % 'Compatible' external input signal modules (e.g.
        % optogenetic circuits only work with light signals)
        % If this is left empty, the module is assumed compatible with any set of synthetic genes.
        prerequisite_exts;
    end

    methods (Access = public)
        % CONSTRUCTOR
        function obj = no_het(obj)
            obj.module_name='no_het';

            % -------------------------------------------------------------
            % SPECIFY GENE INFO--------------------------------------------

            obj.prerequisite_exts={};

            % SPECIFY GENE AND MISC. SPECIES NAMES HERE
            obj.names={};

            obj.misc_names={};

            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------


            % initialise the map storing all genes' parameters and fill with default gene expression parameters
            obj.parameters=containers.Map('KeyType', 'char', ...
                    'ValueType', 'double');
             for i=1:size(obj.names,2)
                obj.parameters(['c_',obj.names{i}]) = 1; % copy no. (nM)
                obj.parameters(['a_',obj.names{i}]) = 100; % max transcription rate (/h)
                obj.parameters(['b_',obj.names{i}]) = 6; % mRNA decay rate (/h)
                obj.parameters(['k+_',obj.names{i}]) = 60; % ribosome binding rate (/h/nM)
                obj.parameters(['k-_',obj.names{i}]) = 60; % ribosome unbinding rate (/h)
                obj.parameters(['n_',obj.names{i}]) = 300; % protein length (aa)
             end

            % initialise the map storing initial conditions and fill with default values
            obj.init_conditions=containers.Map('KeyType', 'char', ...
                    'ValueType', 'double');
             for i=1:size(obj.names,2)
                obj.init_conditions(['m_',obj.names{i}]) = 0;
                obj.init_conditions(['p_',obj.names{i}]) = 0;
             end
             for i=1:size(obj.misc_names,2)
                obj.init_conditions(obj.misc_names{i}) = 0;
             end


            % -------------------------------------------------------------
            % SPECIFY GENE INFO--------------------------------------------

            % SPECIFY NON-DEFAULT PARAMETERS HERE

            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------
        end
        
        % regulation function for heterologous gene transcription
        % for the gene called gene_name -
        % depends on the system's state x and external input ext_inp
        function F = regulation(obj,gene_name,x,ext_inp)
            het_par=obj.parameters;
            x_het=x( 10 : 9+size(obj.names,2)*2 ); % get heterologous gene info
            misc=x(10+size(obj.names,2)*2:end); % get miscellaneous species info

            % -------------------------------------------------------------
            % SPECIFY REGULATORY FUNCTIONS---------------------------------
            
            % if-else statements for every gene name in obj.names
            if strcmp(gene_name,'')
                F=1;

            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------
            else
                F=1; % constitutive expression by default
            end
        end

        % any extra terms in mRNA ODEs apart from synthesis & growth dilution
        % for the gene called gene_name
        function extra_term=extra_m_term(obj,gene_name,x,ext_inp)
            het_par=obj.parameters;
            x_het=x( 10 : 9+size(obj.names,2)*2 ); % get heterologous gene info
            misc=x(10+size(obj.names,2)*2:end); % get miscellaneous species info

            % -------------------------------------------------------------
            % SPECIFY TERMS---------------------------------

            if strcmp(gene_name,'xtra')
                extra_term=0;

            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------

            else
                extra_term=0; % none by default
            end
        end

        % any extra terms in protein ODEs apart from synthesis & growth dilution
        % for the gene called gene_name
        function extra_term=extra_p_term(obj,gene_name,x,ext_inp)
            het_par=obj.parameters;
            x_het=x( 10 : 9+size(obj.names,2)*2 ); % get heterologous gene info
            misc=x(10+size(obj.names,2)*2:end); % get miscellaneous species info

            % -------------------------------------------------------------
            % SPECIFY REGULATORY FUNCTIONS---------------------------------

            if strcmp(gene_name,'xtra')
                extra_term=0;

            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------
            
            else
                extra_term=0; % none by default
            end
        end

        % ODEs for miscellaneous species in the system
        function dxdt=misc_ode(obj,t,x,ext_inp,l)
            het_par=obj.parameters;
            x_het=x( 10 : 9+size(obj.names,2)*2 ); % get heterologous gene info
            misc=x(10+size(obj.names,2)*2:end); % get miscellaneous species info

            % -------------------------------------------------------------
            % SPECIFY  MISCELLANEOUS SPECIES' ODES-------------------------
            
            dxdt=[]; % no miscellaneous species

            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------
        end
        
    end
end