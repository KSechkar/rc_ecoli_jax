%% aif_controller.m
% Describes heterologous genes expressed in the cell
% Here, AN ANTITHETIC INTEGRAL FEEDBACK CONTROLLER maintaining a constant extent of ribosomal
% competition in cells (plus an extra 'distrubing' synthetic gene whose
% expression is regulated by an external input - to test the controller's performance)

%%

classdef aif_controller
    % describe genes and their parameters
    properties (SetAccess = public)
        module_name='aif_controller'; % name of the heterologous gene module
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
        function obj = aif_controller(obj)
            % -------------------------------------------------------------
            % SPECIFY GENE INFO--------------------------------------------

            obj.prerequisite_exts={'constant_inducer','pulse_inducer','step_inducer','oscillating_inducer'}; % disturbing gene expression is chemically induced

            % SPECIFY GENE NAMES HERE
            obj.names={'sens', ... % burden sensor, activates m_anti exp
                'anti', ... % antisense RNA, annihilates m_act
                'act', ... % actuator, activates amplifier expression
                'amp',... % amplifier, affects cell burden
                'dist',... % disturbing gene
                };

            obj.misc_names={'bound'}; % actuator and annihilator bound to each other and inactivated

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
            
            % HIGH COPY-NUMBER PLASMID
            obj.parameters('c_sens')=100;
            obj.parameters('c_act')=100;
            obj.parameters('c_anti')=100;
            obj.parameters('c_amp')=100;
            obj.parameters('c_dist')=100;

            % SPECIFY NON-DEFAULT PARAMETERS HERE
            % Hill function for p_sens-DNA binding
            obj.parameters('K_dna(anti)-sens')=1000; % Hill constant
            obj.parameters('eta_dna(anti)-sens')=1; % Hill coefficient

            % Hill function for p_act-DNA binding
            obj.parameters('K_dna(amp)-act')=1000; % Hill constant
            obj.parameters('eta_dna(amp)-act')=1; % Hill coefficient

            % m_anti-m_act binding
            obj.parameters('kb_anti')=300; % binding rate constant
            obj.parameters('b_bound')=6; % antisense degradation rate

            % m_anti NOT transcribed
            obj.parameters('k+_anti')=0; % i.e. cannot bind ribosomes

            % max transcription rates
            obj.parameters('a_sens')=1;
            obj.parameters('a_anti')=50;
            obj.parameters('a_act')=25;

            % --- Disturbance ---
          
            % max transcription rate (arbitrary)
            obj.parameters('a_dist')=100;

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
            
            % regulating m_anti expression
            if strcmp(gene_name,'anti')
                p_sens=x_het(6); % get sensor protein conc.

                % Hill REPRESSION function
                F = het_par('K_dna(anti)-sens').^het_par('eta_dna(anti)-sens')./ ...
                    (het_par('K_dna(anti)-sens').^het_par('eta_dna(anti)-sens') + p_sens.^het_par('eta_dna(anti)-sens'));

            elseif strcmp(gene_name,'amp')
                p_act=x_het(8); % get sensor protein conc.

                % Hill activation function
                F = p_act.^het_par('eta_dna(amp)-act')./ ...
                    (het_par('K_dna(amp)-act').^het_par('eta_dna(amp)-act') + p_act.^het_par('eta_dna(amp)-act'));

            elseif (strcmp(gene_name,'dist'))
                F=ext_inp(1); % expression proportional to inducer conc.           

            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------
            else
                F=1; % constitutive expression by default
            end
        end

        % any extra terms in mRNA ODEs apart from synthesis & growth dilution
        function extra_term=extra_m_term(obj,gene_name,x,ext_inp)
            het_par=obj.parameters;
            x_het=x( 10 : 9+size(obj.names,2)*2 ); % get heterologous gene info
            misc=x(10+size(obj.names,2)*2:end); % get miscellaneous species info

            % -------------------------------------------------------------
            % SPECIFY TERMS---------------------------------
            
            % extra term for m_act-m_anti binding in actuator's ODE
            if strcmp(gene_name,'act')
                m_anti=x_het(2); % get annihilator concentration
                m_act=x_het(3); % get actuator concentration
                extra_term=-het_par('kb_anti').*m_anti.*m_act; % annihilator and actuator binding

            % etra term for m_act-m_anti binding in anihilator's ODE
            elseif strcmp(gene_name,'anti')
                m_anti=x_het(2); % get annihilator concentration
                m_act=x_het(3); % get actuator concentration
                extra_term=-het_par('kb_anti').*m_anti.*m_act;

            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------

            else
                extra_term=0; % none by default
            end
        end

        % any extra terms in protein ODEs apart from synthesis & growth dilution
        function extra_term=extra_p_term(obj,gene_name,x,ext_inp)

            % -------------------------------------------------------------
            % SPECIFY REGULATORY FUNCTIONS---------------------------------
            
            % no extra terms for protein ODEs in all cases
            if strcmp(gene_name,'sens')
                extra_term=0;
            elseif strcmp(gene_name,'anti')
                extra_term=0;
            elseif strcmp(gene_name,'act')
                extra_term=0;
            elseif strcmp(gene_name,'dist')
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
            
            % we have 1 such species - bound actuator-annihilator pair

            m_anti=x_het(2); % get annihliator conc.
            m_act=x_het(3); % get actuator conc.
%             disp([m_anti, m_act]);

            anti=misc(1);

            dxdt=het_par('kb_anti').*m_anti.*m_act - (het_par('b_bound')+l).*anti; % forms due to actuator and annihilator binding, is diluted by growth and by RNA-degrading machinery
            
            % END OF USER SPEC---------------------------------------------
            % -------------------------------------------------------------
            
        end
    end
end