%% cell_params.m
% values of the host cell model parameters

%%

function params=cell_params(~)
    params = containers.Map('KeyType', 'char', ...
                'ValueType', 'double');
    
    % GENERAL PARAMETERS
    params('M') = 11.9*10^8; % cell mass (aa) - taken for 1 div/h for an order-of-magnitude-estimate [1]
    params('phi_q')=0.59; % constant housekeeping protein mass fraction [2]

    % GENE EXPRESSION PARAMETERS  
    % metabolic/aminoacylating genes
    params('c_a') = 1; % copy no. (nM) - convention
    params('b_a') = 6; % mRNA decay rate (/h) [3]
    params('k+_a') = 60; % ribosome binding rate (/h/nM) [3]
    params('k-_a') = 60; % ribosome unbinding rate (/h) [3]
    params('n_a') = 300; % protein length (aa) [3]

    % ribosomal gene
    params('c_r') = 1; % copy no. (nM) - convention
    params('b_r') = 6; % mRNA decay rate (/h) [3]
    params('k+_r') = 60; % ribosome binding rate (/h/nM) [3]
    params('k-_r') = 60; % ribosome unbinding rate (/h) [3]
    params('n_r') = 7459; % protein length (aa) [3]

    % ACTIVATION & RATE FUNCTION PARAMETERS
    params('e_max')=20*3600; % max elongation rate (aa/h) [4]
    params('psi_max')= 1080000 / 2.5; % max synthesis rate (aa/h) [4]   
    params('tau')= 1; % ppGpp sensitivity (ribosome transc. and tRNA synth. Hill const) [4]

    % used in testing - fix the ppGpp level to make ribosome expression effectively cosntitutive
    params('is_fixed_T') = 0; % 1 if fixed
    params('fixed_T') = 1; % the fixed value of T decsribing the ppGpp level

    % used for simulating nutrient upshifts
    params('is_upshift') = 0; % 1 if simulating a nutrient upshift scenario
    params('s_postshift') = 0; % post-upshift steady-state value of nutirent quality (zero as no upshift by default)

    % FITTED PARAMETERS
    params('a_a') = 394464.6979;  % metabolic gene transcription rate (/h) 
    params('a_r') = 1.0318.*params('a_a'); % ribosomal gene transcription rate (/h) 
    params('nu_max')= 4.0469e+03; % max tRNA amioacylation rate (/h)
    params('K_nut')= 1.2397e+03; %5992.78; % tRNA charging rate Michaelis-Menten constant (nM) 
    params('K_e')= 1.2397e+03; %5992.78; % translation elongation rate Michaelis-Menten constant (nM) 
    params('kcm')= 0.000356139;  % chloramphenical binding rate constant (/h/nM)   
end

%% REFERENCES:
% [1] - Bremer H et al. 2008 Modulation of Chemical Composition and Other Parameters of the Cell at Different Exponential Growth Rates
% [2] - Hui et al. 2015 Quantitative proteomic analysis reveals a simple strategy of global resource allocation in bacteria
% [3] - Weiße AY et al. 2015 Mechanistic links between cellular trade-offs, gene expression, and growth
% [4] - Chure G et al. 2022 An Optimal Regulation of Fluxes Dictates Microbial Growth In and Out of Steady-State