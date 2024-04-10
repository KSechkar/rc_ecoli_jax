%% cell_init_conds.m
% default initial conditions for the host cell

%%
function init_conds=cell_init_conds(par)
    init_conds = containers.Map('KeyType', 'char', ...
            'ValueType', 'double');
    
    % mRNA concentrations - non-zero to avoid being stuck at lambda=0
    init_conds('m_a')=1000; % metabolic
    init_conds('m_r')=0.01; % ribosomal
    
    % protein concentrations - start with 50/50 a/R allocation as a convention
    init_conds('p_a')=par('M').*(1-par('phi_q'))./(2.*par('n_a')); % metabolic *
    init_conds('R')=par('M').*(1-par('phi_q'))./(2.*par('n_r')); % ribosomal *

    % tRNA concentrations - 3E-5 abundance units in Chure and Cremer 2022 are equivalent to 80 uM = 80000 nM
    init_conds('tc')=80000; % charged tRNAs
    init_conds('tu')=80000; % free tRNAs

    % concentration of ribosomes inactivated by chloramphenicol
    init_conds('Bcm')=0;

    % nutrient quality s and chloramphenicol concentration h
    init_conds('s')=0.5;
    init_conds('h')=0; % no translation inhibition assumed by default
    
end