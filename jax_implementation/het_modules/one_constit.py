## one_constit.py
# Describes heterologous genes expressed in the cell
# here, ONE CONSTITUTIVE GENE

## PACKAGE IMPORTS
import jax.numpy as jnp
import jax.lax as lax
import numpy as np


## INITIALISE all the necessary parameters to simulate the circuit
# (returning the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette)
def initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['xtra']  # names of genes in the circuit
    miscs = []  # names of miscellaneous species involved in the circuit (e.g. metabolites)
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_xtra']] will return the concentration of mRNA of the gene 'xtra'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] =  i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] =  i  # transcription regulation functions (in F, not x!!!)
    for i in range(0, len(genes)):
        name2pos['mscale_' + genes[i]] =  i  # mRNA count scaling factors (in mRNA_count_scales, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes: # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["#0072BD", "#D95319", "#4DBEEE", "#A2142F", "#FF00FF"]
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles = {'colours': {}, 'dashes': {}}  # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

## transcription regulation functions
def F_calc(t ,x, par, name2pos):
    F_xtra = 1 # constitutive gene
    return jnp.array([F_xtra])

## ODE
def ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # RETURN THE ODE
    return [# mRNAs
            par['func_xtra'] * l * F[name2pos['F_xtra']] * par['c_xtra'] * par['a_xtra'] - (par['b_xtra'] + l) * x[name2pos['m_xtra']] \
                - par['kcm']*par['h_ext']*(x[name2pos['m_xtra']] / k_het[name2pos['k_xtra']] / D) * R,
            # proteins
            (e / par['n_xtra']) * (x[name2pos['m_xtra']] / k_het[name2pos['k_xtra']] / D) * R - (l + par['d_xtra']) * x[name2pos['p_xtra']]
    ]

## stochastic reaction propensities for hybrid tau-leaping simulations
def v(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            mRNA_count_scales, # scaling factors for mRNA counts
            par,  # system parameters
            name2pos
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # RETURN THE PROPENSITIES
    return [
            # synthesis, degradation, dilution of xtra gene mRNA
            par['func_xtra'] * l * F[name2pos['F_xtra']] * par['c_xtra'] * par['a_xtra'] / mRNA_count_scales[name2pos['mscale_xtra']],
            par['b_xtra'] * x[name2pos['m_xtra']] / mRNA_count_scales[name2pos['mscale_xtra']],
            l * x[name2pos['m_xtra']] / mRNA_count_scales[name2pos['mscale_xtra']],
            # mRNA removal due to chloramphenicol action
            par['kcm'] * par['h_ext'] * (x[name2pos['m_xtra']] / k_het[name2pos['k_xtra']] / D) * R / mRNA_count_scales[name2pos['mscale_xtra']],
            # synthesis, degradation, dilution of xtra gene protein
            (e / par['n_xtra']) * (x[name2pos['m_xtra']] / k_het[name2pos['k_xtra']] / D) * R,
            par['d_xtra'] * x[name2pos['p_xtra']],
            l * x[name2pos['p_xtra']]
    ]