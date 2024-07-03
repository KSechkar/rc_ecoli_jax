## nojax_no_het.py
# Describes heterologous genes expressed in the cell - in Python without JAX
# here, NO SYNTHETIC GENES PRESENT

## PACKAGE IMPORTS
import numpy as np


## INITIALISE all the necessary parameters to simulate the circuit
# (returning the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette)
def initialise():
    default_par={'cat_gene_present':0} # chloramphenicol resistance gene present
    default_init_conds={}
    genes={}
    miscs={}
    name2pos={'p_cat':0, } # placeholders, will never be used but required for correct execution'}
    circuit_colours={}
    return default_par, default_init_conds, genes, miscs, name2pos, circuit_colours

# transcription regulation functions
def F_calc(t ,x, par, name2pos):
    return []

# ode
def ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # RETURN THE ODE
    return []


# stochastic reaction propensities for hybrid tau-leaping simulations
def v(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            mRNA_count_scales, # scaling factors for mRNA counts
            par,  # system parameters
            name2pos
            ):
    # RETURN THE PROPENSITIES
    return np.array([])
