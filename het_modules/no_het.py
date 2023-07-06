'''
NO_HET.PY - describe heterologous genes expressed in the cell
Here, NO SYNTHETIC GENES
'''
# By Kirill Sechkar

# PACKAGE IMPORTS
import jax.numpy as jnp
import jax.lax as lax


# initialise the circuit, its parameters and plotting colours for the genes
def initialise():
    default_par = {}
    default_init_conds = {}
    genes = {}
    miscs = {}
    name2pos = {}
    circuit_colours = {}
    return default_par, default_init_conds, genes, miscs, name2pos, circuit_colours


# transcription regulation functions
def F_calc(t, x, par, name2pos):
    return jnp.array([])


# ode
def ode(F_calc,  # calculating the transcription regulation functions
        t, x,  # time, cell state, external inputs
        e, l,  # translation elongation rate, growth rate
        R,  # ribosome count in the cell, resource
        k_het, D,
        # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
        par,  # system parameters
        name2pos  # name to position decoder
        ):
    # RETURN THE ODE
    return []
