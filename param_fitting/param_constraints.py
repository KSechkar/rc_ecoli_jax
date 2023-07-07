'''
GROWTH_PHENOMENA_PREDICTION.PY: Change the free parameters pairwise to look for possible constraints reducing their number.
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
# multiprocessing - must be imported and handled first!
import os
import multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())

import numpy as np
import jax
import jax.numpy as jnp
import functools
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, SteadyStateEvent
from sklearn.neighbors import KernelDensity

import pickle
import pandas as pd
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts, palettes as bkpalettes, transform as bktransform
from math import pi

import time

# CIRCUIT IMPORTS ------------------------------------------------------------------------------------------------------
# get top path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# actually import circuit modules
from cell_model.cell_model import *
import het_modules.no_het as nocircuit  # import the 'no heterologous genes' module
from param_fitting.mcmc_fit import get_l_phir, minus_sos_for_parvec, ode_fit


# PLOTTING HEATMAPS ----------------------------------------------------------------------------------------------------
def plot_heatmap(loglikes,  # physiological variable to be plotted
                 par_names, # parameter names
                 par_x_range_jnp,  # range of nutrient qualities considered (jnp array)
                 par_y_range_jnp,  # range of heterologous gene expression rates considered (jnp array)
                 colourbar_range = (None,),  # range of the colour scale
                 dimensions=(480,450)  # dimensions of the plot (width, height)
                 ):
    # make sure nutrient qualities and gene concentration are numpy arrays and NOT logs
    par_x_range = np.exp(np.array(par_x_range_jnp))
    par_y_range = np.exp(np.array(par_y_range_jnp))
    
    # get a meshgrid of the parameter ranges
    par_x_mesh, par_y_mesh = np.meshgrid(par_x_range, par_y_range)
    par_x_mesh_ravel = par_x_mesh.ravel()
    par_y_mesh_ravel = par_y_mesh.ravel()

    # unravel the loglikes
    loglikes_ravel=loglikes.ravel()

    # calculate widths and heights for heatmap rectangles - unintuitive due to log scale
    rect_widths_along_x_axis = np.zeros(len(par_x_range))
    rect_widths_along_x_axis[0] = par_x_range[1] - par_x_range[0]
    for i in range(1, len(par_x_range)):
        rect_widths_along_x_axis[i] = ((par_x_range[i] - par_x_range[i - 1]) -
                                       rect_widths_along_x_axis[i - 1] / 2) * 2

    rect_heights_along_y_axis = np.zeros(len(par_y_range))
    rect_heights_along_y_axis[0] = par_y_range[1] - par_y_range[0]
    for i in range(1, len(par_y_range)):
        rect_heights_along_y_axis[i] = ((par_y_range[i] - par_y_range[i - 1]) - rect_heights_along_y_axis[i - 1] / 2) * 2

    rect_widths_ravel = np.zeros(par_x_mesh_ravel.shape)
    rect_heights_ravel = np.zeros(par_y_mesh_ravel.shape)
    for i in range(0, len(par_x_mesh_ravel)):
        par_x_where = np.argwhere(
            par_x_range == par_x_mesh_ravel[i])  # locate the baseline value in the baseline range
        rect_widths_ravel[i] = rect_widths_along_x_axis[par_x_where[0][0]] * 1.25
        par_y_where = np.argwhere(par_y_range == par_y_mesh_ravel[i])  # locate the par_y value in the par_y range
        rect_heights_ravel[i] = rect_heights_along_y_axis[par_y_where[0][0]] * 1.25

    # make a dataframe for the heatmap of guaranteed fold changes
    heatmap_df = pd.DataFrame({par_names[0]: par_x_mesh_ravel, par_names[1]: par_y_mesh_ravel,
                               'log(likelihood)': loglikes_ravel,
                               'rect_width': rect_widths_ravel, 'rect_height': rect_heights_ravel})

    # PLOT guaranteed fold-changes
    figure = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="Maximum-to-baseline expression ratio",
        y_axis_label="Hill coefficient",
        x_range=(min(par_x_range), max(par_x_range)),
        y_range=(min(par_y_range), max(par_y_range)),
        x_axis_type="log",
        y_axis_type="log",
        # title="F_switch value GF-changes",
        tools='pan,box_zoom,reset,save'
    )

    # plot the heatmap itself
    rect = figure.rect(x=par_names[0], y=par_names[1], source=heatmap_df,
                       width='rect_width', height='rect_height',
                       fill_color=bktransform.log_cmap('log(likelihood)',
                                                       bkpalettes.Turbo256,
                                                       low=min(loglikes_ravel),
                                                       high=max(loglikes_ravel)),
                       line_width=0, line_alpha=0)
    # add colour bar
    figure.add_layout(rect.construct_color_bar(
        major_label_text_font_size="8pt",
        formatter=bkmodels.PrintfTickFormatter(format="%d"),
        label_standoff=6,
        border_line_color=None,
        padding=5
        ), 'right')

    # set fonts
    figure.xaxis.axis_label_text_font_size = "8pt"
    figure.xaxis.major_label_text_font_size = "8pt"
    figure.yaxis.axis_label_text_font_size = "8pt"
    figure.yaxis.major_label_text_font_size = "8pt"

    return figure

# MAIN FUNCTION --------------------------------------------------------------------------------------------------------
def main():
    # PREPARE: SET UP JAX ----------------------------------------------------------------------------------------------
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)

    # PREPARE: INITIALISE THE CELL MODEL -----------------------------------------------------------------------------------
    # initialise cell model
    cellmodel_auxil = CellModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes
    par = cellmodel_auxil.default_params()  # get default parameter values
    init_conds = cellmodel_auxil.default_init_conds(par)  # get default initial conditions

    # load synthetic gene circuit
    ode_with_circuit, circuit_F_calc, par, init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles = cellmodel_auxil.add_circuit(
        nocircuit.initialise,
        nocircuit.ode,
        nocircuit.F_calc,
        par, init_conds)  # load the circuit - or here, the absence thereof

    # PREPARE: IMPORT EXPERIMENTAL DATA FROM SCOTT ET AL. 2010 -------------------------------------------------------------
    cutoff_growthrate = 0.3  # data points with growth rates slower than this will not be considered - setting the bar high for now just to check how optimisation works
    # setups and measurements
    dataset = pd.read_csv('../data/growth_rib_fit_notext.csv',
                          header=None).values  # read the experimental dataset (eq2 strain of Scott 2010)
    par_xs = np.logspace(np.log10(0.08), np.log10(0.5), 6)  # nutrient qualities are equally log-spaced points
    read_setups = []  # initialise setups array: (s,h) pairs
    read_measurements = []  # intialise measurements array: (l,phi_r) pairs
    for i in range(0, dataset.shape[0]):
        if (dataset[i, 0] > cutoff_growthrate):
            # inputs
            par_x = par_xs[int(i / 5)]  # records start from worst nutrient quality
            h = dataset[
                    i, 3] * 1000  # all h values for same nutr quality same go one after another. Convert to nM from uM!
            read_setups.append([par_x, h])

            # outputs
            l = dataset[i, 0]  # growth rate (1/h)
            phi_r = dataset[i, 2]  # ribosome mass fraction
            read_measurements.append([l, phi_r])
    setups = jnp.array(read_setups)
    exp_measurements = jnp.array(read_measurements)

    # measurement errors, scaled by 1/sqrt(no. replicates)
    read_unscaled_errors = []  # measurement errors (stdevs of samples) for (l, phi_r)
    read_replicates = []  # replicate numbers for (l, phi_r)
    error_dataset = pd.read_csv('../data/growth_rib_fit_errors_notext.csv',
                                header=None).values  # read the experimental dataset (eq2 strain of Scott 2010)
    for i in range(0, dataset.shape[0]):
        if (dataset[i, 0] > cutoff_growthrate):
            # errors
            read_unscaled_errors.append([error_dataset[i, 0], error_dataset[i, 2]])
            # replicates
            read_replicates.append([error_dataset[i, 4], error_dataset[i, 5]])
    unscaled_errors = jnp.array(read_unscaled_errors)
    replicates = jnp.array(read_replicates)
    # exp_errors = jnp.divide(unscaled_errors,jnp.sqrt(replicates))    # scale the errors
    # OR use errors that we used for matlab fitting
    exp_errors = jnp.ones(unscaled_errors.shape) * jnp.array([[error_dataset[:, 0].mean(), error_dataset[:, 2].mean()]])

    # PREPARE: DEFINE FUNCTIONS USED IN MCMC FITTING -----------------------------------------------------------------------
    # construct initial conditions based on experimental setups -  that is, based on (s,h) pairs
    x0_default = cellmodel_auxil.x0_from_init_conds(init_conds, circuit_genes, circuit_miscs)
    x0s_unswapped = jnp.multiply(np.ones((setups.shape[0], len(x0_default))), x0_default)
    x0s_swapped_s_values = x0s_unswapped.at[:, 7].set(setups[:, 0])  # set s values in x0s
    x0s = x0s_swapped_s_values.at[:, 8].set(setups[:, 1])  # set h values in x0s

    # specify simulation parameters
    tf = (0, 48)  # simulation time frame - assume that the cell is close to steady state after 1000h
    dt_max = 0.1  # maximum integration step
    rtol = 1e-6  # relative tolerance for the ODE solver
    atol = 1e-6  # absolute tolerance for the ODE solver

    # define the objective function in terms of fitted parameter vector
    vector_field = lambda t, y, args: ode_fit(t, y, args)
    term = ODETerm(vector_field)  # ODE term
    args = (
        par,  # model parameters
        circuit_name2pos,  # gene name - position in circuit vector decoder
        len(circuit_genes), len(circuit_miscs),  # number of genes and miscellaneous species in the circuit
        cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes)
        # relevant synthetic gene parameters in jax.array form
    )
    solver = Dopri5()  # solver
    stepsize_controller = PIDController(rtol=rtol, atol=atol)  # step size controller
    steady_state_stop = SteadyStateEvent(rtol=0.001, atol=0.001)  # stop simulation prematurely if steady state is reached
    diffeqsolve_forx0 = lambda x0: diffeqsolve(term, solver,
                                               args=args,
                                               t0=tf[0], t1=tf[1], dt0=0.1, y0=x0,
                                               max_steps=None,
                                               discrete_terminating_event=steady_state_stop,
                                               stepsize_controller=stepsize_controller)  # ODE integrator for given x0

    vmapped_diffeqsolve_forx0s = jax.jit(
        jax.vmap(diffeqsolve_forx0, in_axes=0))  # vmapped ODE integrator for several x0s in parallel
    pmapped_diffeqsolve_forx0s = jax.pmap(diffeqsolve_forx0,
                                          in_axes=0)  # pmapped ODE integrator for several x0s in parallel

    get_l_phir_forxs = lambda xs_ss: get_l_phir(xs_ss,
                                                args)  # getting (l, phi_r) pairs from steady state x vector values
    minus_sos = lambda parvec: minus_sos_for_parvec(parvec,
                                                    vmapped_diffeqsolve_forx0s, get_l_phir_forxs,
                                                    x0s, exp_measurements,
                                                    exp_errors)  # objective function (returns SOS)

    # CONSTRUCT PARAMETER VECTORS FOR PAIRWISE CONSIDERATION -----------------------------------------------------------
    # default parameter vector
    parvec_default = jnp.log(jnp.array([par['a_r']/par['a_a'], par['K_e'], par['nu_max'], par['kcm']]))

    # define testing ranges
    K_range = jnp.linspace(jnp.log(0.1), jnp.log(1.5), 11)+jnp.log(par['K_e'])
    nu_max_range = jnp.linspace(jnp.log(0.1), jnp.log(1.5), 11)+jnp.log(par['nu_max'])

    # SIMULATE ---------------------------------------------------------------------------------------------------------
    simulate  = False
    if(simulate):
        # initialise the output array
        loglikes = np.zeros((K_range.shape[0], nu_max_range.shape[0]))

        for i in range(0,K_range.shape[0]):
            for j in range(0,nu_max_range.shape[0]):
                loglikes[i,j] = minus_sos((parvec_default.at[1].set(K_range[i])).at[2].set(nu_max_range[j]))
                print((i,j))
        pickle_file_name = 'fit_outcomes/param_constraints.pkl'
        pickle_file = open(pickle_file_name, 'wb')
        pickle.dump(loglikes, file=pickle_file)
        pickle_file.close()
    else:
        pickle_file_name = 'fit_outcomes/param_constraints.pkl'
        pickle_file = open(pickle_file_name, 'rb')
        loglikes = pickle.load(pickle_file)
        pickle_file.close()

    # PLOT -------------------------------------------------------------------------------------------------------------
    bkplot.output_file('fit_eval_figures/param_constraints.html')
    hmap_fig=plot_heatmap(loglikes,('K_e=K_nu','nu_max'),K_range,nu_max_range)
    bkplot.save(hmap_fig)

    return

# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()