'''
GROWTH_PHENOMENA_PREDICTION.PY: Generate figures showing how the model predicts experimentally observed growth phenomena
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
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts

import time

# CIRCUIT IMPORTS ------------------------------------------------------------------------------------------------------
# get top path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# actually import circuit modules
from cell_model.cell_model import *
import het_modules.no_het as nocircuit  # import the 'no heterologous genes' module
from param_fitting.mcmc_fit import get_l_phir, minus_sos_for_parvec, ode_fit

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

    # PREPARE: IMPORT EXPERIMENTAL DATA FROM CHURE ET AL. 2023 -------------------------------------------------------------
    # ribosomal mass fractions vs growth rates
    rvl_dataset = pd.read_csv('../data/exp_meas_ribosomes.csv', header=None).values
    rvl_ls = rvl_dataset[:, 0]
    rvl_phirs = rvl_dataset[:, 1]

    # translation elongation rates vs growth rates
    evl_dataset = pd.read_csv('../data/exp_meas_elongation.csv', header=None).values
    evl_ls = evl_dataset[:, 0]
    evl_es = evl_dataset[:, 1]

    # ppGpp levels vs growth rates
    ppgppvl_dataset = pd.read_csv('../data/exp_meas_ppGpp.csv', header=None).values
    ppgppvl_ls = ppgppvl_dataset[:, 0]
    ppgppvl_ppgpps = ppgppvl_dataset[:, 1]

    # PREPARE: IMPORT EXPERIMENTAL DATA FROM SCOTT ET AL. 2010 -------------------------------------------------------------
    cutoff_growthrate=0.3    # data points with growth rates slower than this will not be considered - setting the bar high for now just to check how optimisation works
    # setups and measurements
    dataset = pd.read_csv('../data/growth_rib_fit_notext.csv', header=None).values # read the experimental dataset (eq2 strain of Scott 2010)
    nutr_quals = np.logspace(np.log10(0.08), np.log10(0.5), 6) # nutrient qualities are equally log-spaced points
    read_setups = []  # initialise setups array: (s,h_ext) pairs
    read_measurements = []  # intialise measurements array: (l,phi_r) pairs
    for i in range(0, dataset.shape[0]):
        if (dataset[i, 0] > cutoff_growthrate):
            # inputs
            nutr_qual = nutr_quals[int(i / 5)]  # records start from worst nutrient quality
            h = dataset[i, 3] * 1000  # all h values for same nutr quality same go one after another. Convert to nM from uM!
            read_setups.append([nutr_qual, h])

            # outputs
            l = dataset[i, 0]  # growth rate (1/h)
            phi_r = dataset[i, 2]  # ribosome mass fraction
            read_measurements.append([l, phi_r])
    setups=jnp.array(read_setups)
    exp_measurements=jnp.array(read_measurements)

    # measurement errors, scaled by 1/sqrt(no. replicates)
    read_unscaled_errors = []  # measurement errors (stdevs of samples) for (l, phi_r)
    read_replicates = []  # replicate numbers for (l, phi_r)
    error_dataset = pd.read_csv('../data/growth_rib_fit_errors_notext.csv', header=None).values # read the experimental dataset (eq2 strain of Scott 2010)
    for i in range(0, dataset.shape[0]):
        if (dataset[i, 0] > cutoff_growthrate):
            # errors
            read_unscaled_errors.append([error_dataset[i,0], error_dataset[i,2]])
            # replicates
            read_replicates.append([error_dataset[i,4],error_dataset[i,5]])
    unscaled_errors=jnp.array(read_unscaled_errors)
    replicates=jnp.array(read_replicates)
    # exp_errors = jnp.divide(unscaled_errors,jnp.sqrt(replicates))    # scale the errors
    # OR use errors that we used for matlab fitting
    exp_errors = jnp.ones(unscaled_errors.shape) * jnp.array([[error_dataset[:, 0].mean(), error_dataset[:, 2].mean()]])

    # PREPARE: DEFINE FUNCTIONS USED IN MCMC FITTING -----------------------------------------------------------------------
    # construct initial conditions based on experimental setups -  that is, based on (s,h_ext) pairs
    x0_default=cellmodel_auxil.x0_from_init_conds(init_conds,circuit_genes,circuit_miscs)   # get default x0 value
    x0s_unswapped=jnp.multiply(np.ones((setups.shape[0], len(x0_default))), x0_default)
    x0s_swapped_s_values=x0s_unswapped.at[:,7].set(setups[:,0]) # set s values in x0s
    x0s = x0s_swapped_s_values.at[:,8].set(setups[:,1])# set h_ext values in x0s

    # specify simulation parameters
    tf = (0, 48)  # simulation time frame - assume that the cell is close to steady state after 1000h
    dt_max = 0.1  # maximum integration step
    rtol = 1e-6; atol = 1e-6  # relative and absolute tolerances for the ODE solver

    # define the objective function in terms of fitted parameter vector
    vector_field=lambda t,y,args: ode_fit(t,y,args); term = ODETerm(vector_field)   # ODE term
    args= (
            par,  # model parameters
            circuit_name2pos, # gene name - position in circuit vector decoder
            len(circuit_genes), len(circuit_miscs), # number of genes and miscellaneous species in the circuit
            cellmodel_auxil.synth_gene_params_for_jax(par,circuit_genes) # relevant synthetic gene parameters in jax.array form
           )
    solver = Dopri5()   # solver
    stepsize_controller = PIDController(rtol=rtol, atol=atol)  # step size controller
    steady_state_stop = SteadyStateEvent(rtol=0.001, atol=0.001)  # stop simulation prematurely if steady state is reached
    diffeqsolve_forx0 = lambda x0: diffeqsolve(term, solver,
                                               args=args,
                                               t0=tf[0], t1=tf[1], dt0=0.1, y0=x0,
                                               max_steps=None,
                                               discrete_terminating_event=steady_state_stop,
                                               stepsize_controller=stepsize_controller)  # ODE integrator for given x0

    vmapped_diffeqsolve_forx0s=jax.jit(jax.vmap(diffeqsolve_forx0,in_axes=0))    # vmapped ODE integrator for several x0s in parallel
    pmapped_diffeqsolve_forx0s=jax.pmap(diffeqsolve_forx0,in_axes=0)    # pmapped ODE integrator for several x0s in parallel

    get_l_phir_forxs = lambda xs_ss: get_l_phir(xs_ss,args)     # getting (l, phi_r) pairs from steady state x vector values

    # PLOT: COMPARISON OF FITTED MODEL PREDICTIONS WITH EXPERIMENTAL DATA FROM SCOTT ET AL. 2010 -----------------------
    # get model predictions for the experimental setups
    parvec = jnp.log(jnp.array([par['a_r']/par['a_a'], par['K_e'], par['nu_max'], par['kcm']]))
    x0s_with_parvec = jnp.concatenate((x0s, jnp.multiply(np.ones((x0s.shape[0], len(parvec))), parvec)), axis=1)
    fitted_model_sol = vmapped_diffeqsolve_forx0s(x0s_with_parvec)
    model_predictions = get_l_phir_forxs(fitted_model_sol.ys)
    print('SOS errors for model predictions:'+str(np.sum(np.square(np.subtract(model_predictions,exp_measurements)/exp_errors))))

    # initialise the plot
    bkplot.output_file('figure_plots/model_vs_scott2010.html')
    fvd_fig = bkplot.figure(
        frame_width=640,
        frame_height=480,
        x_axis_label="Growth rate, 1/h",
        y_axis_label="Rib. mass frac.",
        x_range=(0, 1.8),
        y_range=(0, 0.41),
        tools="box_zoom,pan,hover,reset"
    )
    colours = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"] # plot colours
    exp_measurements_forplot=np.array(exp_measurements)
    fitted_predictions_forplot=np.array(model_predictions)
    exp_errors_forplot=np.array(exp_errors)

    #PLOT: COMPARISON OF MODEL PREDICTIONS WITH EXPERIMENTAL DATA FROM CHURE ET AL. 2023 -------------------------------
    # plot data and model predictions by nutrient quality
    colourind = 0
    last_nutr_qual = setups[0, 0]
    for i in range(0, setups.shape[0]):
        if (setups[i, 0] != last_nutr_qual):
            colourind += 1
            last_nutr_qual = setups[i, 0]

        fvd_fig.circle(exp_measurements_forplot[i, 0], exp_measurements_forplot[i, 1], size=10, line_width=2, line_color=colours[colourind],
                      fill_color=(0, 0, 0, 0))
        fvd_fig.x(fitted_predictions_forplot[i, 0], fitted_predictions_forplot[i, 1], size=10, line_width=3, line_color=colours[colourind])
        # add error bars - growth rates
        fvd_fig.segment(x0=exp_measurements_forplot[i, 0] - exp_errors_forplot[i, 0],
                        x1=exp_measurements_forplot[i, 0] + exp_errors_forplot[i, 0],
                        y0=exp_measurements_forplot[i, 1], y1=exp_measurements_forplot[i, 1],
                        line_color=colours[colourind])
        # add error bars - ribosome mass fractions
        fvd_fig.segment(x0=exp_measurements_forplot[i, 0],x1=exp_measurements_forplot[i, 0],
                        y0=exp_measurements_forplot[i, 1] - exp_errors_forplot[i, 1],
                        y1=exp_measurements_forplot[i, 1] + exp_errors_forplot[i, 1],
                        line_color=colours[colourind])

    # ADD GROWTH LAW FITS
    # first law
    xs_1 = [[], [], [], []]
    ys_1 = [[], [], [], []]
    nutrind = 0
    chlorind = 0
    last_nutr_qual = setups[0, 0]
    for i in range(0, len(setups)):
        if (setups[i][0] != last_nutr_qual):
            nutrind += 1
            chlorind = 0
            last_nutr_qual = setups[i][0]
        xs_1[chlorind].append(fitted_predictions_forplot[i][0])
        ys_1[chlorind].append(fitted_predictions_forplot[i][1])
        chlorind += 1
    fit_coeffs = np.zeros((len(xs_1), 2))
    for chlorind in range(0, len(xs_1)):
        if (len(xs_1[chlorind]) >= 2):
            linfit = np.polyfit(xs_1[chlorind], ys_1[chlorind], 1)
            fit_coeffs[chlorind, 0] = linfit[0]
            fit_coeffs[chlorind, 1] = linfit[1]
    dashings = ['dashed', 'dashdot', 'dotted', 'dotdash']
    for chlorind in [3, 2, 1, 0]:
        if (fit_coeffs[chlorind, 1] != 0):
            xpoints = np.linspace(0, xs_1[chlorind][-1] * 1.1, 100)
            ypoints = np.polyval(fit_coeffs[chlorind, :], xpoints)
            fvd_fig.line(xpoints, ypoints, color='black', line_dash=dashings[chlorind], line_width=1)

    # second law
    # group data by nutrient quality
    xs_2 = [[]]
    ys_2 = [[]]
    nutrind = 0
    chlorind = 0
    last_nutr_qual = setups[0, 0]
    for i in range(0, len(setups)):
        if (setups[i, 0] != last_nutr_qual):
            nutrind += 1
            last_nutr_qual = setups[i, 0]
            xs_2.append([])
            ys_2.append([])
        xs_2[nutrind].append(fitted_predictions_forplot[i, 0])
        ys_2[nutrind].append(fitted_predictions_forplot[i, 1])
    fit_coeffs = np.zeros((len(xs_2), 2))
    for nutrind in range(0, len(xs_2)):
        linfit = np.polyfit(xs_2[nutrind], ys_2[nutrind], 1)
        fit_coeffs[nutrind, 0] = linfit[0]
        fit_coeffs[nutrind, 1] = linfit[1]
    for nutrind in np.flip(range(0, len(xs_2))):
        if (fit_coeffs[nutrind, 0] != 0):
            xpoints = np.linspace(0, xs_2[nutrind][0] * 1.1, 100)
            ypoints = np.polyval(fit_coeffs[nutrind, :], xpoints)
            fvd_fig.line(xpoints, ypoints, line_color=colours[nutrind], line_dash='solid', line_width=1)

    # save plot
    bkplot.save(fvd_fig)

    # MORE REALITY VS MODEL PREDICTION COMPARISONS: GET MODEL PREDICTIONS FOR THEM -----------------------------------------
    # define nutrient qualities
    vl_nutr_quals=np.logspace(-2,0,10)

    # initialsie output arrays
    vl_ls=np.zeros(len(vl_nutr_quals))
    vl_phirs=np.zeros(len(vl_nutr_quals))
    vl_es=np.zeros(len(vl_nutr_quals))
    vl_ppgpps=np.zeros(len(vl_nutr_quals))

    # time frame for comparison simulations - longer than the one used for fitting to avoid showing non-steady state values
    vl_tf=[0,48]

    vector_field = lambda t, y, args: ode_with_circuit(t, y, args)
    term = ODETerm(vector_field)
    for i in range(0, len(vl_nutr_quals)):
        # define initial and culture conditions
        vl_x0 = jnp.array(cellmodel_auxil.x0_from_init_conds(init_conds, circuit_genes, circuit_miscs)
                          ).at[7].set(vl_nutr_quals[i]) # define initial and culture conditions

        # simulate the model
        sol = diffeqsolve(term, solver,
                    args=args,
                    t0=vl_tf[0], t1=vl_tf[1], dt0=0.1, y0=vl_x0,
                    discrete_terminating_event=steady_state_stop,
                    max_steps=None,
                    stepsize_controller=stepsize_controller) # integrate the ODE

        # retrieve the steady-state growth rate, ribosomal mass fraction, translation elongation rate and ppGpp level
        e,l,_,_,_,T,_,_=cellmodel_auxil.get_e_l_Fr_nu_psi_T_D_Dnohet(sol.ts, sol.ys,
                                                                     par,
                                                                     circuit_genes, circuit_miscs, circuit_name2pos)
        vl_es[i]=np.array(e)[0]  # record steady-state translation elongation rate
        vl_ls[i]=np.array(l)[0]  # record steady-state growth rate
        vl_ppgpps[i] = 1 / np.array(T)[0] # record (relative) steady-state ppGpp level
        vl_phirs[i]=np.array(sol.ys)[0,3]*par['n_r']/par['M'] # record steady-state ribosomal mass fraction

        print(i)

    # MORE REALITY VS MODEL PREDICTION COMPARISONS: MAKE PLOTS -------------------------------------------------------------
    bkplot.output_file('figure_plots/model_vs_chure2023.html')
    # plot ribosomal mass fractions vs growth rates
    rvl_fig = bkplot.figure(frame_width=400,
                            frame_height=400,
                            x_axis_label='Growth rate [1/h]',
                            y_axis_label='Ribosomal mass fraction',
                            x_range=(0,2),
                            y_range=(0,0.3)
                            )
    rvl_fig.circle(rvl_ls, rvl_phirs, size=3, line_width=0, line_color='black', fill_color='blue',legend_label='Exp. data')
    rvl_fig.line(vl_ls, vl_phirs, line_width=2, line_color='red', legend_label='Model')
    rvl_fig.legend.location = 'bottom_right'

    # plot translation elongation rates vs growth rates
    evl_fig = bkplot.figure(frame_width=400,
                            frame_height=400,
                            x_axis_label='Growth rate [1/h]',
                            y_axis_label='Translation elongation rate [aa/s]',
                            x_range=(0,2),
                            y_range=(0,20)
                            )
    evl_fig.circle(evl_ls, evl_es, size=3, line_width=0, line_color='black', fill_color='blue',legend_label='Exp. data')
    evl_fig.line(vl_ls, vl_es/3600, line_width=2, line_color='red', legend_label='Model')
    evl_fig.legend.location = 'bottom_right'

    # plot ppGpp levels vs growth rates - need to get a reference ppGpp level first
    distance_to_ref = 100  # initialise the distance to reference growth rate with unreasonably high number
    for i in range(0,len(vl_ls)):
        if (abs(vl_ls[i] - 1) < abs(distance_to_ref)):  # if the current distance is the smallest so far, this is the new reference
            distance_to_ref = vl_ls[i] - 1
            closest_i = i
    ppgpp_ref =vl_ppgpps[closest_i]
    ppgppvl_fig = bkplot.figure(frame_width=400,
                                frame_height=400,
                                x_axis_label='Growth rate [1/h]',
                                y_axis_label='Relative ppGpp level',
                                x_range=(0, 2),
                                y_range=(0.1, 13),
                                y_axis_type='log'
                                )
    ppgppvl_fig.circle(ppgppvl_ls, ppgppvl_ppgpps, size=3, line_width=0, line_color='black', fill_color='blue',legend_label='Exp. data')
    ppgppvl_fig.line(vl_ls, vl_ppgpps/ppgpp_ref, line_width=2, line_color='red', legend_label='Model')
    # save plots
    bkplot.save(bklayouts.grid([[rvl_fig,evl_fig],[ppgppvl_fig,None]]))
    return

# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()


