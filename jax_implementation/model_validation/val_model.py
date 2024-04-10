## val_model.py
# validate the predictions of the JAX implementation of the model against experimental data

## IMPORT PACKAGES
# multiprocessing - must be imported and handled first!
import os
import multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())

import numpy as np
import jax
import jax.numpy as jnp

import pandas as pd
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts

import time

## SET UP JAX
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
print(jax.lib.xla_bridge.get_backend().platform)

## IMPORT CELL AND CIRCUIT SIMULATORS
import sys, os
sys.path.append(os.path.abspath('..'))
from jax_implementation.jax_cell_simulator import *
from jax_implementation.het_modules.no_het import initialise as nohet_init, ode as nohet_ode, F_calc as nohet_F_calc, v as nohet_v

## MAIN FUNCTION
def main():
    ## PREPARE: INITIALISE THE CELL MODEL
    cellmodel_auxil = CellModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes
    par = cellmodel_auxil.default_params()  # get default parameter values
    init_conds = cellmodel_auxil.default_init_conds(par)  # get default initial conditions
    ode_with_circuit, circuit_F_calc, par, init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles, circuit_v = cellmodel_auxil.add_circuit(
        nohet_init,
        nohet_ode,
        nohet_F_calc,
        par, init_conds)  # load the circuit

    ## PREPARE: IMPORT EXPERIMENTAL DATA FROM CHURE ET AL. 2023
    # ribosomal mass fractions vs growth rates
    rvl_dataset = pd.read_csv('data/exp_meas_ribosomes.csv', header=None).values
    rvl_ls = rvl_dataset[:, 0]
    rvl_phirs = rvl_dataset[:, 1]

    # translation elongation rates vs growth rates
    evl_dataset = pd.read_csv('data/exp_meas_elongation.csv', header=None).values
    evl_ls = evl_dataset[:, 0]
    evl_es = evl_dataset[:, 1]

    # ppGpp levels vs growth rates
    ppgppvl_dataset = pd.read_csv('data/exp_meas_ppGpp.csv', header=None).values
    ppgppvl_ls = ppgppvl_dataset[:, 0]
    ppgppvl_ppgpps = ppgppvl_dataset[:, 1]

    ## PREPARE: IMPORT EXPERIMENTAL DATA FROM SCOTT ET AL. 2010
    cutoff_growthrate=0.3    # data points with growth rates slower than this will not be considered - setting the bar high for now just to check how optimisation works
    # setups and measurements
    dataset = pd.read_csv('data/growth_rib_fit_notext.csv', header=None).values # read the experimental dataset (eq2 strain of Scott 2010)
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

    # measurement errors - average over all measurements
    read_unscaled_errors = []  # measurement errors (stdevs of samples) for (l, phi_r)
    read_replicates = []  # replicate numbers for (l, phi_r)
    error_dataset = pd.read_csv('data/growth_rib_fit_errors_notext.csv', header=None).values # read the experimental dataset (eq2 strain of Scott 2010)
    exp_errors = jnp.ones(exp_measurements.shape) * jnp.array([[error_dataset[:, 0].mean(), error_dataset[:, 2].mean()]])

    ## SPECIFY SIMULATION PARAMETERS
    tf = (0, 480)  # simulation time frame - assume that the cell is close to steady state after 1000h
    rtol = 1e-6; atol = 1e-6  # relative and absolute tolerances for the ODE solver

    ## SIMULATE
    model_predictions = []
    for i in range(0,len(setups)):
        # set the medium nutrient quality
        sim_init_conds = cellmodel_auxil.default_init_conds(par)  # get default initial conditions
        sim_init_conds['s'] = setups[i, 0]

        # set the external chloramphenicol conc.
        sim_par = par.copy()
        sim_par['h_ext'] = setups[i, 1]

        sol = ode_sim(sim_par,  # dictionary with model parameters
                      ode_with_circuit,  # ODE function for the cell with synthetic circuit
                      cellmodel_auxil.x0_from_init_conds(sim_init_conds, circuit_genes, circuit_miscs),
                      # initial condition VECTOR
                      len(circuit_genes), len(circuit_miscs), circuit_name2pos,
                      # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                      cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes),
                      # synthetic gene parameters for calculating k values
                      tf, jnp.array([tf[1]]),  # just saving the final (steady) state of the system
                      rtol,
                      atol)  # simulation parameters: time frame, save time step, relative and absolute tolerances
        # get growth rate
        _, ls, _, _,_,_,_ = cellmodel_auxil.get_e_l_Fr_nu_psi_T_D_Dnohet(sol.ts, sol.ys, par, circuit_genes, circuit_miscs, circuit_name2pos)
        l=ls[-1]
        # get ribosomal mass fraction
        phi_r = float((sol.ys[-1,3]+sol.ys[-1,7])*par['n_r']/par['M'])
        # record
        model_predictions.append([l, phi_r])

    ## PLOT: COMPARISON OF FITTED MODEL PREDICTIONS WITH EXPERIMENTAL DATA (BEING FITTED)
    bkplot.output_file('model_vs_scott2010.html')
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

    ## ADD GROWTH LAW FITS
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

    ## MORE REALITY VS MODEL PREDICTION COMPARISONS: GET MODEL PREDICTIONS FOR THEM
    # define nutrient qualities
    vl_nutr_quals=np.logspace(-2,0,32)

    # initialsie output arrays
    vl_ls=np.zeros(len(vl_nutr_quals))
    vl_phirs=np.zeros(len(vl_nutr_quals))
    vl_es=np.zeros(len(vl_nutr_quals))
    vl_ppgpps=np.zeros(len(vl_nutr_quals))

    # time frame for comparison simulations - longer than the one used for fitting to avoid showing non-steady state values
    vl_tf=[0,48]

    for i in range(0, len(vl_nutr_quals)):
        print(vl_nutr_quals[i])
        # set the medium nutrient quality
        vl_init_conds = cellmodel_auxil.default_init_conds(par)  # get default initial conditions
        vl_init_conds['s'] = vl_nutr_quals[i]  # set nutrient quality

        sol = ode_sim(par,  # dictionary with model parameters
                      ode_with_circuit,  # ODE function for the cell with synthetic circuit
                      cellmodel_auxil.x0_from_init_conds(vl_init_conds, circuit_genes, circuit_miscs),
                      # initial condition VECTOR
                      len(circuit_genes), len(circuit_miscs), circuit_name2pos,
                      # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                      cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes),
                      # synthetic gene parameters for calculating k values
                      tf, jnp.array([tf[1]]),  # just saving the final (steady) state of the system
                      rtol,
                      atol)  # simulation parameters: time frame, save time step, relative and absolute tolerances
        # translation elongation rate, growth rate and inverse of ppGpp level
        es, ls, _, _, _, Ts, _ = cellmodel_auxil.get_e_l_Fr_nu_psi_T_D_Dnohet(sol.ts, sol.ys, par, circuit_genes,
                                                                               circuit_miscs, circuit_name2pos)
        vl_es[i] = es[-1]
        vl_ls[i] = ls[-1]
        vl_ppgpps[i] = 1 / Ts[-1]
        # get ribosomal mass fraction
        vl_phirs[i]=float(sol.ys[-1, 3] * par['n_r'] / par['M'])

    ## MORE REALITY VS MODEL PREDICTION COMPARISONS: MAKE PLOTS
    bkplot.output_file('model_vs_chure2023.html')
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

## MAIN CALL
if __name__ == '__main__':
    main()