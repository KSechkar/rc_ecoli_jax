##sim_script.py
# Stochastic simulation of an antithetic integral feedback controller's performance

## IMPORT PACKAGES
import numpy as np
import jax
import jax.numpy as jnp
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts

import time

## SYNTHETIC GENE CIRCUIT IMPORTS
from het_modules.aif_controller import initialise as aif_init, ode as aif_ode, F_calc as aif_F_calc, v as aif_v

## IMPORT CELL AND CIRCUIT SIMULATORS
from jax_cell_simulator import *
from het_modules.aif_controller import initialise as aif_init, ode as aif_ode, F_calc as aif_F_calc, v as aif_v

## MAIN FUNCTION
def main():
    ## SET UP  jax
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)

    ## SPECIFY number of stochastic trajectories
    num_trajs = 1

    timer = time.time()
    ## INITIALISE cell model
    cellmodel_auxil = CellModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes
    par = cellmodel_auxil.default_params()  # get default parameter values
    init_conds = cellmodel_auxil.default_init_conds(par)  # get default initial conditions

    ## LOAD synthetic gene circuit - WITH HYBRID SIMULATION SUPPORT
    ode_with_circuit, circuit_F_calc, par, init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles, circuit_v = cellmodel_auxil.add_circuit(
        aif_init,
        aif_ode,
        aif_F_calc,
        par, init_conds,
        # propensity calculation function for hybrid simulations
        aif_v)

    ## PARAMETERISE the circuit and culture conditions
    init_conds['s'] = 0.5 # nutrient quality
    # disturbance parameters
    par['c_dist'] = 100  # gene copy number
    par['a_dist'] = 500  # promoter strength
    t_dist_on_since_stoch = 7.5 # disturbance timing
    # Integral controller parameters for sensor protein-DNA binding
    par['K_dna(anti)-sens'] = 7000
    par['eta_dna(anti)-sens'] = 1
    # Integral controller parameters for sensor protein-DNA binding
    par['K_dna(amp)-act'] = 700
    par['eta_dna(amp)-act'] = 1
    # Integral controller parameters for actuator-annihilator binding
    par['kb_anti'] = 300
    # Sensor gene concentration and promoter strength
    par['c_sens'] = 100
    par['a_sens'] = 50
    # Annihilator gene concentration and promoter strength
    par['c_anti'] = 100
    par['a_anti'] = 800
    # Actuator gene concentration and promoter strength
    par['c_act'] = 100
    par['a_act'] = 400
    # Integral controller amplifier gene concentration and promoter strength
    par['c_amp'] = 100
    par['a_amp'] = 4000

    ## DEFINE DETERMINISTIC simulation parameters
    tf = (0, 72)  # simulation time frame
    rtol = 1e-6  # relative tolerance for the ODE solver
    atol = 1e-6  # absolute tolerance for the ODE solver

    ## DEFINE STOCHASTIC simulation parameters
    stoch_sim_duration=1e-2 # simulation time frame
    tau_savetimestep = 1e-3  # save time step a multiple of tau
    tau = 1e-6  # simulation time step
    tau_odestep = 1e-7  # number of ODE integration steps in a single tau-leap step (smaller than tau)

    ## CLOSED LOOP: RUN DETERMINISTIC simulation
    sol = ode_sim(par,  # dictionary with model parameters
                  ode_with_circuit,  # ODE function for the cell with synthetic circuit
                  cellmodel_auxil.x0_from_init_conds(init_conds, circuit_genes, circuit_miscs),
                  # initial condition VECTOR
                  len(circuit_genes), len(circuit_miscs), circuit_name2pos,
                  # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                  cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes),
                  # synthetic gene parameters for calculating k values
                  tf, jnp.arange(tf[0], tf[1] + tau_savetimestep/2, tau_savetimestep),
                  rtol = rtol,
                  atol = atol)  # simulation parameters: when to save the system's state, relative and absolute tolerances)   # simulation parameters: time frame, save time step, relative and absolute tolerances
    ts = np.array(sol.ts)
    xs = np.array(sol.ys)
    det_steady_x = sol.ys[-1, :]
    print('closed loop deterministic simulation complete')

    ## CLOSED LOOP: RUN STOCHASTIC simulation
    tf_stoch = (tf[1],tf[1]+stoch_sim_duration) # simulation time frame
    par['t_dist_on']=tf_stoch[0]+t_dist_on_since_stoch # disturbance timing
    mRNA_count_scales, S, x0_tauleap, circuit_synpos2genename, keys0 = tauleap_sim_prep(par, len(circuit_genes),
                                                                                        len(circuit_miscs),
                                                                                        circuit_name2pos, det_steady_x,
                                                                                        key_seeds=jnp.arange(0,num_trajs,1))
    ts_jnp, xs_jnp, final_keys = tauleap_sim(par,  # dictionary with model parameters
                                             circuit_v,  # circuit reaction propensity calculator
                                             x0_tauleap,
                                             # initial condition VECTOR (processed to make sure random variables are appropriate integers)
                                             len(circuit_genes), len(circuit_miscs), circuit_name2pos,
                                             cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes),
                                             # synthetic gene parameters for calculating k values
                                             tf_stoch, tau, tau_odestep, tau_savetimestep,
                                             # simulation parameters: time frame, tau leap step size, number of ode integration steps in a single tau leap step
                                             mRNA_count_scales, S, circuit_synpos2genename,
                                             # mRNA count scaling factor, stoichiometry matrix, synthetic gene number in list of synth. genes to name decoder
                                             keys0)  # starting random number generation key
    # concatenate the results with the deterministic simulation
    ts = np.concatenate((ts, np.array(ts_jnp)))
    xs_first = np.concatenate(
        (xs, np.array(xs_jnp[1])))  # getting the results from the first random number generator key in vmap
    xss = np.concatenate((xs * np.ones((keys0.shape[0], 1, 1)), np.array(xs_jnp)),
                         axis=1)  # getting the results from all vmapped trajectories
    print('closed loop hybrid simulation complete')

    ## CLOSED LOOP: RECORD TRAJECTORIES
    psens_trajs = xss[:, :, circuit_name2pos['p_sens']]  # sensor protein concentrations
    ls=np.zeros_like(psens_trajs)
    Ds=np.zeros_like(psens_trajs)
    for i in range(0,num_trajs):
        _, ls_i, _, _, _, _, Ds_i = cellmodel_auxil.get_e_l_Fr_nu_psi_T_D_Dnohet(ts, xss[i], par, circuit_genes, circuit_miscs,
                                                                   circuit_name2pos)
        ls[i,:]=ls_i
        Ds[i,:]=Ds_i

    ## OPEN LOOP: RUN DETERMINISTIC simulation
    # controller gene concentrations set to zero
    par_openloop = par.copy()
    par_openloop['c_anti'] = 0
    par_openloop['c_act'] = 0
    par_openloop['c_amp'] = 0
    sol = ode_sim(par_openloop,  # dictionary with model parameters
                  ode_with_circuit,  # ODE function for the cell with synthetic circuit
                  cellmodel_auxil.x0_from_init_conds(init_conds, circuit_genes, circuit_miscs),
                  # initial condition VECTOR
                  len(circuit_genes), len(circuit_miscs), circuit_name2pos,
                  # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                  cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes),
                  # synthetic gene parameters for calculating k values
                  tf, jnp.arange(tf[0], tf[1] + tau_savetimestep/2, tau_savetimestep),
                  rtol=rtol,
                  atol=atol)  # simulation parameters: when to save the system's state, relative and absolute tolerances)   # simulation parameters: time frame, save time step, relative and absolute tolerances
    ts_openloop = np.array(sol.ts)
    xs_openloop = np.array(sol.ys)
    det_steady_x_openloop = sol.ys[-1, :]
    print('open loop deterministic simulation complete')

    ## OPEN LOOP: RUN STOCHASTIC simulation
    mRNA_count_scales, S, x0_tauleap, circuit_synpos2genename, keys0 = tauleap_sim_prep(par_openloop, len(circuit_genes),
                                                                                        len(circuit_miscs),
                                                                                        circuit_name2pos, det_steady_x_openloop,
                                                                                        key_seeds=jnp.arange(0,num_trajs,1))
    ts_jnp_openloop, xs_jnp_openloop, final_keys = tauleap_sim(par_openloop,  # dictionary with model parameters
                                                circuit_v,  # circuit reaction propensity calculator
                                                x0_tauleap,
                                                # initial condition VECTOR (processed to make sure random variables are appropriate integers)
                                                len(circuit_genes), len(circuit_miscs), circuit_name2pos,
                                                cellmodel_auxil.synth_gene_params_for_jax(par_openloop, circuit_genes),
                                                # synthetic gene parameters for calculating k values
                                                tf_stoch, tau, tau_odestep, tau_savetimestep,
                                                # simulation parameters: time frame, tau leap step size, number of ode integration steps in a single tau leap step
                                                mRNA_count_scales, S, circuit_synpos2genename,
                                                # mRNA count scaling factor, stoichiometry matrix, synthetic gene number in list of synth. genes to name decoder
                                                keys0)  # starting random number generation key
    # concatenate the results with the deterministic simulation
    ts_openloop = np.concatenate((ts_openloop, np.array(ts_jnp_openloop)))
    xs_first_openloop = np.concatenate(
        (xs_openloop, np.array(xs_jnp_openloop[1])))  # getting the results from the first random number generator key in vmap
    xss_openloop = np.concatenate((xs_openloop * np.ones((keys0.shape[0], 1, 1)), np.array(xs_jnp_openloop)),
                                  axis=1)  # getting the results from all vmapped trajectories
    print('open loop hybrid simulation complete')

    ## OPEN LOOP: RECORD TRAJECTORIES
    psens_trajs_openloop = xss_openloop[:, :, circuit_name2pos['p_sens']]  # sensor protein concentrations
    ls_openloop = np.zeros_like(psens_trajs_openloop)
    Ds_openloop = np.zeros_like(psens_trajs_openloop)
    for i in range(0, num_trajs):
        _, ls_i, _, _, _, _, Ds_i = cellmodel_auxil.get_e_l_Fr_nu_psi_T_D_Dnohet(ts_openloop, xss_openloop[i], par_openloop,
                                                                   circuit_genes, circuit_miscs, circuit_name2pos)
        ls_openloop[i, :] = ls_i
        Ds_openloop[i, :] = Ds_i

    print('Simulation complete in: ', time.time()-timer,' seconds')
    ## FIND PRE-DISTURBANCE MEANS TO PLOT RELATIVE TRAJECTORIES
    ref_predist_times=(0,7.5)
    # p_sens - closed loop
    ref_psens = np.mean(psens_trajs[:,(ts>=ref_predist_times[0]) & (ts<=ref_predist_times[1])])
    # p_sens - open loop
    ref_psens_openloop = np.mean(psens_trajs_openloop[:,(ts_openloop>=ref_predist_times[0]) & (ts_openloop<=ref_predist_times[1])])
    # l - closed loop
    ref_l = np.mean(ls[:,(ts>=ref_predist_times[0]) & (ts<=ref_predist_times[1])])
    # l - open loop
    ref_l_openloop = np.mean(ls_openloop[:,(ts_openloop>=ref_predist_times[0]) & (ts_openloop<=ref_predist_times[1])])
    # D - closed loop
    ref_D = np.mean(Ds[:,(ts>=ref_predist_times[0]) & (ts<=ref_predist_times[1])])
    # D - open loop
    ref_D_openloop = np.mean(Ds_openloop[:,(ts_openloop>=ref_predist_times[0]) & (ts_openloop<=ref_predist_times[1])])

    ## PLOT: SENSOR PROTEIN concentration
    bkplot.output_file('aif_plots.html')

    psens_plot = bkplot.figure(title='Sensor protein concentration',
                               x_axis_label='Time since disturbance (h)',
                               y_axis_label='Concentration (nM)',
                               width=400,
                               height=400,
                               x_range=(-t_dist_on_since_stoch,stoch_sim_duration-t_dist_on_since_stoch),
                               y_range=(0, 1.5 * np.max(psens_trajs)))
    # closed loop trajectories
    for i in range(0, num_trajs):
        psens_plot.line(ts - tf_stoch[0] - t_dist_on_since_stoch,
                        psens_trajs[i]/ref_psens,
                        line_width=1, color='blue', alpha=0.5)
    # open loop trajectories
    for i in range(0, num_trajs):
        psens_plot.line(ts_openloop - tf_stoch[0] - t_dist_on_since_stoch,
                        psens_trajs_openloop[i]/ref_psens_openloop,
                        line_width=1, color='red', alpha=0.5)
    # closed loop mean
    # psens_plot.line(ts - tf_stoch[0] - t_dist_on_since_stoch,
    #                 np.mean(psens_trajs, axis=0)/ref_psens,
    #                 line_width=2, color='blue', legend_label='closed loop')
    # open loop mean
    # psens_plot.line(ts_openloop - tf_stoch[0] - t_dist_on_since_stoch,
    #                 np.mean(psens_trajs_openloop, axis=0)/ref_psens_openloop,
    #                 line_width=2, color='red', legend_label='open loop')
    bkplot.save(psens_plot)

    return

## MAIN CALL


