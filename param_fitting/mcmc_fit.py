'''
PARAMETER_FITTING.PY: Fit parameter values for the
Python/Jax implementation of the coarse-grained resource-aware E.coli model
(Initial conditions and fitted parameter values are passed as x0 values)
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

# CELL MODEL ELEMENTS MODIFIED FOR FITTING -----------------------------------------------------------------------------
# mostly used to treat the fitted parameters as initial conditions (in order to facilitate vectorisation)

# translation elongation rate
def e_calc_fit(par, tc, K_e):
    return par['e_max'] * tc / (tc + K_e)

# tRNA charging rate
def nu_calc_fit(par, tu, s, K_nu, nu_max):
    return nu_max * s * (tu / (tu + K_nu))

# ODE
def ode_fit(t, x, args):
    # unpack the args
    par = args[0]  # model parameters
    circuit_name2pos = args[1]  # gene name - position in circuit vector decoder
    num_circuit_genes = args[2]; num_circuit_miscs = args[3]  # number of genes and miscellaneous species in the circuit
    kplus_het, kminus_het, n_het, d_het = args[4]  # unpack jax-arrayed synthetic gene parameters for calculating k values

    # give the state vector entries meaningful names
    m_a = x[0]  # metabolic gene mRNA
    m_r = x[1]  # ribosomal gene mRNA
    p_a = x[2]  # metabolic proteins
    R = x[3]  # non-inactivated ribosomes
    tc = x[4]  # charged tRNAs
    tu = x[5]  # uncharged tRNAs
    Bcm = x[6]  # chlorampenicol-bound ribosomes
    s = x[7]  # nutrient quality (constant)
    h = x[8]  # chloramphenicol concentration in the medium (constant)
    
    # parameter values passed with x0 for vmapping purposes
    par_ar_aa_ratio = x[9]
    par_K=x[10]
    par_nu_max=x[11]
    par_kcm=x[12]
    
    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = e_calc_fit(par,tc,K_e=par_K)

    # ribosome inactivation due to chloramphenicol
    kcmh = par_kcm * h

    # ribosome dissociation constants
    k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'], kcmh)
    k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'],kcmh)
    k_het = k_calc(e, kplus_het, kminus_het, n_het,kcmh)

    T = tc / tu  # ratio of charged to uncharged tRNAs

    x_het_div_k_het = x[9:9+num_circuit_genes] / k_het  # competitoion for ribosomes from heterologous genes

    # resource competition denominator
    D = 1 + (1 / (1 - par['phi_q']))*(m_a / k_a + m_r / k_r + x_het_div_k_het.sum())
    B = R * (1 - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

    nu = nu_calc_fit(par, tu, s, K_nu=par_K, nu_max=par_nu_max)  # tRNA charging rate

    l = l_calc(par, e, B)  # growth rate

    psi = psi_calc(par, T, l)  # tRNA synthesis rate - AMENDED

    # return dx/dt for the host cell
    return jnp.array([
        # mRNAs
        l * par['c_a'] * par['a_a'] - (par['b_a'] + l) * m_a,
        l * Fr_calc(par,T) * par['c_r'] * par['a_r'] - (par['b_r'] + l) * m_r,
        # metabolic protein p_a
        (e / par['n_a']) * (m_a / k_a / D) * R - l * p_a,
        # ribosomes
        (e / par['n_r']) * (m_r / k_r / D) * R - l * R - kcmh * B,
        # tRNAs
        nu * p_a - l * tc - e * B,
        psi - l * tu - nu * p_a + e * B,
        # ribosomes inactivated by chloramphenicol
        kcmh*B-l*Bcm,
        # nutrient quality assumed constant
        0,
        # chloramphenicol concentration in the medium assumed constant
        0,
        # parameters passed with x0 for vmapping purposes
        0, 0, 0, 0
    ])


# SOS CALCULATION ------------------------------------------------------------------------------------------------------
# return MINUS sum of squared errors between experimental and simulated data (scaled by measurement errors)
#@functools.partial(jax.jit, static_argnums=(1,2))
def minus_sos_for_parvec(parvec,  # vector of parameter values being fitted
                   vmapped_diffeqsolve_forx0s,  # function for getting the steady states
                   get_l_phir_forxs,# function for getting the (l, phi_r) pairs from steady states with given args
                   x0s,  # initial conditions giving rise to data points
                   exp_measurements, exp_errors  # experimental measurements and errors
                   ):
    # get concatenated initial condition + fitted parameter vector
    x0s_with_parvec = jnp.concatenate((x0s, jnp.multiply(np.ones((x0s.shape[0], len(parvec))), parvec)), axis=1)

    # get modelled system's steady states for all the initial conditions
    model_sol=vmapped_diffeqsolve_forx0s(x0s_with_parvec)

    # return MINUS sum of squared differences of predicted and real measurements, each divided by corresponding measurement errors
    return -jnp.sum(jnp.square(jnp.divide(get_l_phir_forxs(model_sol.ys)-exp_measurements, exp_errors)))

# get the (l,phi_r) pairs from steady-state x vectors
# NOTE: no heterologous genes when we fit the model, so the corresponding steps have been scrapped to enable correct jax compilation
def get_l_phir(xs_ss,   # steady-state x vector values
               args  # non-fitted parameters, which we here need for the calculation
               ):
    # unpack the args
    par = args[0]  # model parameters
    circuit_name2pos = args[1]  # gene name - position in circuit vector decoder
    num_circuit_genes = args[2]; num_circuit_miscs = args[3]  # number of genes and miscellaneous species in the circuit
    kplus_het, kminus_het, n_het, d_het = args[4]  # unpack jax-arrayed synthetic gene parameters for calculating k values

    # unpack
    m_a = xs_ss[:,0,0]  # metabolic gene mRNA
    m_r = xs_ss[:,0,1]  # ribosomal gene mRNA
    p_a = xs_ss[:,0,2]  # metabolic proteins
    R = xs_ss[:,0,3]  # non-inactivated ribosomes
    tc = xs_ss[:,0,4]  # charged tRNAs
    tu = xs_ss[:,0,5]  # uncharged tRNAs
    Bcm = xs_ss[:,0,6]  # chloramphenicol-bound ribosomes
    s = xs_ss[:,0,7]  # nutrient quality (constant)
    h = xs_ss[:,0,8]  # chloramphenicol concentration in the medium (constant)
    
    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = e_calc(par, tc)

    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = e_calc(par, tc)

    # ribosome inactivation due to chloramphenicol
    kcmh = par['kcm'] * h

    # ribosome dissociation constants
    k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'], kcmh)
    k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'], kcmh)

    # resource competition denominator
    D = 1 + (1 / (1 - par['phi_q'])) * (m_a / k_a + m_r / k_r)
    B = R * (1 - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

    # GET GROWTH RATES
    ls = l_calc(par, e, B)


    # GET RIBOSOMAL MASS FRACTIONS - including chloramphenicol-inactivated ribosomes
    phi_rs = (R+Bcm)*par['n_r']/par['M']

    # return
    return jnp.stack((ls, phi_rs),axis=1)

# MCMC FITTING ---------------------------------------------------------------------------------------------------------
# run MCMC fitting - NOT JITTED! which allows to benefit from fast pmapping
def run_mcmc_fit(init_parvec,  # initial parameter vector
                 loglike_fun,  # log(likelihood) function, does not need to be scaled
                 bound_min, bound_max,  # bounds for the parameter values
                 stdevs,  # standard deviations for the normal distributions
                 num_steps,  # number of samples to be drawn
                 key,  # key for the jax random number generator
                 save_every_steps = 10000 # save the chain every this many steps
                 ):
    # split the key to get an individual pair (proposal+acceptance) for every step
    keys=jax.random.split(key, 2*num_steps)

    # initialise the array storing the chain and associated log(likelihoods)
    parvecs=np.zeros((num_steps,len(init_parvec)))
    loglikes=np.zeros(num_steps)

    # record the initial parameter vector and its associated log(likelihood)
    parvecs[0,:]=init_parvec
    loglikes[0]=loglike_fun(init_parvec)

    # run the MCMC simulation
    for i in range(1,num_steps):
        parvec,loglike=mcmc_step(parvecs[i-1,:],loglikes[i-1],bound_min,bound_max,stdevs,loglike_fun,keys[2*i:2*i+2,:])  # make a step
        parvecs[i,:] = np.array(parvec); loglikes[i]=loglike    # record the step
        
        # show progress
        if(i%100==0):
            print(i)
            
        # save a backup of the chain
        if(i%save_every_steps==0):
            pickle_file_name = 'fit_outcomes/backup_save.pkl'
            pickle_file = open(pickle_file_name, 'wb')
            pickle.dump((parvecs, loglikes), file=pickle_file)
            pickle_file.close()

    # return
    return parvecs, loglikes

# MCMC step function
def mcmc_step(parvec,  # current parameter vector
              loglike, # current log(likelihood) value
              bound_min, bound_max,  # bounds for the parameter values
              stdevs,  # standard deviations for the normal distributions
              loglike_fun,  # log(likelihood) function, does not need to be scaled
              keys  # TWO key for the jax random number generator first for proposal, then for acceptance
              ):
    # propose a new parameter vector (withv the first key)
    proposed_parvec, propdist_ratio = propose_norm_fold(parvec, bound_min, bound_max, stdevs, keys[0,:])

    # get its log(likelihood)
    proposed_loglike = loglike_fun(proposed_parvec)

    # accept with a probability that's proportional to the ratio of likelihoods
    prob_acceptance = jnp.exp(proposed_loglike-loglike)*propdist_ratio # get the probability of acceptance
    random_draw = jax.random.uniform(keys[1,:])   # make a random draw (with the second key)
    new_parvec=jax.lax.select(random_draw>prob_acceptance,parvec,proposed_parvec)   # get new parameter vector accordingly
    new_loglike=jax.lax.select(random_draw>prob_acceptance,loglike,proposed_loglike)# get new log(likelihhod) accordingly

    # return the new parameter vector and the associated log(likelihood) value
    return new_parvec, new_loglike

# propose a new parameter vector according to a normal distribution (centred around the current value)
# treating out-of-bounds values by folding them (see the DREAM manual)
def propose_norm_fold(parvec,  # current parameter vector value
                      bound_min, bound_max,  # bounds for the parameter values
                      stdevs,  # standard deviations for the normal distributions
                      key  # key for the jax random number generator
                      ):
    # draw a sample from the normal distirbution
    proposed_parvec_nofold = jnp.multiply(jax.random.normal(key,shape=parvec.shape),stdevs)+parvec

    # fold the values that are out of bounds
    proposed_parvec = jnp.remainder(proposed_parvec_nofold - bound_min, bound_max - bound_min) + bound_min

    # find the probability of the proposed value under the normal distribution - or so we would if the ration wasn't just 1 in this case
    #prob_current_to_proposed = jnp.prod(jax.scipy.stats.norm.pdf(proposed_parvec_nofold,parvec,stdevs))

    # return the proposed parameter vector and the ratio of the probabilities for Metropolis-Hastings calculation
    return proposed_parvec, 1   # prob_proposed_to_current/prob_current_to_proposed

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
    cutoff_growthrate=0.3    # data points with growth rates slower than this will not be considered - setting the bar high for now just to check how optimisation works
    # setups and measurements
    dataset = pd.read_csv('../data/growth_rib_fit_notext.csv', header=None).values # read the experimental dataset (eq2 strain of Scott 2010)
    nutr_quals = np.logspace(np.log10(0.08), np.log10(0.5), 6) # nutrient qualities are equally log-spaced points
    read_setups = []  # initialise setups array: (s,h) pairs
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
    steady_state_stop = SteadyStateEvent(rtol=0.01, atol=0.01)  # stop simulation prematurely if steady state is reached
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
                                                    pmapped_diffeqsolve_forx0s, get_l_phir_forxs,
                                                    x0s, exp_measurements,
                                                    exp_errors)  # objective function (returns SOS)

    # PREPARE: SET MCMC PARAMETERS -------------------------------------------------------------------------------------
    # Perpare to run MCMC
    num_steps = 10**5 # define number of MCMC steps
    prng_seed = 0 # define seed for the random number generator
    key = jax.random.PRNGKey(prng_seed) # get a key for the random number generator

    init_parvec=jnp.array([0.953427, 80000, 6000, 0.000353953]) # initial value of the parameter vector
    #bound_min=jnp.zeros(init_parvec.shape)  # minimum boundary: non-negative parameter values
    bound_min = init_parvec/100  # minimum boundary
    bound_max = init_parvec*100  # maximum boundary

    proposal_stdevs=init_parvec/5 # define the normal (with folding) proposal distribution's standard deviation

    # RUN! -------------------------------------------------------------------------------------------------------------
    print('Running MCMC...')
    print('Cutoff growth rate: '+str(cutoff_growthrate)+' Seed: '+str(prng_seed)+'  Steps: '+str(num_steps))
    start_time=time.time()
    parvecs,loglikes = run_mcmc_fit(init_parvec,minus_sos,bound_min,bound_max,proposal_stdevs,num_steps,key) # run MCMC
    print('MCMC run complete! ('+str(num_steps)+' steps; '+str(time.time()-start_time)+' s)')

    # save the outcome of the MCMC run
    pickle_file_name = 'fit_outcomes/mcmc_outcome_cutoff'+str(cutoff_growthrate)+'_seed'+str(prng_seed)+'_'+str(num_steps)+'steps.pkl'
    pickle_file = open(pickle_file_name, 'wb')
    pickle.dump((parvecs,loglikes),file=pickle_file)
    pickle_file.close()
    return

# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()