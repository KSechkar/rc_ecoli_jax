## nojax_aif_controller.m
# Describes heterologous genes expressed in the cell - in Python without JAX
# Here, AN ANTITHETIC INTEGRAL FEEDBACK CONTROLLER maintaining a constant extent of ribosomal
# competition in cells (plus an extra 'distrubing' synthetic gene whose
# expression is regulated by an external input - to test the controller's performance)

## PACKAGE IMPORTS
import numpy as np

## INITIALISE all the necessary parameters to simulate the circuit
def initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['sens',    # burden sensor, activates m_anti exp
             'anti',    # antisense RNA, annihilates m_act
             'act',     # actuator, activates amplifier expression
             'amp',     # amplifier, affects cell burden
             'dist']    # disturbing gene

    miscs = ['bound']  # actuator and annihilator bound to each other and inactivated
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
        name2pos['k_' + genes[i]] = i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)
    for i in range(0, len(genes)):
        name2pos['mscale_' + genes[i]] = i  # mRNA count scaling factors (in mRNA_count_scales, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes:  # gene parameters
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
    # Hill function for p_sens-DNA binding
    default_par['K_dna(anti)-sens'] = 1000  # Hill constant
    default_par['eta_dna(anti)-sens'] = 1  # Hill coefficient

    # Hill function for p_act-DNA binding
    default_par['K_dna(amp)-act'] = 1000  # Hill constant
    default_par['eta_dna(amp)-act'] = 1  # Hill coefficient

    # m_anti-m_act binding
    default_par['kb_anti'] = 300  # binding rate constant
    default_par['b_bound'] = 6  # antisense degradation rate

    # m_anti NOT transcribed
    default_par['k+_anti'] = 0  # i.e. cannot bind ribosomes

    # max transcription rates
    default_par['a_sens'] = 1
    default_par['a_anti'] = 50
    default_par['a_act'] = 25

    # Disturbance
    default_par['a_dist'] = 100 # max transcription rate (arbitrary)
    default_par['t_dist_on'] = 100   # time of activation of disturbance

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


## TRANSCRIPTION REGULATION FUNCTIONS
def F_calc(t ,x, par, name2pos):
    # sensor gene: constitutive
    F_sens=1

    # annihilator RNA: repressed by the sensor protein
    p_sens = x[name2pos['p_sens']]
    F_anti = par['a_anti'] ** par['eta_dna(anti)-sens'] / \
             (par['K_dna(anti)-sens'] ** par['eta_dna(anti)-sens'] + p_sens ** par['eta_dna(anti)-sens'])

    # actuator gene: constitutive
    F_act=1

    # amplifier gene: activated by the actuator protein
    p_act = x[name2pos['p_act']]
    F_amp = p_act ** par['eta_dna(amp)-act'] / \
            (par['K_dna(amp)-act'] ** par['eta_dna(amp)-act'] + p_act ** par['eta_dna(amp)-act'])

    # disturbance gene: activated after certain time
    F_dist = int(t>par['t_dist_on'])

    return [F_sens,    # burden sensor, activates m_anti exp
             F_anti,    # antisense RNA, annihilates m_act
             F_act,     # actuator, activates amplifier expression
             F_amp,     # amplifier, affects cell burden
             F_dist]    # disturbing gene

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
    return [# mRNAs: sensor, annihilator, actuator, amplifier, disturbance
        par['func_sens'] * l * F[name2pos['F_sens']] * par['c_sens'] * par['a_sens'] \
            - (par['b_sens'] + l) * x[name2pos['m_sens']] \
            - par['kcm'] * par['h_ext'] * (x[name2pos['m_sens']] / k_het[name2pos['k_sens']] / D) * R,
        par['func_anti'] * l * F[name2pos['F_anti']] * par['c_anti'] * par['a_anti'] \
            - (par['b_anti'] + l) * x[name2pos['m_anti']] \
            - par['kcm'] * par['h_ext'] * (x[name2pos['m_anti']] / k_het[name2pos['k_anti']] / D) * R \
            - par['kb_anti'] * x[name2pos['m_anti']] * x[name2pos['m_act']], # additional term for m_anti-m_act binding
        par['func_act'] * l * F[name2pos['F_act']] * par['c_act'] * par['a_act'] \
            - (par['b_act'] + l) * x[name2pos['m_act']] \
            - par['kcm'] * par['h_ext'] * (x[name2pos['m_act']] / k_het[name2pos['k_act']] / D) * R \
            - par['kb_anti'] * x[name2pos['m_anti']] * x[name2pos['m_act']], # additional term for m_anti-m_act binding
        par['func_amp'] * l * F[name2pos['F_amp']] * par['c_amp'] * par['a_amp'] \
            - (par['b_amp'] + l) * x[name2pos['m_amp']] \
            - par['kcm'] * par['h_ext'] * (x[name2pos['m_amp']] / k_het[name2pos['k_amp']] / D) * R,
        par['func_dist'] * l * F[name2pos['F_dist']] * par['c_dist'] * par['a_dist'] \
            - (par['b_dist'] + l) * x[name2pos['m_dist']] \
            - par['kcm'] * par['h_ext'] * (x[name2pos['m_dist']] / k_het[name2pos['k_dist']] / D) * R,
        # proteins: sensor, annihilator (will be zero as not translated), actuator, amplifier, disturbance
        (e/par['n_sens']) * (x[name2pos['m_sens']] / k_het[name2pos['k_sens']] / D) * R - (l + par['d_sens']) * x[name2pos['p_sens']],
        (e/par['n_anti']) * (x[name2pos['m_anti']] / k_het[name2pos['k_anti']] / D) * R - (l + par['d_anti']) * x[name2pos['p_anti']],
        (e/par['n_act']) * (x[name2pos['m_act']] / k_het[name2pos['k_act']] / D) * R - (l + par['d_act']) * x[name2pos['p_act']],
        (e/par['n_amp']) * (x[name2pos['m_amp']] / k_het[name2pos['k_amp']] / D) * R - (l + par['d_amp']) * x[name2pos['p_amp']],
        (e/par['n_dist']) * (x[name2pos['m_dist']] / k_het[name2pos['k_dist']] / D) * R - (l + par['d_dist']) * x[name2pos['p_dist']],
        # bound species: actuator and annihilator bound to each other and inactivated
        par['kb_anti'] * x[name2pos['m_anti']] * x[name2pos['m_act']] - (l+par['b_bound']) * x[name2pos['bound']]
    ]

## STOCHASTIC REACTION PROPENSITIES FOR HYBRID TAU-LEAPING SIMULATIONS
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
    return [ # common reactions for all RNAs
        # synthesis, degradation, dilution, chloramphenicol-enabled removal of sensor gene mRNA
        par['func_sens'] * l * F[name2pos['F_sens']] * par['c_sens'] * par['a_sens'] / mRNA_count_scales[name2pos['mscale_sens']],
        par['b_sens'] * x[name2pos['m_sens']] / mRNA_count_scales[name2pos['mscale_sens']],
        l * x[name2pos['m_sens']] / mRNA_count_scales[name2pos['mscale_sens']],
        par['kcm'] * par['h_ext'] * (x[name2pos['m_sens']] / k_het[name2pos['k_sens']] / D) * R / mRNA_count_scales[ name2pos['mscale_sens']],
        # synthesis, degradation, dilution, chloramphenicol-enabled removal of annihilator gene mRNA
        par['func_anti'] * l * F[name2pos['F_anti']] * par['c_anti'] * par['a_anti'] / mRNA_count_scales[name2pos['mscale_anti']],
        par['b_anti'] * x[name2pos['m_anti']] / mRNA_count_scales[name2pos['mscale_anti']],
        l * x[name2pos['m_anti']] / mRNA_count_scales[name2pos['mscale_anti']],
        par['kcm'] * par['h_ext'] * (x[name2pos['m_anti']] / k_het[name2pos['k_anti']] / D) * R / mRNA_count_scales[ name2pos['mscale_anti']],
        # synthesis, degradation, dilution, chloramphenicol-enabled removal of actuator gene mRNA
        par['func_act'] * l * F[name2pos['F_act']] * par['c_act'] * par['a_act'] / mRNA_count_scales[name2pos['mscale_act']],
        par['b_act'] * x[name2pos['m_act']] / mRNA_count_scales[name2pos['mscale_act']],
        l * x[name2pos['m_act']] / mRNA_count_scales[name2pos['mscale_act']],
        par['kcm'] * par['h_ext'] * (x[name2pos['m_act']] / k_het[name2pos['k_act']] / D) * R / mRNA_count_scales[ name2pos['mscale_act']],
        # synthesis, degradation, dilution, chloramphenicol-enabled removal of amplifier gene mRNA
        par['func_amp'] * l * F[name2pos['F_amp']] * par['c_amp'] * par['a_amp'] / mRNA_count_scales[name2pos['mscale_amp']],
        par['b_amp'] * x[name2pos['m_amp']] / mRNA_count_scales[name2pos['mscale_amp']],
        l * x[name2pos['m_amp']] / mRNA_count_scales[name2pos['mscale_amp']],
        par['kcm'] * par['h_ext'] * (x[name2pos['m_amp']] / k_het[name2pos['k_amp']] / D) * R / mRNA_count_scales[ name2pos['mscale_amp']],
        # synthesis, degradation, dilution, chloramphenicol-enabled removal of disturbance gene mRNA
        par['func_dist'] * l * F[name2pos['F_dist']] * par['c_dist'] * par['a_dist'] / mRNA_count_scales[name2pos['mscale_dist']],
        par['b_dist'] * x[name2pos['m_dist']] / mRNA_count_scales[name2pos['mscale_dist']],
        l * x[name2pos['m_dist']] / mRNA_count_scales[name2pos['mscale_dist']],
        par['kcm'] * par['h_ext'] * (x[name2pos['m_dist']] / k_het[name2pos['k_dist']] / D) * R / mRNA_count_scales[ name2pos['mscale_dist']],
        # AIF motif
        # actuator and annihilator binding
        par['kb_anti'] * x[name2pos['m_anti']] * x[name2pos['m_act']] / mRNA_count_scales[name2pos['mscale_anti']],
        # common reactions for all proteins
        # synthesis, degradation, dilution of sensor gene protein
        (e / par['n_sens']) * (x[name2pos['m_sens']] / k_het[name2pos['k_sens']] / D) * R,
        par['d_sens'] * x[name2pos['p_sens']],
        l * x[name2pos['p_sens']],
        # synthesis, degradation, dilution of annihilator gene protein (will all be zero as m_anti not translated)
        (e / par['n_anti']) * (x[name2pos['m_anti']] / k_het[name2pos['k_anti']] / D) * R,
        par['d_anti'] * x[name2pos['p_anti']],
        l * x[name2pos['p_anti']],
        # synthesis, degradation, dilution of actuator gene protein
        (e / par['n_act']) * (x[name2pos['m_act']] / k_het[name2pos['k_act']] / D) * R,
        par['d_act'] * x[name2pos['p_act']],
        l * x[name2pos['p_act']],
        # synthesis, degradation, dilution of amplifier gene protein
        (e / par['n_amp']) * (x[name2pos['m_amp']] / k_het[name2pos['k_amp']] / D) * R,
        par['d_amp'] * x[name2pos['p_amp']],
        l * x[name2pos['p_amp']],
        # synthesis, degradation, dilution of disturbance gene protein
        (e / par['n_dist']) * (x[name2pos['m_dist']] / k_het[name2pos['k_dist']] / D) * R,
        par['d_dist'] * x[name2pos['p_dist']],
        l * x[name2pos['p_dist']],
        # reactions for the actuator-annihilator bound complex
        # degradation and dilution
        par['b_bound'] * x[name2pos['bound']],
        l * x[name2pos['bound']]
    ]