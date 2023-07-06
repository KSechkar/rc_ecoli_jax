'''
CELL_MODEL.PY: Python/Jax implementation of the coarse-grained resource-aware E.coli model
Class to enable resource-aware simulation of synthetic gene expression in the cell
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import functools
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, SteadyStateEvent
import pandas as pd
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts

import time

# CIRCUIT IMPORTS ------------------------------------------------------------------------------------------------------
# get top path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# actually import circuit modules
import het_modules.no_het as nocircuit


# CELL MODEL FUNCTIONS -------------------------------------------------------------------------------------------------
# Definitions of functions appearing in the cell model ODEs
# apparent mRNA-ribosome dissociation constant
def k_calc(e, kplus, kminus, n, kcmh):
    return (kminus + e / n + kcmh) / kplus

# translation elongation rate
def e_calc(par, tc):
    return par['e_max'] * tc / (tc + par['K_e'])

# growth rate
def l_calc(par, e, B):
    return e * B / par['M']

# tRNA charging rate
def nu_calc(par, tu, s):
    return par['nu_max'] * s * (tu / (tu + par['K_nu']))

# tRNA synthesis rate
def psi_calc(par, T, l):
    return par['psi_max'] * T / (T + par['tau']) * l

# ribosomal gene transcription regulation function
def Fr_calc(par, T):
    return T / (T + par['tau'])


# CELL MODEL AUXILIARIES -----------------------------------------------------------------------------------------------
# Auxiliries for the cell model - set up default parameters and initial conditions, plot simulation outcomes
class CellModelAuxiliary:
    # INITIALISE
    def __init__(self):
        # plotting colours
        self.gene_colours = {'a': "#EDB120", 'r': "#7E2F8E", 'q': '#C0C0C0', # colours for metabolic, riboozyme, housekeeping genes
                        'het': "#0072BD", 'Bcm': "#77AC30"}               # colours for heterologous genes and ribosomes inactivated by chloramphenicol
        self.tRNA_colours = {'tc': "#000000", 'tu': "#ABABAB"} # colours for charged and uncharged tRNAs
        return

    # PROCESS SYNTHETIC CIRCUIT MODULE
    # add synthetic circuit to the cell model
    def add_circuit(self,
                    circuit_initialiser,  # function initialising the circuit
                    circuit_ode,  # function defining the circuit ODEs
                    circuit_F_calc,  # function calculating the circuit genes' transcription regulation functions
                    cellmodel_par, cellmodel_init_conds  # host cell model parameters and initial conditions
                    ):
        # call circuit initialiser
        circuit_par, circuit_init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles = circuit_initialiser()

        # update parameter, initial condition and colour dictionaries
        cellmodel_par.update(circuit_par)
        cellmodel_init_conds.update(circuit_init_conds)

        # join the circuit ODEs with the transcription regulation functions
        circuit_ode_with_F_calc = lambda t,x,e,l,R,k_het,D,par,name2pos: circuit_ode(circuit_F_calc,
                                                                                     t,x,e,l,R,k_het,D,par,name2pos)

        # add the ciorcuit ODEs to that of the host cell model
        cellmodel_ode = lambda t,x,args: ode(t, x, circuit_ode_with_F_calc, args)

        # return updated ode and parameter, initial conditions, circuit gene (and miscellaneous specie) names
        # name - position in state vector decoder and colours for plotting the circuit's time evolution
        return cellmodel_ode, circuit_F_calc, cellmodel_par, cellmodel_init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles

    # package synthetic gene parameters into jax arrays for calculating k values
    def synth_gene_params_for_jax(self, par,    # system parameters
                                  circuit_genes     # circuit gene names
                                  ):
        # initialise parameter arrays
        kplus_het = np.zeros(len(circuit_genes))
        kminus_het = np.zeros(len(circuit_genes))
        n_het = np.zeros(len(circuit_genes))
        d_het = np.zeros(len(circuit_genes))

        # fill parameter arrays
        for i in range(0, len(circuit_genes)):
            kplus_het[i] = par['k+_' + circuit_genes[i]]
            kminus_het[i] = par['k-_' + circuit_genes[i]]
            n_het[i] = par['n_' + circuit_genes[i]]
            d_het[i] = par['d_' + circuit_genes[i]]

        # return as a tuple of arrays
        return (jnp.array(kplus_het), jnp.array(kminus_het), jnp.array(n_het), jnp.array(d_het))

    # SET DEFAULTS
    # set default parameters
    def default_params(self):
        '''
        References for default parameter values:
        [1] - Bremer H et al. 2008 Modulation of Chemical Composition and Other parameters of the Cell at Different Exponential Growth Rates
        [2] - Hui et al. 2015 Quantitative proteomic analxsis reveals a simple strategy of global resource allocation in bacteria
        [3] - Weiße AY et al. 2015 Mechanistic links between cellular trade-offs, gene expression, and growth
        [4] - Chure G et al. 2022 An Optimal Regulation of Fluxes Dictates Microbial Growth In and Out of Steady-State
        [5] - Gutiérrez Mena J et al. 2022 Dynamic cybergenetic control of bacterial co-culture composition via optogenetic feedback
        '''

        params = {} # initialise

        # GENERAL PARAMETERS
        params['M'] = 11.9 * 10 ** 8  # cell mass (aa) - taken for 1 div/h for an order-of-magnitude-estimate [1]
        params['phi_q'] = 0.59  # constant housekeeping protein mass fraction [2]

        # GENE EXPRESSION parAMETERS
        # metabolic/aminoacylating genes
        params['c_a'] = 1.0  # copy no. (nM) - convention
        params['b_a'] = 6.0  # mRNA decay rate (/h) [3]
        params['k+_a'] = 60.0  # ribosome binding rate (/h/nM) [3]
        params['k-_a'] = 60.0  # ribosome unbinding rate (/h) [3]
        params['n_a'] = 300.0  # protein length (aa) [3]

        # ribosomal gene
        params['c_r'] = 1.0  # copy no. (nM) - convention
        params['b_r'] = 6.0  # mRNA decay rate (/h) [3]
        params['k+_r'] = 60.0  # ribosome binding rate (/h/nM) [3]
        params['k-_r'] = 60.0  # ribosome unbinding rate (/h) [3]
        params['n_r'] = 7459.0  # protein length (aa) [3]

        # ACTIVATION & RATE FUNCTION PARAMETERS
        params['e_max'] = 20.0 * 3600.0  # max elongation rate (aa/h) [4]
        params['psi_max'] = 1080000.0/2.5  # max synthesis rate (aa) - estimate from [4] divided by the growth rate in 'very rich media' for which the estimate was made
        params['tau'] = 1.0  # ppGpp sensitivity (ribosome transc. and tRNA synth. Hill const) [4]

        # FITTED PARAMETERS
        params['a_a'] = 3.881e5 # metabolic gene transcription rate (/h)
        params['a_r'] = 0.953427 * params['a_a']  # ribosomal gene transcription rate (/h)
        params['K_nu'] = 5992.78  # tRNA charging rate Michaelis-Menten constant (nM)
        params['K_e'] = 5992.78 # translation elongation rate Michaelis-Menten constant (nM)
        params['nu_max'] = 4165.26  # max tRNA amioacylation rate (/h)
        params['kcm'] = 0.000353953 # chloramphenicol binding rate constant (/h/nM)
        return params

    # set default initial conditions
    def default_init_conds(self, par):
        init_conds = {} # initialise

        # mRNA concentrations - non-zero to avoid being stuck at lambda=0
        init_conds['m_a'] = 1000.0 # metabolic
        init_conds['m_r'] = 0.01  # ribosomal

        # protein concentrations - start with 50/50 a/R allocation as a convention
        init_conds['p_a'] = par['M'] * (1 - par['phi_q']) / (2 * par['n_a'])  # metabolic *
        init_conds['R'] = par['M'] * (1 - par['phi_q']) / (2 * par['n_r'])  # ribosomal *

        # tRNA concentrations - 3E-5 abundance units in Chure and Cremer 2022 are equivalent to 80 uM = 80000 nM
        init_conds['tc'] = 80000.0  # charged tRNAs
        init_conds['tu'] = 80000.0  # free tRNAs

        # chloramphenicol - bound ribosomes
        init_conds['Bcm'] = 0.0

        # nutrient quality s and chloramphenicol concentration h
        init_conds['s'] = 0.5
        init_conds['h'] = 0.0  # no translation inhibition assumed by default
        return init_conds

    # PREPARE FOR SIMULATIONS
    # set default initial condition vector
    def x0_from_init_conds(self,init_conds,circuit_genes,circuit_miscs):
        # NATIVE GENES
        x0 = [
            # mRNAs;
            init_conds['m_a'],  # metabolic gene transcripts
            init_conds['m_r'],  # ribosomal gene transcripts

            # proteins
            init_conds['p_a'],  # metabolic proteins
            init_conds['R'],  # non-inactivated ribosomes

            # tRNAs
            init_conds['tc'],  # charged
            init_conds['tu'],  # uncharged

            # chloramphenicol-bound ribosomes
            init_conds['Bcm'],  # chloramphenicol-bound ribosomes

            # culture medium's nutrient quality and chloramphenicol concentration
            init_conds['s'],  # nutrient quality
            init_conds['h'],  # chloramphenicol levels IN THE CELL
        ]
        # SYNTHETIC CIRCUIT
        for gene in circuit_genes:  # mRNAs
            x0.append(init_conds['m_' + gene])
        for gene in circuit_genes:  # proteins
            x0.append(init_conds['p_' + gene])

        # MISCELLANEOUS SPECIES
        for misc in circuit_miscs:  # miscellanous species
            x0.append(init_conds[misc])


        return jnp.array(x0)

    # PLOT RESULTS, CALCULATE CELLULAR VARIABLES
    # plot protein composition of the cell by mass over time
    def plot_protein_masses(self,ts,xs,
                            par, circuit_genes,  # model parameters, list of circuit genes
                            dimensions=(320,180), tspan=None):
        # set default time span if unspecified
        if(tspan==None):
            tspan = (ts[0],ts[-1])

        # create figure
        mass_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Protein mass, aa",
            x_range=tspan,
            title='Protein masses',
            tools="box_zoom,pan,hover,reset"
        )

        flip_t = np.flip(ts) # flipped time axis for patch plotting

        # plot heterologous protein mass - if there are any heterologous proteins to begin with
        if(len(circuit_genes)!=0):
            bottom_line = np.zeros(xs.shape[0])
            top_line = bottom_line + np.sum(xs[:,9+len(circuit_genes):9+len(circuit_genes)*2]*np.array(self.synth_gene_params_for_jax(par,circuit_genes)[2],ndmin=2),axis=1)
            mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                              line_width=0.5, line_color='black', fill_color=self.gene_colours['het'], legend_label='het')
        else:
            top_line = np.zeros(xs.shape[0])

        # plot mass of inactivated ribosomes
        bottom_line = top_line
        top_line = bottom_line + xs[:,6] * par['n_r']
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                          line_width=0.5, line_color='black', fill_color=self.gene_colours['Bcm'], legend_label='R:h')

        # plot mass of active ribosomes - only if there are any to begin with
        bottom_line = top_line
        top_line = bottom_line + xs[:,3] * par['n_r']
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                          line_width=0.5, line_color='black', fill_color=self.gene_colours['r'], legend_label='R (free)')

        # plot metabolic protein mass
        bottom_line = top_line
        top_line = bottom_line + xs[:,2]*par['n_a']
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                            line_width=0.5, line_color='black', fill_color=self.gene_colours['a'], legend_label='p_a')

        # plot housekeeping protein mass
        bottom_line = top_line
        top_line = bottom_line/(1-par['phi_q'])
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                            line_width=0.5, line_color='black', fill_color=self.gene_colours['q'], legend_label='p_q')

        # add legend
        mass_figure.legend.label_text_font_size = "8pt"
        mass_figure.legend.location = "top_right"

        return mass_figure

    # plot mRNA, protein and tRNA concentrations over time
    def plot_native_concentrations(self,ts,xs,
                            par, circuit_genes,  # model parameters, list of circuit genes
                            dimensions=(320,180), tspan=None):
        # set default time span if unspecified
        if(tspan==None):
            tspan = (ts[0],ts[-1])

        # Create a ColumnDataSource object for the plot
        source = bkmodels.ColumnDataSource(data={
            't': ts,
            'm_a': xs[:,0],  # metabolic mRNA
            'm_r': xs[:,1],  # ribosomal mRNA
            'p_a': xs[:,2],  # metabolic protein
            'R': xs[:,3],  # ribosomal protein
            'tc': xs[:,4],  # charged tRNA
            'tu': xs[:,5],  # uncharged tRNA
            's': xs[:,6],  # nutrient quality
            'h': xs[:,7],  # chloramphenicol concentration
            'm_het': np.sum(xs[:,9:9+len(circuit_genes)], axis=1),  # heterologous mRNA
            'p_het': np.sum(xs[:,9+len(circuit_genes):9+len(circuit_genes)*2], axis=1),  # heterologous protein
        })

        # PLOT mRNA CONCENTRATIONS
        mRNA_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="mRNA conc., nM",
            x_range=tspan,
            title='mRNA concentrations',
            tools="box_zoom,pan,hover,reset"
        )
        mRNA_figure.line(x='t',y='m_a', source=source, line_width=1.5, line_color=self.gene_colours['a'], legend_label='m_a') # plot metabolic mRNA concentrations
        mRNA_figure.line(x='t',y='m_r', source=source, line_width=1.5, line_color=self.gene_colours['r'], legend_label='m_r') # plot ribosomal mRNA concentrations
        mRNA_figure.line(x='t',y='m_het', source=source, line_width=1.5, line_color=self.gene_colours['het'], legend_label='m_het') # plot heterologous mRNA concentrations
        mRNA_figure.legend.label_text_font_size = "8pt"
        mRNA_figure.legend.location = "top_right"

        # PLOT protein CONCENTRATIONS
        protein_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Protein conc., nM",
            x_range=tspan,
            title='Protein concentrations',
            tools="box_zoom,pan,hover,reset"
        )
        protein_figure.line(x='t',y='p_a', source=source, line_width=1.5, line_color=self.gene_colours['a'], legend_label='p_a') # plot metabolic protein concentrations
        protein_figure.line(x='t',y='R', source=source, line_width=1.5, line_color=self.gene_colours['r'], legend_label='R') # plot ribosomal protein concentrations
        protein_figure.line(x='t',y='p_het', source=source, line_width=1.5, line_color=self.gene_colours['het'], legend_label='p_het') # plot heterologous protein concentrations
        protein_figure.legend.label_text_font_size = "8pt"
        protein_figure.legend.location = "top_right"

        # PLOT tRNA CONCENTRATIONS
        tRNA_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="tRNA conc., nM",
            x_range=tspan,
            title='tRNA concentrations',
            tools="box_zoom,pan,hover,reset"
        )
        tRNA_figure.line(x='t',y='tc', source=source, line_width=1.5, line_color=self.tRNA_colours['tc'], legend_label='tc') # plot charged tRNA concentrations
        tRNA_figure.line(x='t',y='tu', source=source, line_width=1.5, line_color=self.tRNA_colours['tu'], legend_label='tu') # plot uncharged tRNA concentrations
        protein_figure.legend.label_text_font_size = "8pt"
        protein_figure.legend.location = "top_right"

        # PLOT EXTRACELLULAR CHLORAMPHENICOL CONCENTRATION
        h_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="h, nM",
            x_range=tspan,
            title='Intracellular chloramphenicol concentration',
            tools="box_zoom,pan,hover,reset"
        )
        h_figure.line(x='t',y='h', source=source, line_width=1.5, line_color=self.gene_colours['Bcm'], legend_label='h') # plot intracellular chloramphenicol concentration

        return mRNA_figure, protein_figure, tRNA_figure, h_figure

    # plot concentrations for the synthetic circuits
    def plot_circuit_concentrations(self, ts, xs,
                                    par, circuit_genes, circuit_miscs, circuit_name2pos,    # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                    circuit_styles, # colours for the circuit plots
                                    dimensions=(320, 180), tspan=None):
        # if no circuitry at all, return no plots
        if(len(circuit_genes)+len(circuit_miscs)==0):
            return None, None, None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # Create a ColumnDataSource object for the plot
        data_for_column={'t': ts} # initialise with time axis
        # record synthetic mRNA and protein concentrations
        for i in range(0,len(circuit_genes)):
            data_for_column['m_'+circuit_genes[i]] = xs[:, 9+i]
            data_for_column['p_'+circuit_genes[i]] = xs[:, 9+len(circuit_genes)+i]
        # record miscellaneous species' concentrations
        for i in range(0,len(circuit_miscs)):
            data_for_column[circuit_miscs[i]] = xs[:, 9+len(circuit_genes)*2+i]
        source = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT mRNA and PROTEIN CONCENTRATIONS (IF ANY)
        if(len(circuit_genes)>0):
            # mRNAs
            mRNA_figure = bkplot.figure(
                frame_width=dimensions[0],
                frame_height=dimensions[1],
                x_axis_label="t, hours",
                y_axis_label="mRNA conc., nM",
                x_range=tspan,
                title='mRNA concentrations',
                tools="box_zoom,pan,hover,reset"
            )
            for gene in circuit_genes:
                mRNA_figure.line(x='t', y='m_'+gene, source=source, line_width=1.5,
                                 line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                                 legend_label='m_'+gene)
            mRNA_figure.legend.label_text_font_size = "8pt"
            mRNA_figure.legend.location = "top_right"

            # proteins
            protein_figure = bkplot.figure(
                frame_width=dimensions[0],
                frame_height=dimensions[1],
                x_axis_label="t, hours",
                y_axis_label="Protein conc., nM",
                x_range=tspan,
                title='Protein concentrations',
                tools="box_zoom,pan,hover,reset"
            )
            for gene in circuit_genes:
                protein_figure.line(x='t', y='p_'+gene, source=source, line_width=1.5,
                                    line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                                    legend_label='p_'+gene)
            protein_figure.legend.label_text_font_size = "8pt"
            protein_figure.legend.location = "top_right"
        else:
            mRNA_figure = None
            protein_figure = None

        # PLOT MISCELLANEOUS SPECIES' CONCENTRATIONS (IF ANY)
        if (len(circuit_miscs) > 0):
            misc_figure = bkplot.figure(
                frame_width=dimensions[0],
                frame_height=dimensions[1],
                x_axis_label="t, hours",
                y_axis_label="Conc., nM",
                x_range=tspan,
                title='Miscellaneous species concentrations',
                tools="box_zoom,pan,hover,reset"
            )
            for misc in circuit_miscs:
                misc_figure.line(x='t', y=misc, source=source, line_width=1.5,
                                 line_color=circuit_styles['colours'][misc], line_dash=circuit_styles['dashes'][misc],
                                 legend_label=misc)
            misc_figure.legend.label_text_font_size = "8pt"
            misc_figure.legend.location = "top_right"
        else:
            misc_figure = None

        return mRNA_figure, protein_figure, misc_figure

    # plot transcription regulation function values for the circuit's genes
    def plot_circuit_regulation(self, ts, xs,
                                circuit_F_calc,  # function calcul;ating the transcription regulation functions for the circuit
                                par, circuit_genes, circuit_miscs, circuit_name2pos, # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                circuit_styles,  # colours for the circuit plots
                                dimensions=(320, 180), tspan=None):
        # if no circuitry, return no plots
        if (len(circuit_genes)==0):
            return None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # find values of gene transcription regulation functions
        Fs=np.zeros((len(ts),len(circuit_genes))) # initialise
        for i in range(0,len(ts)):
            Fs[i,:]=np.array(circuit_F_calc(ts[i],xs[i,:],par,circuit_name2pos)[:])

        # Create ColumnDataSource object for the plot
        data_for_column={'t': ts} # initialise with time axis
        for i in range(0,len(circuit_genes)):
            data_for_column['F_'+str(circuit_genes[i])]=Fs[:,i]

        # PLOT
        F_figure=bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Transc. reg. funcs. F",
            x_range=tspan,
            y_range=(0,1.05),
            title='Gene transcription regulation',
            tools="box_zoom,pan,hover,reset"
        )
        for gene in circuit_genes:
            F_figure.line(x='t', y='F_'+gene, source=data_for_column, line_width=1.5,
                          line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                          legend_label='F_'+gene)
        F_figure.legend.label_text_font_size = "8pt"
        F_figure.legend.location = "top_right"

        return F_figure

    # plot physiological variables: growth rate, translation elongation rate, ribosomal gene transcription regulation function, ppGpp concentration, tRNA charging rate, RC denominator
    def plot_phys_variables(self,ts,xs,
                            par, circuit_genes, circuit_miscs, circuit_name2pos, # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                            dimensions=(320,180), tspan=None):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # get cell variables' values over time
        e, l, F_r, nu, _, T, D, D_nohet = self.get_e_l_Fr_nu_psi_T_D_Dnohet(ts, xs, par, circuit_genes, circuit_miscs, circuit_name2pos)
        

        # Create a ColumnDataSource object for the plot
        data_for_column={'t': np.array(ts), 'e': np.array(e), 'l': np.array(l), 'F_r': np.array(F_r), 'ppGpp': np.array(1/T), 'nu': np.array(nu), 'D': np.array(D), 'D_nohet':np.array(D_nohet)}
        source = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT GROWTH RATE
        l_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Growth rate, 1/h",
            x_range=tspan,
            y_range=(0,2),
            title='Growth rate',
            tools="box_zoom,pan,hover,reset"
        )
        l_figure.line(x='t', y='l', source=source, line_width=1.5, line_color='blue', legend_label='l')
        l_figure.legend.label_text_font_size = "8pt"
        l_figure.legend.location = "top_right"

        # PLOT TRANSLATION ELONGATION RATE
        e_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Translation elongation rate, 1/h",
            x_range=tspan,
            y_range=(0,par['e_max']),
            title='Translation elongation rate',
            tools="box_zoom,pan,hover,reset"
        )
        e_figure.line(x='t', y='e', source=source, line_width=1.5, line_color='blue', legend_label='e')
        e_figure.legend.label_text_font_size = "8pt"
        e_figure.legend.location = "top_right"

        # PLOT RIBOSOMAL GENE TRANSCRIPTION REGULATION FUNCTION
        Fr_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Ribosomal gene transcription regulation function",
            x_range=tspan,
            y_range=(0,1),
            title='Ribosomal gene transcription regulation function',
            tools="box_zoom,pan,hover,reset"
        )
        Fr_figure.line(x='t', y='F_r', source=source, line_width=1.5, line_color='blue', legend_label='F_r')
        Fr_figure.legend.label_text_font_size = "8pt"
        Fr_figure.legend.location = "top_right"

        # PLOT ppGpp CONCENTRATION
        ppGpp_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Rel. ppGpp conc. = 1/T",
            x_range=tspan,
            title='ppGpp concentration',
            tools="box_zoom,pan,hover,reset"
        )
        ppGpp_figure.line(x='t', y='ppGpp', source=source, line_width=1.5, line_color='blue', legend_label='ppGpp')
        ppGpp_figure.legend.label_text_font_size = "8pt"
        ppGpp_figure.legend.location = "top_right"

        # PLOT tRNA CHARGING RATE
        nu_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="tRNA charging rate, aa/h",
            x_range=tspan,
            title='tRNA charging rate',
            tools="box_zoom,pan,hover,reset"
        )
        nu_figure.line(x='t', y='nu', source=source, line_width=1.5, line_color='blue', legend_label='nu')
        nu_figure.legend.label_text_font_size = "8pt"
        nu_figure.legend.location = "top_right"

        # PLOT RC DENOMINATOR
        D_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="RC denominator D",
            x_range=tspan,
            title='Resource Competition denominator',
            tools="box_zoom,pan,hover,reset"
        )
        D_figure.line(x='t', y='D', source=source, line_width=1.5, line_color='blue', legend_label='D')
        D_figure.legend.label_text_font_size = "8pt"
        D_figure.legend.location = "top_right"

        return l_figure, e_figure, Fr_figure, ppGpp_figure, nu_figure, D_figure

    # find values of different cellular variables
    def get_e_l_Fr_nu_psi_T_D_Dnohet(self, t, x,
                                     par, circuit_genes, circuit_miscs, circuit_name2pos):
        # give the state vector entries meaningful names
        m_a = x[:, 0]  # metabolic gene mRNA
        m_r = x[:, 1]  # ribosomal gene mRNA
        p_a = x[:, 2]  # metabolic proteins
        R = x[:, 3]  # non-inactivated ribosomes
        tc = x[:, 4]  # charged tRNAs
        tu = x[:, 5]  # uncharged tRNAs
        Bcm = x[:,6] # chloramphenicol-bound ribosomes
        s = x[:, 7]  # nutrient quality (constant)
        h = x[:, 8]  # chloramphenicol concentration (constant)
        x_het = x[:, 9:9 + 2*len(circuit_genes)]  # heterologous protein concentrations
        misc = x[:, 9 + 2*len(circuit_genes):9 + 2*len(circuit_genes) + len(circuit_miscs)]  # miscellaneous species

        # vector of Synthetic Gene Parameters 4 JAX
        sgp4j = self.synth_gene_params_for_jax(par,circuit_genes)
        kplus_het, kminus_het, n_het, d_het = sgp4j


        # CALCULATE PHYSIOLOGICAL VARIABLES
        # translation elongation rate
        e = e_calc(par,tc)

        # ribosome inactivation due to chloramphenicol
        kcmh = par['kcm']*h

        # ribosome dissociation constants
        k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'],kcmh)    # metabolic genes
        k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'],kcmh)    # ribosomal genes
        k_het = k_calc((jnp.atleast_2d(jnp.array(e)*jnp.ones((len(circuit_genes),1)))).T,
                       jnp.atleast_2d(kplus_het)*jnp.ones((len(e),1)),
                       jnp.atleast_2d(kminus_het)*jnp.ones((len(e),1)),
                       jnp.atleast_2d(n_het)*jnp.ones((len(e),1)),
                       (jnp.atleast_2d(jnp.array(kcmh)*jnp.ones((len(circuit_genes),1)))).T)         # heterologous genes


        T = tc / tu  # ratio of charged to uncharged tRNAs


        # resource competition denominator
        D = 1 + (1 / (1 - par['phi_q'])) * (m_a / k_a + m_r / k_r + (x_het[:,0:len(circuit_genes)] / k_het).sum(axis=1))
        B = R * (1 - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

        nu = nu_calc(par, tu, s)  # tRNA charging rate

        l = l_calc(par,e, B)  # growth rate

        psi = psi_calc(par, T, l)  # tRNA synthesis rate - AMENDED

        F_r=Fr_calc(par,T) # ribosomal gene transcription regulation function

        # RC denominator, as it would be without heterologous genes
        D_nohet = 1 + (1 / (1 - par['phi_q'])) * (m_a / k_a + m_r / k_r)
        return e, l, F_r, nu, psi, T, D, D_nohet

# SIMULATION -----------------------------------------------------------------------------------------------------------
# ODE simulator with DIFFRAX
@functools.partial(jax.jit, static_argnums=(1,3,4))
def ode_sim(par,    # dictionary with model parameters
            ode_with_circuit,  # ODE function for the cell with the synthetic gene circuit
            x0,  # initial condition VECTOR
            num_circuit_genes, num_circuit_miscs, circuit_name2pos, sgp4j, # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder, relevant synthetic gene parameters in jax.array form
            tf, ts, rtol, atol    # simulation parameters: time frame, when to save the system's state, relative and absolute tolerances
            ):
    # define the ODE term
    vector_field=lambda t,y,args: ode_with_circuit(t,y,args)
    term = ODETerm(vector_field)

    # define arguments of the ODE term
    args= (
            par,  # model parameters
            circuit_name2pos, # gene name - position in circuit vector decoder
            num_circuit_genes, num_circuit_miscs, # number of genes and miscellaneous species in the circuit
            sgp4j # relevant synthetic gene parameters in jax.array form
           )

    # define the solver
    solver = Dopri5()

    # define the time points at which we save the solution
    stepsize_controller = PIDController(rtol=rtol, atol=atol)
    saveat = SaveAt(ts=ts)

    # solve the ODE
    sol = diffeqsolve(term, solver,
                      args=args,
                      t0=tf[0], t1=tf[1], dt0=0.1, y0=x0, saveat=saveat,
                      max_steps=None,
                      stepsize_controller=stepsize_controller)

    # convert jax arrays into numpy arrays
    return sol

# ode
def ode(t,x,circuit_ode,args):
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
    Bcm = x[6]  # chloramphenicol-bound ribosomes
    s = x[7]  # nutrient quality (constant)
    h = x[8]  # chloramphenicol concentration in the medium (constant)
    # synthetic circuit genes and miscellaneous species can be accessed directly from x with circuit_name2pos

    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = e_calc(par,tc)

    # ribosome inactivation due to chloramphenicol
    kcmh = par['kcm'] * h

    # ribosome dissociation constants
    k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'], kcmh)
    k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'],kcmh)
    k_het = k_calc(e, kplus_het, kminus_het, n_het,kcmh)

    T = tc / tu  # ratio of charged to uncharged tRNAs

    x_het_div_k_het = x[9:9+num_circuit_genes] / k_het  # competitoion for ribosomes from heterologous genes

    # resource competition denominator
    D = 1 + (1 / (1 - par['phi_q']))*(m_a / k_a + m_r / k_r + x_het_div_k_het.sum())
    B = R * (1 - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

    nu = nu_calc(par, tu, s)  # tRNA charging rate

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
        0
    ] +
    circuit_ode(t,x,e,l,R,k_het,D,par,circuit_name2pos))


# MAIN FUNCTION (FOR TESTING) ------------------------------------------------------------------------------------------
def main():
    # set up jax
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)

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

    # set up the simualtion
    # set simulation parameters
    tf = (0, 48)  # simulation time frame
    savetimestep = 0.1  # save time step
    dt_max = 0.1  # maximum integration step
    rtol = 1e-6  # relative tolerance for the ODE solver
    atol = 1e-6  # absolute tolerance for the ODE solver

    # simulate
    sol=ode_sim(par,    # dictionary with model parameters
                ode_with_circuit,   #  ODE function for the cell with synthetic circuit
                cellmodel_auxil.x0_from_init_conds(init_conds,circuit_genes,circuit_miscs),  # initial condition VECTOR
                len(circuit_genes), len(circuit_miscs), circuit_name2pos, # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                cellmodel_auxil.synth_gene_params_for_jax(par,circuit_genes), # synthetic gene parameters for calculating k values
                tf, jnp.arange(tf[0], tf[1], savetimestep), # time axis for saving the system's state
                rtol, atol)    # simulation parameters: when to save the system's state, relative and absolute tolerances)   # simulation parameters: time frame, save time step, relative and absolute tolerances
    ts=np.array(sol.ts)
    xs=np.array(sol.ys)

    # PLOT - HOST CELL MODEL
    bkplot.output_file(filename="../test_plots/cellmodel_sim.html", title="Cell Model Simulation") # set up bokeh output file
    mass_fig=cellmodel_auxil.plot_protein_masses(ts,xs,par,circuit_genes) # plot simulation results
    nat_mrna_fig,nat_prot_fig,nat_trna_fig,h_fig = cellmodel_auxil.plot_native_concentrations(ts, xs, par, circuit_genes)  # plot simulation results
    l_figure, e_figure, Fr_figure, ppGpp_figure, nu_figure, D_figure = cellmodel_auxil.plot_phys_variables(ts, xs, par, circuit_genes, circuit_miscs, circuit_name2pos)  # plot simulation results
    bkplot.save(bklayouts.grid([[mass_fig, nat_mrna_fig, nat_prot_fig],
                                [nat_trna_fig, h_fig, l_figure],
                                [e_figure, Fr_figure, D_figure]]))

    # PLOT - SYNTHETIC GENE CIRCUIT
    bkplot.output_file(filename="../test_plots/circuit_sim.html", title="Synthetic Gene Circuit Simulation") # set up bokeh output file
    het_mrna_fig, het_prot_fig, misc_fig = cellmodel_auxil.plot_circuit_concentrations(ts, xs, par, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles)  # plot simulation results
    F_fig = cellmodel_auxil.plot_circuit_regulation(ts, xs, circuit_F_calc, par, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles)  # plot simulation results
    bkplot.save(bklayouts.grid([[het_mrna_fig, het_prot_fig, misc_fig],
                                [F_fig, None, None]]))

    return


# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()