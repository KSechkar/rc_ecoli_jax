## jax_cell_simulator.py
# JAX/python implementation of the cell simulator

## PACKAGE IMPORTS
import numpy as np
import jax
import jax.numpy as jnp
import functools
import diffrax
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, SteadyStateEvent
import pandas as pd
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts

import time

## SYNTHETIC GENE CIRCUIT IMPORTS
from het_modules.one_constit import initialise as oneconstit_init, ode as oneconstit_ode, F_calc as oneconstit_F_calc, v as oneconstit_v


## RATE AND ACTIVATION FUNCTIONS
# apparent mRNA-ribosome dissociation constant
def k_calc(e, kplus, kminus, n, kcmh):
    return (kminus + e / n + kcmh) / kplus


# translation elongation rate
def e_calc(par, tc):
    return par['e_max'] * tc / (tc + par['K_e'])


# growth rate
def l_calc(par, e, B, prodeflux):
    return (e * B - prodeflux) / par['M']


# tRNA charging rate
def nu_calc(par, tu, s):
    return par['nu_max'] * s * (tu / (tu + par['K_nu']))


# tRNA synthesis rate
def psi_calc(par, T):
    return par['psi_max'] * T / (T + par['tau'])


# ribosomal gene transcription regulation function
def Fr_calc(par, T):
    return T / (T + par['tau'])


## AUXILIARY FUNCTIONS
class CellModelAuxiliary:
    # INITIALISE
    def __init__(self):
        # plotting colours
        self.gene_colours = {'a': "#EDB120", 'r': "#7E2F8E", 'q': '#C0C0C0',
                             # colours for metabolic, riboozyme, housekeeping genes
                             'het': "#0072BD",
                             'h': "#77AC30"}  # colours for heterologous genes and intracellular chloramphenicol level
        self.tRNA_colours = {'tc': "#000000", 'tu': "#ABABAB"}  # colours for charged and uncharged tRNAs
        return

    # PROCESS SYNTHETIC CIRCUIT MODULE
    # add synthetic circuit to the cell model
    def add_circuit(self,
                    circuit_initialiser,  # function initialising the circuit
                    circuit_ode,  # function defining the circuit ODEs
                    circuit_F_calc,  # function calculating the circuit genes' transcription regulation functions
                    cellmodel_par, cellmodel_init_conds,  # host cell model parameters and initial conditions
                    # optional support for hybrid simulations
                    circuit_v=None
                    ):
        # call circuit initialiser
        circuit_par, circuit_init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles = circuit_initialiser()

        # update parameter, initial condition and colour dictionaries
        cellmodel_par.update(circuit_par)
        cellmodel_init_conds.update(circuit_init_conds)

        # join the circuit ODEs with the transcription regulation functions
        circuit_ode_with_F_calc = lambda t, x, e, l, R, k_het, D, par, name2pos: circuit_ode(circuit_F_calc,
                                                                                             t, x, e, l, R, k_het, D,
                                                                                             par, name2pos)

        # IF stochastic component specified, predefine F_calc for it as well
        if (circuit_v != None):
            circuit_v_with_F_calc = lambda t, x, e, l, R, k_het, D, mRNA_count_scales, par, name2pos: circuit_v(
                circuit_F_calc,
                t, x, e, l, R, k_het, D,
                mRNA_count_scales,
                par, name2pos)
        else:
            circuit_v_with_F_calc = None

        # add the ciorcuit ODEs to that of the host cell model
        cellmodel_ode = lambda t, x, args: ode(t, x, circuit_ode_with_F_calc, args)

        # return updated ode and parameter, initial conditions, circuit gene (and miscellaneous specie) names
        # name - position in state vector decoder and colours for plotting the circuit's time evolution
        return cellmodel_ode, circuit_F_calc, cellmodel_par, cellmodel_init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles, circuit_v_with_F_calc

    # package synthetic gene parameters into jax arrays for calculating k values
    def synth_gene_params_for_jax(self, par,  # system parameters
                                  circuit_genes  # circuit gene names
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

        params = {}  # initialise

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
        params['psi_max'] = 1080000.0 / 2.5  # max synthesis rate (aa/h) [4]
        params['tau'] = 1.0  # ppGpp sensitivity (ribosome transc. and tRNA synth. Hill const) [4]

        # EXTERNAL CHLORAMPHENICOL CONCENTRATION
        params['h_ext'] = 0.0  # external chloramphenicol concentration (nM)

        # FITTED PARAMETERS
        params['a_a'] = 394464.6979  # metabolic gene transcription rate (/h)
        params['a_r'] = 1.0318*params['a_a']  # ribosomal gene transcription rate (/h)
        params['nu_max'] = 4.0469e3  # max tRNA amioacylation rate (/h)
        params['K_nu'] = 1.2397e3  # tRNA charging rate Michaelis-Menten constant (nM)
        params['K_e'] = 1.2397e3  # translation elongation rate Michaelis-Menten constant (nM)
        params['kcm'] = 0.000356139  # chloramphenicol binding rate (1/nM/h)#  chloramphenical binding rate constant (/h/nM)

        # RETURN
        return params

    # set default initial conditions
    def default_init_conds(self, par):
        init_conds = {}  # initialise

        # mRNA concentrations - non-zero to avoid being stuck at lambda=0
        init_conds['m_a'] = 1000.0  # metabolic
        init_conds['m_r'] = 0.01  # ribosomal

        # protein concentrations - start with 50/50 a/R allocation as a convention
        init_conds['p_a'] = par['M'] * (1 - par['phi_q']) / (2 * par['n_a'])  # metabolic *
        init_conds['R'] = par['M'] * (1 - par['phi_q']) / (2 * par['n_r'])  # ribosomal *

        # tRNA concentrations - 3E-5 abundance units in Chure and Cremer 2022 are equivalent to 80 uM = 80000 nM
        init_conds['tc'] = 80000.0  # charged tRNAs
        init_conds['tu'] = 80000.0  # free tRNAs

        # nutrient quality s and chloramphenicol concentration h
        init_conds['s'] = 0.5
        init_conds['Bcm'] = 0.0  # no translation inhibition assumed by default
        return init_conds

    # PREPARE FOR SIMULATIONS
    # set default initial condition vector
    def x0_from_init_conds(self, init_conds, circuit_genes, circuit_miscs):
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

            # culture medium's nutrient quality and chloramphenicol concentration
            init_conds['s'],  # nutrient quality
            init_conds['Bcm'],  # chloramphenicol levels IN THE CELL
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
    def plot_protein_masses(self, ts, xs,
                            par, circuit_genes,  # model parameters, list of circuit genes
                            dimensions=(320, 180), tspan=None):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

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

        flip_t = np.flip(ts)  # flipped time axis for patch plotting

        # plot heterologous protein mass - if there are any heterologous proteins to begin with
        if (len(circuit_genes) != 0):
            bottom_line = np.zeros(xs.shape[0])
            top_line = bottom_line + np.sum(xs[:, 8 + len(circuit_genes):8 + len(circuit_genes) * 2] * np.array(
                self.synth_gene_params_for_jax(par, circuit_genes)[2], ndmin=2), axis=1)
            mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                              line_width=0.5, line_color='black', fill_color=self.gene_colours['het'],
                              legend_label='het')
        else:
            top_line = np.zeros(xs.shape[0])

        # plot mass of inactivated ribosomes
        if ((xs[:, 7] != 0).any()):
            bottom_line = top_line
            top_line = bottom_line + xs[:, 7]
            mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                              line_width=0.5, line_color='black', fill_color=self.gene_colours['h'], legend_label='R:h')

        # plot mass of active ribosomes - only if there are any to begin with
        bottom_line = top_line
        top_line = bottom_line + xs[:, 3]
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                          line_width=0.5, line_color='black', fill_color=self.gene_colours['r'],
                          legend_label='R (free)')

        # plot metabolic protein mass
        bottom_line = top_line
        top_line = bottom_line + xs[:, 2] * par['n_a']
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                          line_width=0.5, line_color='black', fill_color=self.gene_colours['a'], legend_label='p_a')

        # plot housekeeping protein mass
        bottom_line = top_line
        top_line = bottom_line / (1 - par['phi_q'])
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                          line_width=0.5, line_color='black', fill_color=self.gene_colours['q'], legend_label='p_q')

        # add legend
        mass_figure.legend.label_text_font_size = "8pt"
        mass_figure.legend.location = "top_right"

        return mass_figure

    # plot mRNA, protein and tRNA concentrations over time
    def plot_native_concentrations(self, ts, xs,
                                   par, circuit_genes,  # model parameters, list of circuit genes
                                   dimensions=(320, 180), tspan=None):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # Create a ColumnDataSource object for the plot
        source = bkmodels.ColumnDataSource(data={
            't': ts,
            'm_a': xs[:, 0],  # metabolic mRNA
            'm_r': xs[:, 1],  # ribosomal mRNA
            'p_a': xs[:, 2],  # metabolic protein
            'R': xs[:, 3],  # ribosomal protein
            'tc': xs[:, 4],  # charged tRNA
            'tu': xs[:, 5],  # uncharged tRNA
            's': xs[:, 6],  # nutrient quality
            'h': xs[:, 7],  # chloramphenicol concentration
            'm_het': np.sum(xs[:, 8:8 + len(circuit_genes)], axis=1),  # heterologous mRNA
            'p_het': np.sum(xs[:, 8 + len(circuit_genes):8 + len(circuit_genes) * 2], axis=1),  # heterologous protein
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
        mRNA_figure.line(x='t', y='m_a', source=source, line_width=1.5, line_color=self.gene_colours['a'],
                         legend_label='m_a')  # plot metabolic mRNA concentrations
        mRNA_figure.line(x='t', y='m_r', source=source, line_width=1.5, line_color=self.gene_colours['r'],
                         legend_label='m_r')  # plot ribosomal mRNA concentrations
        mRNA_figure.line(x='t', y='m_het', source=source, line_width=1.5, line_color=self.gene_colours['het'],
                         legend_label='m_het')  # plot heterologous mRNA concentrations
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
        protein_figure.line(x='t', y='p_a', source=source, line_width=1.5, line_color=self.gene_colours['a'],
                            legend_label='p_a')  # plot metabolic protein concentrations
        protein_figure.line(x='t', y='R', source=source, line_width=1.5, line_color=self.gene_colours['r'],
                            legend_label='R')  # plot ribosomal protein concentrations
        protein_figure.line(x='t', y='p_het', source=source, line_width=1.5, line_color=self.gene_colours['het'],
                            legend_label='p_het')  # plot heterologous protein concentrations
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
        tRNA_figure.line(x='t', y='tc', source=source, line_width=1.5, line_color=self.tRNA_colours['tc'],
                         legend_label='tc')  # plot charged tRNA concentrations
        tRNA_figure.line(x='t', y='tu', source=source, line_width=1.5, line_color=self.tRNA_colours['tu'],
                         legend_label='tu')  # plot uncharged tRNA concentrations
        protein_figure.legend.label_text_font_size = "8pt"
        protein_figure.legend.location = "top_right"

        # PLOT INTRACELLULAR CHLORAMPHENICOL CONCENTRATION
        h_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="h, nM",
            x_range=tspan,
            title='Intracellular chloramphenicol concentration',
            tools="box_zoom,pan,hover,reset"
        )
        h_figure.line(x='t', y='h', source=source, line_width=1.5, line_color=self.gene_colours['h'],
                      legend_label='h')  # plot intracellular chloramphenicol concentration

        return mRNA_figure, protein_figure, tRNA_figure, h_figure

    # plot concentrations for the synthetic circuits
    def plot_circuit_concentrations(self, ts, xs,
                                    par, circuit_genes, circuit_miscs, circuit_name2pos,
                                    # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                    circuit_styles,  # colours for the circuit plots
                                    dimensions=(320, 180), tspan=None):
        # if no circuitry at all, return no plots
        if (len(circuit_genes) + len(circuit_miscs) == 0):
            return None, None, None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # Create a ColumnDataSource object for the plot
        data_for_column = {'t': ts}  # initialise with time axis
        # record synthetic mRNA and protein concentrations
        for i in range(0, len(circuit_genes)):
            data_for_column['m_' + circuit_genes[i]] = xs[:, 8 + i]
            data_for_column['p_' + circuit_genes[i]] = xs[:, 8 + len(circuit_genes) + i]
        # record miscellaneous species' concentrations
        for i in range(0, len(circuit_miscs)):
            data_for_column[circuit_miscs[i]] = xs[:, 8 + len(circuit_genes) * 2 + i]
        source = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT mRNA and PROTEIN CONCENTRATIONS (IF ANY)
        if (len(circuit_genes) > 0):
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
                mRNA_figure.line(x='t', y='m_' + gene, source=source, line_width=1.5,
                                 line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                                 legend_label='m_' + gene)
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
                protein_figure.line(x='t', y='p_' + gene, source=source, line_width=1.5,
                                    line_color=circuit_styles['colours'][gene],
                                    line_dash=circuit_styles['dashes'][gene],
                                    legend_label='p_' + gene)
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
                                circuit_F_calc,
                                # function calculating the transcription regulation functions for the circuit
                                par, circuit_genes, circuit_miscs, circuit_name2pos,
                                # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                circuit_styles,  # colours for the circuit plots
                                dimensions=(320, 180), tspan=None):
        # if no circuitry, return no plots
        if (len(circuit_genes) == 0):
            return None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # find values of gene transcription regulation functions
        Fs = np.zeros((len(ts), len(circuit_genes)))  # initialise
        for i in range(0, len(ts)):
            Fs[i, :] = np.array(circuit_F_calc(ts[i], xs[i, :], par, circuit_name2pos)[:])

        # Create ColumnDataSource object for the plot
        data_for_column = {'t': ts}  # initialise with time axis
        for i in range(0, len(circuit_genes)):
            data_for_column['F_' + str(circuit_genes[i])] = Fs[:, i]

        # PLOT
        F_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Transc. reg. funcs. F",
            x_range=tspan,
            y_range=(0, 1.05),
            title='Gene transcription regulation',
            tools="box_zoom,pan,hover,reset"
        )
        for gene in circuit_genes:
            F_figure.line(x='t', y='F_' + gene, source=data_for_column, line_width=1.5,
                          line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                          legend_label='F_' + gene)
        F_figure.legend.label_text_font_size = "8pt"
        F_figure.legend.location = "top_right"

        return F_figure

    # plot physiological variables: growth rate, translation elongation rate, ribosomal gene transcription regulation function, ppGpp concentration, tRNA charging rate, RC denominator
    def plot_phys_variables(self, ts, xs,
                            par, circuit_genes, circuit_miscs, circuit_name2pos,
                            # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                            dimensions=(320, 180), tspan=None):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # get cell variables' values over time
        e, l, F_r, nu, _, T, D  = self.get_e_l_Fr_nu_psi_T_D_Dnohet(ts, xs, par, circuit_genes, circuit_miscs,
                                                                            circuit_name2pos)

        # Create a ColumnDataSource object for the plot
        data_for_column = {'t': np.array(ts), 'e': np.array(e), 'l': np.array(l), 'F_r': np.array(F_r),
                           'ppGpp': np.array(1 / T), 'nu': np.array(nu), 'D': np.array(D)}
        source = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT GROWTH RATE
        l_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Growth rate, 1/h",
            x_range=tspan,
            y_range=(0, 2),
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
            y_range=(0, par['e_max']),
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
            y_range=(0, 1),
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
        s = x[:, 6]  # nutrient quality (constant)
        Bcm = x[:, 7]  # chloramphenicol concentration (constant)
        x_het = x[:, 8:8 + 2 * len(circuit_genes)]  # heterologous protein concentrations
        misc = x[:, 8 + 2 * len(circuit_genes):8 + 2 * len(circuit_genes) + len(circuit_miscs)]  # miscellaneous species

        # vector of Synthetic Gene Parameters 4 JAX
        sgp4j = self.synth_gene_params_for_jax(par, circuit_genes)
        kplus_het, kminus_het, n_het, d_het = sgp4j

        # CALCULATE PHYSIOLOGICAL VARIABLES
        # translation elongation rate
        e = e_calc(par, tc)

        # ribosome inactivation rate due to chloramphenicol
        kcmh = par['kcm'] * par['h_ext']

        # ribosome dissociation constants
        k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'], kcmh)  # metabolic genes
        k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'], kcmh)  # ribosomal genes
        k_het = k_calc((jnp.atleast_2d(jnp.array(e) * jnp.ones((len(circuit_genes), 1)))).T,
                       jnp.atleast_2d(kplus_het) * jnp.ones((len(e), 1)),
                       jnp.atleast_2d(kminus_het) * jnp.ones((len(e), 1)),
                       jnp.atleast_2d(n_het) * jnp.ones((len(e), 1)), kcmh)  # heterologous genes

        # overall protein degradation flux for all synthetic genes
        prodeflux = jnp.sum(
            d_het * n_het * x[:, 8 + len(circuit_genes):8 + len(circuit_genes) * 2],
            axis=1)  # heterologous protein degradation flux
        prodeflux_div_eR = prodeflux / (e * R)  # heterologous protein degradation flux divided by eR

        # resource competition denominator
        m_het_div_k_het = jnp.sum(x[:, 8:8 + len(circuit_genes)] / k_het, axis=1)  # heterologous protein synthesis flux
        sum_mk_not_q = m_a / k_a + m_r / k_r + m_het_div_k_het
        mq_div_kq = (par['phi_q'] * (1 - prodeflux_div_eR) * sum_mk_not_q -
                     par['phi_q'] * prodeflux_div_eR) / (1 - par['phi_q'] * (1 - prodeflux_div_eR))
        D = (1 + mq_div_kq + sum_mk_not_q)  # resource competition denominator

        T = tc / tu  # ratio of charged to uncharged tRNAs
        B = R * (1 - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

        nu = nu_calc(par, tu, s)  # tRNA charging rate

        l = l_calc(par, e, B, prodeflux)  # growth rate

        psi = psi_calc(par, T)  # tRNA synthesis rate - AMENDED

        F_r = Fr_calc(par, T)  # ribosomal gene transcription regulation function

        # RC denominator, as it would be without heterologous genes
        return e, l, F_r, nu, jnp.multiply(psi, l), T, D

    # PLOT RESULTS FOR SEVERAL TRAJECTORIES AT ONCE (SAME TIME AXIS)
    # plot mRNA, protein and tRNA concentrations over time
    def plot_native_concentrations_multiple(self, ts, xss,
                                            par, circuit_genes,  # model parameters, list of circuit genes
                                            dimensions=(320, 180), tspan=None,
                                            simtraj_alpha=0.1):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # Create ColumnDataSource objects for the plot
        sources = {}
        for i in range(0, len(xss)):
            # Create a ColumnDataSource object for the plot
            sources[i] = bkmodels.ColumnDataSource(data={
                't': ts,
                'm_a': xss[i, :, 0],  # metabolic mRNA
                'm_r': xss[i, :, 1],  # ribosomal mRNA
                'p_a': xss[i, :, 2],  # metabolic protein
                'R': xss[i, :, 3],  # ribosomal protein
                'tc': xss[i, :, 4],  # charged tRNA
                'tu': xss[i, :, 5],  # uncharged tRNA
                's': xss[i, :, 6],  # nutrient quality
                'h': xss[i, :, 7],  # chloramphenicol concentration
                'm_het': np.sum(xss[i, :, 8:8 + len(circuit_genes)], axis=1),  # heterologous mRNA
                'p_het': np.sum(xss[i, :, 8 + len(circuit_genes):8 + len(circuit_genes) * 2], axis=1),
                # heterologous protein
            })

        # Create a ColumnDataSource object for plotting the average trajectory
        source_avg = bkmodels.ColumnDataSource(data={
            't': ts,
            'm_a': np.mean(xss[:, :, 0], axis=0),  # metabolic mRNA
            'm_r': np.mean(xss[:, :, 1], axis=0),  # ribosomal mRNA
            'p_a': np.mean(xss[:, :, 2], axis=0),  # metabolic protein
            'R': np.mean(xss[:, :, 3], axis=0),  # ribosomal protein
            'tc': np.mean(xss[:, :, 4], axis=0),  # charged tRNA
            'tu': np.mean(xss[:, :, 5], axis=0),  # uncharged tRNA
            's': np.mean(xss[:, :, 6], axis=0),  # nutrient quality
            'h': np.mean(xss[:, :, 7], axis=0),  # chloramphenicol concentration
            'm_het': np.sum(np.mean(xss[:, :, 8:8 + len(circuit_genes)], axis=0), axis=1),  # heterologous mRNA
            'p_het': np.sum(np.mean(xss[:, :, 8 + len(circuit_genes):8 + len(circuit_genes) * 2], axis=0), axis=1),
            # heterologous protein
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
        # plot simulated trajectories
        for i in range(0, len(xss)):
            mRNA_figure.line(x='t', y='m_a', source=sources[i], line_width=1.5, line_color=self.gene_colours['a'],
                             legend_label='m_a', line_alpha=simtraj_alpha)  # plot metabolic mRNA concentrations
            mRNA_figure.line(x='t', y='m_r', source=sources[i], line_width=1.5, line_color=self.gene_colours['r'],
                             legend_label='m_r', line_alpha=simtraj_alpha)  # plot ribosomal mRNA concentrations
            mRNA_figure.line(x='t', y='m_het', source=sources[i], line_width=1.5, line_color=self.gene_colours['het'],
                             legend_label='m_het', line_alpha=simtraj_alpha)  # plot heterologous mRNA concentrations
        # plot average trajectory
        mRNA_figure.line(x='t', y='m_a', source=source_avg, line_width=1.5, line_color=self.gene_colours['a'],
                         legend_label='m_a')  # plot metabolic mRNA concentrations
        mRNA_figure.line(x='t', y='m_r', source=source_avg, line_width=1.5, line_color=self.gene_colours['r'],
                         legend_label='m_r')  # plot ribosomal mRNA concentrations
        mRNA_figure.line(x='t', y='m_het', source=source_avg, line_width=1.5, line_color=self.gene_colours['het'],
                         legend_label='m_het')  # plot heterologous mRNA concentrations
        # add and format the legend
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
        # plot simulated trajectories
        for i in range(0, len(xss)):
            protein_figure.line(x='t', y='p_a', source=sources[i], line_width=1.5, line_color=self.gene_colours['a'],
                                legend_label='p_a', line_alpha=simtraj_alpha)
            protein_figure.line(x='t', y='R', source=sources[i], line_width=1.5, line_color=self.gene_colours['r'],
                                legend_label='R', line_alpha=simtraj_alpha)
            protein_figure.line(x='t', y='p_het', source=sources[i], line_width=1.5,
                                line_color=self.gene_colours['het'],
                                legend_label='p_het', line_alpha=simtraj_alpha)
        # plot average trajectory
        protein_figure.line(x='t', y='p_a', source=source_avg, line_width=1.5, line_color=self.gene_colours['a'],
                            legend_label='p_a')
        protein_figure.line(x='t', y='R', source=source_avg, line_width=1.5, line_color=self.gene_colours['r'],
                            legend_label='R')
        protein_figure.line(x='t', y='p_het', source=source_avg, line_width=1.5, line_color=self.gene_colours['het'],
                            legend_label='p_het')
        # add and format the legend
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
        # plot simulated trajectories
        for i in range(0, len(xss)):
            tRNA_figure.line(x='t', y='tc', source=sources[i], line_width=1.5, line_color=self.tRNA_colours['tc'],
                             legend_label='tc', line_alpha=simtraj_alpha)
            tRNA_figure.line(x='t', y='tu', source=sources[i], line_width=1.5, line_color=self.tRNA_colours['tu'],
                             legend_label='tu', line_alpha=simtraj_alpha)
        # plot average trajectory
        tRNA_figure.line(x='t', y='tc', source=source_avg, line_width=1.5, line_color=self.tRNA_colours['tc'],
                         legend_label='tc')
        tRNA_figure.line(x='t', y='tu', source=source_avg, line_width=1.5, line_color=self.tRNA_colours['tu'],
                         legend_label='tu')
        # add and format the legend
        tRNA_figure.legend.label_text_font_size = "8pt"
        tRNA_figure.legend.location = "top_right"

        # PLOT INTRACELLULAR CHLORAMPHENICOL CONCENTRATION
        h_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="h, nM",
            x_range=tspan,
            title='Intracellular chloramphenicol concentration',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            h_figure.line(x='t', y='h', source=sources[i], line_width=1.5, line_color=self.gene_colours['h'],
                          legend_label='h', line_alpha=simtraj_alpha)
        # plot average trajectory
        h_figure.line(x='t', y='h', source=source_avg, line_width=1.5, line_color=self.gene_colours['h'],
                      legend_label='h')
        # add and format the legend
        h_figure.legend.label_text_font_size = "8pt"
        h_figure.legend.location = "top_right"

        return mRNA_figure, protein_figure, tRNA_figure, h_figure

    # plot concentrations for the synthetic circuits
    def plot_circuit_concentrations_multiple(self, ts, xss,
                                             par, circuit_genes, circuit_miscs, circuit_name2pos,
                                             # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                             circuit_styles,  # colours for the circuit plots
                                             dimensions=(320, 180), tspan=None,
                                             simtraj_alpha=0.1):
        # if no circuitry at all, return no plots
        if (len(circuit_genes) + len(circuit_miscs) == 0):
            return None, None, None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # Create ColumnDataSource objects for the plot
        sources = {}
        for i in range(0, len(xss)):
            # Create a ColumnDataSource object for the plot
            data_for_column = {'t': ts}
            # record synthetic mRNA and protein concentrations
            for j in range(0, len(circuit_genes)):
                data_for_column['m_' + circuit_genes[j]] = xss[i, :, 8 + j]
                data_for_column['p_' + circuit_genes[j]] = xss[i, :, 8 + len(circuit_genes) + j]
            # record miscellaneous species' concentrations
            for j in range(0, len(circuit_miscs)):
                data_for_column[circuit_miscs[j]] = xss[i, :, 8 + len(circuit_genes) * 2 + j]
            sources[i] = bkmodels.ColumnDataSource(data=data_for_column)

        # Create a ColumnDataSource object for plotting the average trajectory
        data_for_column = {'t': ts}
        # record synthetic mRNA and protein concentrations
        for j in range(0, len(circuit_genes)):
            data_for_column['m_' + circuit_genes[j]] = np.mean(xss[:, :, 8 + j], axis=0)
            data_for_column['p_' + circuit_genes[j]] = np.mean(xss[:, :, 8 + len(circuit_genes) + j], axis=0)
        # record miscellaneous species' concentrations
        for j in range(0, len(circuit_miscs)):
            data_for_column[circuit_miscs[j]] = np.mean(xss[:, :, 8 + len(circuit_genes) * 2 + j], axis=0)
        source_avg = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT mRNA and PROTEIN CONCENTRATIONS (IF ANY)
        if (len(circuit_genes) > 0):
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
            # plot simulated trajectories
            for i in range(0, len(xss)):
                for gene in circuit_genes:
                    mRNA_figure.line(x='t', y='m_' + gene, source=sources[i], line_width=1.5,
                                     line_color=circuit_styles['colours'][gene],
                                     line_dash=circuit_styles['dashes'][gene],
                                     legend_label='m_' + gene, line_alpha=simtraj_alpha)
            # plot average trajectory
            for gene in circuit_genes:
                mRNA_figure.line(x='t', y='m_' + gene, source=source_avg, line_width=1.5,
                                 line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                                 legend_label='m_' + gene)
            # add and format the legend
            mRNA_figure.legend.label_text_font_size = "8pt"
            mRNA_figure.legend.location = 'top_left'

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
            # plot simulated trajectories
            for i in range(0, len(xss)):
                for gene in circuit_genes:
                    protein_figure.line(x='t', y='p_' + gene, source=sources[i], line_width=1.5,
                                        line_color=circuit_styles['colours'][gene],
                                        line_dash=circuit_styles['dashes'][gene],
                                        legend_label='p_' + gene, line_alpha=simtraj_alpha)
            # plot average trajectory
            for gene in circuit_genes:
                protein_figure.line(x='t', y='p_' + gene, source=source_avg, line_width=1.5,
                                    line_color=circuit_styles['colours'][gene],
                                    line_dash=circuit_styles['dashes'][gene],
                                    legend_label='p_' + gene)
            # add and format the legend
            protein_figure.legend.label_text_font_size = "8pt"
            protein_figure.legend.location = 'top_left'
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
            # plot simulated trajectories
            for i in range(0, len(xss)):
                for misc in circuit_miscs:
                    misc_figure.line(x='t', y=misc, source=sources[i], line_width=1.5,
                                     line_color=circuit_styles['colours'][misc],
                                     line_dash=circuit_styles['dashes'][misc],
                                     legend_label=misc, line_alpha=simtraj_alpha)
            # plot average trajectory
            for misc in circuit_miscs:
                misc_figure.line(x='t', y=misc, source=source_avg, line_width=1.5,
                                 line_color=circuit_styles['colours'][misc], line_dash=circuit_styles['dashes'][misc],
                                 legend_label=misc)
            # add and format the legend
            misc_figure.legend.label_text_font_size = "8pt"
            misc_figure.legend.location = 'top_left'
        else:
            misc_figure = None

        return mRNA_figure, protein_figure, misc_figure

    # plot transcription regulation function values for the circuit's genes
    def plot_circuit_regulation_multiple(self, ts, xss,
                                         par, circuit_F_calc,
                                         # function calculating the transcription regulation functions for the circuit
                                         circuit_genes, circuit_miscs, circuit_name2pos,
                                         # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                         circuit_styles,  # colours for the circuit plots
                                         dimensions=(320, 180), tspan=None,
                                         simtraj_alpha=0.1):
        # if no circuitry at all, return no plots
        if (len(circuit_genes) + len(circuit_miscs) == 0):
            return None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # find values of gene transcription regulation functions and create ColumnDataSource objects for the plot
        sources = {}
        for i in range(0, len(xss)):
            Fs = np.zeros((len(ts), len(circuit_genes)))  # initialise
            for k in range(0, len(ts)):
                Fs[k, :] = np.array(circuit_F_calc(ts[k], xss[i, k, :], par, circuit_name2pos)[:])

            # Create a ColumnDataSource object for the plot
            data_for_column = {'t': ts}
            for j in range(0, len(circuit_genes)):
                data_for_column['F_' + circuit_genes[j]] = Fs[:, j]
            sources[i] = bkmodels.ColumnDataSource(data=data_for_column)

        # Create a ColumnDataSource object for plotting the average trajectory
        data_for_column = {'t': ts}
        for j in range(0, len(circuit_genes)):
            data_for_column['F_' + circuit_genes[j]] = np.zeros_like(ts)
        # add gene transcription regulation functions for different trajectories together
        for i in range(0, len(xss)):
            for j in range(0, len(circuit_genes)):
                data_for_column['F_' + circuit_genes[j]] += np.array(sources[i].data['F_' + circuit_genes[j]])
        # divide by the number of trajectories to get the average
        for j in range(0, len(circuit_genes)):
            data_for_column['F_' + circuit_genes[j]] /= len(xss)
        source_avg = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT TRANSCRIPTION REGULATION FUNCTIONS
        F_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Transc. reg. funcs. F",
            x_range=tspan,
            y_range=(0, 1.05),
            title='Gene transcription regulation',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            for gene in circuit_genes:
                F_figure.line(x='t', y='F_' + gene, source=sources[i], line_width=1.5,
                              line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                              legend_label='F_' + gene, line_alpha=simtraj_alpha)
        # plot average trajectory
        for gene in circuit_genes:
            F_figure.line(x='t', y='F_' + gene, source=source_avg, line_width=1.5,
                          line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                          legend_label='F_' + gene)
        # add and format the legend
        F_figure.legend.label_text_font_size = "8pt"
        F_figure.legend.location = 'top_left'

        return F_figure

    # plot physiological variables: growth rate, translation elongation rate, ribosomal gene transcription regulation function, ppGpp concentration, tRNA charging rate, RC denominator
    def plot_phys_variables_multiple(self, ts, xss,
                                     par, circuit_genes, circuit_miscs, circuit_name2pos,
                                     # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                     dimensions=(320, 180), tspan=None,
                                     simtraj_alpha=0.1):
        # if no circuitry at all, return no plots
        if (len(circuit_genes) + len(circuit_miscs) == 0):
            return None, None, None, None, None, None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # get cell variables' values over time and create ColumnDataSource objects for the plot
        sources = {}
        for i in range(0, len(xss)):
            e, l, F_r, nu, psi, T, D = self.get_e_l_Fr_nu_psi_T_D_Dnohet(ts, xss[i, :, :],
                                                                                  par, circuit_genes, circuit_miscs,
                                                                                  circuit_name2pos)
            # Create a ColumnDataSource object for the plot
            sources[i] = bkmodels.ColumnDataSource(data={
                't': np.array(ts),
                'e': np.array(e),
                'l': np.array(l),
                'F_r': np.array(F_r),
                'nu': np.array(nu),
                'psi': np.array(psi),
                '1/T': np.array(1 / T),
                'D': np.array(D)
            })

        # Create a ColumnDataSource object for plotting the average trajectory
        data_for_column = {'t': ts,
                           'e': np.zeros_like(ts),
                           'l': np.zeros_like(ts),
                           'F_r': np.zeros_like(ts),
                           'nu': np.zeros_like(ts),
                           'psi': np.zeros_like(ts),
                           '1/T': np.zeros_like(ts),
                           'D': np.zeros_like(ts)}
        # add physiological variables for different trajectories together
        for i in range(0, len(xss)):
            data_for_column['e'] += np.array(sources[i].data['e'])
            data_for_column['l'] += np.array(sources[i].data['l'])
            data_for_column['F_r'] += np.array(sources[i].data['F_r'])
            data_for_column['nu'] += np.array(sources[i].data['nu'])
            data_for_column['psi'] += np.array(sources[i].data['psi'])
            data_for_column['1/T'] += np.array(sources[i].data['1/T'])
            data_for_column['D'] += np.array(sources[i].data['D'])
        # divide by the number of trajectories to get the average
        data_for_column['e'] /= len(xss)
        data_for_column['l'] /= len(xss)
        data_for_column['F_r'] /= len(xss)
        data_for_column['nu'] /= len(xss)
        data_for_column['psi'] /= len(xss)
        data_for_column['1/T'] /= len(xss)
        data_for_column['D'] /= len(xss)
        source_avg = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT GROWTH RATE
        l_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Growth rate, 1/h",
            x_range=tspan,
            title='Growth rate',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            l_figure.line(x='t', y='l', source=sources[i], line_width=1.5, line_color='blue', legend_label='l',
                          line_alpha=simtraj_alpha)
        # plot average trajectory
        l_figure.line(x='t', y='l', source=source_avg, line_width=1.5, line_color='blue', legend_label='l')
        # add and format the legend
        l_figure.legend.label_text_font_size = "8pt"
        l_figure.legend.location = 'top_left'

        # PLOT TRANSLATION ELONGATION RATE
        e_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Translation elongation rate, aa/s",
            x_range=tspan,
            title='Translation elongation rate',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            e_figure.line(x='t', y='e', source=sources[i], line_width=1.5, line_color='blue', legend_label='e',
                          line_alpha=simtraj_alpha)
        # plot average trajectory
        e_figure.line(x='t', y='e', source=source_avg, line_width=1.5, line_color='blue', legend_label='e')
        # add and format the legend
        e_figure.legend.label_text_font_size = "8pt"
        e_figure.legend.location = 'top_left'

        # PLOT RIBOSOMAL GENE TRANSCRIPTION REGULATION FUNCTION
        F_r_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Ribosomal gene transc. reg. func. F_r",
            x_range=tspan,
            y_range=(0, 1.05),
            title='Ribosomal gene transcription regulation function',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            F_r_figure.line(x='t', y='F_r', source=sources[i], line_width=1.5, line_color='blue', legend_label='F_r',
                            line_alpha=simtraj_alpha)
        # plot average trajectory
        F_r_figure.line(x='t', y='F_r', source=source_avg, line_width=1.5, line_color='blue', legend_label='F_r')
        # add and format the legend
        F_r_figure.legend.label_text_font_size = "8pt"
        F_r_figure.legend.location = 'top_left'

        # PLOT ppGpp CONCENTRATION
        pppGpp_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="[ppGpp], nM",
            x_range=tspan,
            title='ppGpp concentration',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            pppGpp_figure.line(x='t', y='1/T', source=sources[i], line_width=1.5, line_color='blue', legend_label='1/T',
                               line_alpha=simtraj_alpha)
        # plot average trajectory
        pppGpp_figure.line(x='t', y='1/T', source=source_avg, line_width=1.5, line_color='blue', legend_label='1/T')
        # add and format the legend
        pppGpp_figure.legend.label_text_font_size = "8pt"
        pppGpp_figure.legend.location = 'top_left'

        # PLOT tRNA CHARGING RATE
        nu_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="tRNA charging rate, 1/s",
            x_range=tspan,
            title='tRNA charging rate',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            nu_figure.line(x='t', y='nu', source=sources[i], line_width=1.5, line_color='blue', legend_label='nu',
                           line_alpha=simtraj_alpha)
        # plot average trajectory
        nu_figure.line(x='t', y='nu', source=source_avg, line_width=1.5, line_color='blue', legend_label='nu')

        # PLOT RC DENOMINATOR
        D_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="RC denominator",
            x_range=tspan,
            title='RC denominator',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            D_figure.line(x='t', y='D', source=sources[i], line_width=1.5, line_color='blue', legend_label='D',
                          line_alpha=simtraj_alpha)
        # plot average trajectory
        D_figure.line(x='t', y='D', source=source_avg, line_width=1.5, line_color='blue', legend_label='D')
        # add and format the legend
        D_figure.legend.label_text_font_size = "8pt"
        D_figure.legend.location = 'top_left'

        return l_figure, e_figure, F_r_figure, pppGpp_figure, nu_figure, D_figure


## DETERMINISTIC SIMULATOR
# ODE simulator with DIFFRAX
@functools.partial(jax.jit, static_argnums=(1,3,4))
def ode_sim(par,  # dictionary with model parameters
            ode_with_circuit,  # ODE function for the cell with the synthetic gene circuit
            x0,  # initial condition VECTOR
            num_circuit_genes, num_circuit_miscs, circuit_name2pos, sgp4j,
            # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder, relevant synthetic gene parameters in jax.array form
            tf, ts, rtol, atol
            # simulation parameters: time frame, when to save the system's state, relative and absolute tolerances
            ):
    # define the ODE term
    vector_field = lambda t, y, args: ode_with_circuit(t, y, args)
    term = ODETerm(vector_field)

    # define arguments of the ODE term
    args = (
        par,  # model parameters
        circuit_name2pos,  # gene name - position in circuit vector decoder
        num_circuit_genes, num_circuit_miscs,  # number of genes and miscellaneous species in the circuit
        sgp4j  # relevant synthetic gene parameters in jax.array form
    )

    # define the solver
    solver = diffrax.Ralston()

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
def ode(t, x, circuit_ode, args):
    # unpack the args
    par = args[0]  # model parameters
    circuit_name2pos = args[1]  # gene name - position in circuit vector decoder
    num_circuit_genes = args[2];
    num_circuit_miscs = args[3]  # number of genes and miscellaneous species in the circuit
    kplus_het, kminus_het, n_het, d_het = args[
        4]  # unpack jax-arrayed synthetic gene parameters for calculating k values

    # give the state vector entries meaningful names
    m_a = x[0]  # metabolic gene mRNA
    m_r = x[1]  # ribosomal gene mRNA
    p_a = x[2]  # metabolic proteins
    R = x[3]  # non-inactivated ribosomes
    tc = x[4]  # charged tRNAs
    tu = x[5]  # uncharged tRNAs
    s = x[6]  # nutrient quality (constant)
    Bcm = x[7]  # INTERNAL chloramphenicol concentration (varies)
    # synthetic circuit genes and miscellaneous species can be accessed directly from x with circuit_name2pos

    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = e_calc(par, tc)

    # ribosome inactivation rate due to chloramphenicol
    kcmh = par['kcm'] * par['h_ext']

    # ribosome dissociation constants
    k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'],kcmh)
    k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'],kcmh)
    k_het = k_calc(e, kplus_het, kminus_het, n_het,kcmh)

    # overall protein degradation flux for all synthetic genes
    prodeflux = jnp.sum(
        d_het * n_het * x[8 + num_circuit_genes:8 + num_circuit_genes * 2])  # heterologous protein degradation flux
    prodeflux_div_eR = prodeflux / (e * R)  # heterologous protein degradation flux divided by eR

    # resource competition denominator
    m_het_div_k_het = jnp.sum(x[8:8 + num_circuit_genes] / k_het)  # heterologous protein synthesis flux
    sum_mk_not_q = m_a / k_a + m_r / k_r + m_het_div_k_het
    mq_div_kq = (par['phi_q'] * (1 - prodeflux_div_eR) * sum_mk_not_q -
                 par['phi_q'] * prodeflux_div_eR) / (1 - par['phi_q'] * (1 - prodeflux_div_eR))
    D = (1 + mq_div_kq + sum_mk_not_q)  # resource competition denominator

    T = tc / tu  # ratio of charged to uncharged tRNAs
    B = R * (1 - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

    nu = nu_calc(par, tu, s)  # tRNA charging rate

    l = l_calc(par, e, B, prodeflux)  # growth rate

    psi = psi_calc(par, T)  # tRNA synthesis rate - AMENDED

    # return dx/dt for the host cell
    dxdt = jnp.array([
                         # mRNAs
                         l * par['c_a'] * par['a_a'] - (par['b_a'] + l) * m_a - kcmh*(m_a/k_a/D)*R,
                         l * Fr_calc(par, T) * par['c_r'] * par['a_r'] - (par['b_r'] + l) * m_r - kcmh*(m_r/k_r/D)*R,
                         # metabolic protein p_a
                         (e / par['n_a']) * (m_a / k_a / D) * R - l * p_a,
                         # ribosomes
                         (e / par['n_r']) * (m_r / k_r / D) * R - l * R - kcmh*B,
                         # tRNAs
                         nu * p_a - l * tc - e * B,
                         l * psi - l * tu - nu * p_a + e * B,
                         # nutrient quality assumed constant
                         0,
                         # ribosomes inactivated by chloramphenicol
                         kcmh * B - l * Bcm
                     ] +
                     circuit_ode(t, x, e, l, R, k_het, D, par, circuit_name2pos)
                     )
    return dxdt

## HYBRID TAU-LEAPING SIMULATOR
def tauleap_sim(par,  # dictionary with model parameters
                circuit_v,  # calculating the propensity vector for stochastic simulation of circuit expression
                x0,  # initial condition VECTOR
                num_circuit_genes, num_circuit_miscs, circuit_name2pos, # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder,
                sgp4j, # relevant synthetic gene parameters in jax.array form
                tf, tau, tau_odestep, tau_savetimestep, # simulation parameters: time frame, tau-leap time step, number of ODE steps in each tau-leap step
                mRNA_count_scales, S, circuit_synpos2genename, keys0, # parameter vectors for efficient simulation and reaction stoichiometry info - determined by tauleap_sim_prep!!
                avg_dynamics = False # true if considering the deterministic cas with average dynamics of random variables
                ):
    # define the arguments for finding the next state vector
    args = (
        par,  # model parameters
        circuit_name2pos,  # gene name - position in circuit vector decoder
        num_circuit_genes, num_circuit_miscs,  # number of genes and miscellaneous species in the circuit
        sgp4j,  # relevant synthetic gene parameters in jax.array form
        mRNA_count_scales, S, circuit_synpos2genename  # parameters for stochastic simulation
    )

    # time points at which we save the solution
    ts = jnp.arange(tf[0], tf[1] + tau_savetimestep/2, tau_savetimestep)

    # number of ODE steps in each tau-leap step
    ode_steps_in_tau = int(tau / tau_odestep)

    # make the retrieval of next x a lambda-function for jax.lax.scanning
    scan_step = lambda sim_state, t: tauleap_record_x(circuit_v, sim_state, t, tau, ode_steps_in_tau, args)

    # define the jac.lax.scan function
    tauleap_scan = lambda sim_state_rec0, ts: jax.lax.scan(scan_step, sim_state_rec0, ts)
    tauleap_scan_jit = jax.jit(tauleap_scan)

    # get initial conditions
    if(len(x0.shape)==1): # if x0 common for all trajectories, copy it for each trajectory
        x0s = jnp.repeat(x0.reshape((1, x0.shape[0])), keys0.shape[0], axis=0)
    else: # otherwise, leave initial conditions as is
        x0s=x0

    # initalise the simulator state: (t, x, sim_step_cntr, record_step_cntr, key, tf, xs)
    sim_state_rec0={'t': tf[0], 'x': x0s, # time, state vector
                'key': keys0, # random number generator key
                'tf': tf, # overall simulation time frame
                'save_every_n_steps': int(tau_savetimestep/tau), # tau-leap steps between record points
                'avg_dynamics': avg_dynamics # true if considering the deterministic cas with average dynamics of random variables
                }

    # vmapping - specify that we vmap over the random number generation keys
    sim_state_vmap_axes = {'t': None, 'x': 0, # time, state vector
                'key': 0, # random number generator key
                'tf': None, # overall simulation time frame
                'save_every_n_steps': None, # tau-leap steps between record points
                'avg_dynamics': None # true if considering the deterministic cas with average dynamics of random variables
                           }

    # simulate (with vmapping)
    sim_state_rec_final, xs = jax.jit(jax.vmap(tauleap_scan_jit, (sim_state_vmap_axes, None)))(sim_state_rec0, ts)

    return ts, xs, sim_state_rec_final['key']


# log the next trajectory point
def tauleap_record_x(circuit_v, # calculating the propensity vector for stochastic simulation of circuit expression
                     sim_state_record,  # simulator state
                     t,  # time of last record
                     tau,  # time step
                     ode_steps_in_tau,  # number of ODE integration steps in each tau-leap step
                     args):
    # DEFINITION OF THE ACTUAL TAU-LEAP SIMULATION STEP
    def tauleap_next_x(step_cntr,sim_state_tauleap):
        # update t
        next_t = sim_state_tauleap['t'] + tau

        # update x
        # find deterministic change in x
        det_update = tauleap_integrate_ode(sim_state_tauleap['t'], sim_state_tauleap['x'], tau, ode_steps_in_tau, args)
        # find stochastic change in x
        stoch_update = tauleap_update_stochastically(sim_state_tauleap['t'], sim_state_tauleap['x'],
                                                     tau, args, circuit_v,
                                                     sim_state_tauleap['key'], sim_state_tauleap['avg_dynamics'])
        # find next x
        next_x_tentative = sim_state_tauleap['x'] + det_update + stoch_update
        # make sure x has no negative entries
        next_x = jax.lax.select(next_x_tentative < 0, jnp.zeros_like(next_x_tentative), next_x_tentative)

        # update key
        next_key, _ = jax.random.split(sim_state_tauleap['key'], 2)

        return {
            # entries updated over the course of the tau-leap step
            't': next_t, 'x': next_x,
            'key': next_key,
            # entries unchanged over the course of the tau-leap step
            'tf': sim_state_tauleap['tf'],
            'save_every_n_steps': sim_state_tauleap['save_every_n_steps'],
            'avg_dynamics': sim_state_tauleap['avg_dynamics']
        }


    # FUNCTION BODY
    # run tau-leap integration until the next state is to be saved
    next_state_bytauleap = jax.lax.fori_loop(0,sim_state_record['save_every_n_steps'], tauleap_next_x, sim_state_record)

    # update the overall simulator state
    next_sim_state_record = {
        # entries updated over the course of the tau-leap step
        't': next_state_bytauleap['t'], 'x': next_state_bytauleap['x'],
        'key': next_state_bytauleap['key'],
        # entries unchanged
        'tf': sim_state_record['tf'],
        'save_every_n_steps': sim_state_record['save_every_n_steps'],
        'avg_dynamics': sim_state_record['avg_dynamics']
    }

    return next_sim_state_record, sim_state_record['x']


# ode integration - Euler method
def tauleap_integrate_ode(t, x, tau, ode_steps_in_tau, args):
    # solve the ODE
    def euler_step(ode_step, x):
        return x + tauleap_ode(t + ode_step_size * ode_step, x, args) * ode_step_size

    ode_step_size = tau / ode_steps_in_tau
    # integrate the ODE
    x_new= jax.lax.fori_loop(0, ode_steps_in_tau, euler_step, x)

    return x_new - x


# ode for the deterministic part of the tau-leaping simulation
def tauleap_ode(t, x, args):
    # unpack the args
    par = args[0]  # model parameters
    circuit_name2pos = args[1]  # gene name - position in circuit vector decoder
    num_circuit_genes = args[2]  # number of genes in the circuit
    num_circuit_miscs = args[3]  # number of miscellaneous species in the circuit
    kplus_het, kminus_het, n_het, d_het = args[
        4]  # unpack jax-arrayed synthetic gene parameters for calculating k values

    # give the state vector entries meaningful names
    m_a = x[0]  # metabolic gene mRNA
    m_r = x[1]  # ribosomal gene mRNA
    p_a = x[2]  # metabolic proteins
    R = x[3]  # non-inactivated ribosomes
    tc = x[4]  # charged tRNAs
    tu = x[5]  # uncharged tRNAs
    s = x[6]  # nutrient quality (constant)
    Bcm = x[7]  # INTERNAL chloramphenicol concentration (varies)
    # synthetic circuit genes and miscellaneous species can be accessed directly from x with circuit_name2pos

    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = e_calc(par, tc)

    # ribosome inactivation rate due to chloramphenicol
    kcmh = par['kcm'] * par['h_ext']

    # ribosome dissociation constants
    k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'], kcmh)
    k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'], kcmh)
    k_het = k_calc(e, kplus_het, kminus_het, n_het, kcmh)

    # overall protein degradation flux for all synthetic genes
    prodeflux = jnp.sum(
        d_het * n_het * x[8 + num_circuit_genes:8 + num_circuit_genes * 2])  # heterologous protein degradation flux
    prodeflux_div_eR = prodeflux / (e * R)  # heterologous protein degradation flux divided by eR

    # resource competition denominator
    m_het_div_k_het = jnp.sum(x[8:8 + num_circuit_genes] / k_het)  # heterologous protein synthesis flux
    sum_mk_not_q = m_a / k_a + m_r / k_r + m_het_div_k_het
    mq_div_kq = (par['phi_q'] * (1 - prodeflux_div_eR) * sum_mk_not_q -
                 par['phi_q'] * prodeflux_div_eR) / (1 - par['phi_q'] * (1 - prodeflux_div_eR))
    D = (1 + mq_div_kq + sum_mk_not_q)  # resource competition denominator

    T = tc / tu  # ratio of charged to uncharged tRNAs
    B = R * (1 - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

    nu = nu_calc(par, tu, s)  # tRNA charging rate

    l = l_calc(par, e, B, prodeflux)  # growth rate

    psi = psi_calc(par, T)  # tRNA synthesis rate - AMENDED

    # continuously, we only consider charged aa-tRNA consumption by NATIVE protein translation
    B_cont = (D - 1 - m_het_div_k_het) / D * R

    # return dx/dt for the host cell
    return jnp.array([
                         # mRNAs
                         l * par['c_a'] * par['a_a'] - (par['b_a'] + l) * m_a - kcmh * (m_a / k_a / D) * R,
                         l * Fr_calc(par, T) * par['c_r'] * par['a_r'] - (par['b_r'] + l) * m_r - kcmh * (m_r / k_r / D) * R,
                         # metabolic protein p_a
                         (e / par['n_a']) * (m_a / k_a / D) * R - l * p_a,
                         # ribosomes
                         (e / par['n_r']) * (m_r / k_r / D) * R - l * R - kcmh * B_cont,
                         # tRNAs
                         nu * p_a - l * tc - e * B_cont,
                         l * psi - l * tu - nu * p_a + e * B_cont,
                         # nutrient quality assumed constant
                         0,
                         # ribosomes inactivated by chloramphenicol
                         kcmh * B_cont - l * Bcm
                     ] +
                     [0] * (2 * num_circuit_genes) + [0] * num_circuit_miscs
                     # synthetic gene expression considered stochastically
                     )


def tauleap_update_stochastically(t, x, tau, args, circuit_v,
                                  key, avg_dynamics):
    # PREPARATION
    # unpack the arguments
    par = args[0]  # model parameters
    circuit_name2pos = args[1]  # gene name - position in circuit vector decoder
    num_circuit_genes = args[2]  # number of genes in the circuit
    num_circuit_miscs = args[3]  # number of miscellaneous species in the circuit
    kplus_het, kminus_het, n_het, d_het = args[
        4]  # unpack jax-arrayed synthetic gene parameters for calculating k values
    # stochastic simulation arguments
    mRNA_count_scales = args[5]
    S = args[6]
    circuit_synpos2genename = args[7]  # parameter vectors for efficient simulation and reaction stoichiometry info

    # give the state vector entries meaningful names
    m_a = x[0]  # metabolic gene mRNA
    m_r = x[1]  # ribosomal gene mRNA
    p_a = x[2]  # metabolic proteins
    R = x[3]  # non-inactivated ribosomes
    tc = x[4]  # charged tRNAs
    tu = x[5]  # uncharged tRNAs
    s = x[6]  # nutrient quality (constant)
    Bcm = x[7]  # INTERNAL chloramphenicol concentration (varies)
    # synthetic circuit genes and miscellaneous species can be accessed directly from x with circuit_name2pos

    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = e_calc(par, tc)

    # ribosome inactivation rate due to chloramphenicol
    kcmh = par['kcm'] * par['h_ext']

    # ribosome dissociation constants
    k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'], kcmh)
    k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'], kcmh)
    k_het = k_calc(e, kplus_het, kminus_het, n_het, kcmh)

    # overall protein degradation flux for all synthetic genes
    prodeflux = jnp.sum(
        d_het * n_het * x[8 + num_circuit_genes:8 + num_circuit_genes * 2])  # heterologous protein degradation flux
    prodeflux_div_eR = prodeflux / (e * R)  # heterologous protein degradation flux divided by eR

    # resource competition denominator
    m_het_div_k_het = jnp.sum(x[8:8 + num_circuit_genes] / k_het)  # heterologous protein synthesis flux
    sum_mk_not_q = m_a / k_a + m_r / k_r + m_het_div_k_het
    mq_div_kq = (par['phi_q'] * (1 - prodeflux_div_eR) * sum_mk_not_q -
                 par['phi_q'] * prodeflux_div_eR) / (1 - par['phi_q'] * (1 - prodeflux_div_eR))
    D = (1 + mq_div_kq + sum_mk_not_q)  # resource competition denominator

    T = tc / tu  # ratio of charged to uncharged tRNAs
    B = R * (1 - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

    nu = nu_calc(par, tu, s)  # tRNA charging rate

    l = l_calc(par, e, B, prodeflux)  # growth rate

    psi = psi_calc(par, T)  # tRNA synthesis rate - AMENDED

    # continuously, we only consider charged aa-tRNA consumption by NATIVE protein translation
    B_cont = (D - 1 - m_het_div_k_het) / D * R

    # FIND REACTION PROPENSITIES
    v = jnp.array(circuit_v(t, x,  # time, cell state, external inputs
                            e, l,  # translation elongation rate, growth rate
                            R,  # ribosome count in the cell, resource
                            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
                            mRNA_count_scales,
                            par,  # system parameters
                            circuit_name2pos
                            ))

    # RETURN THE NUMBER OF TIMES THAT EACH REACTION HAS OCCURRED
    # or take the average if we're considering the deterministic case
    return jax.lax.select(avg_dynamics, jnp.matmul(S, v * tau), jnp.matmul(S, jax.random.poisson(key=key, lam=v * tau)))


# preparatory step creating the objects necessary for tau-leap simulation - DO NOT JIT
def tauleap_sim_prep(par,  # dictionary with model parameters
                     num_circuit_genes, num_circuit_miscs, circuit_name2pos,
                     x0_det,  # initial condition found deterministically
                     key_seeds=[0] # random key seed(s)
                     ):
    # DICTIONARY mapping positions of a synthetic gene in the list of all synthetic genes to their names
    circuit_synpos2genename = {}
    for name in circuit_name2pos.keys():
        if (name[0] == 'm'):
            circuit_synpos2genename[circuit_name2pos[name] - 8] = name[2:]

    # PARAMETER VECTORS FOR EFFICIENT SIMULATION
    # mRNA count scalings
    mRNA_count_scales_np = np.zeros((num_circuit_genes))
    for i in range(0, num_circuit_genes):
        mRNA_count_scales_np[i] = par['n_' + circuit_synpos2genename[i]] / 25
    mRNA_count_scales = jnp.array(mRNA_count_scales_np)

    # STOICHIOMETRY MATRIX
    S = gen_stoich_mat(par,
                       circuit_name2pos, circuit_synpos2genename,
                       num_circuit_genes, num_circuit_miscs,
                       mRNA_count_scales)

    # MAKE THE INITIAL CONDITION FOR TAU-LEAPING WITH APPROPRIATE INTEGER VALUES OF RANDOM VARIABLES
    x0_det_np = np.array(x0_det)  # convert to numpy array for convenience
    x0_tauleap_mrnas = np.multiply(np.round(x0_det_np[8:8 + num_circuit_genes] / mRNA_count_scales_np),
                                   mRNA_count_scales_np)  # synthetic mRNA counts - must be a multiple of the corresponding scaling factors that acount for translation by multiple ribosomes
    x0_tauleap_prots_and_miscs = np.round(
        x0_det_np[8 + num_circuit_genes:])  # synthetic protein and miscellaneous species counts
    x0_tauleap = np.concatenate((x0_det_np[0:8], x0_tauleap_mrnas, x0_tauleap_prots_and_miscs), axis=0)

    # STARTING RANDOM NUMBER GENERATION KEY
    key_seeds_jnp = jnp.array(key_seeds)
    keys0 = jax.vmap(jax.random.PRNGKey)(key_seeds_jnp)

    return mRNA_count_scales, S, x0_tauleap, circuit_synpos2genename, keys0


# generate the stocihiometry matrix for heterologous genes (plus return total number of stochastic reactions) - DO NOT JIT
def gen_stoich_mat(par,
                   circuit_name2pos, circuit_synpos2genename,
                   num_circuit_genes, num_circuit_miscs,
                   mRNA_count_scales
                   ):
    # unpack stochionmetry args
    mRNA_count_scales_np = np.array(mRNA_count_scales)

    # find the number of stochastic reactions that can occur
    num_stoch_reactions = num_circuit_genes * (3 +  # synthesis/degradation/dilution of mRNA
                                               1 +  # ribosome removal due to chloramphenicol
                                               3)  # synthesis/degradation/dilution of protein
    if (('m_anti' in circuit_name2pos.keys()) and ('m_act' in circuit_name2pos.keys())):  # plus, might need to model mutual annihilation of sRNAs
        num_stoch_reactions += 3 # m_act:m_anti binding, degradation and dilution of the bound complex

    # initialise (in numpy format)
    S = np.zeros((8 + 2 * num_circuit_genes + num_circuit_miscs, num_stoch_reactions))

    # initialise thge counter of reactions in S
    reaction_cntr = 0

    # mRNA - reactions common for all genes
    for i in range(0, num_circuit_genes):
        S[8 + i, reaction_cntr] = mRNA_count_scales_np[i]  # mRNA synthesis
        reaction_cntr += 1
        S[8 + i, reaction_cntr] = -mRNA_count_scales_np[i]  # mRNA degradation
        reaction_cntr += 1
        S[8 + i, reaction_cntr] = -mRNA_count_scales_np[i]  # mRNA dilution
        reaction_cntr += 1
        # ribosome and mRNA removal due to chloramphenicol
        S[8 + i, reaction_cntr] = -mRNA_count_scales_np[i] # mRNA removal
        S[3, reaction_cntr] = -mRNA_count_scales_np[i] # ribosome removal
        reaction_cntr +=1

    # m_act:m_anti binding (if AIF motif present)
    if (('m_anti' in circuit_name2pos.keys()) and ('m_act' in circuit_name2pos.keys())):
        S[circuit_name2pos['m_act'], reaction_cntr] = -mRNA_count_scales_np[circuit_name2pos['mscale_act']]  # actuator mRNA removal
        S[circuit_name2pos['m_anti'], reaction_cntr] = -mRNA_count_scales_np[circuit_name2pos['mscale_anti']]    # annihilator mRNA removal
        reaction_cntr += 1

    # protein - reactions common for all genes
    for i in range(0, num_circuit_genes):
        S[8 + num_circuit_genes + i, reaction_cntr] = 1  # protein synthesis
        S[4, reaction_cntr] = -par[
            'n_' + circuit_synpos2genename[i]]  # includes tRNA unchraging during translation(-tc)
        S[5, reaction_cntr] = par['n_' + circuit_synpos2genename[i]]  # includes tRNA unchraging during translation(+tu)
        reaction_cntr += 1
        S[8 + num_circuit_genes + i, reaction_cntr] = -1  # protein degradation
        reaction_cntr += 1
        S[8 + num_circuit_genes + i, reaction_cntr] = -1  # protein dilution
        reaction_cntr += 1

    # bound actuator-annihilator complex (if AIF motif present)
    if ('bound' in circuit_name2pos.keys()):
        # degradation of the bound complex
        S[circuit_name2pos['bound'], reaction_cntr] = -1
        reaction_cntr += 1
        # dilution of the bound complex
        S[circuit_name2pos['bound'], reaction_cntr] = -1
        reaction_cntr += 1

    return jnp.array(S)


## MAIN FUNCTION (for testing)
def main():
    # set up jax
    jax.config.update('jax_platform_name', 'gpu')
    jax.config.update("jax_enable_x64", True)

    # initialise cell model
    cellmodel_auxil = CellModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes
    par = cellmodel_auxil.default_params()  # get default parameter values
    init_conds = cellmodel_auxil.default_init_conds(par)  # get default initial conditions

    # load synthetic gene circuit - WITH HYBRID SIMULATION SUPPORT
    ode_with_circuit, circuit_F_calc, par, init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles, circuit_v = cellmodel_auxil.add_circuit(
        oneconstit_init,
        oneconstit_ode,
        oneconstit_F_calc,
        par, init_conds,
        # propensity calculation function for hybrid simulations
        oneconstit_v)

    # burdensome gene
    par['c_xtra'] = 100.0
    par['a_xtra'] = 5000.0

    # culture medium
    init_conds['s'] = 0.5
    #par['h_ext']=2000
    # DETERMINISTIC SIMULATION TO FIND THE STARTING STEADY STATE
    # set simulation parameters
    tf = (0, 50)  # simulation time frame
    savetimestep = 0.1  # save time step
    dt_max = 0.1  # maximum integration step
    rtol = 1e-6  # relative tolerance for the ODE solver
    atol = 1e-6  # absolute tolerance for the ODE solver

    # simulate
    sol = ode_sim(par,  # dictionary with model parameters
                  ode_with_circuit,  # ODE function for the cell with synthetic circuit
                  cellmodel_auxil.x0_from_init_conds(init_conds, circuit_genes, circuit_miscs),
                  # initial condition VECTOR
                  len(circuit_genes), len(circuit_miscs), circuit_name2pos,
                  # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                  cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes),
                  # synthetic gene parameters for calculating k values
                  tf, jnp.arange(tf[0], tf[1], savetimestep),  # time axis for saving the system's state
                  rtol,
                  atol)  # simulation parameters: when to save the system's state, relative and absolute tolerances)   # simulation parameters: time frame, save time step, relative and absolute tolerances
    ts = np.array(sol.ts)
    xs = np.array(sol.ys)
    # det_steady_x = jnp.concatenate((sol.ys[-1, 0:8], jnp.round(sol.ys[-1, 8:])))
    det_steady_x = sol.ys[-1, :]

    # HYBRID SIMULATION
    # tau-leap hybrid simulation parameters
    tf_hybrid = (tf[-1], tf[-1] + 1)  # simulation time frame
    tau = 1e-6  # simulation time step
    tau_odestep = 1e-7  # number of ODE integration steps in a single tau-leap step (smaller than tau)
    tau_savetimestep = 1e-2  # save time step a multiple of tau

    # simulate
    timer = time.time()
    mRNA_count_scales, S, x0_tauleap, circuit_synpos2genename, keys0 = tauleap_sim_prep(par, len(circuit_genes),
                                                                                        len(circuit_miscs),
                                                                                        circuit_name2pos, det_steady_x,
                                                                                        key_seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8 ,9])
    ts_jnp, xs_jnp, final_keys = tauleap_sim(par,  # dictionary with model parameters
                                 circuit_v,  # circuit reaction propensity calculator
                                 x0_tauleap, # initial condition VECTOR (processed to make sure random variables are appropriate integers)
                                 len(circuit_genes), len(circuit_miscs), circuit_name2pos,
                                 cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes), # synthetic gene parameters for calculating k values
                                 tf_hybrid, tau, tau_odestep, tau_savetimestep,  # simulation parameters: time frame, tau leap step size, number of ode integration steps in a single tau leap step
                                 mRNA_count_scales, S, circuit_synpos2genename, # mRNA count scaling factor, stoichiometry matrix, synthetic gene number in list of synth. genes to name decoder
                                 keys0) # starting random number genereation key

    # concatenate the results with the deterministic simulation
    ts = np.concatenate((ts, np.array(ts_jnp)))
    xs_first = np.concatenate((xs, np.array(xs_jnp[1]))) # getting the results from the first random number generator key in vmap
    xss = np.concatenate((xs*np.ones((keys0.shape[0],1,1)),np.array(xs_jnp)),axis=1) # getting the results from all vmapped trajectories

    print('tau-leap simulation time: ', time.time() - timer)

    # PLOT - HOST CELL MODEL
    bkplot.output_file(filename="cellmodel_sim.html",
                       title="Cell Model Simulation")  # set up bokeh output file
    mass_fig = cellmodel_auxil.plot_protein_masses(ts, xs_first, par, circuit_genes)  # plot simulation results
    nat_mrna_fig, nat_prot_fig, nat_trna_fig, h_fig = cellmodel_auxil.plot_native_concentrations_multiple(ts, xss, par,
                                                                                                 circuit_genes,
                                                                                              tspan=(tf[-1] - (tf_hybrid[-1] - tf_hybrid[0]), tf_hybrid[-1]),
                                                                                                          simtraj_alpha=0.1)  # plot simulation results
    l_figure, e_figure, Fr_figure, ppGpp_figure, nu_figure, D_figure = cellmodel_auxil.plot_phys_variables_multiple(ts, xss, par,
                                                                                                           circuit_genes,
                                                                                                           circuit_miscs,
                                                                                                           circuit_name2pos,
                                                                                                           tspan=(tf[-1] - (tf_hybrid[-1] -tf_hybrid[0]),tf_hybrid[-1]),
                                                                                                        simtraj_alpha=0.1)  # plot simulation results
    bkplot.save(bklayouts.grid([[None, nat_mrna_fig, nat_prot_fig],
                                [nat_trna_fig, h_fig, l_figure],
                                [e_figure, Fr_figure, D_figure]]))

    # PLOT - SYNTHETIC GENE CIRCUIT
    bkplot.output_file(filename="circuit_sim.html",
                       title="Synthetic Gene Circuit Simulation")  # set up bokeh output file
    het_mrna_fig, het_prot_fig, misc_fig = cellmodel_auxil.plot_circuit_concentrations_multiple(ts, xss, par, circuit_genes,
                                                                                       circuit_miscs, circuit_name2pos,
                                                                                       circuit_styles, tspan=( tf[-1] - (tf_hybrid[-1] - tf_hybrid[0]), tf_hybrid[-1]),
                                                                                                simtraj_alpha=0.1)  # plot simulation results
    F_fig = cellmodel_auxil.plot_circuit_regulation_multiple(ts, xss, par, circuit_F_calc, circuit_genes, circuit_miscs,
                                                    circuit_name2pos, circuit_styles, tspan=(tf[-1] - (tf_hybrid[-1] - tf_hybrid[0]), tf_hybrid[-1]),
                                                    simtraj_alpha=0.1)  # plot simulation results
    bkplot.save(bklayouts.grid([[het_mrna_fig, het_prot_fig, misc_fig],
                                [F_fig, None, None]]))

    return


## MAIN CALL
if __name__ == '__main__':
    main()