# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 14:13:03 2015

Simulation of a network of brain regions, where each region is symbolized by
a neuron and its firing rate. Fitting to data of convoluted spike serie;
equivalent to firing rate.

Functions:
 - read_nested_dict_to_list(dictionary)
 - write_nested_dict_from_list(list_value, old_dictionary)
 - fitter_params(params_dictionary)
     -- error(pop, params_dictionary)
     -- resf(pop_vect)


The global error is the square sum between the simulation and the data
or a basal activity (mean of the data)


@author: alexandre
"""

import solve_cython as sc # solve file

import matplotlib.pylab as plt
import numpy as np
import math
# General functions #
import sys
sys.path.append('/Users/alexandre/Desktop/INRIA/Prog/Python/')

from utils import plot_data_frame, pydot_networkx_neato_graph
from utils.dictionary import read_nested_dict_to_list, write_nested_dict_from_list, merge_two_dicts, read_nested_dict_to_list_of_unique_key

from parameters_simu import * # global parameters

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ----------- MAIN ----------- #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


if __name__ == "__main__":

    # Simulation of the model over time; dataFrame #
    r = sc.solve(params_dictionary,step,t,regions,connexion_ignored,dict_get_default,ext_stim,it_to_steady_state)

    # numpy.ndarray() to DataFrame
    r = pd.DataFrame(r, index = t, columns = regions)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # ----------- PLOT ----------- #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Simulation synchronized with Asier data #
    simu_data = r.copy()
    simu_data.index = (r.index.values - t_adjust)
    simu_data = simu_data[simu_data.index > 0]

    # DataFrames to plot #
    #    df_to_plot = [simu_data[region_fitted],
    #            data_to_fit.iloc[0:int(500/time_step)],
    #            simu_data[stim_regions[0]]]

    df_to_plot = [
            data_to_fit.iloc[0:int(500/time_step)]
                 ]

    # Plot of Asier data and simulation fit #
    plot_data_frame(df_to_plot, 'Triphasic fit', 'Time in msec', 'Activity', xlim = (time_first_artefact-5.0, time_first_artefact+60.0), legend_label=['SNR fit','SNR data','Cortex fit'])


    # -------------------------- Histograms with standard error envelop ------------------ #

    import seaborn as sns
    datas_hist_list = [] # Initialisation
    datas_hist = datas[list_names_cell]

    for i in xrange(datas_hist.shape[1]): # To exclude convol mean column
        datas_hist_list += [datas_hist.iloc[:,i].values]

    datas_hist_array = np.array(datas_hist_list)
    #    axis = sns.tsplot(data=datas_hist_array, time=datas.index.astype(float))
    #    axis.set(xlim=(time_first_artefact-5.0, time_first_artefact+60.0))
    axis = sns.tsplot(data=datas_hist_array, time=datas.index.astype(float), ci=[68, 95]) # Respectively one std and two std from mean (std = sigma)


