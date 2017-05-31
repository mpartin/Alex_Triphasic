# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 16:50:16 2015

Parameters to include in Convoluted_Asier_data, Model_Basal_ganglia_Triphasic and solve

Functions:
    -params_dict()
    -plot_data_frame(list_data,
                     Title='Empty',
                     xlabel='Empty',
                     ylabel='Empty',
                     fontsize = 20,
                     legend = True,
                     plot_style = 'ggplot')


@author: alexandre
"""

from __future__ import division
import pandas as pd
#import playdoh
import cPickle as pickle
import numpy as np
from copy import deepcopy

from scipy.ndimage.filters import gaussian_filter

# General functions #
import sys
sys.path.append('/Users/alexandre/Desktop/INRIA/Prog/Python/')

from utils import plot_data_frame, pydot_graph
from utils.dictionary import pop_nested_dictionary, read_nested_dict_to_list, write_nested_dict_from_list, merge_two_dicts
from utils import Rect

###############################################################################
# --------------------------------- PARAMS --------------------------------- #
###############################################################################
# ----------- Fitting ----------- #
fit = False # Boolean for fitting or not #

# ----------- Meta-fitting ----------- #
# When we want to do several time the same fit to have a distribution #
parameters_distribution_with_fit = False # Distribution of parameter over fit or not
nb_repetition_of_fit = 2 # Number of time we call fit function

# ----------- Plotting ----------- #
graph = False # Boolean for graph or not #

# String return if the key is not found with dict.get() #
dict_get_default = 'default'

# --------------------------------------------------------------------------- #
# ---------------------------- Read data file ------------------------- #
# --------------------------------------------------------------------------- #

cell_type = 'dyskinetic_cells'

#input_ampl_volt = '_25V' # Between 5V - 25V
input_ampl_volt = '_all'

# -------- Cells we want to exclude ------- #
#removing_input_ampl_volt = [] # Between 5V - 25V
removing_input_ampl_volt = ['_5V', '_25V'] # Between 5V - 25V


if input_ampl_volt == '_all':
    input_voltage = 1.0 # Voltage has no impact on simulation
else:
    input_voltage = input_ampl_volt.replace('_','')
    input_voltage = float(input_voltage.replace('V',''))


# --------- Choose cells to fit --------- #
# ------------- Inhibition ------------ #
#cells_to_fit =['Dysk6_neuron1', 'Dysk6_neuron2', 'Dysk9_neuron8', 'Dysk25_neuron2', 'Dysk25_neuron4', 'Dysk35_neuron1', 'Dysk38_neuron2', 'Dysk39_neuron5', 'Dysk39_neuron7', 'Dysk40_neuron6']

# ----------- Triphasic ------------- #
#cells_to_fit =['Dysk6_neuron3', 'Dysk6_neuron6', 'Dysk6_neuron7', 'Dysk9_neuron_exp', 'Dysk24_neuron1', 'Dysk24_neuron6', 'Dysk39_neuron4', 'Dysk39_neuron_exp']

# -------------- Early excitation -------------- #
#cells_to_fit =['Dysk6_neuron4', 'Dysk9_neuron3', 'Dysk35_neuron5']

# ----------------- Excitation ----------------- #
#cells_to_fit =['Dysk24_neuron4', 'Dysk25_neuron5', 'Dysk35_neuron_exp', 'Dysk40_neuron2', 'Dysk40_neuron3']

# -------------- Inhibition + late excitation --------------- #
#cells_to_fit =['Dysk6_neuron_exp', 'Dysk6_neuron5', 'Dysk9_neuron4', 'Dysk9_neuron5', 'Dysk24_neuron5', 'Dysk24_neuron_exp', 'Dysk25_neuron8', 'Dysk35_neuron6', 'Dysk39_neuron1', 'Dysk39_neuron6', 'Dysk39_neuron8', 'Dysk39_neuron9']

# -------------- Late excitation -------------- #
#cells_to_fit =['Dysk25_neuron7', 'Dysk35_neuron4', 'Dysk38_neuron7', 'Dysk38_neuron_exp']

# ----------- non-responder -------------- #
#cells_to_fit =['Dysk9_neuron1', 'Dysk24_neuron3', 'Dysk25_neuron1', 'Dysk25_neuron3', 'Dysk25_neuron6', 'Dysk35_neuron3', 'Dysk35_neuron7', 'Dysk38_neuron1', 'Dysk38_neuron3', 'Dysk38_neuron4', 'Dysk38_neuron5', 'Dysk38_neuron6', 'Dysk39_neuron2', 'Dysk40_neuron1', 'Dysk40_neuron4', 'Dysk40_neuron5']

# ------------ Early excitation + inhibition -------------- #
#cells_to_fit =['Dysk6_neuron8', 'Dysk9_neuron2', 'Dysk24_neuron2', 'Dysk25_neuron9', 'Dysk25_neuron_exp', 'Dysk35_neuron2', 'Dysk35_neuron8', 'Dysk39_neuron3', 'Dysk40_neuron_exp']



cells_to_fit = []

params_fitted_dictionary_to_load = "params_fitted_dictionary_" +cell_type.upper() +input_ampl_volt +'.p'

params_fitted_dictionary_to_save = params_fitted_dictionary_to_load

# ---------------------------- Input and ouput path ----------------------------#

rel_path_data = '/../../Data/2016_01/' +cell_type.upper() +'/'
pickle_convolution_data = 'Convolution_data_' +cell_type.upper() +'.p'

# ---------------------------- Name for mean of recording cells -------------- #
mean_data = cell_type.upper() + '_mean'

#time_first_artefact = 25.0 # msec
time_first_artefact = 20.0 # msec

if time_first_artefact:
    pickle_convolution_data = 'Convolution_data_' +cell_type.upper() +'.p'


# Duration of a cycle is 1 second #
max_cycle_dur = 1010.0 # ms | To be sure to get entire cycle

time_step = 0.1 # msec
t_start = 0.0 # msec

# ---------------------------- Information ----------------------------------- #

# Stimulus in Asier data are at 50 ms #
t_stim_Asier_data = time_first_artefact # ms

# Region of the brain recorded #
region_fitted = 'SNR'

datas = pd.read_pickle(pickle_convolution_data)

list_names_cell = [] # Initialisation

for cell_input_name in datas.columns: # Loop on each cell

    if input_ampl_volt == '_all':
        list_names_cell.append(cell_input_name)
    else:
        if cell_input_name.find(input_ampl_volt) >= 0: # find() return -1 when it doesn't find the good one
            list_names_cell.append(cell_input_name)

    if len(removing_input_ampl_volt) > 0:
        for remov_ampl in removing_input_ampl_volt:
            if cell_input_name.find(remov_ampl) >= 0:
                list_names_cell.remove(cell_input_name)



# ------------ If there are specific cells to fit ------------ #
if len(cells_to_fit):
    list_names_cell_temp = [] # Initialisation
    for cell_input_name in list_names_cell: # Loop on each cell
        for cell_to_fit in cells_to_fit:
            if cell_input_name.find(cell_to_fit) >= 0: # find() return -1 when it doesn't find the good one
                list_names_cell_temp.append(cell_input_name)

    list_names_cell = list_names_cell_temp

data_to_fit = datas[list_names_cell].mean(1) # Mean of all cells
    # Mean of data is the last column #
    #data_to_fit = datas.iloc[:,-1].dropna() # dropna() to avoid RuntimeWarning because of NaNs.
    #    plt.plot(data_to_fit.index, data_to_fit)


# ----------------------------------------------------------------------------- #
# --------------------------- Simulation -------------------------------------- #
# ----------------------------------------------------------------------------- #

# --------------------------- Sigmoid function -------------------------------------- #
n = 2.0 # Power
K = 0.1 # Pente
A = 0.5 # Amplitude max of instantaneous frequency in ms-1


# --------------------------- Stimulation -------------------------------------- #

# Time to the system to reach steady state #
t_to_steady_state = 300.0 # ms
# Equivalent of t_to_steady_state in iteration #
it_to_steady_state = int(t_to_steady_state/time_step)


# --------------------------------- Time vector --------------------------------- #
step = time_step # Time step of simulation in msec
time_simulation = 500.0
t_st = 0.0 # Starting time of the simulation in msec
t_end = t_to_steady_state +time_simulation # Ending time of the simulation in msec

# Time vector in step#
t = np.arange(t_st,t_end,step)

# ------------------------- Parameters fittable --------------------------------- #

# --------------------- Parameter's dictionary of the network ------------------ #
#params_dictionary = params_dict()

# --------------------- Load a dictionary from a pickle file --------------------- #
#params_dictionary = pickle.load(open(params_fitted_dictionary_to_load, "rb"))

params_dictionary = pickle.load(open("params_dictionary_fixed_values_A.p", "rb"))

# Names of regions considered #
regions = params_dictionary['I0'].keys()
# -------------------------------- External stimulus ------------------------- #
# Starting time of the stimulations in msec #
# Duration time of the stimulations in msec #
# Maximum amplitude of the stimulations in ???#
# It's a dictionary in case there are several regions stimulated #
ext_stim_dict ={}


ext_stim_dict = {
            'Cortex': {'t_stim': [t_to_steady_state], 'dur_stim': [0.6], 'max_ampl_stim': [5.5]},
                }

# ----------------------------------------------------------------------------#


# List of stimulated regions #
stim_regions = ext_stim_dict.keys()

# Region of the brain stimulated #
region_stimulated = stim_regions[0]
ind_region_stimulated = regions.index(stim_regions[0])

# Time to adjust the simulation to data if no stimulation #
t_adjust = t_to_steady_state
# List of stimulation time to get the min #
t_stim_temp = []

# External stimulations for each region at each iteration #
ext_stim = np.zeros((len(t),len(regions)))

# Loop for all regions stimulated #
for zone, params_signal in ext_stim_dict.items():
    for i in range(len(params_signal['t_stim'])):
        # List of all times of stimulation #
        t_stim_temp.append(params_signal['t_stim'][i])
        # Gaussian filter on a rectangular signal (same size of the simulation) #
        #        ext_stim_temp = gaussian_filter(Rect(t,params_signal['t_stim'][i], params_signal['t_stim'][i]+params_signal['dur_stim'][i]), 1/step)
        ext_stim_temp = Rect(t,params_signal['t_stim'][i], params_signal['t_stim'][i]+params_signal['dur_stim'][i])

        # Normalization and set of the amplitude #
        ext_stim[:,regions.index(zone)] += (ext_stim_temp * params_signal['max_ampl_stim'][i] / max(ext_stim_temp))
    # The for loop works as a if; if there is no stimulation, t_adjust stay to its init value, else we keep the early one #
    t_adjust = min(t_stim_temp) - t_stim_Asier_data


# -------------- Connexions to ignore; by keeping them at steady-state  --------- #
# We keep the initial value #
connexion_ignored = {} # Empty dictionary to avoid error in solve()
# Experiment to ignore pathway; values need to be array in case there are several of them #
#connexion_ignored = {'Striatum' : ['GPe']} # Indirect pathway ignored
#connexion_ignored = {'Cortex' : ['STN']} # Hyperirect pathway ignored
#connexion_ignored = {'Striatum' : ['SNR']} # Direct pathway ignored
#connexion_ignored = {'STN' : ['SNR']} # Indiret and hyperdirect pathways ignored
#connexion_ignored = {'Cortex' : ['STN'], 'Striatum' : ['GPe','SNR']} # All pathway ignored
#connexion_ignored = {'Th' : ['Cortex']}
#connexion_ignored = {'GPe' : ['STN']}
