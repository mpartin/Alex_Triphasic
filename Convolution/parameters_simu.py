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

# SHAM or 60HDA cells #
#cell_type = 'sham'
cell_type = '6ohda'

#input_ampl_volt = '_15V' # Between 5V - 25V
input_ampl_volt = '_all'

# -------- Cells we want to exclude ------- #
#removing_input_ampl_volt = [] # Between 5V - 25V
removing_input_ampl_volt = ['_5V', '_25V', '_30V'] # Between 5V - 25V


if input_ampl_volt == '_all':
    input_voltage = 1.0 # Voltage has no impact on simulation
else:
    input_voltage = input_ampl_volt.replace('_','')
    input_voltage = float(input_voltage.replace('V',''))


## ------- 6OHDA inhibition+late_excitation Asier ------- #
#cells_to_fit_temp_Asier = ['OHDA3_neuron5', 'OHDA3_neuron10', 'OHDA1_neuron8']
## ------- 6OHDA inhibition Asier ------- #
#cells_to_fit_temp_Asier = ['OHDA2_neuron1', 'OHDA1_neuron2', 'OHDA1_neuron3', 'OHDA1_neuron4', 'OHDA1_neuron5', 'OHDA1_neuron9', 'OHDA1_neuron10', 'OHDA1_neuron13', 'OHDA1_neuron14']
## ------- 6OHDA Non-responder Asier ------- #
#cells_to_fit_temp_Asier = ['OHDA1_neuron6', 'OHDA1_neuron7']
## ------- 6OHDA Excitation Asier ------- #
#cells_to_fit_temp_Asier = ['OHDA1_neuron11']
#
## ------- 6OHDA inhibition+late_excitation Bertrand     ------- #
#cells_to_fit_temp_Bertrand = ['DSK7_6936', 'DSK8_6968', 'DSK_7356', 'DSK6_8164']
## ------- 6OHDA inhibition Bertrand ------- #
#cells_to_fit_temp_Bertrand = ['DSK8_6938', 'DSK7_7210', 'DSK8_7238', 'DSK6_7250', 'DSK_7372']
## ------- 6OHDA excitation Bertrand ------- #
#cells_to_fit_temp_Bertrand = ['DSK8_6994', 'DSK7_7030', 'DSK6_7486', 'DSK7_7944']
## ------- 6OHDA Non-responder Bertrand ------- #
#cells_to_fit_temp_Bertrand = ['DSK8_7006', 'DSK8_7356', 'DSK8_7422', 'DSK6_7822']
## ------- 6OHDA triphasic Bertrand ------- #
cells_to_fit_temp_Bertrand = ['DSK4_7014', 'DSK_7238', 'DSK_8056']
## ------- 6OHDA excitation+inhibition Bertrand ------- #
#cells_to_fit_temp_Bertrand = ['DSK_7234', 'DSK_7326', 'DSK_7418', 'DSK_7536']

## ------- SHAM inhibition+late excitation Asier ------- #
#cells_to_fit_temp_Asier = ['SHAM4_neuron1', 'SHAM4_neuron3', 'SHAM4_neuron8', 'SHAM5_neuron8', 'SHAM5_neuron12', 'SHAM6_neuron7']
## ------- SHAM excitation+inhibition Asier ------- #
#cells_to_fit_temp_Asier = ['SHAM4_neuron2', 'SHAM6_neuron1', 'SHAM6_neuron2', 'SHAM6_neuron4']
## ------- SHAM inhibition Asier ------- #
#cells_to_fit_temp_Asier = ['SHAM4_neuron4', 'SHAM4_neuron5', 'SHAM4_neuron7', 'SHAM5_neuron9', 'SHAM5_neuron13', 'SHAM6_neuron6']
## ------- SHAM Excitation Asier ------- #
#cells_to_fit_temp_Asier = ['SHAM4_neuron6', 'SHAM5_neuron1', 'SHAM5_neuron5', 'SHAM6_neuron5']
## ------- Non-responder Asier ------- #
#cells_to_fit_temp_Asier = ['SHAM5_neuron10']
# ------- Triphasic Asier ------- #
#cells_to_fit_temp_Asier = ['SHAM5_neuron11', 'SHAM6_neuron3']

cells_to_fit_temp_Asier = []

## ------- SHAM inhibition+late excitation Bertrand ------- #
#cells_to_fit_temp_Bertrand = ['DSK4_7698', 'DSK6_7654', 'DSK6_7746']
## ------- SHAM inhibition Bertrand ------- #
#cells_to_fit_temp_Bertrand = ['DSK5_7180', 'DSK5_8272', 'DSK6_7150', 'DSK7_7764', 'DSK7_7798', 'DSK7_7800', 'DSK7_8058']
# ------- Triphasic Bertrand ------- #
#cells_to_fit_temp_Bertrand = ['DSK5_7634', 'DSK5_8030', 'DSK5_8082', 'DSK7_7714', 'DSK5_8272']
## ------- Non-responder Bertrand ------- #
#cells_to_fit_temp_Bertrand = ['DSK6_7858']


#cells_to_fit_temp_Bertrand = []

# ------- BOTH ------- #
cells_to_fit_temp_Both = cells_to_fit_temp_Asier +cells_to_fit_temp_Bertrand

# --------- Choose cells to fit --------- #
cells_to_fit = []
#cells_to_fit = cells_to_fit_temp_Asier
cells_to_fit = cells_to_fit_temp_Both

#maniper_name = '_Asier'
#maniper_name = '_Bertrand'
maniper_name = '_both'


params_fitted_dictionary_to_load = "params_fitted_dictionary_" +cell_type.upper() +maniper_name +input_ampl_volt +'.p'

params_fitted_dictionary_to_save = params_fitted_dictionary_to_load

# ---------------------------- Input and ouput path ----------------------------#

rel_path_data = '/../../Data/2015_10/' +cell_type.upper() +maniper_name +'/'
pickle_convolution_data = 'Convolution_data_' +cell_type.upper() +maniper_name +'.p'

# ---------------------------- Name for mean of recording cells -------------- #
mean_data = cell_type.upper() + '_mean'

# ASIER parameters #
# The first artefact is at 50 ms #
if maniper_name =="_Asier":
    time_first_artefact = 20.0 # msec
elif maniper_name =="_Bertrand":
    time_first_artefact = 20.0 # msec
elif maniper_name =="_both":
    time_first_artefact = 20.0 # msec


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
