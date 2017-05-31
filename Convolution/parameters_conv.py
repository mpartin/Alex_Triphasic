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

# String return if the key is not found with dict.get() #
dict_get_default = 'default'

# --------------------------------------------------------------------------- #
# ---------------------------- Read data file ------------------------- #
# --------------------------------------------------------------------------- #

cell_type = 'dyskinetic_cells'

# ---------------------------- Input and ouput path ----------------------------#

rel_path_data = '/../../Data/2015_10/' +cell_type.upper() +'/'

pickle_convolution_data = 'Convolution_data_' +cell_type.upper() +'.p'

# ---------------------------- Name for mean of recording cells -------------- #
mean_data = cell_type.upper() + '_mean'

# ---------------------------- Extraction of datas ---------------------------- #
sec_to_msec = 1000 # Multiply to transform second into msec.
nb_decim = 1 # Number of decimals
round_decim = '%.'+ str(nb_decim) + 'f' # Magic line which round with nb_decim and save the result in a string


# Detection of artefact and spikes #
threshold_artefact = 2.0 # mV
threshold_spike = 0.3 # mV

# Cleaning of artefact #
low_window_artefact = 0.25 # msec | Duration of the rise of the first peak
up_window_artefact = 7.0 # msec | Duration of the two peaks of the artefact
width_artefact = 1.0 # msec | Width of the first peak of the artefact
start_artefact = 0.0 # msec | it takes generally 0.2 msec to reach the top

# Cleaning of spike #
low_window_spike = 0.2 # msec
up_window_spike = 0.2 # msec

# Indice to differentiate each event #
artefact_ind = 2.0 # No unit
spike_ind = 1.0 # No unit
noise_ind = 0.0 # No unit
default_value = -100.0 # Special index value, easy to exclude


#time_first_artefact = 25.0 # msec
time_first_artefact = 20.0 # msec


# Duration of a cycle is 1 second #
max_cycle_dur = 1010.0 # ms | To be sure to get entire cycle

time_step = 0.1 # msec
t_start = 0.0 # msec

# ------------- Iteration conversion ------------ #
nb_iter_low_window_artefact = int(low_window_artefact/time_step)
nb_iter_up_window_artefact = int(up_window_artefact/time_step)
nb_iter_width_artefact = int(width_artefact/time_step)
nb_iter_start_artefact = int(start_artefact/time_step)
nb_iter_low_window_spike = int(low_window_spike/time_step)
nb_iter_up_window_spike = int(up_window_spike/time_step)

nb_iter_first_artefact = time_first_artefact/time_step # No unit
nb_iter_to_end_cycle = (max_cycle_dur-time_first_artefact)/time_step # No unit


volt_index_cycle = np.arange(t_start, max_cycle_dur, time_step) # Each iteration of the recording in msec
#volt_index_cycle = [round_decim % i for i in volt_index_cycle] # Transform to string otherwise pandas is doing bullshits with index

# ------------- Histogram ------------- #
bin_hist = 1.0 # ms
max_hist = max_cycle_dur # ms

# ---------------------------- Convolution ----------------------------------- #

# ------ Convolution function --------- #
convolv_start = 0.0 # ms | Start of the convolution vector
convolv_step = time_step # ms | Step of the convolution vector
convolv_stop = 1.0 # ms | Stop of the convolution vector
#convolv_stop = 0.10 # ms | Stop of the convolution vector


convolv_decrease = 0.0 # Parameter of decrease of the exponential convolution

len_convolv_range = int((convolv_stop-convolv_start)/convolv_step) # Length of the convolution vector

array_which_convolve = np.zeros(len_convolv_range) # Initialisation of the convolution vector

convolv_range = np.arange(convolv_start,convolv_stop,convolv_step) # Time vector to apply the function of convolution
expo_func = lambda x, a: np.exp(-a*x) # Exponential function

for i in xrange(len_convolv_range): # Loop to fill the convolution vector
    array_which_convolve[i] = expo_func(convolv_range[i], convolv_decrease)

# ------- Convolution data --------- #

conv_start = 0.0 # ms
conv_end = 500.0 # ms
nb_iter_conv_start = int(conv_start/time_step) # No unit
nb_iter_conv_end = int(conv_end/time_step) # No unit

# ---------------------------- Information ----------------------------------- #

# Stimulus in Asier data are at 50 ms #
t_stim_Asier_data = time_first_artefact # ms

# Region of the brain recorded #
region_fitted = 'SNR'
