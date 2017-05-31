# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 16:29:03 2015

Over a period of time, it computes the activity and courants of different "neurons-nodes"
connected to each other; from a set of parameters.

Functions:
- c_solve(ndarray r, int len_t, double *I0_list,
             int len_I0, int nb_edges, int *zone_from,
             int *zone_to, double *weight, double *tau, double * t_delay,
             double step, ndarray ext_stim,
             int *list_zone_from_ignored, int *list_zone_to_ignored,
             double t_to_steady_state, int nb_edges_ignored)
- solve(params_dict,step,t,regions,connexion_ignored,
          dict_get_default,ext_stim,t_to_steady_state)


@author: alexandre
"""

#from cython_gsl cimport *

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from libc.math cimport copysign, ceil

#from cython.view cimport array as cvarray # Access to arrays
from copy import deepcopy

import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

import pandas as pd

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ----------- FUNCTIONS ----------- #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

cdef extern from "numpy/arrayobject.h": # To construct numpy array
    ctypedef int intp
    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef intp *dimensions
        cdef intp *strides
        cdef int flags

### TEST ###
#DTYPE = np.double
#ctypedef np.double_t DTYPE_t

cdef extern from "linear_positive.h": # Linear_positive function in C
    double F_pos(double x)



# Return r after simulation; function to solve ODE with time loop #
cdef c_solve(ndarray r, int len_t, double *I0_list,
             int len_I0, int nb_edges, int *zone_from,
             int *zone_to, double *weight, double *tau, double *t_delay,
             double step, ndarray ext_stim,
             int *list_zone_from_ignored, int *list_zone_to_ignored,
             double t_to_steady_state, int nb_edges_ignored):

### TEST ###
#cdef np.ndarray c_solve(np.ndarray r, int len_t, double *I0_list,
#             int len_I0, int nb_edges, int *zone_from,
#             int *zone_to, double *weight, double *tau, double *t_delay,
#             double step, np.ndarray ext_stim,
#             int *list_zone_from_ignored, int *list_zone_to_ignored,
#             double t_to_steady_state, int nb_edges_ignored):
    # ------------------------ Initialisation ------------------------ #

    cdef double i_connect #
    cdef int ind_delay #
    cdef int it, j, k, i, m # Loop's indices

    # Malloc of current input of each neuron #
    cdef double *I_connections = <double *>malloc((len_I0) * sizeof(double))
    if not I_connections: # If allocation doesn't work
        raise MemoryError()


    # ------------------------ Time loop ------------------------ #

    for it from 1 <= it < len_t:

        # Initialisation loop for current input of each neuron #
        for i from 0 <= i < len_I0:
            I_connections[i] = 0.0

        # Computation of all current input for each neuron #
        for j from 0 <= j < nb_edges: # Through all edges

            ind_delay = <int>(ceil(<double>(it) - t_delay[j])) # Int type of the time iteration

            for m from 0 <= m < nb_edges_ignored: # Through all edges ignored
                # If edge ignored and time of simulation is enough to reach steady_state #
                if (zone_from[j] == list_zone_from_ignored[m]) and (zone_to[j] == list_zone_to_ignored[m]) and (it > t_to_steady_state):
                    ind_delay = <int> (t_to_steady_state)    # Time value to steady_state
                    break # The last value of the loop must be the ignored edge (if it exists)

            if ind_delay < 1: # If the time iteration is negative beauce of delay
                ind_delay = 1 # Because we do ind_delay - 1
            if ind_delay >= len_t: # When delay are negative; which is not realistic
                #print('Warning t_delay = %f \n', t_delay[j])
                ind_delay = len_t - 1
                for i from 0 <= i < len_I0:
                    I_connections[i] = 1e300 # Huge error
                break

            # Connections
            I_connections[zone_to[j]] += r[ind_delay-1, zone_from[j]] * weight[j]

        # Computation of r for each region
        for k from 0 <= k < len_I0:
            r[it, k] = r[it-1,k] + (-r[it-1,k] + F_pos(I0_list[k] + I_connections[k] + ext_stim[it,k])) *step/tau[k]

    # Free allocations #
    free(I_connections)


    return r

# Wrapper
def solve(params_dict,step,t,regions,connexion_ignored,
          dict_get_default,ext_stim,t_to_steady_state):


    # ------------------- INITIALISATION/PARAMETERS ------------------------ #

    WT = params_dict['WT']

    # ------------------------ To count the number of edges ------------------------ #

    cdef int nb_edges = 0
    for items in WT.values():
        nb_edges += len(items.keys())

    # Connections ignored #
    cdef int nb_edges_ignored = 0
    for items in connexion_ignored.values():
        nb_edges_ignored += len(items)

    # ------------------------ Parameters ------------------------ #
    I0 = params_dict['I0']
    #    tau_EPSC = params_dict['tau_EPSC']

    cdef int len_I0 = int(len(regions)) # Length of the list of init

    # ------------------------ Initialisation of malloc ------------------------ #

    # Allocate number * sizeof(double) bytes of memory #
    cdef double *I0_list = <double *>malloc((len_I0) * sizeof(double))
    if not I0_list: # If allocation doesn't work
        raise MemoryError()

    cdef int *list_zone_from = <int *>malloc((nb_edges) * sizeof(int))
    if not list_zone_from: # If allocation doesn't work
        raise MemoryError()
    cdef int *list_zone_to = <int *>malloc((nb_edges) * sizeof(int))
    if not list_zone_to: # If allocation doesn't work
        raise MemoryError()
    cdef double *list_weight = <double *>malloc((nb_edges) * sizeof(double))
    if not list_weight: # If allocation doesn't work
        raise MemoryError()
    cdef double *list_tau = <double *>malloc((nb_edges) * sizeof(double))
    if not list_tau: # If allocation doesn't work
        raise MemoryError()
    cdef double *list_t_delay = <double *>malloc((nb_edges) * sizeof(double))
    if not list_t_delay: # If allocation doesn't work
        raise MemoryError()



    cdef int *list_zone_from_ignored = <int *>malloc((nb_edges_ignored) * sizeof(int))
    if not list_zone_from_ignored: # If allocation doesn't work
        raise MemoryError()
    cdef int *list_zone_to_ignored = <int *>malloc((nb_edges_ignored) * sizeof(int))
    if not list_zone_to_ignored: # If allocation doesn't work
        raise MemoryError()

    # ------------------------ List of connections ignored ------------------------ #

    cdef int counter = 0
    for zone_from, dict_zone_to in WT.iteritems():
        for zone_to, dict_weight_tau in dict_zone_to.iteritems():

            list_zone_from[counter] = regions.index(zone_from)
            list_zone_to[counter] = regions.index(zone_to)
            list_weight[counter] = dict_weight_tau['w']
            list_tau[counter] = dict_weight_tau['tau']
            list_t_delay[counter] = dict_weight_tau['t_delay']/step

            counter += 1

    # ------------------------ List of connections ignored ------------------------ #

    cdef int counter_ignored = 0
    for zone_from_ignored, list_ignored_zone_to in connexion_ignored.iteritems():
        for zone_to_ignored in list_ignored_zone_to:
            list_zone_from_ignored[counter_ignored] = regions.index(zone_from_ignored)
            list_zone_to_ignored[counter_ignored] = regions.index(zone_to_ignored)

            counter_ignored += 1


    # ------------------------ Initialisation of numpy.array ------------------------ #

    cdef int len_t = int(len(t)) # Length of the list of time

    # Initialisation of an empty np.ndarray #

    ### TEST ###
    #    cdef np.ndarray[DTYPE_t, ndim=2] r = np.zeros([len_t, len_I0], dtype=DTYPE)
    #    cdef np.ndarray[DTYPE_t, ndim=2] ext_stimulation = ext_stim

    cdef np.ndarray[np.float64_t, ndim = 2] r = np.zeros((len_t, len_I0), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] ext_stimulation = ext_stim

    cdef int l
    for l from 0 <= l < len_I0: # Fill the rest of the array of initialisation

        if I0.values()[l] > 0:
            I0_list[l] = I0.values()[l]
        else:
            I0_list[l] = 0.0

        r[0,l] = I0_list[l]

    # ------------------------ Call to time loop function ------------------------ #
    #    r_temp = c_solve(r, len_t, I0_list, len_I0, nb_edges,
    #                     list_zone_from, list_zone_to, list_weight,
    #                     list_tau, list_t_delay, step, ext_stimulation,
    #                     list_zone_from_ignored, list_zone_to_ignored,
    #                     t_to_steady_state, nb_edges_ignored)

    ### TEST ###
    r = c_solve(r, len_t, I0_list, len_I0, nb_edges,
                 list_zone_from, list_zone_to, list_weight,
                 list_tau, list_t_delay, step, ext_stimulation,
                 list_zone_from_ignored, list_zone_to_ignored,
                 t_to_steady_state, nb_edges_ignored)

    # ------------------------ Free allocation ------------------------ #
    free(I0_list)
    free(list_zone_from)
    free(list_zone_to)
    free(list_weight)
    free(list_tau)
    free(list_t_delay)
    free(list_zone_from_ignored)
    free(list_zone_to_ignored)


    #    print(type(r_temp))
    #r_temp = np.array(r_temp) # To have a python object wich is destroy in a proper way
    #    print(type(r_temp))

    #r_temp = deepcopy(r)
    #PyArray_CopyInto(r_temp, r)

    r_temp = np.zeros((len_t, len_I0))
    ### TEST ###
    for i_1 from 0 <= i_1 < len_t:
        for i_2 from 0 <= i_2 < len_I0:
            value = r[i_1, i_2]
            r_temp[i_1,i_2] = value



    ### TEST ###
    #    for i_1 from 0 <= i_1 < len_t:
    #        for i_2 from 0 <= i_2 < len_I0:
    #            value = r_temp[i_1, i_2]
    #            r[i_1,i_2] = value
    #
    #    memcpy(<double*> h.data,r_temp,2*fft*fc*sizeof(double))

    ### TEST ###
    #    cdef DTYPE_t value
    #
    #    for i_1 from 0 <= i_1 < len_t:
    #        for i_2 from 0 <= i_2 < len_I0:
    #            value = r_temp[i_1, i_2]
    #            r[i_1,i_2] = value

    #    del(r_temp)

    #    return l
    #    return r_temp
    return r