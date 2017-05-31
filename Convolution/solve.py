# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 16:29:03 2015

Over a period of time, it computes the activity and courants of different "neurons-nodes"
connected to each other; from a set of parameters.

Functions:
- S(x)
- Sb(s, n=2.0, K=0.1, A=0.5)
- H
- Rect
- solve(params_dict)


@author: alexandre
"""

import numpy as np
import pandas as pd
from math import ceil

# General functions #
import sys
sys.path.append('/Users/alexandre/Desktop/INRIA/Prog/Python/')

from utils import F

def solve(params_dict,step,t,regions,connexion_ignored,dict_get_default,ext_stim,t_to_steady_state):

    # ----------- PARAMS ----------- #
    
    # Initial firing rate, time constant EPSC and ??? #
    I0, tauEPSC = params_dict['I0'], params_dict['tauEPSC']
    
    for I0_key, I0_value in I0.items():
        if I0_value < 0.0:
            print 'WARNING: I0 value out of bounds set to 0.0', I0_key, I0_value
            I0[I0_key] = 0.0
                   
    # ----------- Weight and tau values of connections between regions ----------- #
    WT = params_dict['WT']

    # ----------- Matrix of connections ----------- #                      
    # pandas.Panel.swapaxes() is for interchanging major and minor axes in a Panel with the proper values #
    pnWT = pd.Panel.from_dict(WT).swapaxes(1, 0)
    
    # ----------- Matrix of weight ----------- #
    pnW = pnWT['w']
    # ----------- Matrix of tau in msec----------- #
    pnT = pnWT['tau']              
    # Matrix of tau in step #
    pnT = pnT / step


    # ----------- INITIALISATION ----------- #
    
    # Arrays r, I and ext_stim have their columns in the order of the vector regions #
    
    # Array of instantaneous firing rate; frequency in 1/msec #
    r = np.zeros((len(t),len(regions)))
    # Initial instantaneous firing rate; F(I0) #
    r[0,:] = [F(I0[zone]) for zone in regions]
    
    # Array of input currents #
    I = np.zeros((len(t),len(regions)))
    # Basal input current #
    I_basal = [I0[zone] for zone in regions]
    
    # Initial instantaneous firing rate; I_basal #
    I[0,:] = I_basal
    
    # Vector of total input for each region at one iteration #
    itot = np.zeros(len(regions))
        
    # ----------- TIME'S LOOP ----------- #
    
    for it in np.arange(1,len(t)):
            # On average, the input from a given pre-synaptic neuron is proportional to its firing rate rj #
            I[it,:] = r[it-1,:]
                    
            # Inputs from connection #
            I_connections = np.zeros((len(regions),len(regions)))               
            # Get the tau and weight vectors for each connection and link it to the r activity #
            for zone_to in pnT.columns:
                    # Connection delays (tau) vector of each departure regions #
                    tau_s = pnT.ix[:,zone_to].dropna()
                    # Connection weights vector of each departure regions #
                    weights = pnW.ix[:,zone_to].dropna()
                    # Get the departure regions vector #                
                    zone_departure = tau_s.keys()
                    
                    # Get the tau and weight values for each connection and link it to the r activity #
                    for zone_from in zone_departure:
                        # Connection delays (tau) value from each regions #
                        tau = tau_s[zone_from]
                        # Connection weights vector from each regions #
                        weight = weights[zone_from]
                        
                        # Correspondant time of the delay #
                        ind_delay = it - tau
                        # ceil() round always to the higher integer; transform to the correspondant iteration of the delay #                    
                        ind_delay = ceil(ind_delay) if ind_delay>=0 else 0    
    
                        try:
                            # To ignore connexion and replace their activity by steady-state activity #
                            # If the zone_from is in the dictionary keys AND  # If the zone_to is in the dictionary value                                   
                            if ((zone_from in connexion_ignored.keys()) and (zone_to in connexion_ignored.get(zone_from, dict_get_default)) and (it > t_to_steady_state)):
                                    # We replace by initial value #
                                    i_connect = I[t_to_steady_state, regions.index(zone_from)] * weight
                            else:
                                # Input from each connections #
                                i_connect = I[ind_delay, regions.index(zone_from)] * weight
       

                        except Exception, e:
                            print e
                            print 'tau', tau
                            print 'weight', weight 
                            print'ind_delay', ind_delay
                            i_connect = 0.0
                        # Array of input from each connections #              
                        I_connections[regions.index(zone_from),regions.index(zone_to)] = i_connect

            # Vector of total input for each region at one iteration #            
            itot = I_basal + ext_stim[it,:] + sum(I_connections)
            
            r[it,:] = r[it-1,:] + (-r[it-1,:] + [F(i_total) for i_total in itot]) *step/tauEPSC
                
                
    # ----------- Array to DataFrame ----------- #
    I = pd.DataFrame(I, index = t, columns = regions)
    r = pd.DataFrame(r, index = t, columns = regions)
    
    return r, I
