# -*- coding: utf-8 -*-
"""
    Created on Jan 2015

    Extract convoluted spike series from Asier's data files.

    Functions:
    - readMyFiles(filePath)
    - readAsierFiles_2014_12_20(list_DataFrame)
    - cycle_treatment(time_serie, cycle_start, cycle_dur)
    - regular_step_serie(round_decim, time_step, cycle_dur, time_serie, cycle_start = 0)
    - pre_treatment(round_decim, time_step)
    - convolution_Asier_data(rel_path_data, output_pickle_data)

    @author: Alexandre
"""


import os # Deal with path
import pylab as plt # Plot
from matplotlib.backends.backend_pdf import PdfPages

from parameters_conv import * # global parameters


# General functions #
import sys
sys.path.append('/Users/alexandre/Desktop/INRIA/Prog/Python/')

from utils import convolution_on_right
from utils import readMyFiles_txt

from matplotlib.backends.backend_pdf import PdfPages


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ----------- FUNCTIONS ----------- #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

#------- Read Asier's files 2015_05 -------#

# Return list of DataFrame of voltage #
def ReadAsierFiles_2015_05(rel_path_data = rel_path_data):
    # ------- Path ------- #
    mypath = os.getcwd()
    mypath += rel_path_data

    list_df_voltage = readMyFiles_txt(mypath)

    return list_df_voltage

def ReadAsierFiles_2015_08_artefact(mypath):

    filePath = os.getcwd()
    filePath += mypath

    list_DataFrame = [] # List of dataframe of each file
    # Get all files in the given folder #
    fileListing = os.listdir(filePath)
    for myFile_name in fileListing:
        # Create the file path #
        myFilePath = os.path.join(filePath, myFile_name)
        # Check it's a file; not a sub folder #
        if (os.path.isfile(myFilePath) and (myFilePath.endswith('.txt') or
                                            myFilePath.endswith('.TXT'))):
            # DataFrame of the text file #
            my_df = pd.read_csv(myFilePath, sep ='\t', header = None)

            my_df.columns = [myFile_name for i in range(0,my_df.shape[1])] # Rename column
            list_DataFrame.append(my_df.iloc[:,1])

    return list_DataFrame

# Return a serie where value is an integer and index is timing #
def spike_serie_index(time_spike_temp, value_spike = 1.0):

    # Round index and keep it in string format; the computer doesn't need to approximate it #
    time_spike_str = pd.Index([round_decim%ind for ind in time_spike_temp.values])
    # Serie with index = time of spike and value = 1 #
    time_spike = pd.Series(value_spike*np.ones(len(time_spike_str)), index = time_spike_str)

    return time_spike

# Return list of artefact and spike series concatenated #
def artefact_and_spike_serie(list_df_artefact, list_df_spike):

    list_artefact_and_spike_serie = []
    # For each cell and input value #
    for i in xrange(len(list_df_artefact)):

        # Sec to msec #
        artefact_serie = spike_serie_index(list_df_artefact[i]*sec_to_msec, value_spike = 2) # Time in index

        # We keep spikes after the first stimulus and we remove NaNs #
        spike_serie_temp = list_df_spike[i][list_df_spike[i]>list_df_artefact[i][0]].dropna()

        # Sec to msec #
        spike_serie = spike_serie_index(spike_serie_temp*sec_to_msec, value_spike = 1) # Time in index

        # Concatenation of artefacts and spikes #
        artefact_and_spike_serie = pd.concat([artefact_serie, spike_serie]) # Concat series | No ordered index because it's strings
        artefact_and_spike_serie = artefact_and_spike_serie.groupby(level=0).max() # If two indexes have the same value, it is because of a round problem. We consider artefact is winning on spike.
        list_artefact_and_spike_serie.append(artefact_and_spike_serie) # Add to the list

    return list_artefact_and_spike_serie

# Return list of spikes series in msec #
def cycle_treatment(list_artefact_and_spike_serie):

    new_list_artefact_and_spike_serie = []
    list_index_time_serie = []

    for serie in list_artefact_and_spike_serie:

        index_time_serie_temp = pd.Series(serie.index).convert_objects(convert_numeric =True) # Convert index to float
        index_time_serie_temp = pd.concat([index_time_serie_temp, pd.Series(serie.values)], axis =1)
        index_time_serie_temp = index_time_serie_temp.sort(columns=0) # Sort by values
        index_time_serie_temp = index_time_serie_temp.reset_index(drop=True) # Reset index without adding a new column

        for i in xrange(len(index_time_serie_temp)):
            if index_time_serie_temp.iloc[i,1] == 2:
                index_time_serie_temp.iloc[i:,0] = index_time_serie_temp.iloc[i:,0] -index_time_serie_temp.iloc[i,0] + time_first_artefact # Substraction and time artefact

        list_index_time_serie.append(index_time_serie_temp) # Add to list


        spike_serie_temp = spike_serie_index(index_time_serie_temp.iloc[:,0], value_spike = 1)

        # Grouped by index values and sum when it's the same but index values are strings so they are not ordered by values #
        spike_serie_temp = spike_serie_temp.groupby(level=0).sum()

        #        spike_serie_temp.ix[20.0] = 0.0 # Remove artefact

        new_list_artefact_and_spike_serie.append(spike_serie_temp)

    return new_list_artefact_and_spike_serie, list_index_time_serie

# Return spike serie with regular time step #
def regular_step_serie(list_spike_serie, max_cycle_dur, time_vector_start = 0.0):

    list_time_serie = []
    # Array with regular time step #
    index_temp = np.arange(time_vector_start, max_cycle_dur, time_step)
    # Round index and keep it in string format; the computer doesn't need to approximate it #
    index_str = pd.Index([round_decim%ind for ind in index_temp])
    # Index with a regular time step #
    zeros_serie = pd.Series(index = index_str)

    for spike_serie in list_spike_serie:

        # Regular time serie; filled with index and zeros #
        time_serie = spike_serie.reindex(zeros_serie.index, fill_value = 0)
        time_serie.ix[round_decim%time_first_artefact] = 0.0
        list_time_serie.append(time_serie)

    return list_time_serie



# Return list of histogram array #
def Histogram_on_series(list_serie, time_step, max_hist, bin_hist = 1.0):

    list_hist_array = [] # Initialisation

    for serie in list_serie: # For each serie in list

        len_hist = int(round(max_hist/bin_hist)) # Length of the histogram
        hist_array = np.zeros(len_hist) # Initialisation

        for i in xrange(int(len_hist)): # For each bin count the number of spike
            hist_array[i] = serie.ix[i*bin_hist:(i+1)*bin_hist-time_step].sum()

        list_hist_array.append(hist_array)

    return list_hist_array

# Histogram and convolution #
def Histograms_on_series_2(list_index_time_serie, df_convolution, num_bins = max_cycle_dur, xlim_min = 15.0, xlim_max = 100.0):

    for i in range(len(list_index_time_serie)):

        with PdfPages('Hist_' +list_names[i].replace('_a.txt','.pdf')) as pdf:

            plt.figure()
            n,bins,patches = plt.hist(list_index_time_serie[i].iloc[:,0], num_bins, range=(time_first_artefact, num_bins +time_first_artefact))
            plt.plot(df_convolution.iloc[:,i].index, df_convolution.iloc[:,i], 'r')
            plt.xlim(xlim_min, xlim_max)
            pdf.savefig()
            plt.close()


# Return dataframe of convolution on Asier data #
def convolution_data(list_serie, list_names):

    df_convolution = pd.DataFrame() # Initialisation

    counter = 0 # Initialisation
    for serie in list_serie: # For each serie in list

        # Convolution needs two arrays #
        convolution = convolution_on_right(array_which_convolve, serie.iloc[nb_iter_conv_start:nb_iter_conv_end].values)

        df_convolution[list_names[counter]] = convolution # Fill the df
        counter +=1 # Increment


    # ------- Mean over all cells ------- #
    convol_mean = df_convolution.mean(axis = 1) # Get the mean on each row of the array
    # Add the mean convolution to the DataFrame #
    df_convolution['convol_mean'] = convol_mean

    df_convolution.index = serie.iloc[nb_iter_conv_start:nb_iter_conv_end].index # Set index

    # Save to pickle file | the location depend of the current workpath #
    df_convolution.to_pickle(pickle_convolution_data)

    return df_convolution



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ----------- MAIN ----------- #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# To not be executed if imported #
if __name__ == "__main__":

    # Get artefact data #
    rel_path_artefact = rel_path_data + 'artefact/'
    list_df_artefact = ReadAsierFiles_2015_08_artefact(rel_path_artefact)

    # Get list of file names in order #
    list_names = [df_artefact.name for df_artefact in list_df_artefact]

    # Get spikes data #
    rel_path_spike = rel_path_data + 'spike/'
    list_df_spike = ReadAsierFiles_2015_05(rel_path_spike)

    # List of artefact and spike series #
    list_artefact_and_spike_serie = artefact_and_spike_serie(list_df_artefact, list_df_spike)
    # List of spike series in cycle #
    list_cycle_spike_serie, list_index_time_serie = cycle_treatment(list_artefact_and_spike_serie)

    # List of spike regular serie in cycle #
    list_cycle_spike_regular_serie = regular_step_serie(list_cycle_spike_serie, max_cycle_dur)

    # DataFrame of convolution #
    df_convolution = convolution_data(list_cycle_spike_regular_serie, list_names)

    # Histograms #
    #    Histograms_on_series_2(list_index_time_serie, df_convolution)

    # Save data in pickle file #
    #pickle.dump(df_convolution, open('pickle_convolution_data.p','wb'))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # ----------- PLOT ----------- #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    #    plt.plot(list_artefact_and_spike_serie[15].index, list_artefact_and_spike_serie[15], '-ro')
    #    plt.plot(list_cycle_spike_regular_serie[15].index, list_cycle_spike_regular_serie[15].values)
    #    plt.plot(df_convolution.iloc[:,15].index, df_convolution.iloc[:,15])

    #    pos = list_names.index('SHAM6_neuron3_5V_artefact_timing.txt')
    #
    #    list_artefact_and_spike_serie[pos][(list_artefact_and_spike_serie[pos] == 2).values]
    #
    #    plt.plot(list_artefact_and_spike_serie[pos].index, list_artefact_and_spike_serie[pos], '-ro')
    #    plt.plot(list_cycle_spike_serie[pos].index, list_cycle_spike_serie[pos].values)
    #    plt.plot(list_cycle_spike_regular_serie[pos].index, list_cycle_spike_regular_serie[pos].values)
    #    plt.plot(df_convolution.iloc[:,pos].index, df_convolution.iloc[:,pos])





