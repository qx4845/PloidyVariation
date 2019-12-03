#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:38:43 2017

@author: stephenwedekind

"""
### Initialize: 
import numpy as np
#import scipy.linalg as LA
#from numpy import linalg as NLA
#import ipdb
#import scipy.sparse
from scipy.optimize import minimize
#from itertools import repeat
#from scipy.misc import comb
#import os
#import scipy.io as sio
import matplotlib.pyplot as plt
#from joblib import Parallel, delayed
import time
from datetime import timedelta
start_time = time.time()

modelNum = 1
crop_hrs = 10

global deltaT, timeRes, t_f, maxT, nChromo

deltaT = 1800 #600   ### Number of seconds between measurements
timeRes = deltaT

#t_f = 54000 #15hrs = 54000sec       ### Maximum time for MONTE CARLO simulation in units of SECONDS
t_f = 72000 #20hrs = 72000sec       ### Maximum time for MONTE CARLO simulation in units of SECONDS
#t_f = 90000 #25hrs = 90000sec       ### Maximum time for MONTE CARLO simulation in units of SECONDS
maxT = int(t_f/deltaT)              ### Maximum time for MONTE CARLO simulation in units of FRAMES
#maxT = 100
MCMC_maxT_hours = int(t_f/3600)

version = "%shr_PLM_Predict_%shr_MCMC_M%s_0.2.5" %(str(int(crop_hrs)), str(int(MCMC_maxT_hours)), str(int(modelNum)))

global cap
cap = 1 #6 ### 'M' a.k.a. maximum number of timesteps for l_gamma to cycle through before dividing (for a given cell)

#### Initial Guesses for Lagrange Multipliers (use below notation for M1, M2, M3):  ### In Taylor's notation (use for M4, M5, M6, M7):
#h_C = -10      #-6.251127895058754          ### Cell Division                      ### 'h_g' or 'h_gamma'
#h_alpha = -10  #-3.7606289000541437         ### Chromosome Duplication             ### 'h_C'
#h_A = -0.1                                  ### Partitioning                       ### 'h_P'
#h_K_Ca = -0.1                               ### Crosstalk                          ### 'K_gC' or 'K_gammaC'
#### Initial Guesses for Lagrange Multipliers (Taylor's notation):
h_C = -10
h_g = -10
h_P = -0.1
h_KgC = -0.1 
h_KgC2 = -0.1

global numFits
numFits = 3

comparison_crowbar = 1

#### Lagrange multipliers to be used in Monte Carlo simulation (if bypassing inferred multipliers):
#global MC_lagrange
### M1: [h_C, h_a]
#MC_lagrange = [-6.947, -2.688, 0]
### M2: [h_C, h_a, h_A]
#MC_lagrange = [-6.291, -2.679, -0.757]
### M3: [h_C, h_a, h_A, h_K_Ca]
#MC_lagrange = [-7.432, -2.708, -0.777, 0.162]
### M4: [h_C, h_g]
##MC_lagrange = [-2.689, 0.088]
### M5: [h_C, h_g, h_P]
##MC_lagrange = [-2.681, 0.092, -0.777]
### M6: [h_C, h_g, h_P, h_KgC]
#MC_lagrange = [-2.880, -1.546, -0.782, 0.309]
### M7: [h_C, h_g, h_P, h_KgC, h_KgC2]
#MC_lagrange = [-2.826, -5.435, -0.771, 1.674, -0.111]

global num_MCMC_trials, max_MCMC_NC
num_MCMC_trials = 1000
max_MCMC_NC = 300
#numTrials = 100
#numTries = numTrials

test_div_time = 0    ### Division-Time range [test_div_time, test_div_time+1) for which to test the number of occurances in both Exp and MCMC data. 

#firstGuess = [h_alpha, h_beta, h_A, h_B, K_A_beta, K_B_alpha]
#firstGuess = [h_C, h_alpha, h_A, h_K_Ca]

if modelNum == 7:
    firstGuess = [h_C, h_g, h_P, h_KgC, h_KgC2]
elif modelNum == 6 or modelNum == 3:
    firstGuess = [h_C, h_g, h_P, h_KgC]
elif modelNum == 5 or modelNum == 2:
    firstGuess = [h_C, h_g, h_P]
elif modelNum == 4 or modelNum == 1:
    firstGuess = [h_C, h_g]

#print('\nInitial Guesses for Lagrange Multipliers: \n[h_C, h_alpha, h_A] = ' + str(bestGuess) + '\n')

print("**************************************************************")
print("Begin: Experimental data analysis ...")
print("**************************************************************\n")
#print("______________________________________________________________\n")

#print("Frame Rate:\n'deltaT' = 30 minutes = 1800 seconds.\n")
print("Frame Rate: 30 minutes = 1800 seconds.\n")

#######################################################################################################
### Load the data from which to derive Likelihoods:                                                 ###
#######################################################################################################
global numChrom
### Load Synthetic Data:
#temp_file = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/MaxCal_ChromosomeCopying_CellDividing_ChromosomePartitioning/Gillespie_Data/saving_every_10_mins/CellDiv_Gillespie_chromosome_count_v3.6.npz')
# Load Experimental Data:
temp_file = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/Experimental_Data/ExperimentalData_PersistenceFilter.npz')
#temp_file = np.load('/home/swedeki3/PloidyVariation/ExperimentalData/ExperimentalData_PersistenceFilter.npz')

numChrom = temp_file['n_C_Exp']

## (For now) Taylor says:
### Weird division in the last movie... Fix it later... ###
#n_C_Exp = n_C_Exp[:-1]
## Which translates to:
numChrom = numChrom[:-1]

### Now*, Taylor says (*but when, though? (not here... for now -SW 01/03/2019)) :
#### Weird division in the last movie... ###
#numChrom[-1] = numChrom[-1][:,:-4]
#numChrom[-1] = numChrom[-1][~np.all(np.isnan(numChrom[-1]),axis=1),:]
##### Eliminating 1 to 3 N_C transitions ### ### (a.k.a. Taylor's Crowbar)
##for numTrial in range(len(numChrom)):
##    for numCell in range(len(numChrom[numTrial])):
##        for numStep in range(len(numChrom[numTrial][numCell]) - 1):
##            while numChrom[numTrial][numCell,numStep + 1] > 2*numChrom[numTrial][numCell,numStep]:
##                numChrom[numTrial][numCell,numStep + 1] -= 1
##        del numStep
##    del numCell
##del numTrial

########
global len_numChrom
len_numChrom = len(numChrom)

######################################################################################################################################################################################################
### For plotting entire Experimental Data set for COMPARISON ONLY (Even though only 'crop_hrs' worth was used for inference)
######################################################################################################################################################################################################
temp_file_plot = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/Experimental_Data/ExperimentalData_PersistenceFilter.npz')
numChrom_plot = temp_file_plot['n_C_Exp']
#numChrom_plot = temp_file['n_C_Exp']
numChrom_plot = numChrom_plot[:-1]

if comparison_crowbar == 1:
##########################################################################################################
#### Crowbar COMPARISON data to only have MaxCal-possible transitions (i.e. never more than doubles):  ###
##########################################################################################################
#### Exp. Crowbar:               
    for nTry in range(len(numChrom_plot)):
        for tSlice in range(numChrom_plot[nTry].shape[1] -1):
    #        for cellNum in range(len(N_Chrom[nTry][tSlice])):
            for cellNum in range(numChrom_plot[nTry].shape[0]):
                if numChrom_plot[nTry][cellNum, tSlice+1] > 2*numChrom_plot[nTry][cellNum, tSlice]:
                    numChrom_plot[nTry][cellNum, tSlice+1] = 2*numChrom_plot[nTry][cellNum, tSlice]
    del nTry, tSlice, cellNum

#########################################################################################################
### Crowbar data to only have MaxCal-possible transitions (i.e. never more than doubles):             ###
#########################################################################################################
### Exp. Crowbar:               
for nTry in range(len(numChrom)):
    for tSlice in range(numChrom[nTry].shape[1] -1):
#        for cellNum in range(len(N_Chrom[nTry][tSlice])):
        for cellNum in range(numChrom[nTry].shape[0]):
            if numChrom[nTry][cellNum, tSlice+1] > 2*numChrom[nTry][cellNum, tSlice]:
                numChrom[nTry][cellNum, tSlice+1] = 2*numChrom[nTry][cellNum, tSlice]
del nTry, tSlice, cellNum

#########################################################################################################
### Crop data to include only the first 'crop_hrs' # of hours:                                                         ###
#########################################################################################################
death_rows = [[] for x in range(len(numChrom))]
for nTry in range(len(numChrom)):
    if numChrom[nTry].shape[1] > crop_hrs*(3600/deltaT):
        numChrom[nTry] = numChrom[nTry][:, :int(crop_hrs*(3600/deltaT))]
        for cellNum in range(numChrom[nTry].shape[0]):
            if np.all(np.isnan(numChrom[nTry][cellNum, :]), axis=0):
                death_rows[nTry].append(cellNum)
#                numChrom[nTry] = np.delete(numChrom[nTry], cellNum, 0)
#del nTry, cellNum      

for nTry in range(len(numChrom)):
    if len(death_rows[nTry]) > 0:
        numChrom[nTry] = np.delete(numChrom[nTry], death_rows[nTry], 0)
        
###########################################################################################################################
### Find number of choromosomes in each dividing cell (experiment):
###########################################################################################################################
#num_chrom_in_div_cell_exp = []
#for nTry in range(len(numChrom)):
#    for cellNum in range(numChrom[nTry].shape[0]):               
#        for tSlice in range(numChrom[nTry].shape[1] - 1):
#            if np.all([~np.isnan(numChrom[nTry][cellNum, tSlice]), np.isnan(numChrom[nTry][cellNum, tSlice + 1])], axis=0):
#                num_chrom_in_div_cell_exp.append(numChrom[nTry][cellNum, tSlice])
#del nTry, tSlice, cellNum   
num_chrom_in_div_cell_exp = []
for nTry in range(len(numChrom_plot)):
    for cellNum in range(numChrom_plot[nTry].shape[0]):               
        for tSlice in range(numChrom_plot[nTry].shape[1] - 1):
            if np.all([~np.isnan(numChrom_plot[nTry][cellNum, tSlice]), np.isnan(numChrom_plot[nTry][cellNum, tSlice + 1])], axis=0):
                num_chrom_in_div_cell_exp.append(numChrom_plot[nTry][cellNum, tSlice])
del nTry, tSlice, cellNum         

###########################################################################################################################
numTrials = len(numChrom)
numTries = numTrials
#num_MCMC_trials = numTrials

frameLengths = []
#max_numFrames = 0
#### Calculate max trajectory time:
for nTry in range(len(numChrom)):
    frameLengths.append(numChrom[nTry].shape[1])
#    if numChrom[nTry].shape[1] > max_numFrames:
#        max_numFrames = numChrom[nTry].shape[1]
max_numFrames = max(frameLengths)

max_exp_time_secs = max_numFrames * deltaT
max_exp_time_hrs = max_numFrames * deltaT / 3600

print("Maximum experimental trajectory time:\n" + str(np.round(max_exp_time_hrs, 5)) + " hours \n= " + str(np.round(max_exp_time_secs, 5)) + " seconds\n= " + str(max_numFrames) + " frames.\n")

#global maxT
#t_f = int(max_exp_time_secs) #25hrs = 90000sec       ### Maximum time for MONTE CARLO simulation in units of SECONDS
#maxT = int(t_f/deltaT)    

#########################################################################################################
### Count the number of measurements made in utilized experimental data set:                          ###
#########################################################################################################
dataCounter = []
for nTry in range(len(numChrom)):
    for tSlice in np.arange(1, numChrom[nTry].shape[1]):
        dataCounter.append( len(np.where(~np.isnan(numChrom[nTry][:,tSlice]))[0]) )
totalDataPoints = np.sum(dataCounter)

print("Number of truncated experimental measurements:\n" + str(int(totalDataPoints)) + " data-points.\n")

dataCounter_plot = []
for nTry in range(len(numChrom_plot)):
    for tSlice in np.arange(1, numChrom_plot[nTry].shape[1]):
        dataCounter_plot.append( len(np.where(~np.isnan(numChrom_plot[nTry][:,tSlice]))[0]) )
totalDataPoints_plot = np.sum(dataCounter_plot)

print("Total number of experimental measurements:\n" + str(int(totalDataPoints_plot)) + " data-points.\n")

###########################################################################################################################
#### Establish maximum observed number of chromosomes (Experiment):                                                     ###
###########################################################################################################################
#global NCG_max, max_NC, NC_plot_max
#NCG_max = 0   
#for z in range(numTries):
#    for y in range(numChrom[z].shape[1]):
#        if int(np.nanmax(numChrom[z][:, y])) > NCG_max:
#            NCG_max = int(np.nanmax(numChrom[z][:, y]))
#            
#print("Maximum number of chromosomes observed in truncated experimental data set:\n" + str(NCG_max) + " chromosomes.\n")
#
#NC_plot_max = 0
#for z in range(numTries):
#    for y in range(numChrom_plot[z].shape[1]):
#        if int(np.nanmax(numChrom_plot[z][:, y])) > NC_plot_max:
#            NC_plot_max = int(np.nanmax(numChrom_plot[z][:, y]))
#            
#print("Maximum number of chromosomes observed experimentally:\n" + str(NC_plot_max) + " chromosomes.\n")
#
##### Taylor uses 200:
##NCG_max = 200
##print("Maximum number of chromosomes used in Likelihood computation:\n" + str(NCG_max) + " chromosomes.\n")
#
##NCG_max += 1
#maxN_C = NCG_max
#max_NC = maxN_C

#################################################################################################
### Compute and plot histogram for division-times:                                            ###
#################################################################################################  
print("**************************************************************")
print("Analyzing Experimental division times...")   
print("**************************************************************\n")  
def divisionTimes(N_C):
    div_times = []
    for nSim in range(len(N_C)):
        for nTry in range(len(N_C[nSim])):
            for cellNum in range(N_C[nSim][nTry].shape[0]):
                if np.isnan(N_C[nSim][nTry][cellNum, -1]):
                    div_times.append(np.sum(~np.isnan(N_C[nSim][nTry][cellNum, :])))
    return np.array(div_times)

div_times_truncated = divisionTimes([numChrom])
div_times_exp = divisionTimes([numChrom_plot])

print("Average Truncated Experimental Division Time: \n    " + str(np.average(div_times_truncated * (timeRes/3600))) + " hours" )
print("Average Full Experimental Division Time: \n    " + str(np.average(div_times_exp * (timeRes/3600))) + " hours" )

div_vals_exp, div_inds_exp =  np.histogram((timeRes/3600)*np.array(div_times_exp), bins = np.arange(0,48,1))

div_vals_exp = div_vals_exp / sum(div_vals_exp)
       
print('\nFinished analyzing Experimental division times!\n')

########################################################################################################
### Function to compute the Bayesian Information Criteria (BIC):                                     ###
########################################################################################################
def BIC_calc(num_LagrangeVals, num_ExpMeasurements, negative_log_Likelihood):
    return num_LagrangeVals * np.log(num_ExpMeasurements) + 2*negative_log_Likelihood

########################################################################################################
#### For systems NOT using FSP, instead of 'transitionCounts' calculate 'inputProbs' as follows:     ###
########################################################################################################
#inputProbs = [np.zeros((n_chrom + 1, 2, 2*n_chrom + 1)) for n_chrom in range(max_NC+1)]
#for nTry in range(len(numChrom)):
#    for cellNum in range(numChrom[nTry].shape[0]):
#        birthFrame = np.where(~np.isnan(numChrom[nTry][cellNum, :]))[0][0]
#        divFrame = np.where(~np.isnan(numChrom[nTry][cellNum, :]))[0][-1]
#        for dtSlice in range(birthFrame, divFrame):
#            inputProbs[int(numChrom[nTry][cellNum, dtSlice])][int(numChrom[nTry][cellNum, dtSlice+1] - numChrom[nTry][cellNum, dtSlice]), 0, int(numChrom[nTry][cellNum, dtSlice+1])] += 1
#    del dtSlice, cellNum
#    for tSlice in range(1, numChrom[nTry].shape[1]):
#        motherInds = np.where(np.all([~np.isnan(numChrom[nTry][:, tSlice-1]), np.isnan(numChrom[nTry][:, tSlice])], axis=0))[0]
#        daughterInds = np.where(np.all([np.isnan(numChrom[nTry][:, tSlice-1]), ~np.isnan(numChrom[nTry][:, tSlice])], axis=0))[0]
#        for dcellNum in range(int(len(daughterInds)/2)):
#            inputProbs[int(numChrom[nTry][motherInds[dcellNum], tSlice-1])][int(numChrom[nTry][daughterInds[2*dcellNum], tSlice] + numChrom[nTry][daughterInds[2*dcellNum+1], tSlice] - numChrom[nTry][motherInds[dcellNum], tSlice-1]), 1, int(numChrom[nTry][daughterInds[2*dcellNum], tSlice])] += 1
#    del tSlice, dcellNum, motherInds, daughterInds
#del nTry

#################################################################################################
### Functions required for MaxCal:                                                            ###
#################################################################################################
def logFactorial(value):
    if all([value > 0,abs(round(value) - value) < 0.000001,value <= 34]):
        return float(sum(np.log(range(1,int(value) + 1))))
    elif all([value > 0,abs(round(value) - value) < 0.000001,value > 34]):
        return float(value)*np.log(float(value)) - float(value) + \
        0.5*np.log(2.0*np.pi*float(value)) - 1.0/(12.0*float(value))
    elif value == 0:
        return float(0)
    else:
        return float('nan')

print("Computing and populating 'factMat' ...")
factMatStartTime = time.time()
global factMat
factMat = []
##for ind1 in range(int(2* maxN_C + 1)):
for ind1 in range(int(3*max_MCMC_NC)):
#for ind1 in range(int(2*max_MCMC_NC)):  ## Can probably get away from this, but it's only one extra second to calculate the above (3*) version.
    factMat.append([])
    for ind2 in range(ind1 + 1):
        factMat[ind1].append(logFactorial(ind1) - logFactorial(ind2) - logFactorial(ind1 - ind2))
    del ind2
    factMat[ind1] = np.array(factMat[ind1])
del ind1
factMatCalcTime = time.time() - factMatStartTime
print("'factMat' computation took: %s secs\n" % timedelta(seconds=round(factMatCalcTime)))
    
def int2string(num,base):
    if num == 0:
        return '0'
    digits = []
    for numDig in range(int(np.floor(np.log(num)/np.log(base))) + 1):
        digits.append(str(int(num%base)))
        num //= base
    return ''.join(digits[::-1])      

########################################################################################################
#### Function to compute path probabilities with Delayed Division (DD):                              ###
########################################################################################################
#def pathProbs(lagrangeVals, N_C, N_g):  ## My version of Taylor's 'transitionProbs'
#    global factMat, cap
#    logWeight = -float('Inf')*np.ones([N_C + 1, cap + 1, 2*N_C + 1])
#    for templ_C in np.arange(0, N_C + 1, 1):
#        for templ_g in np.arange(N_g, N_g + 2, 1):
#            for templ_P in np.arange((templ_g != cap)*(templ_C + N_C), (templ_C + N_C)+1, 1):
#                
#                logWeight[templ_C, templ_g, templ_P] = \
#                factMat[N_C + templ_C][templ_P] + factMat[N_C][templ_C] + \
#                lagrangeVals[0]*templ_C + lagrangeVals[1]*(templ_g - N_g) #+ \
##                lagrangeVals[2]*(templ_g == cap)*(((N_C + templ_C)/2 - templ_P)**2) + \
##                lagrangeVals[3]*(templ_g - N_g)*(N_C + templ_C) + \
##                lagrangeVals[4]*(templ_g - N_g)*((N_C + templ_C)**2)
#                
#    W = np.amax(logWeight)
#    logWeight = np.exp(logWeight - W)
#    Q = np.sum(logWeight)
#    logWeight = logWeight/Q
##    tProbSum = np.sum(logWeight)
#    return logWeight
##    return Q, W, tProbSum
    
#######################################################################################################
### Function to compute path probabilities with Delayed Division (DD):                              ###
#######################################################################################################
def pathProbs(lagrangeVals, N_C, N_g):  ## My version of Taylor's 'transitionProbs'
    global factMat, cap
    logWeight = -float('Inf')*np.ones([N_C + 1, cap + 1, 2*N_C + 1])
    for templ_C in np.arange(0, N_C + 1, 1):
        for templ_g in np.arange(N_g, N_g + 2, 1):
            for templ_P in np.arange((templ_g != cap)*(templ_C + N_C), (templ_C + N_C)+1, 1):
                
                if modelNum == 1 or modelNum == 4:
                    logWeight[templ_C, templ_g, templ_P] = \
                    factMat[N_C + templ_C][templ_P] + factMat[N_C][templ_C] + \
                    lagrangeVals[0]*templ_C + lagrangeVals[1]*(templ_g - N_g)
                elif modelNum == 2 or modelNum == 5:
                    logWeight[templ_C, templ_g, templ_P] = \
                    factMat[N_C + templ_C][templ_P] + factMat[N_C][templ_C] + \
                    lagrangeVals[0]*templ_C + lagrangeVals[1]*(templ_g - N_g) + \
                    lagrangeVals[2]*(templ_g == cap)*(((N_C + templ_C)/2 - templ_P)**2)
                elif modelNum == 3 or modelNum == 6:
                    logWeight[templ_C, templ_g, templ_P] = \
                    factMat[N_C + templ_C][templ_P] + factMat[N_C][templ_C] + \
                    lagrangeVals[0]*templ_C + lagrangeVals[1]*(templ_g - N_g) + \
                    lagrangeVals[2]*(templ_g == cap)*(((N_C + templ_C)/2 - templ_P)**2) + \
                    lagrangeVals[3]*(templ_g - N_g)*(N_C + templ_C)
                elif modelNum == 7:
                    logWeight[templ_C, templ_g, templ_P] = \
                    factMat[N_C + templ_C][templ_P] + factMat[N_C][templ_C] + \
                    lagrangeVals[0]*templ_C + lagrangeVals[1]*(templ_g - N_g) + \
                    lagrangeVals[2]*(templ_g == cap)*(((N_C + templ_C)/2 - templ_P)**2) + \
                    lagrangeVals[3]*(templ_g - N_g)*(N_C + templ_C) + \
                    lagrangeVals[4]*(templ_g - N_g)*((N_C + templ_C)**2)
    
    W = np.amax(logWeight)
    logWeight = np.exp(logWeight - W)
    Q = np.sum(logWeight)
    logWeight = logWeight/Q
#    tProbSum = np.sum(logWeight)
    return logWeight
#    return Q, W, tProbSum

########################################################################################################
#### Function to compute path probabilities:                                                         ###
########################################################################################################
##def pathProbs(lagrangeVals, N_C):  ## My version of Taylor's 'transitionProbs'
##    global factMat, cap
##    logWeight = -float('Inf')*np.ones([N_C + 1, cap + 1, 2*N_C + 1])
##    for templ_C in np.arange(0, N_C + 1, 1):
##        for templ_g in np.arange(0,2,1):
##            for templ_P in np.arange((1 - templ_g)*(templ_C + N_C),(templ_C + N_C)+1,1):
##                
##                logWeight[templ_C, templ_g, templ_P] = \
##                factMat[N_C + templ_C][templ_P] + factMat[N_C][templ_C] + \
##                lagrangeVals[0]*templ_C + lagrangeVals[1]*templ_g #+ \
###                lagrangeVals[2]*templ_g*(((N_C + templ_C)/2 - templ_P)**2) + \
###                lagrangeVals[3]*templ_g*(N_C + templ_C)
##                
##    W = np.amax(logWeight)
##    logWeight = np.exp(logWeight - W)
##    Q = np.sum(logWeight)
##    logWeight = logWeight/Q
###    tProbSum = np.sum(logWeight)
##    return logWeight
###    return Q, W, tProbSum
#def pathProbs(lagrangeVals, n_C):  ## My version of Taylor's 'transitionProbs_Test'
#    global factMat
#    logWeight = -float('Inf')*np.ones([2, n_C + 1, 2*n_C + 1])
##    templ_C = np.array([[indC*np.ones(np.size(logWeight, axis=2)) for inda in range(np.size(logWeight, axis=1))] for indC in range(np.size(logWeight, axis=0))],dtype=int).reshape((np.size(logWeight, axis=0))*(np.size(logWeight, axis=1))*(np.size(logWeight, axis=2)))
##    templ_a = np.array([[inda*np.ones(np.size(logWeight, axis=2)) for inda in range(np.size(logWeight, axis=1))] for indC in range(np.size(logWeight, axis=0))],dtype=int).reshape((np.size(logWeight, axis=0))*(np.size(logWeight, axis=1))*(np.size(logWeight, axis=2)))
##    templ_A = np.array([[np.arange(np.size(logWeight, axis=2)) for inda in range(np.size(logWeight, axis=1))] for indC in range(np.size(logWeight, axis=0))],dtype=int).reshape((np.size(logWeight, axis=0))*(np.size(logWeight, axis=1))*(np.size(logWeight, axis=2)))
#    for templ_C in np.arange(0,2,1):
#        for templ_a in np.arange(0,n_C + 1,1):
#            for templ_A in np.arange((1 - templ_C)*(templ_a + n_C),(templ_a + n_C)+1,1):
#
#                logWeight[templ_C, templ_a, templ_A] = \
#                factMat[n_C + templ_a][templ_A] + factMat[n_C][templ_a] + \
#                lagrangeVals[0]*templ_C + lagrangeVals[1]*templ_a + \
#                lagrangeVals[2]*templ_C*(((n_C + templ_a)/2 - templ_A)**2) + \
#                lagrangeVals[3]*templ_C*(n_C + templ_a)
#                
#    W = np.amax(logWeight)
#    logWeight = np.exp(logWeight - W)
#    Q = np.sum(logWeight)
#    logWeight = logWeight/Q
##    tProbSum = np.sum(logWeight)
#    return logWeight
##    return Q, W, tProbSum
    
########################################################################################################
#### For systems NOT using FSP, instead of 'transitionCounts' calculate 'inputProbs' as follows:     ###
########################################################################################################
##inputProbs = [np.zeros((2, n_chrom + 1, 2*n_chrom + 1)) for n_chrom in range(max_NC+1)]
##for nTry in range(len(numChrom)):
##    for cellNum in range(numChrom[nTry].shape[0]):
##        birthFrame = np.where(~np.isnan(numChrom[nTry][cellNum, :]))[0][0]
##        divFrame = np.where(~np.isnan(numChrom[nTry][cellNum, :]))[0][-1]
##        for dtSlice in range(birthFrame, divFrame):
##            inputProbs[int(numChrom[nTry][cellNum, dtSlice])][0, int(numChrom[nTry][cellNum, dtSlice+1] - numChrom[nTry][cellNum, dtSlice]), int(numChrom[nTry][cellNum, dtSlice+1])] += 1
##    del dtSlice, cellNum
##    for tSlice in range(1, numChrom[nTry].shape[1]):
##        motherInds = np.where(np.all([~np.isnan(numChrom[nTry][:, tSlice-1]), np.isnan(numChrom[nTry][:, tSlice])], axis=0))[0]
##        daughterInds = np.where(np.all([np.isnan(numChrom[nTry][:, tSlice-1]), ~np.isnan(numChrom[nTry][:, tSlice])], axis=0))[0]
##        for dcellNum in range(int(len(daughterInds)/2)):
##            inputProbs[int(numChrom[nTry][motherInds[dcellNum], tSlice-1])][1, int(numChrom[nTry][daughterInds[2*dcellNum], tSlice] + numChrom[nTry][daughterInds[2*dcellNum+1], tSlice] - numChrom[nTry][motherInds[dcellNum], tSlice-1]), int(numChrom[nTry][daughterInds[2*dcellNum], tSlice])] += 1
##    del tSlice, dcellNum, motherInds, daughterInds
##del nTry
##inputProbs = [np.zeros((n_chrom + 1, 2, 2*n_chrom + 1)) for n_chrom in range(max_NC+1)]
##for nTry in range(len(numChrom)):
##    for cellNum in range(numChrom[nTry].shape[0]):
##        birthFrame = np.where(~np.isnan(numChrom[nTry][cellNum, :]))[0][0]
##        divFrame = np.where(~np.isnan(numChrom[nTry][cellNum, :]))[0][-1]
##        for dtSlice in range(birthFrame, divFrame):
##            inputProbs[int(numChrom[nTry][cellNum, dtSlice])][int(numChrom[nTry][cellNum, dtSlice+1] - numChrom[nTry][cellNum, dtSlice]), 0, int(numChrom[nTry][cellNum, dtSlice+1])] += 1
##    del dtSlice, cellNum
##    for tSlice in range(1, numChrom[nTry].shape[1]):
##        motherInds = np.where(np.all([~np.isnan(numChrom[nTry][:, tSlice-1]), np.isnan(numChrom[nTry][:, tSlice])], axis=0))[0]
##        daughterInds = np.where(np.all([np.isnan(numChrom[nTry][:, tSlice-1]), ~np.isnan(numChrom[nTry][:, tSlice])], axis=0))[0]
##        for dcellNum in range(int(len(daughterInds)/2)):
##            inputProbs[int(numChrom[nTry][motherInds[dcellNum], tSlice-1])][int(numChrom[nTry][daughterInds[2*dcellNum], tSlice] + numChrom[nTry][daughterInds[2*dcellNum+1], tSlice] - numChrom[nTry][motherInds[dcellNum], tSlice-1]), 1, int(numChrom[nTry][daughterInds[2*dcellNum], tSlice])] += 1
##    del tSlice, dcellNum, motherInds, daughterInds
##del nTry
#inputProbs = [np.zeros((2, n_chrom + 1, 2*n_chrom + 1)) for n_chrom in range(max_NC+1)]
#for nTry in range(len(numChrom)):
#    for cellNum in range(numChrom[nTry].shape[0]):
#        birthFrame = np.where(~np.isnan(numChrom[nTry][cellNum, :]))[0][0]
#        divFrame = np.where(~np.isnan(numChrom[nTry][cellNum, :]))[0][-1]
#        for dtSlice in range(birthFrame, divFrame):
#            inputProbs[int(numChrom[nTry][cellNum, dtSlice])][0, int(numChrom[nTry][cellNum, dtSlice+1] - numChrom[nTry][cellNum, dtSlice]), int(numChrom[nTry][cellNum, dtSlice+1])] += 1
#    del dtSlice, cellNum
#    for tSlice in range(1, numChrom[nTry].shape[1]):
#        motherInds = np.where(np.all([~np.isnan(numChrom[nTry][:, tSlice-1]), np.isnan(numChrom[nTry][:, tSlice])], axis=0))[0]
#        daughterInds = np.where(np.all([np.isnan(numChrom[nTry][:, tSlice-1]), ~np.isnan(numChrom[nTry][:, tSlice])], axis=0))[0]
#        for dcellNum in range(int(len(daughterInds)/2)):
#            inputProbs[int(numChrom[nTry][motherInds[dcellNum], tSlice-1])][1, int(numChrom[nTry][daughterInds[2*dcellNum], tSlice] + numChrom[nTry][daughterInds[2*dcellNum+1], tSlice] - numChrom[nTry][motherInds[dcellNum], tSlice-1]), int(numChrom[nTry][daughterInds[2*dcellNum], tSlice])] += 1
#    del tSlice, dcellNum, motherInds, daughterInds
#del nTry
#    
########################################################################################################
#### Function to compute the -log(Likelihood) using MaxCal + ML estimation                           ###
########################################################################################################
#def MaxCal_MLe(lagrangeVals):
#    logLike = 0
#    for n_Cb in range(int(NCG_max+1)):          ### n_Cb: index for BEFORE number of chromosomes
#        probz = pathProbs(lagrangeVals, n_Cb)
#        if np.sum(probz) > 0:
#            logLike -= np.nansum(np.log(probz)*inputProbs[n_Cb])
#        del probz
#    return logLike

#######################################################################################################
### Function to compute the -log(Likelihood) using MaxCal + ML estimation w/ Delayed Division       ###
#######################################################################################################
def MaxCal_MLe(lagrangeVals):
    global factMat, cap, nChromo
    
    N_max = 0
    for nTry in range(len(nChromo)):
        if np.nanmax(nChromo[nTry]) > N_max:
            N_max = int(np.nanmax(nChromo[nTry]))
    
    probsList = [[[] for ind_g in range(cap)] for ind_C in range(N_max + 1)]
    logLike = 0.0
    for nTry in range(len(nChromo)):
        for cellNum in range(nChromo[nTry].shape[0]):
            if (np.sum(~np.isnan(nChromo[nTry][cellNum, :])) == 1 and ~np.isnan(nChromo[nTry][cellNum, -1])): ### If born on the last frame ...
                continue
#            if ~np.isnan(nChromo[nTry][cellNum, 0]):                                                   ### If already alive in the first frame...
#                continue
            startFrame = np.where(~np.isnan(nChromo[nTry][cellNum,:]))[0][0]
            endFrame = np.where(~np.isnan(nChromo[nTry][cellNum,:]))[0][-1] + 1
            probSum = []
            for gTind in range(2**(endFrame - startFrame)):                         ### Index numbering every possible gamma trajectory
                binary_gTind = int2string(gTind, 2)  ### Converting every possible index into a binary number (since l_g still limited to 0 or 1) allows all combinations of l_g values over time to be accounted for. 
                while len(binary_gTind) < endFrame - startFrame:
                    binary_gTind = '0' + binary_gTind
                lg_list = np.array([int(x) for x in list(binary_gTind)])
                if endFrame == nChromo[nTry].shape[1]:                          ### If NO Division, i.e., if this cell is still alive at the end of the movie ...
                    if np.sum(lg_list) < cap:
                        probSum.append(1.0)
                        for frameNum in np.arange(startFrame, endFrame - 1, 1):
                            if len(probsList[int(nChromo[nTry][cellNum, frameNum])][int(np.sum(lg_list[:frameNum-startFrame]))]) == 0:
                                probsList[int(nChromo[nTry][cellNum, frameNum])][int(np.sum(lg_list[:frameNum-startFrame]))] = \
                                pathProbs(lagrangeVals, int(nChromo[nTry][cellNum, frameNum]), int(np.sum(lg_list[:frameNum-startFrame])))
                            tempProbs = probsList[int(nChromo[nTry][cellNum, frameNum])][int(np.sum(lg_list[:frameNum-startFrame]))]
#                            ipdb.set_trace() # Breakpoint
                            probSum[-1] *= tempProbs[int(nChromo[nTry][cellNum, frameNum + 1] - nChromo[nTry][cellNum, frameNum]), \
                                   int(np.sum(lg_list[:frameNum - startFrame + 1])), int(nChromo[nTry][cellNum, frameNum + 1])]
                elif np.sum(lg_list) == cap and lg_list[-1] == 1:                   ### If Division, i.e., if this cell has met the conditions for division ...
                    probSum.append(1.0)
                    for frameNum in np.arange(startFrame, endFrame, 1):             ### Examine all gamma trajectories that could have led to this division ...
                        if len(probsList[int(nChromo[nTry][cellNum, frameNum])][int(np.sum(lg_list[:frameNum-startFrame]))]) == 0:
                            probsList[int(nChromo[nTry][cellNum, frameNum])][int(np.sum(lg_list[:frameNum-startFrame]))] = \
                            pathProbs(lagrangeVals, int(nChromo[nTry][cellNum, frameNum]), int(np.sum(lg_list[:frameNum-startFrame])))
                        tempProbs = probsList[int(nChromo[nTry][cellNum, frameNum])][int(np.sum(lg_list[:frameNum-startFrame]))]
                        if frameNum < endFrame - 1:                                 ### Consider each point on each gamma trajectory before ultimate division ...
#                            ipdb.set_trace() # Breakpoint
                            probSum[-1] *= tempProbs[int(nChromo[nTry][cellNum, frameNum + 1] - nChromo[nTry][cellNum, frameNum]), \
                                   int(np.sum(lg_list[:frameNum - startFrame + 1])), int(nChromo[nTry][cellNum, frameNum + 1])]
                        else:
                            motherInds = np.where(np.all([~np.isnan(nChromo[nTry][:, frameNum]), np.isnan(nChromo[nTry][:, frameNum + 1])], axis=0))[0]
                            daughterInds = np.where(np.all([np.isnan(nChromo[nTry][:, frameNum]), ~np.isnan(nChromo[nTry][:, frameNum + 1])], axis=0))[0]
                            cellInd = np.where(motherInds == cellNum)[0][0]
#                            ipdb.set_trace() # Breakpoint
                            probSum[-1] *= tempProbs[int(nChromo[nTry][daughterInds[2 * cellInd], frameNum + 1] + nChromo[nTry][daughterInds[2 * cellInd + 1], frameNum + 1] - nChromo[nTry][motherInds[cellInd], frameNum]), \
                                   int(np.sum(lg_list[:frameNum - startFrame + 1])), int(nChromo[nTry][daughterInds[2 * cellInd], frameNum + 1])]
            logLike -= np.log(np.sum(probSum))
    return logLike

#########################################################################################################
### MINIMIZES the output of the MaxCal_ML function which is the negative log(Likelihood) of observing ###
### the loaded trajectories, given the bestGuess lagrangeVals, effectively MAXIMIZING this Likelihood:###
#########################################################################################################
print("**************************************************************")
print("Starting MaxCal Inference ...")
print("**************************************************************\n")
inference_start_time = time.time()

nChromo = numChrom

fitStartGuess = float('NaN')*np.ones((numFits,len(firstGuess)))
fitLagrangeVals  = float('NaN')*np.ones((numFits,len(firstGuess)))
fitLogLike = float('NaN')*np.ones((numFits,1))

initLogLike = MaxCal_MLe(firstGuess)

if modelNum == 7:
    LnameStr = "[h_C, h_g, h_P, K_gC, K_gC2]"
elif modelNum == 6 or modelNum == 3:
    LnameStr = "[h_C, h_g, h_P, K_gC]"
elif modelNum == 5 or modelNum == 2:
    LnameStr = "[h_C, h_g, h_P]"
elif modelNum == 4 or modelNum == 1:
    LnameStr = "[h_C, h_g]"

print("Initial guess Lagrange multipliers: " + LnameStr + " = " + str(firstGuess) + "\nwith corresponding -log(Likelihood) = " + str(initLogLike))
#print("Initial guess Lagrange multipliers: [h_C, h_g, h_P, h_KgC] = " + str(firstGuess) + "\nwith corresponding -log(Likelihood) = " + str(initLogLike))
#print("Initial guess Lagrange multipliers: [h_C, h_g, h_P] = " + str(firstGuess) + "\nwith corresponding -log(Likelihood) = " + str(initLogLike))
for fitNum in range(numFits):
    print("\nIteration # " + str(fitNum+1) + ":")
    if fitNum == 0:
        fitStartGuess[0, :] = firstGuess
    else:
        fitStartGuess[fitNum,:] = fitLagrangeVals[fitNum-1,:]
    res = minimize(MaxCal_MLe,fitStartGuess[fitNum,:],method='nelder-mead',\
    tol=0.01,options={'disp':True,'maxiter':500})
    fitLagrangeVals[fitNum,:] = res['x']
    fitLogLike[fitNum] = res['fun']
    del res
    print("Fitting is " + str(int((fitNum+1)*100/numFits)) + " % complete ...")
del fitNum
bestInd = np.where(fitLogLike == np.nanmin(fitLogLike))[0][0]
bestGuess = fitLagrangeVals[bestInd]
bestLike = fitLogLike[bestInd]
        
#bestLike = MaxCal_MLe(MC_lagrange)
#bestGuess = MC_lagrange

print('Total number of Experimental Data points: ' + str(totalDataPoints) + '\n')

#BIC = len(MC_lagrange)*np.log(totalDataPoints) + 2*bestLike
BIC = len(bestGuess.tolist())*np.log(totalDataPoints) + 2*bestLike

##print('Lagrange Mutlipliers: \n[h_C, h_g, h_P, h_KgC, h_KgC2] = ' + str(MC_lagrange) + ' \nyield -log(Likelihood) = ' + str(Lagr01) + ", and BIC = " + str(BIC) + '\n')
print('Optimized Lagrange Mutlipliers\n(Model # ' + str(int(modelNum)) + ', Truncated data length: ' + str(int(crop_hrs)) + ' hours): \n' + LnameStr + ' = ' + str(np.round(bestGuess.tolist(), 3)) + ' \nyield -log(Likelihood) = ' + str(np.round(bestLike[0], 3)) + ", and BIC = " + str(np.round(BIC[0], 3)) + '\n')

if crop_hrs >= 22:
    if modelNum == 1 or modelNum == 4:
        filename_inference = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/InferenceResults/Full/M%s/Ploidy_Inference_Results_hC=%s_hg=%s_deltaT=%ss_v%s' %(str(int(modelNum)), str(np.round(bestGuess[0], 3)), str(np.round(bestGuess[1], 3)), str(timeRes), str(version))
    elif modelNum == 2 or modelNum == 5:
        filename_inference = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/InferenceResults/Full/M%s/Ploidy_Inference_Results_hC=%s_hg=%s_hP=%s_deltaT=%ss_v%s' %(str(int(modelNum)), str(np.round(bestGuess[0], 3)), str(np.round(bestGuess[1], 3)), str(np.round(bestGuess[2], 3)), str(timeRes), str(version))
    elif modelNum == 3 or modelNum == 6:
        filename_inference = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/InferenceResults/Full/M%s/Ploidy_Inference_Results_hC=%s_hg=%s_hP=%s_KgC=%s_deltaT=%ss_v%s' %(str(int(modelNum)), str(np.round(bestGuess[0], 3)), str(np.round(bestGuess[1], 3)), str(np.round(bestGuess[2], 3)), str(np.round(bestGuess[3], 3)), str(timeRes), str(version))
    elif modelNum == 7:
        filename_inference = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/InferenceResults/Full/M%s/Ploidy_Inference_Results_hC=%s_hg=%s_hP=%s_KgC=%s_KgC2=%s_deltaT=%ss_v%s' %(str(int(modelNum)), str(np.round(bestGuess[0], 3)), str(np.round(bestGuess[1], 3)), str(np.round(bestGuess[2], 3)), str(np.round(bestGuess[3], 3)), str(np.round(bestGuess[4], 3)), str(timeRes), str(version))
else:
    if modelNum == 1 or modelNum == 4:
        filename_inference = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/InferenceResults/%shrs/M%s/Ploidy_Inference_Results_hC=%s_hg=%s_deltaT=%ss_v%s' %(str(int(crop_hrs)), str(int(modelNum)), str(np.round(bestGuess[0], 3)), str(np.round(bestGuess[1], 3)), str(timeRes), str(version))
    elif modelNum == 2 or modelNum == 5:
        filename_inference = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/InferenceResults/%shrs/M%s/Ploidy_Inference_Results_hC=%s_hg=%s_hP=%s_deltaT=%ss_v%s' %(str(int(crop_hrs)), str(int(modelNum)), str(np.round(bestGuess[0], 3)), str(np.round(bestGuess[1], 3)), str(np.round(bestGuess[2], 3)), str(timeRes), str(version))
    elif modelNum == 3 or modelNum == 6:
        filename_inference = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/InferenceResults/%shrs/M%s/Ploidy_Inference_Results_hC=%s_hg=%s_hP=%s_KgC=%s_deltaT=%ss_v%s' %(str(int(crop_hrs)), str(int(modelNum)), str(np.round(bestGuess[0], 3)), str(np.round(bestGuess[1], 3)), str(np.round(bestGuess[2], 3)), str(np.round(bestGuess[3], 3)), str(timeRes), str(version))
    elif modelNum == 7:
        filename_inference = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/InferenceResults/%shrs/M%s/Ploidy_Inference_Results_hC=%s_hg=%s_hP=%s_KgC=%s_KgC2=%s_deltaT=%ss_v%s' %(str(int(crop_hrs)), str(int(modelNum)), str(np.round(bestGuess[0], 3)), str(np.round(bestGuess[1], 3)), str(np.round(bestGuess[2], 3)), str(np.round(bestGuess[3], 3)), str(np.round(bestGuess[4], 3)), str(timeRes), str(version))
    
np.savez(filename_inference, bestGuess = bestGuess, bestLike = bestLike, BIC = BIC, bestInd = bestInd, fitStartGuess = fitStartGuess, fitLagrangeVals = fitLagrangeVals, fitLogLike = fitLogLike, totalDataPoints = totalDataPoints, dataCounter = dataCounter, modelNum = modelNum, crop_hrs = crop_hrs, div_times_exp = div_times_exp)
#np.savez(filename_inference, MC_lagrange = MC_lagrange, MC_like = bestLike, BIC = BIC, totalDataPoints = totalDataPoints, dataCounter = dataCounter, modelNum = modelNum, crop_hrs = crop_hrs, div_times_exp = div_times_exp)

minimize_time = time.time() - inference_start_time
print("\nMaxCal extraction took: %s sec" % timedelta(seconds=round(minimize_time)))
MaxCalExtractionCompleteTime = time.time()
print("\nMaxCal Inference complete!\n")

#### Inference results loading routine:
#
#different_labels = 0
#
#if crop_hrs == 10:
#    if modelNum == 1:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M1/Ploidy_Inference_Results_hC=-2.569_hg=-6.774_deltaT=1800_vPloidy_10hr_PLM_Infer_M1_0.1.6.npz')
#    elif modelNum == 2:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M2/Ploidy_Inference_Results_hC=-2.56_hg=-6.008_hP=-1.083_deltaT=1800_vPloidy_10hr_PLM_Infer_M2_0.1.6.npz')
#    elif modelNum == 3:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M3/Ploidy_Inference_Results_hC=-2.626_hg=-8.638_hP=-1.119_hKgC=0.386_deltaT=1800_vPloidy_10hr_PLM_Infer_M3_0.1.6.npz')
#    elif modelNum == 4:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M4/Ploidy_Inference_Results_hC=-2.575_hg=0.049_deltaT=1800_vPloidy_10hr_PLM_Infer_M4_0.1.6.npz')
#    elif modelNum == 5:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M5/Ploidy_Inference_Results_hC=-2.564_hg=0.051_hP=-1.106_deltaT=1800_vPloidy_10hr_PLM_Infer_M5_0.1.6.npz')
#    elif modelNum == 6:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M6/Ploidy_Inference_Results_hC=-2.814_hg=-1.899_hP=-1.11_hKgC=0.391_deltaT=1800_vPloidy_10hr_PLM_Infer_M6_0.1.6.npz')
#    elif modelNum == 7:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M7/Ploidy_Inference_Results_hC=-2.756_hg=-5.539_hP=-1.103_hKgC=1.739_hKgC2=-0.115_deltaT=1800_vPloidy_10hr_PLM_Infer_M7_0.1.6.npz')
#elif crop_hrs == 16:
#    if modelNum == 1:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M1/Ploidy_Inference_Results_hC=-2.592_hg=-7.071_deltaT=1800_vPloidy_16hr_PLM_Infer_M1_0.1.6.npz')
#    elif modelNum == 2:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M2/Ploidy_Inference_Results_hC=-2.587_hg=-6.472_hP=-0.652_deltaT=1800_vPloidy_16hr_PLM_Infer_M2_0.1.6.npz')
#    elif modelNum == 3:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M3/Ploidy_Inference_Results_hC=-2.608_hg=-7.375_hP=-0.667_hKgC=0.129_deltaT=1800_vPloidy_16hr_PLM_Infer_M3_0.1.6.npz')
#    elif modelNum == 4:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M4/Ploidy_Inference_Results_hC=-2.598_hg=-0.007_deltaT=1800_vPloidy_16hr_PLM_Infer_M4_0.1.6.npz')
#    elif modelNum == 5:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M5/Ploidy_Inference_Results_hC=-2.592_hg=0.003_hP=-0.667_deltaT=1800_vPloidy_16hr_PLM_Infer_M5_0.1.6.npz')
#    elif modelNum == 6:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M6/Ploidy_Inference_Results_hC=-2.77_hg=-1.503_hP=-0.667_hKgC=0.291_deltaT=1800_vPloidy_16hr_PLM_Infer_M6_0.1.6.npz')
#    elif modelNum == 7:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M7/Ploidy_Inference_Results_hC=-2.713_hg=-5.933_hP=-0.66_hKgC=1.876_hKgC2=-0.13_deltaT=1800_vPloidy_16hr_PLM_Infer_M7_0.1.6.npz')
#elif crop_hrs == 22:
#    if modelNum == 1:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/BIC_Comparison/FullData_No_Inference/M1/Ploidy_Inference_Results_hC=-6.947_hg=-2.688_deltaT=1800_vPloidy_FLM_BIC_M1_0.1.6.npz')
#        different_labels = 1
#    elif modelNum == 2:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M2/Ploidy_Inference_Results_hC=-2.679_hg=-6.289_hP=-0.759_deltaT=1800_vPloidy_22hr_PLM_Infer_M2_0.1.6.npz')
#    elif modelNum == 3:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Inference/M3/Ploidy_Inference_Results_hC=-2.708_hg=-7.428_hP=-0.778_hKgC=0.162_deltaT=1800_vPloidy_22hr_PLM_Infer_M3_0.1.6.npz')
#    elif modelNum == 4:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/BIC_Comparison/FullData_No_Inference/M4/Ploidy_Inference_Results_hC=-2.689_hg=0.088_deltaT=1800_vPloidy_FLM_BIC_M4_0.1.6.npz')
#        different_labels = 1
#    elif modelNum == 5:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/BIC_Comparison/FullData_No_Inference/M5/Ploidy_Inference_Results_hC=-2.681_hg=0.092_hP=-0.777_deltaT=1800_vPloidy_FLM_BIC_M5_0.1.6.npz')
#        different_labels = 1
#    elif modelNum == 6:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/BIC_Comparison/FullData_No_Inference/M6/Ploidy_Inference_Results_hC=-2.88_hg=-1.546_hP=-0.782_hKgC=0.309_deltaT=1800_vPloidy_FLM_BIC_M6_0.1.6.npz')
#        different_labels = 1
#    elif modelNum == 7:
#        temp_inf = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/FullData_Prediction/M7/Ploidy_Inference_Results_hC=-2.826_hg=-5.438_hP=-0.77_hKgC=1.675_hKgC2=-0.111_deltaT=1800_vPloidy_FullData_Infer_Predict_M7_0.1.7.2.npz')
#        
#
#else:
#    print('\n... I still need to specify the path to load the inference results\nfile corresponding to the selected parameters:\nmodelNum = ' + str(int(modelNum)) + "\ncrop_hrs = " + str(crop_hrs) + "\n\n... If not available, re-run inference procedure.\n")
#
#if different_labels == 0:
#    bestGuess = temp_inf['bestGuess']
#    bestLike = temp_inf['bestLike']
#    #BIC = temp_inf['BIC']
#    #bestInd = temp_inf['bestInd']
#    #fitStartGuess = temp_inf['fitStartGuess']
#    #fitLagrangeVals = temp_inf['fitLagrangeVals']
#    #fitLogLike = temp_inf['fitLogLike']
#elif different_labels == 1:
#    bestGuess = temp_inf['MC_lagrange']
#    bestLike = temp_inf['MC_like']
#
#print("Loading MaxCal Inference data is complete!\n")

#############################################################################################################
### Calculate -ln(Likelihoods) using PLMs (Lagrange Multipliers inferred from truncated data)             ###
### and the entire (~20 hr.) experimental data set: i.e. "How likely is it that the PLMs reproduce        ###
### the FULL data set" and compare these -ln(Likelihoods) to those calculated with LMs from FULL data set.###
#############################################################################################################
del nChromo
nChromo = numChrom_plot
PF_Like = MaxCal_MLe(bestGuess.tolist())
PF_BIC = BIC_calc(len(bestGuess.tolist()), totalDataPoints_plot, PF_Like)
print("Using Lagrange Multipliers inferred from PARTIAL data, along with FULL experimental data set,\n'20 hour' -ln(Likelihood) = " + str(np.round(PF_Like, 1)) + "\n'20 hour' BIC = " + str(np.round(PF_BIC, 1)))
print("(Model # " + str(int(modelNum)) + ", Truncated Data Length: " + str(crop_hrs) + " hours)\n")
    
filename_PF_Like_BIC = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PF_BIC_Comparison/M%s/20hr_-ln(Likes)_from_%shr_PLMs_M%s' %(str(int(modelNum)), str(int(crop_hrs)), str(int(modelNum)))
np.savez(filename_PF_Like_BIC, FullPLM_Likelihood = PF_Like, FullPLM_BIC = PF_BIC, PLMs = bestGuess)

######################################################################################################
#### MaxCal Monte Carlo simulation:                                                                ###
###################################################################################################### 
#print("**************************************************************")
#print("Starting " + str(int(MCMC_maxT_hours)) + "-hour MaxCal Monte Carlo (MCMC) Prediction Series\nbased on " + str(int(crop_hrs)) + " hours of experimental data ...")
#print("**************************************************************")
#  
##t_f = 72000 #20hrs = 72000sec    ### Maximum time for Monte Carlo simulation
##maxLalpha = max_la
#maxLalpha = 10
#
#max_Qgen_NC = max_NC + 10
#
###MC_lagrange = bestGuess.tolist()
##LagrMCMC = MaxCal_MLe(MC_lagrange)
#LagrMCMC = MaxCal_MLe(bestGuess)
#
#print('\nLagrange Mutlipliers used in this MaxCal Monte Carlo simulation: \n[h_C, h_g, h_P, h_KgC, h_KgC2] = ' + str(np.round(bestGuess,3)) + ' \nwith -log(Likelihood) = ' + str(np.round(LagrMCMC,3)) + ", and BIC = " + str(np.round(BIC_calc(len(bestGuess.tolist()), totalDataPoints, LagrMCMC),3)) + "\n")
#
##t_f=40000000
##t_f = int(2*86400)
#timeRes=deltaT
##timeRes=1
#
#
##M_LB = 0
#MCMC_start_time = time.time()
#
##temp_percent = 1
#
######################################################################################################
#### MCMC preparatory function(s):                                                                 ###
######################################################################################################
#### Collect initial experimental chromosome numbers (t=0) for MCMC starting points
#global NC_init, Ng_init
#NC_init = []
#Ng_init = []
#for trial in range(len(numChrom)):
#    NC_init.append(numChrom[trial][~np.isnan(numChrom[trial][:,0]),0])
##    NC_init[-1].reshape(NC_init[-1].shape[0], 1)
#    NC_init[-1].resize(NC_init[-1].shape[0], 1)
#    Ng_init.append(np.zeros((NC_init[-1].shape)))
#del trial
#
######################################################################################################
#### Single MCMC Simulation:                                                                       ###
######################################################################################################
#global good_chrom_state_executions, good_div_state_executions
#good_chrom_state_executions = 0
#good_div_state_executions = 0
#
#num_chrom_in_div_cell_mcmc = []
#def MCMC_sim(lagrangeVals, chrom_state, div_state):
#    global factMat, num_MCMC_trials, maxT, max_NC, NC_init, NG_init, good_chrom_state_executions, good_div_state_executions
#    probsList = [[[] for g_ind in range(cap)] for C_ind in range(max_MCMC_NC)]
#    for nTry in range(len(chrom_state)):
#        for tSlice in range(maxT):
#            chrom_state[nTry] = np.hstack((chrom_state[nTry], float('NaN')*np.ones([chrom_state[nTry].shape[0],1])))
#            div_state[nTry] = np.hstack((div_state[nTry], float('NaN')*np.ones([div_state[nTry].shape[0],1])))
#            for cellNum in range(len(chrom_state[nTry])):
#                if np.isnan(chrom_state[nTry][cellNum, -2]):
#                    continue
#                if ~np.isnan(chrom_state[nTry][cellNum, -2]):
#                    good_chrom_state_executions += 1
#                if ~np.isnan(div_state[nTry][cellNum, -2]):
#                    good_div_state_executions += 1
#                    
#                if int(chrom_state[nTry][cellNum, -2]) > len(probsList):
#                    print("'chrom_state' = " + str(int(chrom_state[nTry][cellNum, -2])) + " got too big at nTry = " + str(nTry) + ", tSlice = " + str(tSlice) + ", cellNum = " + str(cellNum) + ", and finally: len(chrom_state) = " + str(len(chrom_state[nTry])) + ", and maxT = " + str(maxT) + ".")
#                
#                if len(probsList[int(chrom_state[nTry][cellNum, -2])][int(div_state[nTry][cellNum, -2])]) == 0:
#                    probsList[int(chrom_state[nTry][cellNum, -2])][int(div_state[nTry][cellNum, -2])] = \
#                    pathProbs(bestGuess, int(chrom_state[nTry][cellNum, -2]), int(div_state[nTry][cellNum, -2]))
#                nextProbs = probsList[int(chrom_state[nTry][cellNum, -2])][int(div_state[nTry][cellNum, -2])]
#                probSum = 0
##                R1 =  np.random.uniform(0,tProbSumList[int(chrom_state[nTry][cellNum, -2])])
#                R1 = np.random.uniform(0,1)
#                temp_LC = -1
#                while (R1 > probSum) and ( temp_LC < int(chrom_state[nTry][cellNum, -2]) ):
#                    temp_LC += 1
#                    temp_Lg = -1
#                    while (R1 > probSum) and (temp_Lg < 1):
#                        temp_Lg += 1
#                        temp_LP = -1
#                        while (R1 > probSum) and ( temp_LP < 2*int(chrom_state[nTry][cellNum, -2]) ):
#                            temp_LP += 1
###                            probSum += MCMC_logWeight[int(chrom_state[nTry][cellNum, -2])][temp_Lg, temp_LC, temp_LP]
##                            probSum += nextProbs[temp_LC, temp_Lg, temp_LP]
#                            probSum += nextProbs[temp_LC, temp_Lg + int(div_state[nTry][cellNum, -2]), temp_LP]
##                if temp_Lg == 0:
##                    chrom_state[nTry][cellNum, -1] = chrom_state[nTry][cellNum, -2] + temp_LC
#                if temp_Lg + int(div_state[nTry][cellNum, -2]) == cap:
#                    num_chrom_in_div_cell_mcmc.append(int(chrom_state[nTry][cellNum, -2]))
#                    chrom_state[nTry] = np.vstack((chrom_state[nTry], float('NaN')*np.ones((2, chrom_state[nTry].shape[1]))))
#                    chrom_state[nTry][-2, -1] = temp_LP
#                    chrom_state[nTry][-1, -1] = chrom_state[nTry][cellNum, -2] + temp_LC - temp_LP
#                    div_state[nTry] = np.vstack((div_state[nTry], float('NaN')*np.ones((2, div_state[nTry].shape[1]))))
#                    div_state[nTry][-2:, -1] = 0
#                else:
#                    chrom_state[nTry][cellNum, -1] = chrom_state[nTry][cellNum, -2] + temp_LC
#                    div_state[nTry][cellNum, -1] = div_state[nTry][cellNum, -2] + temp_Lg
#    return chrom_state
#                    
######################################################################################################
#### Execute MCMC series:                                                                          ###
######################################################################################################
#nC_MCMC = [[] for ind in range(num_MCMC_trials)]
#temp_percent = 1
#for simNum in range(num_MCMC_trials):
###    N_C_MCMC_DS2[simNum] = MCMC_sim(np.copy(NC_init).tolist(), MC_lagrange)
###    nC_MCMC[simNum] = MCMC_sim(np.copy(NC_init).tolist(), MC_lagrange)
##    nC_MCMC[simNum] = np.asarray(MCMC_sim(np.copy(NC_init).tolist(), MC_lagrange))
#    nC_MCMC[simNum] = np.asarray(MCMC_sim(bestGuess, np.copy(NC_init).tolist(), np.copy(Ng_init).tolist()))
#
#    if int(100*(simNum+1)/num_MCMC_trials) >= temp_percent:
#        print('MaxCal Monte Carlo (MCMC) simulation is ' + str(int(100*(simNum+1)/num_MCMC_trials)) + ' % complete ...')   
#        temp_percent += 1  
#
###        filename001 = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/MaxCal_ChromosomeCopying_CellDividing_ChromosomePartitioning/MCMC_Data/CellDiv_MCMC_trajectories_hC=%s_ha=%s_hA=%s_deltaT=%s_v%s.npz' %(str(MC_lagrange[0]), str(MC_lagrange[1]), str(MC_lagrange[2]), str(timeRes), str(version))
##        filename001 = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/MaxCal_ChromosomeCopying_CellDividing_ChromosomePartitioning/TaylorComparison/3_Prediction_Check/M2/CellDiv_MCMC_trajectories_hC=%s_ha=%s_hA=%s_deltaT=%s_v%s.npz' %(str(MC_lagrange[0]), str(MC_lagrange[1]), str(MC_lagrange[2]), str(timeRes), str(version))
##        filename001 = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/TaylorComparison/3_Prediction_Check/M7/Ploidy_MCMC_trajectories_hC=%s_hg=%s_hP=%s_hKgC=%s_hKgC2=%s_deltaT=%ss_v%s.npz' %(str(np.round(bestGuess[0], 3)), str(np.round(bestGuess[1], 3)), str(np.round(bestGuess[2], 3)), str(np.round(bestGuess[3], 3)), str(np.round(bestGuess[4], 3)), str(timeRes), str(version))
##        filename001 = '/home/swedeki3/PloidyVariation/M%s/Ploidy_MCMC_trajectories_hC=%s_hg=%s_hP=%s_hKgC=%s_hKgC2=%s_deltaT=%ss_v%s.npz' %(str(int(modelNum)), str(np.round(bestGuess[0], 3)), str(np.round(bestGuess[1], 3)), str(np.round(bestGuess[2], 3)), str(np.round(bestGuess[3], 3)), str(np.round(bestGuess[4], 3)), str(timeRes), str(version))
#
#        filename001 = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Prediction/M%s/Ploidy_MCMC_trajectories_hC=%s_hg=%s_hP=%s_hKgC=%s_hKgC2=%s_deltaT=%ss_v%s.npz' %(str(int(modelNum)), str(np.round(bestGuess[0], 3)), str(np.round(bestGuess[1], 3)), str(np.round(bestGuess[2], 3)), str(np.round(bestGuess[3], 3)), str(np.round(bestGuess[4], 3)), str(timeRes), str(version))
#
###        sio.savemat(filename0, {'N_A': N_A, 'N_alpha_star': N_alpha_star, 'R_A': R_A, 'RXNs': RXNs})
#        np.savez(filename001, nC_MCMC = nC_MCMC, maxT = maxT, MCMC_maxT_hours = MCMC_maxT_hours, numTrials = numTrials, num_MCMC_trials = num_MCMC_trials, total_time_seconds = t_f, timeRes = timeRes, num_chrom_in_div_cell_mcmc = num_chrom_in_div_cell_mcmc)
#
#print("\nMaxCal Monte Carlo series is complete!\n")
#
#### Loading Routine:
##filename001 = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/MaxCal_ChromosomeCopying_CellDividing_ChromosomePartitioning/TaylorComparison/3_Prediction_Check/M2/CellDiv_MCMC_trajectories_hC=%s_ha=%s_hA=%s_deltaT=%s_v%s.npz' %(str(MC_lagrange[0]), str(MC_lagrange[1]), str(MC_lagrange[2]), str(timeRes), str(version))
##temp_load_2 = np.load(filename001)
##nC_MCMC = temp_load_2['nC_MCMC']
##num_chrom_in_div_cell_mcmc = temp_load_2['num_chrom_in_div_cell_mcmc']
##print("Loading the most recent NEW MaxCal Monte Carlo series is complete!\n")
#
################################################################################################################################################################
####### MCMC Analyses to prepare for plotting: #################################################################################################################
################################################################################################################################################################
#print("*************************************************************************")
#print('Begin post-processing for plots...')
#print("*************************************************************************\n")
#
##################################################################################################
#### Compute and plot histogram for division-times for DS2:                                    ###
##################################################################################################
#
#div_times_mcmc = divisionTimes(np.asarray(nC_MCMC))
#
##print("\nAverage OLD MCMC Division Time: \n    " + str(np.average(div_times_mcmc0/2)) + " hours" )
#print("Average MCMC Division Time: \n    " + str(np.average(np.multiply(div_times_mcmc, (timeRes/3600)))) + " hours" )
#print("Average Experimental Division Time: \n    " + str(np.average(np.multiply(div_times_exp, (timeRes/3600)))) + " hours" )
#print("Percent Difference: \n    " + str( ((np.average(np.multiply(div_times_mcmc, (timeRes/3600))) - np.average(np.multiply(div_times_exp, (timeRes/3600)))) / np.average(np.multiply(div_times_exp, (timeRes/3600)))) *100 )  + " %")
#
#div_vals_mcmc, div_inds_mcmc =  np.histogram((timeRes/3600)*np.array(div_times_mcmc), bins = np.arange(0,48,1))
##div_vals, div_inds1 =  np.histogram(np.array(div_times_mcmc), bins = np.arange(0,48,1))
#
#div_vals_mcmc = div_vals_mcmc / sum(div_vals_mcmc)
#       
##print('Finished analyzing division times!\n***************************************************************')
#print('\nFinished analyzing Monte Carlo division times!\n')
#
#global NC_MCMC_max
#NC_MCMC_max = 0
#for nSim in range(len(nC_MCMC)):
#    for nTry in range(len(nC_MCMC[nSim])):
#        for tSlice in range(nC_MCMC[nSim][nTry].shape[1]):
#            for cellNum in range(nC_MCMC[nSim][nTry].shape[0]):
#                if np.isnan(nC_MCMC[nSim][nTry][cellNum, tSlice]) == False:
#                    if nC_MCMC[nSim][nTry][cellNum, tSlice] > NC_MCMC_max:
#                        NC_MCMC_max = nC_MCMC[nSim][nTry][cellNum, tSlice]
#
#if NC_MCMC_max == 0:
#    NC_MCMC_max = 1
#    print('\nMininum number of chromosomes was observed to be ZERO (0) !!!\n')
#    
#chrom_nums_mcmc= []
#        
#for nSim in range(num_MCMC_trials):
#    for nTry in range(numTrials):      
#        chrom_num_idx = np.where(~np.isnan(nC_MCMC[nSim][nTry][:, -1]))[0]
#        for c_idx in range(len(chrom_num_idx)):
#            chrom_nums_mcmc.append(nC_MCMC[nSim][nTry][chrom_num_idx[c_idx], -1])
#        
#chr_vals_mcmc, chr_inds_mcmc =  np.histogram(np.array(chrom_nums_mcmc), bins = np.arange(0,NC_MCMC_max+1,1))
#    
#chr_vals_mcmc = chr_vals_mcmc / sum(chr_vals_mcmc)
#
#####################################################################################################################################################################################################################################################################################################################################################################################################################################################################
####### Plots #######################################################################################################################################################################################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################################################################################################################################################################################################
#        
#####################################################################################################################################################################################################
########### PLOTS FOR EXPERIMENTAL DATA: ############################################################################################################################################################
#####################################################################################################################################################################################################
#        
###figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
##fig0 = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
####fig1, ax1 = plt.subplots()
###ax1 = plt.subplot2grid((2,6), (0,0), colspan = 2)
###ax2 = plt.subplot2grid((2,6), (0,2), colspan = 2)
###ax3 = plt.subplot2grid((2,6), (0,4), colspan = 2)
###ax4 = plt.subplot2grid((2,6), (1,0), colspan = 3)
###ax5 = plt.subplot2grid((2,6), (1,3), colspan = 3)
##
##textSize = 14
##
##plt.suptitle("Experimental Graphical Trajectory Analysis", fontsize = textSize + 2)
##
##
####################################################################################################
###### EXP: Plot Division Time Distribution: #######################################################
####################################################################################################
##ax5 = plt.subplot2grid((2,6), (1,3), colspan = 3)
####ax1.plot(inds[:-1]+0.5,vals,c='b',linewidth=4.0)
####ax1.bar(inds[:-1]+0.5,vals)
##ax5.bar(div_inds_exp[:-1],div_vals_exp, 0.5, color='b')
####ax1.bar(inds[:-1]+0.4,vals, 0.3, color='b')
###ax1.axis([0,14,0,0.21])
##ax5.axis([0,14,0,0.5])
##ax5.grid(True)
###plt.xlabel(r'$\tau \ (hours)$', fontsize=textSize)
###plt.ylabel(r'$P(\tau)$', fontsize=textSize)
##plt.xlabel('Division Time (hours)', fontsize=textSize)
##plt.ylabel('Probability', fontsize=textSize)
##plt.xticks(fontsize=textSize-1, rotation=0)
##plt.yticks(fontsize=textSize-1, rotation=0)
###plt.tight_layout()
###ax1.title.set_text("Experimental Cell Division Time Distribution")
#
##################################################################################################
#### EXP: Compute histogram for chromosome-numbers:                                            ###
##################################################################################################
##chrom_nums_exp= []
##        
##for nTry in range(numTries):      
##    chrom_num_idx = np.where(~np.isnan(numChrom[nTry][:, -1]))[0]
##    for c_idx in range(len(chrom_num_idx)):
##        chrom_nums_exp.append(numChrom[nTry][chrom_num_idx[c_idx], -1])
##        
##chr_vals_exp, chr_inds_exp =  np.histogram(np.array(chrom_nums_exp), bins = np.arange(0,NCG_max+1,1))
##    
##chr_vals_exp = chr_vals_exp / sum(chr_vals_exp)
#chrom_nums_exp= []
#        
#for nTry in range(numTries):      
#    chrom_num_idx = np.where(~np.isnan(numChrom_plot[nTry][:, -1]))[0]
#    for c_idx in range(len(chrom_num_idx)):
#        chrom_nums_exp.append(numChrom_plot[nTry][chrom_num_idx[c_idx], -1])
#        
#chr_vals_exp, chr_inds_exp =  np.histogram(np.array(chrom_nums_exp), bins = np.arange(0,NCG_max+1,1))
#    
#chr_vals_exp = chr_vals_exp / sum(chr_vals_exp)
#
###################################################################################################
##### EXP: Plot Chromosome Number Distribution: ###################################################
###################################################################################################
##ax2 = plt.subplot2grid((2,6), (0,2), colspan = 2)
###fig = plt.figure()
###fig, ax = plt.subplots()
####ax.plot(PfA[0:41],'r--',linewidth=4.0,label='Inferred')
###ax.plot(np.arange(0,np.size(AplotGood),1),np.transpose(AplotGood[0,0:np.size(AplotGood, axis=1)]),c='b',linewidth=4.0, label='SingleRun')
###ax.plot(np.arange(0,np.size(AplotGoodCollectionAVG),1),np.transpose(AplotGoodCollectionAVG),c='b',linewidth=4.0, label='Average')
##ax2.bar(chr_inds_exp[:-1],chr_vals_exp, 0.5, color='b')
##ax2.grid(True)
###plt.xlabel(r'$N_C$', fontsize=textSize)
###plt.ylabel(r'$P(N_C)$', fontsize=textSize)
##plt.xlabel('# of chromosomes', fontsize=textSize)
##plt.xticks(fontsize=textSize-1, rotation=0)
##plt.yticks(fontsize=textSize-1, rotation=0)
##plt.ylabel('Probability', fontsize=textSize)
###plt.tight_layout()
###ax2.title.set_text("Experimental Chromosome Number Distribution")
##
##### Do the same for Chromosomes In Dividing Cells:
##ax3 = plt.subplot2grid((2,6), (0,4), colspan = 2)
##div_chr_vals_exp, div_chr_inds_exp =  np.histogram(np.array(num_chrom_in_div_cell_exp), bins = np.arange(0,NCG_max,1))
##div_chr_vals_exp = div_chr_vals_exp / sum(div_chr_vals_exp)
##
##ax3.bar(div_chr_inds_exp[:-1], div_chr_vals_exp, 0.5, color='b')
##ax3.grid(True)
###plt.xlabel(r'$N_C$', fontsize=textSize)
###plt.ylabel(r'$P(N_C)$', fontsize=textSize)
##plt.xlabel('# of chromosomes in dividing cell', fontsize=textSize)
##plt.ylabel('Probability', fontsize=textSize)
##plt.xticks(fontsize=textSize-1, rotation=0)
##plt.yticks(fontsize=textSize-1, rotation=0)
###plt.tight_layout()
#
###################################################################################################
##### EXP: Taylor's ploting routine: ##############################################################
###################################################################################################
#n_C_Exp = numChrom_plot
#
##numCells = float('NaN')*np.ones((len(n_C_Exp),np.max(stopFrames)))
#numCells_exp = float('NaN')*np.ones((len(n_C_Exp), max(n_C_Exp[i].shape[1] for i in range(len(n_C_Exp)))))
#numChromo_Exp = []
#for nTry in range(len(n_C_Exp)):
#    numCells_exp[nTry,:n_C_Exp[nTry].shape[1]] = np.sum(~np.isnan(n_C_Exp[nTry]),axis=0)
#    
#    """ Persistence Filter (i.e. only trust an increase in chromosomes """
#    """ if it persists for at least two frames; don't trust decreases) """
#    
#    for cellInd in range(n_C_Exp[nTry].shape[0]):
#        inds = np.where(~np.isnan(n_C_Exp[nTry][cellInd,:]))[0]
#        if inds[0] < n_C_Exp[nTry].shape[1] - 1:
#            numChromo = min(n_C_Exp[nTry][cellInd,inds[0]],n_C_Exp[nTry][cellInd,inds[0] + 1])
#            for frameInd in range(inds[0],inds[-1]):
#                while n_C_Exp[nTry][cellInd,frameInd] > numChromo and \
#                n_C_Exp[nTry][cellInd,frameInd + 1] > numChromo:
#                    numChromo += 1
#                while n_C_Exp[nTry][cellInd,frameInd] > numChromo:
#                    n_C_Exp[nTry][cellInd,frameInd] -= 1
#                while n_C_Exp[nTry][cellInd,frameInd] < numChromo:
#                    n_C_Exp[nTry][cellInd,frameInd] += 1
#            while n_C_Exp[nTry][cellInd,inds[-1]] > numChromo:
#                n_C_Exp[nTry][cellInd,inds[-1]] -= 1
#            while n_C_Exp[nTry][cellInd,inds[-1]] < numChromo:
#                n_C_Exp[nTry][cellInd,inds[-1]] += 1
#        del inds
#    del cellInd
#    del frameInd
#    del numChromo
#    
#    numChromo_Exp.extend(n_C_Exp[nTry][~np.isnan(n_C_Exp[nTry][:,-1]),-1].tolist())
#
#""" Assessing experimental results graphically """
#
#partitionPerc_exp = []
#for simInd in range(len(n_C_Exp)):
#    for frameInd in range(1,n_C_Exp[simInd].shape[1]):
#        splitInds = np.where(np.all([np.isnan(n_C_Exp[simInd][:,frameInd - 1]),\
#        ~np.isnan(n_C_Exp[simInd][:,frameInd])],axis=0))[0]
#        for cellInd in range(int(len(splitInds)/2)):
#            partitionPerc_exp.append(n_C_Exp[simInd][splitInds[2*cellInd],frameInd]/\
#            (n_C_Exp[simInd][splitInds[2*cellInd],frameInd] + n_C_Exp[simInd][splitInds[2*cellInd + 1],frameInd]))
#            partitionPerc_exp.append(1 - partitionPerc_exp[-1])
#    del frameInd
#    del splitInds
#    del cellInd
#del simInd
#
###import matplotlib.pyplot as plt
##import matplotlib as mpl
##mpl.rc('font',family='Arial')
##mpl.rc('font',size=12)
##mpl.rcParams['xtick.labelsize'] = 15
##mpl.rcParams['ytick.labelsize'] = 15
###plt.rc('font',weight='bold')
#
#### EXP Plot: Preliminary calculations:
#avgCells_exp = np.nanmean(numCells_exp,axis=0)
#stDevCells_exp = np.nanstd(numCells_exp,axis=0)
#chromoHist_exp = np.histogram(numChromo_Exp,bins=np.arange(np.max(numChromo_Exp) + 1))[0]
#chromoHist_exp = chromoHist_exp/sum(chromoHist_exp)
#del numChromo_Exp
#
###################################################################################################
##### EXP: Plot Experimental Chromosome Number over Time: #########################################
###################################################################################################
##avgCells = np.nanmean(numCells,axis=0)
##stDevCells = np.nanstd(numCells,axis=0)
##chromoHist = np.histogram(numChromo_Exp,bins=np.arange(np.max(numChromo_Exp) + 1))[0]
##chromoHist = chromoHist/sum(chromoHist)
##del numChromo_Exp
##
###plt.figure()
##ax1 = plt.subplot2grid((2,6), (0,0), colspan = 2)
##ax1.errorbar(np.arange(0,deltaT*len(avgCells),deltaT)/3600,avgCells,stDevCells,color='b')
##ax1.plot(np.arange(0,deltaT*len(avgCells),deltaT)/3600,avgCells,'.b',markersize=2)
##ax1.grid(True)
###plt.xlabel('Time (hours)',fontsize=textSize,fontweight='bold')
###plt.ylabel('# of cells',fontsize=textSize,fontweight='bold')
##plt.xlabel('Time (hours)', fontsize=textSize)
##plt.ylabel('# of cells', fontsize=textSize)
##plt.xticks(fontsize=textSize-1, rotation=0)
##plt.yticks(fontsize=textSize-1, rotation=0)
###plt.tight_layout()
##
##
###################################################################################################
##### EXP: Plot Probability vs. Percentage of chromosomes: ########################################
###################################################################################################
###fig = plt.figure()
###ax = fig.gca()
##
##inc = 1/24
##
##partitionHist,partitionInds = np.histogram(partitionPerc,np.arange(0,1 + inc,inc) - inc/2)
##partitionHist = partitionHist/np.sum(partitionHist)
##
##ax4 = plt.subplot2grid((2,6), (1,0), colspan = 3)
##ax4.bar(partitionInds[:-1] + inc/2 - 0.375*inc,partitionHist,0.75*inc,color='b')
##ax4.axis([0,1,0,0.3])
##plt.xticks(np.arange(0,1.01,0.25))
##ax4.grid(True)
##plt.xlabel('Percentage of chromosomes',fontsize=textSize)
##plt.ylabel('Probability',fontsize=textSize)
##plt.xticks(fontsize=textSize-1, rotation=0)
##plt.yticks(fontsize=textSize-1, rotation=0)
##
##plt.tight_layout()
##plt.subplots_adjust(top=0.9)
##################################################################################################################################################################################################
########### /END: EXPERIMENTAL PLOTS: ############################################################################################################################################################
##################################################################################################################################################################################################
#
#####################################################################################################################################################################################################
########### PLOTS FOR MCMC DATA: ####################################################################################################################################################################
#####################################################################################################################################################################################################
##
###figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
##fig1 = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
####fig1, ax1 = plt.subplots()
###ax1 = plt.subplot2grid((2,6), (0,0), colspan = 2)
###ax2 = plt.subplot2grid((2,6), (0,2), colspan = 2)
###ax3 = plt.subplot2grid((2,6), (0,4), colspan = 2)
###ax4 = plt.subplot2grid((2,6), (1,0), colspan = 3)
###ax5 = plt.subplot2grid((2,6), (1,3), colspan = 3)
##
##textSize = 14
##
##plt.suptitle("MaxCal Monte Carlo (MCMC) Graphical Trajectory Analysis", fontsize = textSize + 1)
##
##
####################################################################################################
###### MCMC: Plot Division Time Distribution: ######################################################
####################################################################################################
##ax05 = plt.subplot2grid((2,6), (1,3), colspan = 3)
####ax1.plot(inds[:-1]+0.5,vals,c='b',linewidth=4.0)
####ax1.bar(inds[:-1]+0.5,vals)
##ax05.bar(div_inds_mcmc[:-1],div_vals_mcmc, 0.5, color='r')
####ax1.bar(inds[:-1]+0.4,vals, 0.3, color='b')
###ax1.axis([0,14,0,0.21])
##ax05.axis([0,14,0,0.5])
##ax05.grid(True)
###plt.xlabel(r'$\tau \ (hours)$', fontsize=textSize)
###plt.ylabel(r'$P(\tau)$', fontsize=textSize)
##plt.xlabel('Division Time (hours)', fontsize=textSize)
##plt.ylabel('Probability', fontsize=textSize)
###plt.tight_layout()
###ax1.title.set_text("Experimental Cell Division Time Distribution")
##
###fig = plt.figure()
#####ax1.plot(inds[:-1]+0.5,vals,c='b',linewidth=4.0)
#####ax1.bar(inds[:-1]+0.5,vals)
###plt.bar(div_inds[:-1],div_vals, 0.5, color='r')
#####ax1.bar(inds[:-1]+0.4,vals, 0.3, color='b')
####ax1.axis([0,14,0,0.21])
###plt.axis([0,14,0,0.5])
###plt.grid(True)
####plt.xlabel(r'$\tau \ (hours)$', fontsize=textSize)
####plt.ylabel(r'$P(\tau)$', fontsize=textSize)
###plt.xlabel('Division Time (hours)', fontsize=textSize)
###plt.ylabel('Probability', fontsize=textSize)
##
###################################################################################################
##### MCMC: Compute and plot histogram for chromosome-numbers (DS1):                            ###
###################################################################################################
###chrom_nums= []
###        
###for nTry in range(numTrials):      
####    chrom_num_idx = np.where(~np.isnan(N_C_MCMC[nTry][:, -1]))[0]
###    chrom_num_idx = np.where(~np.isnan(N_C_MCMC[nTry][-1]))[0]
###
###    for c_idx in range(len(chrom_num_idx)):
###        chrom_nums.append(N_C_MCMC[nTry][-1][chrom_num_idx[c_idx]])
###        
###chr_vals, chr_inds =  np.histogram(np.array(chrom_nums), bins = np.arange(0,NC_MCMC_max,1))
###    
###chr_vals = chr_vals / sum(chr_vals)
##
###################################################################################################
##### MCMC: Plot Chromosome Number Distribution: ##################################################
###################################################################################################
##ax02 = plt.subplot2grid((2,6), (0,2), colspan = 2)
###fig = plt.figure()
###fig, ax = plt.subplots()
####ax.plot(PfA[0:41],'r--',linewidth=4.0,label='Inferred')
###ax.plot(np.arange(0,np.size(AplotGood),1),np.transpose(AplotGood[0,0:np.size(AplotGood, axis=1)]),c='b',linewidth=4.0, label='SingleRun')
###ax.plot(np.arange(0,np.size(AplotGoodCollectionAVG),1),np.transpose(AplotGoodCollectionAVG),c='b',linewidth=4.0, label='Average')
##ax02.bar(chr_inds_mcmc[:-1],chr_vals_mcmc, 0.5, color='r')
##ax02.grid(True)
###plt.xlabel(r'$N_C$', fontsize=textSize)
###plt.ylabel(r'$P(N_C)$', fontsize=textSize)
##plt.xlabel('# of chromosomes', fontsize=textSize)
##plt.ylabel('Probability', fontsize=textSize)
###plt.tight_layout()
###ax2.title.set_text("Experimental Chromosome Number Distribution")
##
##### Do the same for Chromosomes In Dividing Cells:
##ax03 = plt.subplot2grid((2,6), (0,4), colspan = 2)
##div_chr_vals_mcmc, div_chr_inds_mcmc =  np.histogram(np.array(num_chrom_in_div_cell_mcmc), bins = np.arange(0,NC_MCMC_max,1))
##div_chr_vals_mcmc = div_chr_vals_mcmc / sum(div_chr_vals_mcmc)
##
##ax03.bar(div_chr_inds_mcmc[:-1], div_chr_vals_mcmc, 0.5, color='r')
##ax03.grid(True)
###plt.xlabel(r'$N_C$', fontsize=textSize)
###plt.ylabel(r'$P(N_C)$', fontsize=textSize)
##plt.xlabel('# of chromosomes in dividing cell', fontsize=textSize)
##plt.ylabel('Probability', fontsize=textSize)
###plt.tight_layout()
#
###################################################################################################
##### MCMC: Taylor's ploting routine adapted for MaxCal Monte Carlo (MCMC): #######################
###################################################################################################
##n_C_Exp = N_C_MCMC_DS2
#
###numCells = float('NaN')*np.ones((len(n_C_Exp),np.max(stopFrames)))
#
###### Find number of columns of each trial of each MCMC simulation
###mcmc_widths = np.zeros((num_MCMC_trials, len(numChrom)))           
###for nSim in range(num_MCMC_trials):
###    for nTry in range(len(N_C_MCMC_DS2[nSim])):
###        if N_C_MCMC_DS2[nSim][nTry].shape[1] > mcmc_widths[nSim, nTry]:
###            mcmc_widths[nSim, nTry] = N_C_MCMC_DS2[nSim][nTry].shape[1]
###max_mcmc_widths = np.max(mcmc_widths, axis=1).tolist()                          ## Max(#rows) for each MCMC simulation
###
###### Find number of rows of each trial of each MCMC simulation
###mcmc_heights = np.zeros((num_MCMC_trials, len(numChrom)))           
###for nSim in range(num_MCMC_trials):
###    for nTry in range(len(N_C_MCMC_DS2[nSim])):
###        if N_C_MCMC_DS2[nSim][nTry].shape[0] > mcmc_heights[nSim, nTry]:
###            mcmc_heights[nSim, nTry] = N_C_MCMC_DS2[nSim][nTry].shape[0]
###max_mcmc_heights = np.max(mcmc_heights, axis=1).tolist()                        ## Max(#columns) for each MCMC simulation
##
##### Find number of columns of each trial of each MCMC simulation
##mcmc_widths = np.zeros((num_MCMC_trials, len(numChrom)))           
##for nSim in range(num_MCMC_trials):
##    for nTry in range(len(nC_MCMC[nSim])):
##        if nC_MCMC[nSim][nTry].shape[1] > mcmc_widths[nSim, nTry]:
##            mcmc_widths[nSim, nTry] = nC_MCMC[nSim][nTry].shape[1]
##max_mcmc_widths = np.max(mcmc_widths, axis=1).tolist()                          ## Max(#rows) for each MCMC simulation
##
##### Find number of rows of each trial of each MCMC simulation
##mcmc_heights = np.zeros((num_MCMC_trials, len(numChrom)))           
##for nSim in range(num_MCMC_trials):
##    for nTry in range(len(nC_MCMC[nSim])):
##        if nC_MCMC[nSim][nTry].shape[0] > mcmc_heights[nSim, nTry]:
##            mcmc_heights[nSim, nTry] = nC_MCMC[nSim][nTry].shape[0]
##max_mcmc_heights = np.max(mcmc_heights, axis=1).tolist()                        ## Max(#columns) for each MCMC simulation
##
#### Find number of columns of each trial of each MCMC simulation
#mcmc_widths = np.zeros((num_MCMC_trials, len(numChrom_plot)))           
#for nSim in range(num_MCMC_trials):
#    for nTry in range(len(nC_MCMC[nSim])):
#        if nC_MCMC[nSim][nTry].shape[1] > mcmc_widths[nSim, nTry]:
#            mcmc_widths[nSim, nTry] = nC_MCMC[nSim][nTry].shape[1]
#max_mcmc_widths = np.max(mcmc_widths, axis=1).tolist()                          ## Max(#rows) for each MCMC simulation
#
#### Find number of rows of each trial of each MCMC simulation
#mcmc_heights = np.zeros((num_MCMC_trials, len(numChrom_plot)))           
#for nSim in range(num_MCMC_trials):
#    for nTry in range(len(nC_MCMC[nSim])):
#        if nC_MCMC[nSim][nTry].shape[0] > mcmc_heights[nSim, nTry]:
#            mcmc_heights[nSim, nTry] = nC_MCMC[nSim][nTry].shape[0]
#max_mcmc_heights = np.max(mcmc_heights, axis=1).tolist()  
#
###################################################################################################
#
##numCells_MCMC = [[] for nn in range(num_MCMC_trials)]
##for nSim in range(num_MCMC_trials):
##    numCells_MCMC[nSim] = float('NaN')*np.ones((len(N_C_MCMC_DS2[nSim]), int(max_mcmc_widths[nSim])))
##numChromo_MCMC = []
##for nSim in range(len(N_C_MCMC_DS2)):
##    for nTry in range(len(N_C_MCMC_DS2[nSim])):
##        numCells_MCMC[nSim][nTry,:N_C_MCMC_DS2[nSim][nTry].shape[1]] = np.sum(~np.isnan(N_C_MCMC_DS2[nSim][nTry]),axis=0)
##        
##        """ Persistence Filter (i.e. only trust an increase in chromosomes """
##        """ if it persists for at least two frames; don't trust decreases) """
##        
##        for cellInd in range(N_C_MCMC_DS2[nSim][nTry].shape[0]):
##            inds = np.where(~np.isnan(N_C_MCMC_DS2[nSim][nTry][cellInd,:]))[0]
##            if len(inds) > 0:
##                if inds[0] < N_C_MCMC_DS2[nSim][nTry].shape[1] - 1:
##                    numChromo = min(N_C_MCMC_DS2[nSim][nTry][cellInd,inds[0]],N_C_MCMC_DS2[nSim][nTry][cellInd,inds[0] + 1])
##                    for frameInd in range(inds[0],inds[-1]):
##                        while N_C_MCMC_DS2[nSim][nTry][cellInd,frameInd] > numChromo and \
##                        N_C_MCMC_DS2[nSim][nTry][cellInd,frameInd + 1] > numChromo:
##                            numChromo += 1
##                        while N_C_MCMC_DS2[nSim][nTry][cellInd,frameInd] > numChromo:
##                            N_C_MCMC_DS2[nSim][nTry][cellInd,frameInd] -= 1
##                        while N_C_MCMC_DS2[nSim][nTry][cellInd,frameInd] < numChromo:
##                            N_C_MCMC_DS2[nSim][nTry][cellInd,frameInd] += 1
##                    while N_C_MCMC_DS2[nSim][nTry][cellInd,inds[-1]] > numChromo:
##                        N_C_MCMC_DS2[nSim][nTry][cellInd,inds[-1]] -= 1
##                    while N_C_MCMC_DS2[nSim][nTry][cellInd,inds[-1]] < numChromo:
##                        N_C_MCMC_DS2[nSim][nTry][cellInd,inds[-1]] += 1
##                del inds
##        del cellInd
##        del frameInd
##        del numChromo
##        
##        numChromo_MCMC.extend(N_C_MCMC_DS2[nSim][nTry][~np.isnan(N_C_MCMC_DS2[nSim][nTry][:,-1]),-1].tolist())
#
#numCells_MCMC = [[] for nn in range(num_MCMC_trials)]
#for nSim in range(num_MCMC_trials):
#    numCells_MCMC[nSim] = float('NaN')*np.ones((len(nC_MCMC[nSim]), int(max_mcmc_widths[nSim])))
#numChromo_MCMC = []
#for nSim in range(len(nC_MCMC)):
#    for nTry in range(len(nC_MCMC[nSim])):
#        numCells_MCMC[nSim][nTry,:nC_MCMC[nSim][nTry].shape[1]] = np.sum(~np.isnan(nC_MCMC[nSim][nTry]),axis=0)
#        
#        """ Persistence Filter (i.e. only trust an increase in chromosomes """
#        """ if it persists for at least two frames; don't trust decreases) """
#        
#        for cellInd in range(nC_MCMC[nSim][nTry].shape[0]):
#            inds = np.where(~np.isnan(nC_MCMC[nSim][nTry][cellInd,:]))[0]
#            if len(inds) > 0:
#                if inds[0] < nC_MCMC[nSim][nTry].shape[1] - 1:
#                    numChromo = min(nC_MCMC[nSim][nTry][cellInd,inds[0]],nC_MCMC[nSim][nTry][cellInd,inds[0] + 1])
#                    for frameInd in range(inds[0],inds[-1]):
#                        while nC_MCMC[nSim][nTry][cellInd,frameInd] > numChromo and \
#                        nC_MCMC[nSim][nTry][cellInd,frameInd + 1] > numChromo:
#                            numChromo += 1
#                        while nC_MCMC[nSim][nTry][cellInd,frameInd] > numChromo:
#                            nC_MCMC[nSim][nTry][cellInd,frameInd] -= 1
#                        while nC_MCMC[nSim][nTry][cellInd,frameInd] < numChromo:
#                            nC_MCMC[nSim][nTry][cellInd,frameInd] += 1
#                    while nC_MCMC[nSim][nTry][cellInd,inds[-1]] > numChromo:
#                        nC_MCMC[nSim][nTry][cellInd,inds[-1]] -= 1
#                    while nC_MCMC[nSim][nTry][cellInd,inds[-1]] < numChromo:
#                        nC_MCMC[nSim][nTry][cellInd,inds[-1]] += 1
#                del inds
#        del cellInd
#        del frameInd
#        del numChromo
#        
#        numChromo_MCMC.extend(nC_MCMC[nSim][nTry][~np.isnan(nC_MCMC[nSim][nTry][:,-1]),-1].tolist())
###################################################################################################
#
#""" Assessing MCMC results graphically """
#
#partitionPerc_mcmc = []
#for nSim in range(len(nC_MCMC)):
#    for simInd in range(len(nC_MCMC[nSim])):
#        for frameInd in range(1,nC_MCMC[nSim][simInd].shape[1]):
#            splitInds = np.where(np.all([np.isnan(nC_MCMC[nSim][simInd][:,frameInd - 1]),\
#            ~np.isnan(nC_MCMC[nSim][simInd][:,frameInd])],axis=0))[0]
#            for cellInd in range(int(len(splitInds)/2)):
#                if np.isnan(nC_MCMC[nSim][simInd][splitInds[2*cellInd],frameInd] / (nC_MCMC[nSim][simInd][splitInds[2*cellInd],frameInd] + nC_MCMC[nSim][simInd][splitInds[2*cellInd + 1],frameInd])) == False:
#                    partitionPerc_mcmc.append(nC_MCMC[nSim][simInd][splitInds[2*cellInd],frameInd]/\
#                    (nC_MCMC[nSim][simInd][splitInds[2*cellInd],frameInd] + nC_MCMC[nSim][simInd][splitInds[2*cellInd + 1],frameInd]))
#                    partitionPerc_mcmc.append(1 - partitionPerc_mcmc[-1])
##                else: 
##                    print("'partitionPerc_mcmc' has a NaN in it at:\nnTry = " + str(simInd) + ", tSlice = " + str(frameInd) + ", cellNum = " + str(cellInd) + ".")
#        del frameInd
#        del splitInds
#    #    del cellInd
#    del simInd
#del nSim
#
###################################################################################################
#### MCMC Plot: Preliminary calculations:                                                       ###
###################################################################################################
#
####HugeMCMC = float('NaN')*np.ones((num_MCMC_trials*len(numChrom), np.max(max_mcmc_width)))
###HugeMCMC = float('NaN')*np.ones((np.sum(mcmc_heights), np.max(mcmc_widths)))
###
###for simInd in range(num_MCMC_trials):
####    HugeMCMC[len(numChrom)*simInd:len(numChrom)*(simInd+1), 0:N_C_MCMC_DS2[simInd]]
###    for nTry in range(len(numChrom)):
###        HugeMCMC[HugeMCMC.shape[0]:HugeMCMC.shape[0] + N_C_MCMC_DS2[simInd][nTry].shape[0], 0:N_C_MCMC_DS2[simInd][nTry].shape[1]] = N_C_MCMC_DS2[simInd][nTry]
##    
##hugeNumCells = float('NaN')*np.ones((len(numChrom)*num_MCMC_trials, int(np.max(mcmc_widths))))
##
##for simInd in range(num_MCMC_trials):
##    hugeNumCells[len(numChrom)*simInd:len(numChrom)*(simInd+1), 0:numCells_MCMC[simInd].shape[1]] = numCells_MCMC[simInd]
##
#hugeNumCells = float('NaN')*np.ones((len(numChrom_plot)*num_MCMC_trials, int(np.max(mcmc_widths))))
#
#for simInd in range(num_MCMC_trials):
#    hugeNumCells[len(numChrom_plot)*simInd:len(numChrom_plot)*(simInd+1), 0:numCells_MCMC[simInd].shape[1]] = numCells_MCMC[simInd]
#
##avgCells_MCMC = np.nanmean(numCells_MCMC,axis=0)
##stDevCells_MCMC = np.nanstd(numCells_MCMC,axis=0)
#avgCells_MCMC = np.nanmean(hugeNumCells,axis=0)
#stDevCells_MCMC = np.nanstd(hugeNumCells,axis=0)
#chromoHist_MCMC = np.histogram(numChromo_MCMC,bins=np.arange(np.max(numChromo_MCMC) + 1))[0]
#chromoHist_MCMC = chromoHist_MCMC/sum(chromoHist_MCMC)
#del numChromo_MCMC
#
###################################################################################################
##### MCMC: Plot Chromosome Number over Time: #####################################################
###################################################################################################
##avgCells_MCMC = np.nanmean(numCells_MCMC,axis=0)
##stDevCells_MCMC = np.nanstd(numCells_MCMC,axis=0)
##chromoHist_MCMC = np.histogram(numChromo_MCMC,bins=np.arange(np.max(numChromo_MCMC) + 1))[0]
##chromoHist_MCMC = chromoHist_MCMC/sum(chromoHist_MCMC)
##del numChromo_MCMC
##
###plt.figure()
##ax01 = plt.subplot2grid((2,6), (0,0), colspan = 2)
##ax01.errorbar(np.arange(0,deltaT*len(avgCells_MCMC),deltaT)/3600,avgCells_MCMC,stDevCells_MCMC,color='r')
##ax01.plot(np.arange(0,deltaT*len(avgCells_MCMC),deltaT)/3600,avgCells_MCMC,'.r',markersize=2)
##ax01.grid(True)
###plt.xlabel('Time (hours)',fontsize=textSize,fontweight='bold')
###plt.ylabel('# of cells',fontsize=textSize,fontweight='bold')
##plt.xlabel('Time (hours)', fontsize=textSize)
##plt.ylabel('# of cells', fontsize=textSize)
###plt.tight_layout()
##
###################################################################################################
##### MCMC: Plot Probability vs. Percentage of chromosomes: #######################################
###################################################################################################
###fig = plt.figure()
###ax = fig.gca()
##
##inc = 1/24
##
##partitionHist,partitionInds = np.histogram(partitionPerc_mcmc,np.arange(0,1 + inc,inc) - inc/2)
##partitionHist = partitionHist/np.sum(partitionHist)
##
##ax04 = plt.subplot2grid((2,6), (1,0), colspan = 3)
##ax04.bar(partitionInds[:-1] + inc/2 - 0.375*inc,partitionHist,0.75*inc,color='r')
##ax04.axis([0,1,0,0.3])
###ax4.xticks(np.arange(0,1.01,0.25))
##ax04.grid(True)
##plt.xlabel('Percentage of chromosomes',fontsize=textSize)
##plt.ylabel('Probability',fontsize=textSize)
##
##plt.tight_layout()
##plt.subplots_adjust(top=0.9)
##########################################################################################################################################################################################
########### /END: MCMC PLOTS: ############################################################################################################################################################
##########################################################################################################################################################################################
#
##########################################################################################################################################################################################
#### *#*#*# Routine for plotting both sets of plots (above) on a single image:
##########################################################################################################################################################################################
#textSize = 14
#chrom_x_axis_length = 14
#
#fig666 = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
#
#plt.suptitle("Experimental vs. MaxCal Monte Carlo (MCMC) Graphical Trajectory Analysis", fontsize = textSize + 2)
#
###################################################################################################
##### Plot Division Time Distributions: ###########################################################
###################################################################################################
#ax5 = plt.subplot2grid((2,6), (1,3), colspan = 3)
#
##plot_exp_div_time_dist = ax5.bar(np.arange(0,len(div_vals_exp)) - 0.125,div_vals_exp,0.25,color='b')
#plot_exp_div_time_dist = ax5.bar(div_inds_exp[:-1]-0.125, div_vals_exp, 0.25, color='b')
#plot_mcmc_div_time_dist = ax5.bar(div_inds_mcmc[:-1]+0.125, div_vals_mcmc, 0.25, color='r')
##plot_mcmc_div_time_dist = ax5.bar(div_inds_mcmc_asIgo[:-1]+0.125, div_vals_mcmc_asIgo, 0.25, color='r')
#
#ax5.axis([0,14,0,0.5])
#ax5.grid(True)
#plt.xlabel('Division Time (hours)', fontsize=textSize)
#plt.ylabel('Probability', fontsize=textSize)
#plt.xticks(fontsize=textSize-1, rotation=0)
#plt.yticks(fontsize=textSize-1, rotation=0)
#plt.legend( ('Experiment', 'MCMC') )
##plt.tight_layout()
#
##################################################################################################
#### Plot Chromosome Number Distributions: #######################################################
##################################################################################################
#ax2 = plt.subplot2grid((2,6), (0,2), colspan = 2)
#NCG_max
##plot_exp_chrom_num_dist = ax2.bar(chr_inds_exp[:-1]-0.125, chr_vals_exp, 0.25, color='b')
##plot_mcmc_chrom_num_dist = ax2.bar(chr_inds_mcmc[:-1]+0.125, chr_vals_mcmc, 0.25, color='r')
#plot_exp_chrom_num_dist = ax2.bar(chr_inds_exp[:-1]-0.125, chr_vals_exp, 0.25, color='b')
##plot_mcmc_chrom_num_dist = ax2.bar(chr_inds_mcmc[:-1]+0.125, chr_vals_mcmc, 0.25, color='r')
##plot_mcmc_chrom_num_dist = ax2.bar(chr_inds_mcmc[:-(np.shape(chr_inds_mcmc)[0] - np.shape(chr_inds_exp)[0] + 1)]+0.125, chr_vals_mcmc[:-(np.shape(chr_inds_mcmc)[0] - np.shape(chr_inds_exp)[0])], 0.25, color='r')
#plot_mcmc_chrom_num_dist = ax2.bar(chr_inds_mcmc[:chrom_x_axis_length]+0.125, chr_vals_mcmc[:chrom_x_axis_length], 0.25, color='r')
#
#ax2.axis([0,chrom_x_axis_length,0,0.4])
#ax2.grid(True)
#plt.xlabel('# of chromosomes', fontsize=textSize)
#plt.xticks(fontsize=textSize-1, rotation=0)
#plt.yticks(fontsize=textSize-1, rotation=0)
#plt.ylabel('Probability', fontsize=textSize)
#plt.legend( ('Exp.', 'MCMC') , loc='lower left', bbox_to_anchor=(0.5, 0.7))
##plt.tight_layout()
##ax2.title.set_text("Chromosome Number Distribution")
#
##################################################################################################
#### Plot Number of Chromosomes in Dividing Cell Distributions: ##################################
##################################################################################################
#ax3 = plt.subplot2grid((2,6), (0,4), colspan = 2)
#
#### EXP Plot: Preliminary calculations:
#div_chr_vals_exp, div_chr_inds_exp =  np.histogram(np.array(num_chrom_in_div_cell_exp), bins = np.arange(0,NC_plot_max+1,1))
#div_chr_vals_exp = div_chr_vals_exp / sum(div_chr_vals_exp)
#### MCMC Plot: Preliminary calculations:
#div_chr_vals_mcmc, div_chr_inds_mcmc =  np.histogram(np.array(num_chrom_in_div_cell_mcmc), bins = np.arange(0,NC_MCMC_max+1,1))
#div_chr_vals_mcmc = div_chr_vals_mcmc / sum(div_chr_vals_mcmc)
#
#plot_exp_div_cell_chrom_num_dist = ax3.bar(div_chr_inds_exp[:-1]-0.125, div_chr_vals_exp, 0.25, color='b')
##plot_mcmc_div_cell_chrom_num_dist = ax3.bar(div_chr_inds_mcmc[:-1]+0.125, div_chr_vals_mcmc, 0.25, color='r')
##plot_mcmc_div_cell_chrom_num_dist = ax3.bar(div_chr_inds_mcmc[:-(np.shape(div_chr_inds_mcmc)[0] - np.shape(div_chr_inds_exp)[0] + 1)]+0.125, div_chr_vals_mcmc[:-(np.shape(div_chr_vals_mcmc)[0] - np.shape(div_chr_vals_exp)[0])], 0.25, color='r')
#plot_mcmc_div_cell_chrom_num_dist = ax3.bar(div_chr_inds_mcmc[:chrom_x_axis_length]+0.125, div_chr_vals_mcmc[:chrom_x_axis_length], 0.25, color='r')
#
#ax3.axis([0,chrom_x_axis_length,0,0.4])
#ax3.grid(True)
##plt.xlabel(r'$N_C$', fontsize=textSize)
##plt.ylabel(r'$P(N_C)$', fontsize=textSize)
#plt.xlabel('# dividing-cell chromosomes', fontsize=textSize)
#plt.ylabel('Probability', fontsize=textSize)
#plt.xticks(fontsize=textSize-1, rotation=0)
#plt.yticks(fontsize=textSize-1, rotation=0)
#plt.legend( ('Exp.', 'MCMC') , loc='lower left', bbox_to_anchor=(0.51, 0.7))
#
##### Test simpler plotting routine to check for errors (yep: needed to change bin range to " bins = np.arange(0,NCG_max+1,1) " from " bins = np.arange(0,NCG_max,1) ":
##fig0001, ax0001 = plt.subplots()
##div_chr_vals_exp, div_chr_inds_exp =  np.histogram(np.array(num_chrom_in_div_cell_exp), bins = np.arange(0,NCG_max+1,1))
##div_chr_vals_exp = div_chr_vals_exp / sum(div_chr_vals_exp)
##plot_exp_div_cell_chrom_num_dist = ax0001.bar(div_chr_inds_exp[:-1]-0.125, div_chr_vals_exp, 0.25, color='b')
##plt.grid(True)
##plt.xlabel('# of chromosomes in dividing cell', fontsize=textSize)
##plt.ylabel('Probability', fontsize=textSize)
##
##fig0002, ax0002 = plt.subplots()
##div_chr_vals_mcmc, div_chr_inds_mcmc =  np.histogram(np.array(num_chrom_in_div_cell_mcmc), bins = np.arange(0,NC_MCMC_max+1,1))
##div_chr_vals_mcmc = div_chr_vals_mcmc / sum(div_chr_vals_mcmc)
##plot_mcmc_div_cell_chrom_num_dist = ax0002.bar(div_chr_inds_mcmc[:-1]-0.125, div_chr_vals_mcmc, 0.25, color='r')
##plt.grid(True)
##plt.xlabel('# of chromosomes in dividing cell', fontsize=textSize)
##plt.ylabel('Probability', fontsize=textSize)
#
##################################################################################################
#### Plot Chromosome Numbers over Time: ##########################################################
##################################################################################################
#ax1 = plt.subplot2grid((2,6), (0,0), colspan = 2)
#
#### EXP: Plot with error bars:
#ax1.errorbar(np.arange(0,deltaT*len(avgCells_exp),deltaT)/3600,avgCells_exp,stDevCells_exp,color='b', capsize=3, elinewidth=0.5, markeredgewidth=0.5)
#plot_exp_chrom_num_vs_time = ax1.plot(np.arange(0,deltaT*len(avgCells_exp),deltaT)/3600,avgCells_exp,'.b',markersize=5)
#### MCMC: Plot with error bars:
#ax1.errorbar(np.arange(0,deltaT*len(avgCells_MCMC),deltaT)/3600,avgCells_MCMC,stDevCells_MCMC,color='r', capsize=3, elinewidth=0.5, markeredgewidth=0.5)
#plot_mcmc_chrom_num_vs_time = ax1.plot(np.arange(0,deltaT*len(avgCells_MCMC),deltaT)/3600,avgCells_MCMC,'.r',markersize=5)
#
#ax1.grid(True)
#plt.xlabel('Time (hours)', fontsize=textSize)
#plt.ylabel('# of cells', fontsize=textSize)
#plt.xticks(fontsize=textSize-1, rotation=0)
#plt.yticks(fontsize=textSize-1, rotation=0)
#plt.legend( ('Experiment', 'MCMC') )
##plt.tight_layout()
#
###################################################################################################
##### Plot Probabilities vs. Percentages of chromosomes: ##########################################
###################################################################################################
#ax4 = plt.subplot2grid((2,6), (1,0), colspan = 3)
#
#inc = 1/24
#### EXP Plot: Preliminary calculations:
#partitionHist_exp, partitionInds_exp = np.histogram(partitionPerc_exp,np.arange(0,1 + inc,inc) - inc/2)
#partitionHist_exp = partitionHist_exp/np.sum(partitionHist_exp)
#### MCMC Plot: Preliminary calculations:
#partitionHist_mcmc, partitionInds_mcmc = np.histogram(partitionPerc_mcmc,np.arange(0,1 + inc,inc) - inc/2)
#partitionHist_mcmc = partitionHist_mcmc/np.sum(partitionHist_mcmc)
#
#####plot_exp_percentage_dist = ax4.bar(partitionInds_exp[:-1] + inc/2 - 0.375*inc,partitionHist_exp,0.75*inc,color='b')
#####plot_mcmc_percentage_dist = ax4.bar(partitionInds_mcmc[:-1] + inc/2 - 0.375*inc,partitionHist_mcmc,0.75*inc,color='r')
####plot_exp_percentage_dist = ax4.bar((partitionInds_exp[:-1] + inc/2 - 0.375*inc)-0.75*inc/4, partitionHist_exp,0.75*inc/2,color='b')
####plot_mcmc_percentage_dist = ax4.bar((partitionInds_mcmc[:-1] + inc/2 - 0.375*inc)+0.75*inc/4, partitionHist_mcmc,0.75*inc/2,color='r')
#plot_exp_percentage_dist = ax4.bar((partitionInds_exp[:-1] + inc/2 - 0.375*inc/2), partitionHist_exp,0.75*inc/2,color='b')
#plot_mcmc_percentage_dist = ax4.bar((partitionInds_mcmc[:-1] + inc/2 + 0.375*inc/2), partitionHist_mcmc,0.75*inc/2,color='r')
##plot_mcmc_percentage_dist = ax4.bar((partitionInds_mcmc[:-1] + inc/2 + 0.275*inc/2), partitionHist_mcmc,0.75*inc/2,color='r')
###plot_exp_percentage_dist = ax4.bar((partitionInds_exp[:-1] + inc/2 - 0.4*inc/2), partitionHist_exp,0.75*inc/2,color='b')
###plot_mcmc_percentage_dist = ax4.bar((partitionInds_mcmc[:-1] + inc/2 + 0.3*inc/2), partitionHist_mcmc,0.75*inc/2,color='r')
#
#ax4.axis([0,1,0,0.3])
#plt.xticks(np.arange(0,1.01,0.25))
#ax4.grid(True)
#plt.xlabel('Percentage of chromosomes',fontsize=textSize)
#plt.ylabel('Probability',fontsize=textSize)
#plt.xticks(fontsize=textSize-1, rotation=0)
#plt.yticks(fontsize=textSize-1, rotation=0)
#plt.legend( ('Experiment', 'MCMC') )
#
#plt.tight_layout()
#plt.subplots_adjust(top=0.9)
#
######################################################################################################################################################
#### Saving Routine for Gillespie + MCMC distributions on same plot:
##if saving == 2:
##    print('Saving plot data ...')
##    filename_0007 = "/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/MaxCal_ChromosomeCopying_CellDividing_ChromosomePartitioning/Gillespie_Data/cellDiv_Gillespie_data_maxTime=%s_ed#%s"%(str('{:.2e}'.format(maxTime)), str(edition_number))
##    #np.savez(filename_0007, Aplot = Aplot, AplotGood = AplotGood, divTimes = divTimes, divTimePlots = divTimePlots)
##    np.savez(filename_0007, chr_vals = chr_vals, chr_inds = chr_inds , chrom_nums = chrom_nums, div_vals = div_vals, div_inds = div_inds, div_times = div_times)
#    
#########################################################################################################################################################################################
#########################################################################################################################################################################################
#
######################################################################################################################################################
### Normal Plot-Data Saving Routine (may require tweaking for specific cases):
##if saving == 2:
##    print('Saving plot data ...')
##    filename_0007 = "/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/MaxCal_ChromosomeCopying_CellDividing_ChromosomePartitioning/MCMC_Data/cellDiv_MCMC_data_maxTime=%s_v%s"%(str('{:.2e}'.format(maxTime)), str(version))
##    #np.savez(filename_0007, Aplot = Aplot, AplotGood = AplotGood, divTimes = divTimes, divTimePlots = divTimePlots)
##    np.savez(filename_0007, chr_vals = chr_vals, chr_inds = chr_inds , chrom_nums = chrom_nums, div_vals = div_vals, div_inds = div_inds, div_times = div_times)
#print('Saving plot data ...')
##filename_0007 = "/home/swedeki3/PloidyVariation/M%s/cellDiv_MCMC_data_maxTime=%shrs_v%s"%(str(int(modelNum)), str('{:.2e}'.format(t_f)), str(version))
#filename_0007 = "/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Prediction/M%s/Ploidy_MCMC_data_maxTime=%shrs_v%s"%(str(int(modelNum)), str(MCMC_maxT_hours), str(version)) #%(str(int(modelNum)), str('{:.2e}'.format(t_f)), str(version))
##np.savez(filename_0007, div_vals_exp = div_vals_exp, div_inds_exp = div_inds_exp, div_vals_mcmc = div_vals_mcmc, div_inds_mcmc = div_inds_mcmc, chr_vals_exp = chr_vals_exp, chr_inds_exp = chr_inds_exp, chr_vals_mcmc = chr_vals_mcmc, chr_inds_mcmc = chr_inds_mcmc, div_chr_vals_exp = div_chr_vals_exp, div_chr_inds_exp = div_chr_inds_exp, div_chr_vals_mcmc = div_chr_vals_mcmc, div_chr_inds_mcmc = div_chr_inds_mcmc, avgCells_exp = avgCells_exp, avgCells_MCMC = avgCells_MCMC, partitionHist_exp = partitionHist_exp, partitionInds_exp = partitionInds_exp, partitionHist_mcmc = partitionHist_mcmc, partitionInds_mcmc = partitionInds_mcmc, deltaT = deltaT, bestGuess = bestGuess, bestLike = bestLike, numChrom = numChrom)
#np.savez(filename_0007, avgCells_exp = avgCells_exp, stDevCells_exp = stDevCells_exp, chromoHist_exp = chromoHist_exp, avgCells_MCMC = avgCells_MCMC, stDevCells_MCMC = stDevCells_MCMC, chromoHist_MCMC = chromoHist_MCMC, div_vals_exp = div_vals_exp, div_inds_exp = div_inds_exp, div_vals_mcmc = div_vals_mcmc, div_inds_mcmc = div_inds_mcmc, chr_vals_exp = chr_vals_exp, chr_inds_exp = chr_inds_exp, chr_vals_mcmc = chr_vals_mcmc, chr_inds_mcmc = chr_inds_mcmc, div_chr_vals_exp = div_chr_vals_exp, div_chr_inds_exp = div_chr_inds_exp, div_chr_vals_mcmc = div_chr_vals_mcmc, div_chr_inds_mcmc = div_chr_inds_mcmc, partitionHist_exp = partitionHist_exp, partitionInds_exp = partitionInds_exp, partitionHist_mcmc = partitionHist_mcmc, partitionInds_mcmc = partitionInds_mcmc, deltaT = deltaT, bestGuess = bestGuess, bestLike = bestLike, numChrom = numChrom)
#
##########################################################################################
##avg_div_time_exp = np.average(np.divide(div_times_exp,2))
##avg_div_time_mcmc = np.average(np.divide(div_times_mcmc,2))
#
#avg_div_time_exp = np.average(np.multiply(div_times_exp, (timeRes/3600)))
#avg_div_time_mcmc = np.average(np.multiply(div_times_mcmc, (timeRes/3600)))
##avg_div_time_mcmc = np.average((timeRes/3600)*np.array(div_times_mcmc) + 1)
#
#print("*************************************************************************\n")
#print("Average Experimental Division Time: \n    " + str(avg_div_time_exp) + " hours" )
#print("Average MaxCal Monte Carlo Division Time: \n    " + str(avg_div_time_mcmc) + " hours" )
#print("Percent Difference: \n    " + str( ((avg_div_time_mcmc - avg_div_time_exp) / avg_div_time_exp) *100 )  + " %")
#
#
#NumCells_exp = []
#NumCells_mcmc = []
#
#tx = []
#
#for t in np.arange(0, 20, 0.1):
#    NumCells_exp.append(np.exp((t/avg_div_time_exp)*np.log(2)))
#    NumCells_mcmc.append(np.exp((t/avg_div_time_mcmc)*np.log(2)))
#    tx.append(t)
#    
#fig2 = plt.figure()
#exp_cells_line = plt.plot(tx, NumCells_exp, 'b', label = 'exp_cells', linewidth=3)
#mcmc_cells_line = plt.plot(tx, NumCells_mcmc, 'r', label = 'mcmc_cells', linewidth=3)
##plt.plot(tx, NumCells_exp, 'b', label = 'exp_cells')
##plt.plot(tx, NumCells_mcmc, 'r', label = 'mcmc_cells')
#plt.grid(True)
#plt.ylabel("# of cells", fontsize=textSize)
#plt.xlabel("time (hours)", fontsize=textSize)
#plt.title("Exp (blue) vs. MCMC (red) from Avg. Division Times", fontsize=textSize)
#plt.legend( ('Experiment', 'MCMC') )
#
##########################################################################################
##filename_div_times = '/home/swedeki3/PloidyVariation/M%s/Ploidy_MaxCal_Inference+MonteCarlo_comparison_hC=%s_hg=%s_hP=%s_hKgC=%s_hKgC2=%s_deltaT=%s_v%s.npz' %(str(int(modelNum)), str(np.round(bestGuess[0], 3)), str(np.round(bestGuess[1], 3)), str(np.round(bestGuess[2], 3)), str(np.round(bestGuess[3], 3)), str(np.round(bestGuess[4], 3)), str(timeRes), str(version))
#filename_div_times = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/PartialData_Prediction/M%s/Ploidy_MaxCal_Inference+MonteCarlo_comparison_hC=%s_hg=%s_hP=%s_hKgC=%s_hKgC2=%s_deltaT=%s_v%s.npz' %(str(int(modelNum)), str(np.round(bestGuess[0], 3)), str(np.round(bestGuess[1], 3)), str(np.round(bestGuess[2], 3)), str(np.round(bestGuess[3], 3)), str(np.round(bestGuess[4], 3)), str(timeRes), str(version))
###    sio.savemat(filename0, {'N_A': N_A, 'N_alpha_star': N_alpha_star, 'R_A': R_A, 'RXNs': RXNs})
###np.savez(filename_div_times, N_C_MCMC_DS2 = N_C_MCMC_DS2, numChrom = numChrom, Div_Tracker = Div_Tracker, Div_Tracker_mcmc = Div_Tracker_mcmc, Div_Tracker_exp = Div_Tracker_exp, div_times_exp = div_times_exp, div_times_mcmc = div_times_mcmc, maxFrames = max(frameLengths), numTrials = numTrials, num_MCMC_trials = num_MCMC_trials, total_time_seconds = t_f, timeRes = timeRes, num_chrom_in_div_cell_mcmc = num_chrom_in_div_cell_mcmc, num_chrom_in_div_cell_exp = num_chrom_in_div_cell_exp)
#np.savez(filename_div_times, avg_div_time_exp = avg_div_time_exp, avg_div_time_mcmc = avg_div_time_mcmc, NumCells_exp = NumCells_exp, NumCells_mcmc = NumCells_mcmc, nC_MCMC = nC_MCMC, numChrom = numChrom, div_times_exp = div_times_exp, div_times_mcmc = div_times_mcmc, maxFrames = max(frameLengths), numTrials = numTrials, num_MCMC_trials = num_MCMC_trials, total_time_seconds = t_f, timeRes = timeRes, num_chrom_in_div_cell_mcmc = num_chrom_in_div_cell_mcmc, num_chrom_in_div_cell_exp = num_chrom_in_div_cell_exp)
#
############################################################################################################################################################################################################
##### Summary: #############################################################################################################################################################################################
############################################################################################################################################################################################################
##
###LagrBG = MaxCal_FSP_MLe(bestGuess)
##
###print('\n*************************************************************************\nInitial Guesses for Lagrange Multipliers: \n[h_C, h_alpha, h_A] = ' + str(bestGuess)+ ' with -log(Likelihood) = ' + str(LagrBG))
####print('\nMaxCal Extracted Lagrange Mutlipliers: \n[h_C, h_alpha, h_A] = ' + str(res['x']) + "\n*************************************************************************")
###print('\nFirst Iteration: MaxCal Extracted Lagrange Mutlipliers: \n[h_C, h_alpha, h_A] = ' + str(res['x']) + ' with -log(Likelihood) = ' + str(Lagr1))
###print('\nSecond Iteration: MaxCal Extracted Lagrange Mutlipliers: \n[h_C, h_alpha, h_A] = ' + str(res2['x']) + ' with -log(Likelihood) = ' + str(Lagr2))
###print('\nThird Iteration: MaxCal Extracted Lagrange Mutlipliers: \n[h_C, h_alpha, h_A] = ' + str(res3['x']) + ' with -log(Likelihood) = ' + str(Lagr3))
###print('\nLagrange Mutlipliers used in this MaxCal Monte Carlo simulation: \n[h_C, h_alpha, h_A] = ' + str(MC_lagrange) + ' with -log(Likelihood) = ' + str(LagrMCMC) + "\n*************************************************************************")
##
###print('\nSelected Transition happened ' + str(total_counted_divs) + ' times with a probability of ' + str(total_counted_divs/len(div_times_exp)))
##
##########################################################################################################################################################################################################
print("**************************************************************")
elapsed_time_secs = time.time() - start_time
print("Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs)))
print("--- %s seconds ---" % (time.time() - start_time))
##A = [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
##import random
##rows = 20
##columns = 40
##num_tries = 12
##AA = [np.array([[random.randrange(1, 12, 2) for x in range(columns)] for y in range(rows)]) for z in range(num_tries)]