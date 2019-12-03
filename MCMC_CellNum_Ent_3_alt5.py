#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:57:58 2019

Working Chromosome # ONLY Entropy
-SW 04 March 2019

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

version = "MCMC_CellNum_Ent_3_alt5"

modelNum = 7
crop_hrs = 22

#temp_inference = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/InferenceResults/%shrs/M%s/Ploidy_Inference_Results_v%shr_10x_Infer_M%s_0.2.9.npz' %(str(int(crop_hrs)), str(int(modelNum)), str(int(crop_hrs)), str(int(modelNum))))
##bestGuess = temp_inference['bestGuess']
##bestLike = temp_inference['bestLike']
#LMs = temp_inference['bestGuess']

global cap
if 1 <= modelNum <= 3:
    cap = 1
else:
    cap = 6 ### 'M' a.k.a. maximum number of timesteps for l_gamma to cycle through before dividing (for a given cell)

i_max = 10

###################################################################################################
########## Calculate Experimental Entropy: ########################################################
###################################################################################################
### Load Experimental data:
#temp_file_full = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/Experimental_Data/ExperimentalData_PersistenceFilter.npz')
temp_file_exp_data = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/Experimental_Data/ExperimentalData_PersistenceFilter.npz')
EXP_numChrom = temp_file_exp_data['n_C_Exp'][:-1]

### Crowbar data to only have MaxCal-possible transitions (i.e. never more than doubles):          
for nTry in range(len(EXP_numChrom)):
    for tSlice in range(EXP_numChrom[nTry].shape[1] -1):
#        for cellNum in range(len(N_Chrom[nTry][tSlice])):
        for cellNum in range(EXP_numChrom[nTry].shape[0]):
            if EXP_numChrom[nTry][cellNum, tSlice+1] > 2*EXP_numChrom[nTry][cellNum, tSlice]:
                EXP_numChrom[nTry][cellNum, tSlice+1] = 2*EXP_numChrom[nTry][cellNum, tSlice]
del nTry, tSlice, cellNum

### Find maximum number of cells in a given time slice:
EXP_maxCellSlice = 0
for nTry in range(len(EXP_numChrom)):
    for tSlice in range(EXP_numChrom[nTry].shape[1]):
        if len(np.where(~np.isnan(EXP_numChrom[nTry][:, tSlice]))[0]) > EXP_maxCellSlice:
            EXP_maxCellSlice = len(np.where(~np.isnan(EXP_numChrom[nTry][:, tSlice]))[0])

### Count the number of cell number transitions:
EXP_CellCounts = np.zeros((int(EXP_maxCellSlice + 1), int(EXP_maxCellSlice + 1)))
for nTry in range(len(EXP_numChrom)):
    for tSlice in np.arange(0, EXP_numChrom[nTry].shape[1] - 1, 1):
        EXP_CellCounts[int(len(np.where(~np.isnan(EXP_numChrom[nTry][:, tSlice]))[0])), int(len(np.where(~np.isnan(EXP_numChrom[nTry][:, tSlice + 1]))[0]))] += 1

### Calculate Cell # ONLY Entropy for Experimental Data:
#EXP_incEnts = np.zeros((np.shape(EXP_CellCounts)))
EXP_ent = 0
for before in range(EXP_CellCounts.shape[0]):
    for after in range(EXP_CellCounts.shape[1]):
        if EXP_CellCounts[before, after] > 0:
#            EXP_incEnts[before, after] = int(EXP_CellCounts[before, after])/np.sum(EXP_CellCounts) * ( np.log( int(EXP_CellCounts[before, after])/np.sum(EXP_CellCounts)) )
            EXP_ent -= (  int(EXP_CellCounts[before, after])/np.sum(EXP_CellCounts) * ( np.log( int(EXP_CellCounts[before, after])/np.sum(EXP_CellCounts)) )  )
#ex_entropy = - np.sum(EXP_incEnt)
#print('Experimental Entropy: S_ex = ' + str(ex_entropy) + '\n')
print('Experimental Entropy: S_ex = ' + str(EXP_ent) + '\n')

###################################################################################################
########## Calculate MCMC Entropy: ################################################################
###################################################################################################
### Load MCMC data for comparison with Experimental data:
if crop_hrs >= 21.5:
    temp_mcmc_file = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/MCMC_Data/M%s/Ploidy_MCMC_trajectories_vFullData_Infer_Predict_M%s_0.3.npz'%(str(int(modelNum)), str(int(modelNum))))
else:
    temp_mcmc_file = np.load('/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/MCMC_Data/M%s/Ploidy_MCMC_trajectories_v%shrData_Infer_Predict_M%s_0.3.npz'%(str(int(crop_hrs)), str(int(modelNum)), str(int(modelNum))))

bigChrom = temp_mcmc_file['nC_MCMC']

#global MCMC_Ents
MCMC_Ents = []
temp_percent = 1
MCMC_maxCellSlice = 0

for simNum in range(bigChrom.shape[0]):
    MCMC_numChrom = bigChrom[simNum]
    
    ### Crowbar data to only have MaxCal-possible transitions (i.e. never more than doubles):          
    for nTry in range(len(MCMC_numChrom)):
        for tSlice in range(MCMC_numChrom[nTry].shape[1] -1):
    #        for cellNum in range(len(N_Chrom[nTry][tSlice])):
            for cellNum in range(MCMC_numChrom[nTry].shape[0]):
                if MCMC_numChrom[nTry][cellNum, tSlice+1] > 2*MCMC_numChrom[nTry][cellNum, tSlice]:
                    MCMC_numChrom[nTry][cellNum, tSlice+1] = 2*MCMC_numChrom[nTry][cellNum, tSlice]
    del nTry, tSlice, cellNum
    
    ### Find maximum number of cells in a given time slice:
#    MCMC_maxCellSlice = 0
    for nTry in range(len(MCMC_numChrom)):
        for tSlice in range(MCMC_numChrom[nTry].shape[1]):
            if len(np.where(~np.isnan(MCMC_numChrom[nTry][:, tSlice]))[0]) > MCMC_maxCellSlice:
                MCMC_maxCellSlice = len(np.where(~np.isnan(MCMC_numChrom[nTry][:, tSlice]))[0])
    
    ### Count the number of cell number transitions:
    MCMC_CellCounts = np.zeros((int(MCMC_maxCellSlice + 1), int(MCMC_maxCellSlice + 1)))
    for nTry in range(len(MCMC_numChrom)):
        for tSlice in np.arange(0, MCMC_numChrom[nTry].shape[1] - 1, 1):
            MCMC_CellCounts[int(len(np.where(~np.isnan(MCMC_numChrom[nTry][:, tSlice]))[0])), int(len(np.where(~np.isnan(MCMC_numChrom[nTry][:, tSlice + 1]))[0]))] += 1

    ### Calculate Cell # ONLY Entropy for Monte Carlo Data:
    MCMC_incEnts = np.zeros((np.shape(MCMC_CellCounts)))
#    MCMC_incEnt = 0
    for before in range(MCMC_CellCounts.shape[0]):
        for after in range(MCMC_CellCounts.shape[1]):
            if MCMC_CellCounts[before, after] > 0:
                MCMC_incEnts[before, after] = int(MCMC_CellCounts[before, after])/np.sum(MCMC_CellCounts) * ( np.log( int(MCMC_CellCounts[before, after])/np.sum(MCMC_CellCounts)) )
#                MCMC_incEnt -= (  MCMC_CellCounts[before, after]/np.sum(MCMC_CellCounts) * ( np.log( MCMC_CellCounts[before, after]/np.sum(MCMC_CellCounts)) )  )
        MCMC_Ents.append(-np.nansum(MCMC_incEnts))
#    MCMC_Ents.append(MCMC_incEnt)
#    MCMC_Ents.append(-np.nansum(MCMC_incEnts))
    
#    del MCMC_numChrom, MCMC_CellCounts, MCMC_incEnt
    
    if int(100*(simNum+1)/bigChrom.shape[0]) >= temp_percent:
        print('MaxCal Monte Carlo (MCMC) entropy calculation is ' + str(int(100*(simNum+1)/bigChrom.shape[0])) + ' % complete with current avg_S_mc = ' + str(np.average(MCMC_Ents)) + ' ...')   
        if temp_percent == 1:
            temp_percent += 9 
        else:
            temp_percent += 10

###################################################################################################        
        
from PercentDifferenceCalculator import percent_difference

PD = percent_difference(EXP_ent, np.average(MCMC_Ents))

print("Cell Number ONLY Entropy:\n")
print('Using Model # ' + str(int(modelNum)) + ': S_mc is ' + str(np.round(PD, 3)) + ' % different from S_ex\n')
temp_PD = '/Users/stephenwedekind/Dropbox/School/RESEARCH/Ghosh_group/__PROJECTS/PloidyVariation/MCMC_Data/M%s/EXP_vs_MCMC_EntPercentDiff_M%s_v%s.npz'%(str(int(modelNum)), str(int(modelNum)),str(version))
np.savez(temp_PD, MCMC_vs_EXP_Ent_percentDiff = PD, EXP_ent = EXP_ent, MCMC_Ents = MCMC_Ents, modelNum = modelNum, crop_hrs = crop_hrs)

print("**************************************************************")
elapsed_time_secs = time.time() - start_time
print("Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs)))
print("--- %s seconds ---" % (time.time() - start_time))