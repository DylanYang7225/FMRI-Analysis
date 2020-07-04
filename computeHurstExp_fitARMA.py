#!/usr/bin/env python
# coding: utf-8

# # Loading Modules and Reading in FMRI Files

import csv
import time
import math
import numpy as np
import scipy.io
import pandas as pd
from matplotlib import pyplot as plt
from os import listdir
from os.path import dirname, join
from numpy import log, polyfit, sqrt, std, subtract
from hurst import compute_Hc
from statsmodels.tsa.arima_model import ARMA, ARIMA, ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa import stattools
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from datetime import datetime
import sys

# the whole 820 subjects are split into 20 stacks numbered 1-20 and then each stack is assgined separately
# the variable stack_num will receive an argument from the slurm .bash job array to determine which stack
# it process
stack_num = int(sys.argv[1])


def CalcHurstExp(ts):  
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]  
    poly = np.polyfit(np.log(lags), np.log(tau), 1)  
    hurst = poly[0]*2.0  
    return max(hurst,0)

def fisherTrans(r):
    return 0.5*(np.log((1+r)/(1-r)))

def fisherAvg(r_scores):
    avg_z = sum([fisherTrans(r_scores[i]) for i in range(len(r_scores))])/len(r_scores)
    avg_r = (math.exp(2*avg_z)-1)/(math.exp(2*avg_z)+1)
    return avg_r

# Function computeWeights compute the corresponding model weights with their AIC scores
def computeWeights(AIC):
    min_AIC = min(AIC)
    AIC = np.array(AIC)
    prob = []
    for i in range(len(AIC)):
        prob.append(math.exp((min_AIC-AIC[i])/2))
    prob = np.array(prob)
    return prob/sum(prob)

def computeEnsemble(models,weights):
    ensemble=np.array([0 for _ in range(len(models[0]))])
    for i in range(len(models)):
        ensemble = ensemble + np.array(models[i])*weights[i]
    return list(ensemble)


mat_files = listdir('/data')
print("Total number of files loaded: "+str(len(mat_files)))

# loop through all files to extract ROI time-series from each subject and store in the tensor matrix subjects[]
subject_ts=[]
# define file directory
input_files_dir = '/data'
output_files_dir = '/output'
for mat_file in mat_files:
    file_name = join(input_files_dir,mat_file)
    temp = scipy.io.loadmat(file_name)  
    subject_ts.append(temp['tc'])


# # Hurst Exponents

#create one csv for storage of Hurst Exponents
#df = pd.DataFrame(list())
#region_nums = [None]
#for i in range(1,1+len(subject_ts[0][0])):
#    region_nums.append('region '+str(i))
#hurst_csv='/output/HurstExp_stack'+str(stack_num)+'.csv'
#df.to_csv(hurst_csv)
#with open(hurst_csv, 'w',newline='') as file:
#    writer = csv.writer(file)
#    writer.writerow(region_nums)

# Compute Hurst Exponents
#for i in range(41): # only 41 .mat files in each stack
#    subject_H = ['sub'+mat_files[i][-10:-4]]
#    for region in range(len(subject_ts[0][0])):
#        H, c, data = compute_Hc(subject_ts[i][:,region])
#        subject_H.append(H)
#    with open(hurst_csv,'a',newline='') as file:
#        writer = csv.writer(file)
#        writer.writerow(subject_H)


# # Fitting ARMA Models
# The overall idea is to
# 
# (1) fit an array of ARMA$(p,d,q)$ models with $1\leq p\leq p_{max}$ and $0\leq q \leq q_{max}$;
# 
# (2) compute model weights for each model 
# 
# (3) make an ensemble model based on the fitted ARMA models and their correponding weights
# 
# (4) compute the correlation of the ensemble model to the original

df = pd.DataFrame(list())
region_nums = [None]
for i in range(1,len(subject_ts[0][0])+1):
    region_nums.append('region '+str(i))
Rscore_csv='/output/Rscore_'+str(stack_num)+'.csv'
df.to_csv(Rscore_csv)
with open(Rscore_csv, 'w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(region_nums)



d = 0 # Due to stationarity test results our time-series are stationary (possibly with fractional integration order) so we do not difference
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print('Computation started at '+ str(current_time))
p_max = 6
q_max = 6
for sub in range(41): # only 41 subjects in each stack
    # order_and_weight = models_weights[region] stores the weights of ARMA models in a subject
    order_and_weight = [None for _ in range(len(subject_ts[0][0]))]
    # Rscore = Rscore[region] stores the Pearson correlation between fitted ARMA models and original time-series
    Rscore = [None for _ in range(len(subject_ts[0][0]))]
    # ensemble_model = ensemble_model[region] stores the ensemble ARMA models for each region in a subject
    ensemble_ts = [None for _ in range(len(subject_ts[0][0]))]
    # Each of the model orders, paired with its model_weights as a tuple, Rscore and ensemble_model will be updated in each loop step and stored in files before they get updated   
    
    for reg in range(160):
        ts = subject_ts[sub][:,reg]
        #PACF=stattools.pacf(ts) # use partial auto-correlation to determine the differencing order
        #if PACF[1]>0.5:
        #    d=1 # if the lag-1 autocorrelation is strong, take an order 1-differencing
        #else:
        #    d=0
        predictions = [] # a 2D array that stores the predicted values of all fitted models
        AIC = [] # an 1D array that stores AIC scores
        orders = []
        for p in range(1,p_max):
            for q in range(0,q_max):
                try:
                    fitted_model = SARIMAX(ts,order=(p,d,q),enforce_invertibility=False).fit(disp=-1)
                    predictions.append(fitted_model.predict())
                    orders.append((p,d,q))
                    AIC.append(fitted_model.aic)
                    weights = computeWeights(AIC)
                    orders_and_weights = (orders, weights)    
                    print("finished fitting model for subject {} region {} with (p,d,q)=({},{},{})".format(sub, reg,p,d,q))
                except:
                    print("failed to fit model for subject {} region {} with (p,d,q)=({},{},{})".format(sub, reg,p,d,q))
                    continue
        
        ensemble = computeEnsemble(predictions,weights)
        Rscore[reg] = pearsonr(ensemble,ts)[0]
        
        order_and_weight[reg] = orders_and_weights
        ensemble_ts[reg] = ensemble
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('subject {} region {} finished at {}'.format(sub,reg,str(current_time)))

    # each subject will be named according to the index of his FMRI .mat file 
    row = ['sub'+mat_files[sub][-10:-4]]
    row.extend(Rscore)
    with open(Rscore_csv, 'a',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    
    # output the ensemble time-series and model weights for each region, each subject has his own output file
    models = np.array(ensemble_ts)
    ensemble_ts_mat = '/output/ensemble_ts_'+mat_files[sub][-10:]
    scipy.io.savemat(ensemble_ts_mat,mdict={'ensemble_ts':models})
    
    order_and_weight = np.array(order_and_weight) 
    ensemble_weights_mat = '/output/ensemble_orders_and_weights_'+mat_files[sub][-10:]
    scipy.io.savemat(ensemble_weights_mat,mdict={'ensemble_orders_and_weights':order_and_weight})




