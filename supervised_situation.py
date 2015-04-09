# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 18:07:51 2014

@author: Brian
"""

from scipy import special
from scipy.stats.mstats import normaltest
from scipy.stats import gaussian_kde
from compute_dist import compute_distribution
from compute_smooth_dist import compute_distribution_smooth
from numpy import ones,array,exp,log,mean,std,sum,mean,zeros,vstack,isnan,linalg
from math import gamma 
import time 
from sklearn import svm
from numpy import hstack,isnan,fliplr,argsort,where,vstack 
from os import listdir
import pickle
from sklearn.decomposition import PCA,KernelPCA, TruncatedSVD
from scipy.io import loadmat
from sklearn.tree import DecisionTreeClassifier
from random import shuffle
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier, GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.preprocessing import normalize,scale
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
from sklearn.linear_model import SGDClassifier
import os
import numpy as np
from feature_extraction import feat_extract
from random import shuffle
from math import floor

#if you are using linux place your drivers directory here
if os.name == 'posix':
  the_path = '/media/brian/Windows/ForFun/drivers/drivers/'
#if you are using windows place your drivers directory here 
else:
  the_path = 'C:/ForFun/drivers/drivers/'
#get a list of driver directories, shuffle them, then keep the first 10
directories = listdir(the_path)
shuffle(directories)
directories = directories[:10]

#Random helper variables. 
drives = list(range(200))
#this variable specifies the number of drives to take from each drivers data set
#when constructing the 1 vs rest data set for training. All 200 drives from the driver 
#of interest are used as class 1 and the 80*9 drives are taken from the other 9 drivers
#in directories  
num_drives = 80
test_dif_scores = []
test_scores = []
reg_test_scores = []
no_sel_test_scores = []
unreg_test_scores = []
num_feats = 100
threshold = .01
for driver in directories:
    print 'get data for driver  ',driver
    all_stuff = feat_extract(int(driver))
    #all_stuff = compute_distribution_smooth(int(driver))[3]

    #make data matrix here
    #the data matrix for each driver is stored in a dictionary named 
    #driver data. To access the data matrix for a particular driver use the 
    #name of the driver's folder as a key in the driver_data dictionary     
    all_measures = all_stuff
    measures_keys =  sorted(all_measures.keys())
    measures_ranges = {}
    last_index = 0
       
    measures_keys = [x for x in measures_keys]
    measure_dict= {}
    
    for index in range(len(measures_keys)):
        measure_dict[measures_keys[index]] = array(all_measures[measures_keys[index]])
        measures_ranges[measures_keys[index]] = list(range(last_index,last_index+measure_dict[measures_keys[index]].shape[1]))
        last_index += measure_dict[measures_keys[index]].shape[1]
    driver_data[driver] = hstack([measure_dict[key] for key in measures_keys])
  
for driver in  driver_data:    
    print 'making a 1 vs the rest data matrix for driver: ',driver 
    positive_class = driver_data[driver]
    other_drivers = []
    for other_driver in driver_data:
        shuffle(drives)
        if other_driver != driver:
            other_drivers.append(driver_data[other_driver][drives[:num_drives],:])
    negative_class = vstack(other_drivers)  
    
    positive_class = positive_class[:,filtered_ranges]
    negative_class = negative_class[:,filtered_ranges]
    
    #labels for this data set 
    labels = vstack((ones((positive_class.shape[0],1)),zeros((negative_class.shape[0],1))))
    labels = training_labels.reshape(labels.shape[0])
    
    #data for this data set
    all_data = vstack((positive_class,negative_class))
    
    #split data up into training and testing subsets. Xs are data ys are labels 
    X_train, X_test, y_train,y_test = cross_validation.train_test_split(all_data, labels, test_size=0.2, random_state=0)  
    