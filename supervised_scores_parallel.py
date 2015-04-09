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
from numpy import ones,array,exp,log,mean,std,sum,mean,zeros,vstack
from math import gamma 
import time 
from sklearn import svm
from numpy import hstack,isnan,fliplr,argsort
from os import listdir
import pickle
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.io import loadmat,savemat
from random import shuffle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import normalize,scale
from multiprocessing import Process,Array
from feature_extraction import feat_extract
#from mpi4py import MPI



def get_scores(stat_directories,directories,indices,num_drivers,id):
    num_drives = 70
    num_feats = 500
    feat_importances = {}
    prob_dict = {}
    drives = list(range(200))
    
    for the_driver in stat_directories[indices[0]:indices[1]]: 
        shuffle(directories)
        itter_direct = []
        count = 0
        index = 0
        while count<num_drivers:
            if directories[index] != the_driver:
                itter_direct.append(directories[index])
                count += 1
            index+=1

        driver_data = {}
        itter_direct.append(the_driver)
        for driver in itter_direct:
            #print 'get driver data ',driver
            #all_stuff = compute_distribution_smooth(int(driver))
            all_stuff = feat_extract(int(driver))
            all_measures = all_stuff[3]
            measures_keys =  sorted(all_measures.keys())
            measures_keys = [x for x in measures_keys if not x in ['accel_no_norm','stop_accel','go_accel','stop_time','time_between_stops','trip_times','sign_angles','stop_time','turn_accel_hist']]
            measure_dict= {}
            for key in measures_keys:
                measure_dict[key] = array(all_measures[key])
               
            driver_data[driver] = hstack([measure_dict[key] for key in measures_keys])
        #compute feature importance  
        positive_class = driver_data[the_driver]
        other_drivers = []
        for other_driver in [name for name in driver_data if name != the_driver]: 
            shuffle(drives)
            other_drivers.append(driver_data[other_driver][drives[:num_drives],:])
            
        negative_class = vstack(other_drivers)
        training_data = vstack((positive_class,negative_class))
        training_data = scale(training_data)
        compressor = PCA(n_components=50)
        compressor.fit(training_data)
        training_data = compressor.transform(training_data)
        training_labels = vstack((ones((positive_class.shape[0],1)),zeros((negative_class.shape[0],1))))
        training_labels = training_labels.reshape(training_labels.shape[0])
        indices = list(range(training_data.shape[0]))    
        #shuffle(indices)
        #shuffle up training data 
        training_data = training_data[indices,:]
        training_labels = training_labels[indices]
        #find best features
        #clf1 = ExtraTreesClassifier(criterion='entropy',n_estimators=100)
        #clf1.fit(training_data,training_labels)
        #sorted_feats = list(reversed(argsort(clf1.feature_importances_)))
        #features_to_use = sorted_feats[:num_feats]
        features_to_use = list(range(training_data.shape[1]))
        #feat_importances[int(driver)] = sorted_feats
        #set up svm model    
        clf = svm.SVC(kernel='rbf',C=20000,probability=True,gamma=.0001)
        clf.fit(training_data[:,features_to_use],training_labels)    
        #print 'training score ',clf.score(training_data[:,features_to_use],training_labels)
        scores = clf.predict_proba(training_data[:positive_class.shape[0],features_to_use])[:,clf.classes_==1]
        prob_dict[int(driver)] = [scores,all_stuff[4]]
        
    pickle.dump(prob_dict,open('/gpfs/gpfs3/bdm13006/data/predictions/the_predictions_%d.pickle'%id,'wb'))
 
num_nodes = 1
num_processes = 12
rank = 0 
#create mpi comm object
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()

if __name__== '__main__':
    directories = list(sorted([int(x) for x in listdir('/gpfs/gpfs3/bdm13006/data/drivers')]))
    shuf_direct = list(sorted([int(x) for x in listdir('/gpfs/gpfs3/bdm13006/data/drivers')]))
    processes_per_node = num_processes/num_nodes
    drivers_per_process = len(directories)/num_processes
    prob_dict = {}
    indices_list = []
    process_list = []
    for x in range(num_processes):
        if x/processes_per_node==rank:
            if(x==num_processes-1):
                indices_list.append((x*drivers_per_process,len(directories)))
            else:
                indices_list.append((x*drivers_per_process,(x+1)*drivers_per_process ))           
    for index in range(len(indices_list)):
        process_list.append(Process(name=str(rank*processes_per_node+index),target=get_scores,args=(directories,shuf_direct,indices_list[index],10,rank*processes_per_node+index)))
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

