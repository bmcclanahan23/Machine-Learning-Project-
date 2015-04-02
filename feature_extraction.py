# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 12:48:53 2014

@author: Brian
"""
import csv
from matplotlib import pyplot as plt
from math import sqrt,acos
from numpy import histogram,zeros, where,linspace,mean,array,isnan,hstack,nonzero,argmax,histogram2d
from os import listdir
from stop_accels import get_stop_accels
from smoother import smooth 
import os
from scipy.stats import skew
from numpy import std 

#This script extracts features from a driver's trips and organizes them in arrays and vectors. For each feature there is either an array or a vector. If there is an array for the feature, each row of the array 
#corresponds to a single trip. If there is a vector for the feature, each element of the vector corresponds to a single trip. The features are stored in a dictionary and a particular feature can be accessed with
#its corresponding key in the dictionary. 

#To use this script extract the dataset from Kaggle into whatever directory and then change the line in the function below to reference the directory that you extracted the data in. 

#feature descriptions 
    #Histogram features (vector features)- speed, accel, angle, turn_speed, turn_accel, speed_accel, time_between_stops, stop_times, speed_accel_hist, speed_turn_hist
    #                   Each of the histogram features is a histogram recording the number of ocurrences of a  measure(e.g. speed, acceleration, turn angle) within bins that correspond to a certain range of that measure. 
    #                   For example the number of speeds which were between 0-5 mph will be the value of the 1st speed bin and the number of speeds recoreded which were between 5-10 mph (excluding 10) will be in the value of the second bin.
    #                   Each bin is divided by the total number of recording of the measure (the length of the trip in seconds)
    #                   Histogram features are structured as arrays with dimensions n by d where n is the number of driver trips (200) and d is the length of the histogram 
    #speed - speed of car
    #accel - acceleration of car
    #angle - turn angle of the car between one second intervals (takes values between 0-90 direction is not considered)
    #turn_speed - the turn angle of the car multiplied by the driving speed 
    #turn_accel - the turn angle of the car multiplied by the acceleration of the car 
    #speed_accel - the speed of the car multiplied by the acceleration 
    #time_between_stops - the time between a cars stops 
    #stop_times - the amount of time the car is at a standstill after coming to a stop 
    #speed_accel_hist - The speed and acceleration a car has during a measurement. This is a 2d histogram which is flattened to produce one vector of features
    #speed_turn_hist - The speed and turn angle a car has during a measurement. This is a 2d histogram which is flattened to produce one vector of features 
    #
    #Scalar features - avg_angle, avg_speed, avg_speed_ns, avg_accel, avg_deaccel, stand_still_proportion, accel_proportion, deaccel_proportion, constant_speed_proportion, all_turn_speeds_mean, avg_time_between_stops,avg_time_stopped, distance_traveled, avg_distance_between_stops, number_stops, no_turn_accel_prop, no_angle_proportion, no_turn_speed_proportion, no_speed_accel_proportion, speed_skew, speed_std, accel_skew, accel_std
    #                 Each of the scalar features is a single number calculated using the entire trip 
    #                 Scalar features are structured as a vector of length n where n is the number of trips (200)
    #avg_angle - average turn angle
    #avg_speed - average speed 
    #avg_speed_ns - average speed with stops not considered 
    #avg_accel - average acceleration 
    #avg_deaccel - average deacceleration
    #stand_still_proportion - The proportion of a trip that the driver spends at a stand still 
    #accel_proportion - The proportion of a trip that the driver is accelerating "significantly" (I choose an arbitrary threshold for acceleration to define significant accelerations)
    #deaccel_proportion - The proportion of a trip that the driver is deaccelerating "significantly" (same deal as above)
    #constant_speed_proportion - The proportion of a trip that the driver is driving at a constant speed (portions of the trip when the acceleration and deacceleration are below and above a certain threshold respectively)
    #all_turn_speeds_mean - The average of all turn speeds recorded 
    #avg_time_between_stops - The average amount of time between stops 
    #avg_stopped - The average amount of time a car spends stopped after it comes to a stop 
    #distance_traveled - The distance the car traveled during the trip 
    #avg_distance_between_stops -  The average distance between stops 
    #number_stops - The number of times the driver stopped during a trip 
    #no_turn_accel_prop - The proportion of the trip for which the drivers turn angle multiplied by it's acceleration is below a certain threshold 
    #no_speed_accel_proportion - The proportion of the trip for which the drivers speed multiplied by it's acceleration is below a certain threshold 
    #no_turn_speed_proportion - The proportion of the trip for which the drivers speed mulitplied by it's turning angle is below a certain threshold 
    #no_angle_proportion - The proportion of the trip for which the drivers turning angle is below a certain threshold 
    #speed_skew - The skew (asymmetry) of the speed histograph
    #speed_std - The standard deviation of the speed 
    #accel_skew - The skew (asymmetry) of the accel histogram 
    #accel_std - The standard deviation of the accel histogram  

def mph_to_mps(mph):
    meters_per_mile = 1609.34
    return mph*meters_per_mile/3600.0
    
def remove_zero_bin(histo,bins):
    index = 0    
    for x in range(len(bins)-1):
        if 0 >= bins[x] and 0< bins[x+1]:
            index = x
            break
    return [array(list(histo[:index]) + list(histo[index+1:])),histo[index]/float(sum(histo))]
    
def feat_extract(driver):
    
    meters_per_mile = 1609.34
    degrees_per_radian = 57.295
    miles_per_hour =[1] +[x for x in range(0,81,5)][1:]
    miles_per_sec = [x/3600.0 for x in miles_per_hour]
    meters_per_sec =  [x*meters_per_mile for x in miles_per_sec]
    accel_miles_per_hour = list(reversed([-x*.5 for x in range(1,18)]))+[-.2] + [.2]+[x*.5 for x in range(1,18)]
    accel_miles_per_sec = [x/3600.0 for x in accel_miles_per_hour]  
    accel_meters_per_sec = [x*meters_per_mile for x in accel_miles_per_sec]
    accel_miles_per_hour2 = list(reversed([-x*1 for x in range(1,5)]))+[-.2] + [.2]+[x*1 for x in range(1,5)]
    accel_miles_per_sec2 = [x/3600.0 for x in accel_miles_per_hour2]  
    accel_meters_per_sec2 = [x*meters_per_mile for x in accel_miles_per_sec2]
    angle_bins = linspace(0,90,40)
    sign_angle_bins = [x*5 for x in range(-18,19)]
    magnitudes_hist = zeros(len(meters_per_sec)-2)
    accels_hist = zeros(len(accel_meters_per_sec)-2)
    angles_hist = zeros(len(angle_bins)-2)
    #Add your directory here #######################################################################################################
    the_path = 'C:/ForFun/drivers/drivers/'
    directories = listdir('%s%d'%(the_path,driver))
    directory_nums = list(sorted([int(directory.split('.')[0]) for directory in directories]))
    all_speeds = []
    all_accels = []
    all_angles = []
    all_sign_angles = []
    all_turn_speeds = []
    all_stop_accels = []
    all_go_accels = []
    all_stop_times = []
    all_time_bet_stops = []
    trip_times = []
    the_indices = []
    all_angles_mean = []
    all_speeds_mean = []
    all_speeds_mean_ns = []
    all_speeds_unnorm = []
    all_accels_unnorm = []
    all_accels_mean = []
    all_deaccels_mean = []
    all_turn_speeds_mean = []
    all_sign_angles_mean = []
    stand_still_proportion = []
    avg_time_between_stops = []
    avg_time_stopped = []
    accel_proportion = []
    deaccel_proportion = []
    constant_speed_proportion = []
    distance_traveled = []
    avg_distance_between_stops = []
    number_stops = []
    turn_accels = []
    no_accel_proportion = []
    no_speed_accel_proportion = []
    no_turn_speed_proportion = []
    no_angle_proportion = []
    no_turn_accel_proportion = []
    all_turn_accels = []
    all_speed_accels = []
    all_speeds_std = []
    all_speeds_skew = []
    all_accels_skew = []
    all_accels_std = []
    all_speed_turn_histo = []
    all_speed_accel_histo = []
    for directory in directory_nums:
        with open('%s%d/%d.csv'%(the_path,driver,directory)) as csvfile:
            data = list(csv.reader(csvfile))
        the_indices.append(directory)
        data = data[1:]
        x = [float(entry[0]) for entry in data]
        y = [float(entry[1]) for entry in data]
        x_diff = array([x[index]-x[index-1] for index in range(1,len(x))])
        y_diff = array([y[index]-y[index-1] for index in range(1,len(y))])
       
        #calculate speeds    
        magnitudes = array([sqrt(x_diff[index]**2+y_diff[index]**2)  for index in range(len(x_diff))])
        n_magnitudes = smooth(magnitudes)
        n_magnitudes[magnitudes==0] = 0 
        magnitudes = n_magnitudes 
        x_diff = smooth(array(x_diff))
        y_diff = smooth(array(y_diff))
        #calculate accelerations
        accels = [magnitudes[index]-magnitudes[index-1] for index in range(1,len(magnitudes))]        
        #calculate angles between vectors 
        #angles = [ acos(abs((x_diff[index]*x_diff[index-1]+y_diff[index]*y_diff[index-1])/float(magnitudes[index]*magnitudes[index-1])))*degrees_per_radian if magnitudes[index]>0 and magnitudes[index-1]>0 and abs((x_diff[index]*x_diff[index-1]+y_diff[index]*y_diff[index-1])/float(magnitudes[index]*magnitudes[index-1])) <=1.0 else 0.0 for index in range(1,len(magnitudes))]     
        angles = [acos((x_diff[index]*x_diff[index-1]+y_diff[index]*y_diff[index-1])/float(magnitudes[index]*magnitudes[index-1]))*degrees_per_radian if magnitudes[index]>0 and magnitudes[index-1]>0 and abs((x_diff[index]*x_diff[index-1]+y_diff[index]*y_diff[index-1])/float(magnitudes[index]*magnitudes[index-1])) <=1.0 else 0.0 for index in range(1,len(magnitudes))]
        
        signs = [(x_diff[index-1]*y_diff[index]+x_diff[index]*y_diff[index-1])/abs(x_diff[index-1]*y_diff[index]+x_diff[index]*y_diff[index-1]) if (x_diff[index-1]*y_diff[index]+x_diff[index]*y_diff[index-1]) != 0 else 1 for index in range(1,len(magnitudes))]    
        sign_angles = [signs[index-1]*acos((x_diff[index]*x_diff[index-1]+y_diff[index]*y_diff[index-1])/float(magnitudes[index]*magnitudes[index-1]))*degrees_per_radian if magnitudes[index]>0 and magnitudes[index-1]>0 and abs((x_diff[index]*x_diff[index-1]+y_diff[index]*y_diff[index-1])/float(magnitudes[index]*magnitudes[index-1])) <=1.0 else 0.0 for index in range(1,len(magnitudes))]
        #calculate turn speeds
        turn_speeds = [(magnitudes[index-1]+magnitudes[index])/2.0*angles[index-1] for index in range(1,len(magnitudes))]     
        #calculate turn accels
        turn_accels = [accels[index]*angles[index] for index in range(len(accels))]
        #calculate speed accel
        speed_accels = [(magnitudes[index-1]+magnitudes[index])/2.0*accels[index-1] for index in range(1,len(magnitudes))]     
        #calculate speed accels        
        
        #filter speeds above 5mph
        f_magnitudes = [mag for mag in magnitudes if mag > mph_to_mps(5)]
        if len(f_magnitudes)==0:
            f_magnitudes = [0]
        #calculate stop and go accelerations
        stop_and_go_accels = get_stop_accels(magnitudes)    
        #get trip times
        trip_times.append(len(data)/1500.0)
        #get mean of all angles
        all_angles_mean.append(mean([ang for ang in angles if ang >1]))
        if isnan(all_angles_mean[-1]):
            all_angles_mean[-1] = 0
        #get mean of all speeds 
        all_speeds_mean.append(mean(f_magnitudes))
        #get mean of all speeds excluding stops
        speed_thresh = mph_to_mps(5)
        all_speeds_mean_ns.append(mean([entry for entry in magnitudes if entry >=speed_thresh] ))
        if isnan(all_speeds_mean_ns[-1]):
            all_speeds_mean_ns[-1] = 0
        #get mean of all acceleration
        accel_thresh = mph_to_mps(.5)
        all_accels_mean.append(mean([entry for entry in accels if entry >accel_thresh]))
        if isnan(all_accels_mean[-1]):
            all_accels_mean[-1] = 0
        #get mean of all deacceleration
        all_deaccels_mean.append(mean([entry for entry in accels if entry <-accel_thresh]))
        if isnan(all_deaccels_mean[-1]):
            all_deaccels_mean[-1] = 0        
        #proportion of stand still time 
        speed_thresh =  mph_to_mps(1)
        stand_still_proportion.append(sum([1 for entry in magnitudes if entry <speed_thresh])/float(len(magnitudes)))
        #proportion of time accelerating
        accel_proportion.append(len([entry for entry in accels if entry >accel_thresh])/float(len(magnitudes)))  
        #proportion of time deaccelerating
        deaccel_proportion.append(len([entry for entry in accels if entry <-accel_thresh])/float(len(magnitudes))) 
        #proportion of time at constant speed 
        accel_thresh = mph_to_mps(.4)
        constant_speed_proportion.append(len([entry for entry in accels if entry >-accel_thresh and entry <accel_thresh])/float(len(magnitudes)))
        #get mean of turn speeds
        all_turn_speeds_mean.append(mean(turn_speeds))  
        #get mean of signed angles
        all_sign_angles_mean.append(mean(sign_angles))
        #get mean time between stopped
        avg_time_between_stops.append(mean([entry for entry in stop_and_go_accels[6] if entry >5]))
        if isnan(avg_time_between_stops[-1]):
            avg_time_between_stops[-1] = 0
        #get mean time between stopped
        avg_time_stopped.append(mean([entry for entry in stop_and_go_accels[7] if entry >0]))
        if isnan(avg_time_stopped[-1]):
            avg_time_stopped[-1] = 0
        #get mean distance between stopped
        avg_distance_between_stops.append(mean([entry for entry in stop_and_go_accels[8] if entry >10]))
        if isnan(avg_distance_between_stops[-1]):
            avg_distance_between_stops[-1] = 0
        #get total distance traveled 
        distance_traveled.append(sum(magnitudes))
        #get number of stops 
        number_stops.append(stop_and_go_accels[2])
        #calculate speed skew
        all_speeds_skew.append(skew(f_magnitudes))
        #calculate speed standard deviation
        all_speeds_std.append(std(f_magnitudes))
        all_accels_skew.append(skew(accels))
        all_accels_std.append(std(accels))
        
        ###################################################################################################
        #magnitudes hist
        mh = histogram(magnitudes,bins=meters_per_sec)[0]+1
        mh,blank = remove_zero_bin(mh,meters_per_sec)
        
        #accels hist         
        ach = histogram(accels,bins=accel_meters_per_sec )[0]+1
        ach,nacp = remove_zero_bin(ach,accel_meters_per_sec)
        no_accel_proportion.append(nacp)
        
        #angles hist
        anh = histogram(angles,bins=angle_bins)[0]+1
        anh,nap = remove_zero_bin(anh,angle_bins)
        no_angle_proportion.append(nap)
        
        #turn speed hist
        ts = histogram(turn_speeds,bins=linspace(0,800,60))[0]+1
        ts,ntsp = remove_zero_bin(ts,linspace(0,800,60))
        no_turn_speed_proportion.append(ntsp)
        
        #turn accels hist
        ta = histogram(turn_accels,bins=linspace(-60,60,40))[0]+1
        ta,ntap = remove_zero_bin(ta,linspace(-60,60,40))
        no_turn_accel_proportion.append(ntap)
        
        #speed accels hist
        spa = histogram(speed_accels,bins=linspace(-60,60,40))[0]+1
        spa,ntap = remove_zero_bin(spa,linspace(-60,60,40))
        no_speed_accel_proportion.append(ntap)
        
        #turn speed 2dhist
        mean_mags = [(magnitudes[index-1]+magnitudes[index])/2.0 for index in range(1,len(magnitudes))]     
        ts2d = histogram2d(mean_mags,angles,bins=[ meters_per_sec[:-6],angle_bins[:-10]])
        ts2d = ts2d[0].reshape(ts2d[0].shape[0]*ts2d[0].shape[1])+1
        
        #acell speed 2dhisto
        sa2d = histogram2d(mean_mags,accels,bins=[meters_per_sec[:-6],accel_meters_per_sec2])
        sa2d = sa2d[0].reshape(sa2d[0].shape[0]*sa2d[0].shape[1])+1
        #useless stuff 
        sa = histogram(stop_and_go_accels[0],bins=linspace(-5,0,30))[0]+1
        ga = histogram(stop_and_go_accels[1],bins=linspace(0,5,30))[0]+1
        t_stop = histogram(stop_and_go_accels[7],bins=linspace(0,120,60))[0]+1
        t_b_stop = histogram(stop_and_go_accels[6],bins=linspace(0,500,60))[0]+1
        #sign angles hist        
        sanh = histogram(sign_angles,bins=sign_angle_bins)[0]+1
        sanh,rsap = remove_zero_bin(sanh,sign_angle_bins)
        
        #do a little add one smoothing        
        all_speeds.append((mh)/float(sum(mh)))
        all_accels.append((ach)/float(sum(ach)))
        all_angles.append((anh)/float(sum(anh)))   
        all_turn_speeds.append(ts/float(sum(ts)))
        all_stop_times.append(t_stop/float(sum(t_stop)))
        all_time_bet_stops.append(t_b_stop/float(sum(t_b_stop)))
        all_stop_accels.append(sa/float(sum(sa)))
        all_go_accels.append(ga/float(sum(ga)))  
        all_sign_angles.append(sanh/float(sum(sanh)))
        all_turn_accels.append(ta/float(sum(ta)))
        all_speed_accels.append(spa/float(sum(spa)))
        all_speed_turn_histo.append(ts2d/float(sum(ts2d)))
        all_speed_accel_histo.append(sa2d/float(sum(sa2d)))
        #add unormalized features 
        all_speeds_unnorm.append(mh)
        all_accels_unnorm.append(ach)
        magnitudes_hist += mh
        accels_hist += ach
        angles_hist += anh
        #pdb.set_trace()  
        
    magnitudes_hist =  magnitudes_hist/sum(magnitudes_hist)
    accels_hist = accels_hist/sum(accels_hist)
    angles_hist = angles_hist/sum(angles_hist)
   
    return {'speed_accel_hist':all_speed_accel_histo,'speed_turn_hist':all_speed_turn_histo,'turn_accels':all_turn_accels,'speed_accels':all_speed_accels,'speed':all_speeds,'accel':all_accels,'angle':all_angles,'turn_speed':all_turn_speeds,'stop_time':all_stop_times,'time_between_stops':all_time_bet_stops,'avg_angle':array(all_angles_mean).reshape((len(all_angles_mean),1)),'avg_speed':array(all_speeds_mean).reshape((len(all_speeds_mean),1)),'avg_speed_ns':array(all_speeds_mean_ns).reshape((len(all_speeds_mean_ns),1)),'avg_accel':array(all_accels_mean).reshape((len(all_accels_mean),1)),'avg_deaccel':array(all_deaccels_mean).reshape((len(all_deaccels_mean),1)),'stand_still_proportion':array(stand_still_proportion).reshape((len(stand_still_proportion),1)),'accel_proportion':array(accel_proportion).reshape((len(accel_proportion),1)),'deaccel_proportion':array(deaccel_proportion).reshape((len(deaccel_proportion),1)),'constant_speed_proportion':array(constant_speed_proportion).reshape((len(constant_speed_proportion),1)),'all_turn_speeds_mean':array(all_turn_speeds_mean).reshape((len(all_turn_speeds_mean),1)),'avg_time_between_stops':array(avg_time_between_stops).reshape((len(avg_time_between_stops),1)),'avg_time_stopped':array(avg_time_stopped).reshape((len(avg_time_stopped),1)),'distance_traveled':array(distance_traveled).reshape((len(distance_traveled),1)),'avg_distance_between_stops':array(avg_distance_between_stops).reshape((len(avg_distance_between_stops),1)),'number_stops':array(number_stops).reshape((len(number_stops),1)),'no_turn_accel_prop':array(no_turn_accel_proportion).reshape((len(no_turn_accel_proportion),1)),'no_angle_proportion':array(no_angle_proportion).reshape((len(no_angle_proportion),1)),'no_turn_speed_proportion':array(no_turn_speed_proportion).reshape((len(no_turn_speed_proportion),1)),'no_speed_accel_proportion':array(no_speed_accel_proportion).reshape((len(no_speed_accel_proportion),1)),'speed_skew':array(all_speeds_skew).reshape((len(all_speeds_skew),1)),'speed_std':array(all_speeds_std).reshape((len(all_speeds_std),1)),'accel_skew':array(all_accels_skew).reshape((len(all_accels_skew),1)),'accel_std':array(all_accels_std).reshape((len(all_accels_std),1))}
    
    
    