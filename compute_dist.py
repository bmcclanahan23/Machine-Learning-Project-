# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 12:48:53 2014

@author: Brian
"""
import csv
from matplotlib import pyplot as plt
from math import sqrt,acos
from numpy import histogram,zeros, where,linspace,mean,array
from os import listdir
from stop_accels import get_stop_accels
import os
#import pdb

def compute_distribution(driver):
    
    meters_per_mile = 1609.34
    degrees_per_radian = 57.295
    miles_per_hour =[x for x in range(0,110,5)]
    miles_per_sec = [x/3600.0 for x in miles_per_hour]
    meters_per_sec =  [x*meters_per_mile for x in miles_per_sec]
    accel_miles_per_hour = list(reversed([-x*.5 for x in range(1,21)])) + [x*.5 for x in range(21)]
    accel_miles_per_sec = [x/3600.0 for x in accel_miles_per_hour]  
    accel_meters_per_sec = [x*meters_per_mile for x in accel_miles_per_sec]
    angle_bins = [x*5 for x in range(19)]
    sign_angle_bins = [x*5 for x in range(-18,19)]
    magnitudes_hist = zeros(len(meters_per_sec)-1)
    accels_hist = zeros(len(accel_meters_per_sec)-1)
    angles_hist = zeros(len(angle_bins)-1)
    if os.name == 'posix':
        the_path = '/media/brian/Windows/ForFun/drivers/drivers/'
    else:
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
    all_speeds_unnorm = []
    all_accels_unnorm = []
    for directory in directory_nums:
        with open('%s%d/%d.csv'%(the_path,driver,directory)) as csvfile:
            data = list(csv.reader(csvfile))
        the_indices.append(directory)
        data = data[1:]
        x = [float(entry[0]) for entry in data]
        y = [float(entry[1]) for entry in data]
        x_diff = [x[index]-x[index-1] for index in range(1,len(x))]
        y_diff = [y[index]-y[index-1] for index in range(1,len(y))]
        #calculate speeds    
        magnitudes = [sqrt(x_diff[index]**2+y_diff[index]**2)  for index in range(len(x_diff))]
        #calculate accelerations
        accels = [magnitudes[index]-magnitudes[index-1] for index in range(1,len(magnitudes))]    
        #calculate angles between vectors 
        #angles = [ acos(abs((x_diff[index]*x_diff[index-1]+y_diff[index]*y_diff[index-1])/float(magnitudes[index]*magnitudes[index-1])))*degrees_per_radian if magnitudes[index]>0 and magnitudes[index-1]>0 and abs((x_diff[index]*x_diff[index-1]+y_diff[index]*y_diff[index-1])/float(magnitudes[index]*magnitudes[index-1])) <=1.0 else 0.0 for index in range(1,len(magnitudes))]     
        angles = [ acos(abs((x_diff[index]*x_diff[index-1]+y_diff[index]*y_diff[index-1])/float(magnitudes[index]*magnitudes[index-1])))*degrees_per_radian if magnitudes[index]>0 and magnitudes[index-1]>0 and abs((x_diff[index]*x_diff[index-1]+y_diff[index]*y_diff[index-1])/float(magnitudes[index]*magnitudes[index-1])) <=1.0 else 0.0 for index in range(1,len(magnitudes))]             
        signs = [x_diff[index-1]*y_diff[index]+x_diff[index]*y_diff[index-1]/abs(x_diff[index-1]*y_diff[index]+x_diff[index]*y_diff[index-1]) if (x_diff[index-1]*y_diff[index]+x_diff[index]*y_diff[index-1]) != 0 else 1 for index in range(1,len(magnitudes))]    
        sign_angles = [signs[index-1]*acos(abs((x_diff[index]*x_diff[index-1]+y_diff[index]*y_diff[index-1])/float(magnitudes[index]*magnitudes[index-1])))*degrees_per_radian if magnitudes[index]>0 and magnitudes[index-1]>0 and abs((x_diff[index]*x_diff[index-1]+y_diff[index]*y_diff[index-1])/float(magnitudes[index]*magnitudes[index-1])) <=1.0 else 0.0 for index in range(1,len(magnitudes))]
        #calculate turn speeds
        turn_speeds = [(magnitudes[index-1]+magnitudes[index])/2.0*angles[index-1] for index in range(1,len(magnitudes))]     
        #calculate stop and go accelerations
        stop_and_go_accels = get_stop_accels(magnitudes)    
        #get trip times
        trip_times.append(len(data)/1500.0)
        #get mean of all angles
        all_angles_mean.append(mean(angles))
        #get mean of all speeds 
        all_speeds_mean.append(mean(magnitudes))
        
        mh = histogram(magnitudes,bins=meters_per_sec)[0]+1
        ach = histogram(accels,bins=accel_meters_per_sec )[0]+1
        anh = histogram(angles,bins=angle_bins)[0]+1
        ts = histogram(turn_speeds,bins=linspace(0,800,60))[0]+1
        sa = histogram(stop_and_go_accels[0],bins=linspace(-5,0,30))[0]+1
        ga = histogram(stop_and_go_accels[1],bins=linspace(0,5,30))[0]+1
        t_stop = histogram(stop_and_go_accels[7],bins=linspace(0,120,60))[0]+1
        t_b_stop = histogram(stop_and_go_accels[6],bins=linspace(0,500,60))[0]+1
        sanh = histogram(sign_angles,bins=sign_angle_bins)[0]+1
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
    return [magnitudes_hist,accels_hist,angles_hist,{'speed_no_norm':all_speeds_unnorm,'accel_no_norm':all_accels_unnorm,'speed':all_speeds,'accel':all_accels,'angle':all_angles,'sign_angles':all_sign_angles,'turn_speed':all_turn_speeds,'stop_accel':all_stop_accels,'go_accel':all_go_accels,'stop_time':all_stop_times,'time_between_stops':all_time_bet_stops,'avg_angle':array(all_angles_mean).reshape((len(all_angles_mean),1)),'avg_speed':array(all_speeds_mean).reshape((len(all_speeds_mean),1)),'trip_times':array(trip_times).reshape((len(trip_times),1))},the_indices]