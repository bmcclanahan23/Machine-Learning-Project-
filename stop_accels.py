# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 15:11:53 2014

@author: Brian
"""
from numpy import average

def get_stop_accels(speeds):
    break_accels = []
    go_accels = []
    num_stops = 0
    num_gos = 0
    stop_positions = []
    go_positions = []
    time_stopped = []
    stopped = 0
    for index in range(len(speeds)):
        if index-1 >= 0:
            if speeds[index] == 0 and speeds[index-1]>0:
                da_range = list(range(index-6,index)) if index-6>=0 else list(range(0,index))
                start = da_range[0]                
                for index_2 in da_range:
                    if speeds[index_2] == 0:
                        start = index_2+1
                da_range = range(start,index)
                if len(da_range)>0:
                    da_weights = [1 for weight in reversed(range(1,len(da_range)+1))]
                    break_accels.append(average([speeds[x+1]-speeds[x] for x in da_range],weights=da_weights))
                    num_stops +=1
                    stop_positions.append(index)
        if index+1 < len(speeds):        
            if speeds[index] == 0 and speeds[index+1]>0:
                da_range = list(range(index,index+10)) if index+10 < len(speeds)-1 else list(range(index,len(speeds)-1))               
                end = da_range[len(da_range)-1]                
                for index_2 in da_range[1:]:
                    if speeds[index_2] == 0:
                        end = index_2-1
                        break
                da_range = range(index,end)
                if len(da_range)>0:
                    da_weights = [1 for weight in reversed(range(1,len(da_range)+1))]
                    go_accels.append(average([speeds[x+1]-speeds[x] for x in da_range],weights=da_weights))
                    num_gos += 1
                    go_positions.append(index)
        if speeds[index] == 0:
            stopped+=1
        elif stopped != 0:
            time_stopped.append(stopped)
            stopped = 0

    time_between_stops = [stop_positions[ind]-stop_positions[ind-1] for ind in range(1,len(stop_positions))]
    distance_between_stops = [sum(speeds[stop_positions[ind-1]:stop_positions[ind]]) for ind in range(1,len(stop_positions))]
    return [break_accels,go_accels,num_stops,num_gos,stop_positions,go_positions,time_between_stops,time_stopped,distance_between_stops]

   