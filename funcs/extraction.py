#!/usr/bin/env python
# coding: utf-8

import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.patches as patches
from matplotlib.offsetbox import AnchoredText
from sklearn.cluster import DBSCAN
import sys
sys.path.append('/home/yaolinm/Projects/Florence/funcs/')
from dbf import *

def find_local_maxima(matrix,threshold,xrange,yrange):
    
    '''
    threshold: peak has to exceed certain value
    Return: xs and ys of local maximums  
    '''
    
    normalized_matrix=matrix/np.max(matrix)
    
    local_maxima = []
    peak = []
    significance = []
    (rows, cols) = matrix.shape

    for i in range(2,rows-2):
        for j in range(2,cols-2):
            current_value = matrix[i][j]
            
            if normalized_matrix[i][j] >= threshold:

                neighbors = matrix[max(0,i-2):min(rows,i+3),max(0,j-2):min(cols,j+3)]

                if np.all(current_value >= neighbors):
                    local_maxima.append([i, j])
                    peak.append(current_value)
                    significance.append(normalized_matrix[i][j])
    
    if len(local_maxima)==0:
        local_maxima.append(list(np.unravel_index(np.argmax(matrix),matrix.shape)))
        peak.append(np.max(matrix))
        significance.append(1)
        
    local_maxima=np.array(local_maxima)
    
    return peak,significance,xrange[local_maxima[:,0]],yrange[local_maxima[:,1]]

def combine_lists(index_list, value_list):
    
    index_values_dict = {}

    for index, value in zip(index_list, value_list):
        if index in index_values_dict:
            index_values_dict[index].append(value)
        else:
            index_values_dict[index] = [value]

    unique_indices = list(index_values_dict.keys())
    values_sublists = list(index_values_dict.values())

    return unique_indices, values_sublists


def build_matrix(pair,ref,half_beam_width=10,spacing=0.02,smin=0.4,smax=3,sampling_rate=10,
                 path='/scratch/zspica_root/zspica0/yaolinm/Florence/stacked_xcorr2/pws/'):
    
    '''
    
    This is a function to extract stacked xcorr functions to build a ready-to-DBF matrix
    pair: a 1*2 array with two beam centers
    ref: read-in H5 file to look for xcorr pairs, not path/file name
    smin and smax controls the length of xcorr funcs to be loaded
    
    Return: a 3D np array for DBF
    
    '''
    
    batch_interval=5 ### this is fixed due to xcorr calculation
    
    nchannel_apart=pair[1]-pair[0]
    
    ### starting and ending pts
    nend=int(smax*spacing*nchannel_apart*sampling_rate)
    
    beam_width=half_beam_width*2+1
    df=np.zeros((beam_width,beam_width,nend))
    
    sstart=pair[0]-batch_interval*half_beam_width
    sources=np.arange(sstart,sstart+batch_interval*beam_width,batch_interval)
    
    rstart=pair[1]-batch_interval*half_beam_width

    for i,source in enumerate(sources):
        
        source_string=str(sources[i]).zfill(4)

        starting_index=np.where(np.array(ref[source_string]) == rstart)[0][0]
        ending_index=starting_index+beam_width

        dd=np.load(path+source_string+'.npy')
        spt=min(640,1200-nend)
        df[i,:,:]=dd[starting_index:ending_index,spt:spt+nend]
        
    ### normalization
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            df[i,j,:]/=np.max(np.abs(df[i,j,:]))
            
    return df

def normalize(data):
    return data/np.max(np.abs(data))

from scipy import ndimage
def detect_edges(image):
    dx = ndimage.sobel(image, axis=0)
    dy = ndimage.sobel(image, axis=1)
    return np.hypot(dx, dy)

def extract_single(large,frange,
                   single_receiver_vels,single_receiver_freqs,
                   single_receiver_peaks,single_receiver_significances,source=True):

    if source:

        for i in range(len(large)):

            if i <=5:
                peaks,significance,slownesss_r,slownesss_s=find_local_maxima(large[i,:,:],1,np.arange(0.4,3,0.02),np.arange(0.4,3,0.02))
                significance=normalize(significance)
                single_receiver_vels[0].extend(list(1/slownesss_r))
                single_receiver_peaks[0].extend(list(peaks))
                single_receiver_significances[0].extend(list(significance))
                single_receiver_freqs[0].extend(list(frange[i]*np.ones(len(slownesss_r))))   
                
            else:
                peaks,significance,slownesss_r,slownesss_s=find_local_maxima(large[i,:,:],0.9,np.arange(0.4,3,0.02),np.arange(0.4,3,0.02))
                significance=normalize(significance)
                
                if len(significance)==1:
                    single_receiver_vels[0].extend(list(1/slownesss_r))
                    single_receiver_peaks[0].extend(list(peaks))
                    single_receiver_significances[0].extend(list(significance))
                    single_receiver_freqs[0].extend(list(frange[i]*np.ones(len(slownesss_r))))  

                elif len(significance)>1:
                    index=np.where(np.array(significance)==1)[0][0]

                    temp_receover_vels=[]
                    temp_receiver_freqs=[]
                    temp_receiver_peaks=[]
                    temp_receiver_significances=[]

                    for j in range(len(significance)):

                        if slownesss_r[j]>=slownesss_r[index] and slownesss_s[j]>=slownesss_s[index]:
                            temp_receover_vels.append(1/slownesss_r[j])
                            temp_receiver_peaks.append(peaks[j])
                            temp_receiver_significances.append(significance[j])
                            temp_receiver_freqs.append(frange[i])

                        elif slownesss_r[j]<=slownesss_r[index] and slownesss_s[j]<=slownesss_s[index]:
                            temp_receover_vels.append(1/slownesss_r[j])
                            temp_receiver_peaks.append(peaks[j])
                            temp_receiver_significances.append(significance[j])
                            temp_receiver_freqs.append(frange[i])

                    a,b,c,d=zip(*sorted(zip(temp_receover_vels,temp_receiver_peaks,
                                            temp_receiver_significances,temp_receiver_freqs)))

                    a=list(a)
                    b=list(b)
                    c=list(c)
                    d=list(d)

                    while len(a)>len(single_receiver_vels):

                        single_receiver_vels.append([])
                        single_receiver_freqs.append([])
                        single_receiver_peaks.append([])
                        single_receiver_significances.append([])

                    for k in range(len(a)):
                        single_receiver_vels[k].append(a[k])
                        single_receiver_peaks[k].append(b[k])
                        single_receiver_significances[k].append(c[k])
                        single_receiver_freqs[k].append(d[k])
                        
    else:
        
        for i in range(len(large)):

            if i <=5:
                peaks,significance,slownesss_r,slownesss_s=find_local_maxima(large[i,:,:],1,np.arange(0.4,3,0.02),np.arange(0.4,3,0.02))
                significance=normalize(significance)
                single_receiver_vels[0].extend(list(1/slownesss_s))
                single_receiver_peaks[0].extend(list(peaks))
                single_receiver_significances[0].extend(list(significance))
                single_receiver_freqs[0].extend(list(frange[i]*np.ones(len(slownesss_s))))   
            else:
                peaks,significance,slownesss_r,slownesss_s=find_local_maxima(large[i,:,:],0.9,np.arange(0.4,3,0.02),np.arange(0.4,3,0.02))
                significance=normalize(significance)
                
                if len(significance)==1:
                    single_receiver_vels[0].extend(list(1/slownesss_s))
                    single_receiver_peaks[0].extend(list(peaks))
                    single_receiver_significances[0].extend(list(significance))
                    single_receiver_freqs[0].extend(list(frange[i]*np.ones(len(slownesss_s))))  

                elif len(significance)>1:

                    index=np.where(np.array(significance)==1)[0][0]

                    temp_receover_vels=[]
                    temp_receiver_freqs=[]
                    temp_receiver_peaks=[]
                    temp_receiver_significances=[]

                    for j in range(len(significance)):

                        if slownesss_r[j]>=slownesss_r[index] and slownesss_s[j]>=slownesss_s[index]:
                            temp_receover_vels.append(1/slownesss_s[j])
                            temp_receiver_peaks.append(peaks[j])
                            temp_receiver_significances.append(significance[j])
                            temp_receiver_freqs.append(frange[i])

                        elif slownesss_r[j]<=slownesss_r[index] and slownesss_s[j]<=slownesss_s[index]:
                            temp_receover_vels.append(1/slownesss_s[j])
                            temp_receiver_peaks.append(peaks[j])
                            temp_receiver_significances.append(significance[j])
                            temp_receiver_freqs.append(frange[i])

                    a,b,c,d=zip(*sorted(zip(temp_receover_vels,temp_receiver_peaks,
                                            temp_receiver_significances,temp_receiver_freqs)))

                    a=list(a)
                    b=list(b)
                    c=list(c)
                    d=list(d)

                    while len(a)>len(single_receiver_vels):

                        single_receiver_vels.append([])
                        single_receiver_freqs.append([])
                        single_receiver_peaks.append([])
                        single_receiver_significances.append([])

                    for k in range(len(a)):
                        single_receiver_vels[k].append(a[k])
                        single_receiver_peaks[k].append(b[k])
                        single_receiver_significances[k].append(c[k])
                        single_receiver_freqs[k].append(d[k])
                        
    receiver_vels,receiver_peaks,receiver_significances,receiver_freqs=[],[],[],[]
    
    for i in range(len(single_receiver_freqs)):
        
        if len(set(single_receiver_freqs[i]))>=5:
            receiver_vels.append(single_receiver_vels[i])
            receiver_peaks.append(single_receiver_peaks[i])
            receiver_significances.append(single_receiver_significances[i])
            receiver_freqs.append(single_receiver_freqs[i])
        
    return receiver_vels,receiver_peaks,receiver_significances,receiver_freqs

def refine_dispersion(frequencies,velocities):

    unique_frequencies = np.unique(frequencies)
    mean_velocities = {}
    stds=[]

    for freq in unique_frequencies:

        freq_velocities = [vel for f, vel in zip(frequencies, velocities) if f == freq]
        freq_velocities = np.array(freq_velocities)

        mean = np.mean(freq_velocities)
        std = np.std(freq_velocities)
        filtered_velocities = freq_velocities[np.abs(freq_velocities - mean) <= 2*std]
        stds.append(std)

        mean_filtered = np.mean(filtered_velocities)
        mean_velocities[freq] = mean_filtered
   
    return list(mean_velocities.keys()),list(mean_velocities.values()),stds

def filter_modes(freqs,vels,frange,modes_x=None,modes_y=None):
    
    x = np.array(freqs)
    y = np.array(vels)
    
    if modes_x is None and modes_y is None:
    
        modes_x,modes_y=[[]],[[]]

    for i in range(8):

        freq=frange[i]
        single_max=np.max(y[np.where(x==freq)[0]])

        modes_x[0].append(freq)
        modes_y[0].append(single_max)

    for i in range(8,len(frange)):

        freq=frange[i]
        indices=np.where(x==freq)[0]

        if len(indices)>0:

            vels_sorted=np.sort(y[indices])
            used_mode=[]

            for j in range(len(vels_sorted)):

                vel=vels_sorted[j]
                temp_diff=[]
                temp_index_diff=[]
                temp_mode=[]

                for k in range(len(modes_x)):                

                    if k not in used_mode:

                        index_diff=list(frange).index(freq)-list(frange).index(modes_x[k][-1])

                        if index_diff<=5:

                            last_vel=modes_y[k][-1]
                            vel_diff=(vel-last_vel)/last_vel
                            if 0<=vel_diff<=0.1 or -0.25<=vel_diff<=0:
                                temp_diff.append(np.abs(vel_diff))
                                temp_index_diff.append(index_diff)
                                temp_mode.append(k)

                if len(temp_diff)>=1:

                    temp_ind=np.where(temp_index_diff==np.min(temp_index_diff))[0]
                    
                    if len(temp_ind)==1:
                        modes_x[temp_mode[temp_ind[0]]].append(freq)
                        modes_y[temp_mode[temp_ind[0]]].append(vel)
                        used_mode.append(temp_mode[temp_ind[0]])
                    else:
                        temp_temp_ind=np.argmin(np.array(temp_diff)[temp_ind])
                        modes_x[temp_mode[temp_ind[temp_temp_ind]]].append(freq)
                        modes_y[temp_mode[temp_ind[temp_temp_ind]]].append(vel)
                        used_mode.append(temp_mode[temp_ind[temp_temp_ind]])

                else:
                    modes_x.append([freq])
                    modes_y.append([vel])

    modes_freq=[sublist for sublist in modes_x if len(sublist) >= 10]
    modes_vel=[sublist for sublist in modes_y if len(sublist) >= 10]
    
    return modes_freq,modes_vel

