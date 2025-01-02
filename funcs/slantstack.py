#!/usr/bin/env python
# coding: utf-8



import os
import glob
import obspy
import numpy as np
import h5py
from obspy import UTCDateTime as UTC
import matplotlib.pyplot as plt
from scipy import signal
import sys
from obspy.signal.util import smooth

def pws(flist,loadmin,loadmax,v,smoothing):
    
    nf = len(flist)
    npts=np.load(flist[0])[:,loadmin:loadmax].shape[-1]
    nchannels=np.load(flist[0])[:,loadmin:loadmax].shape[0]
    original=np.zeros((nchannels,npts))
    
    c = np.zeros((nchannels,npts), dtype=complex)
    
    for i, f in enumerate(flist):
        print(i)
        
        data=np.load(flist[i])[:,loadmin:loadmax]
        
        ### remove average
#         for i in range(len(data)):
#             data[i,:]/=np.max(np.abs(data[i,:]))
#         for i in range(len(data)):   
#             np.mean(data,axis=0)
            
        original+=data
        
        for ch in range(nchannels):
            h = signal.hilbert(data[ch,:])
            c[ch,:] += h/abs(h)
        
    c = abs(c/nf)
    original/=nf
    original[:,int((loadmax-loadmin)/2)]=np.zeros(nchannels)
    
    operator = np.ones(smoothing)/smoothing
    
    for ch in range(nchannels):
        c[ch,:] = np.convolve(c[ch,:], operator, 'same')
    
    pwsed = original*c**v
    pwsed[:,int((loadmax-loadmin)/2)]=np.zeros(nchannels)
    
    return original/np.max(np.abs(original)),pwsed/np.max(np.abs(pwsed))

def get_freqdomain_info(tdodata, dt = 0.1, lfhf = [0.33, 2]): 

    td_data=tdodata
    num_ts=td_data.shape[1]
    print(td_data.shape)
    td_data = td_data - td_data.mean()
    """ Remove data mean before Fourier transforming"""
    fs=np.fft.fftfreq(num_ts,dt) # frequency 
    fd_data=np.fft.rfft(td_data)
    print(fd_data.shape)
    fs_positive=fs[:fd_data.shape[1]]
    if num_ts%2==0:
        fs_positive[-1]=-1*fs_positive[-1]
    rel_indices=np.intersect1d(np.where(fs_positive>=lfhf[0])[0],np.where(fs_positive<=lfhf[1])[0])
    print("length of rel_indices is ", len(rel_indices) )
    freqpoints = fs_positive[rel_indices]
    afdm = np.matrix(fd_data[:,rel_indices])
    cumphase=np.unwrap(np.angle(afdm))
    return freqpoints, afdm, cumphase
    
def dosinglefreq(absdist,true_phase,ktrials,tempprint=False):
  
    rel_dist=absdist-absdist[0]
    kxmat=np.outer(ktrials,rel_dist)
    apsm=np.exp(+1j*kxmat)
    # apsm stands for applied_phase_shift_matrix
    tpmat=true_phase.reshape(len(true_phase),1)
    #tpmat=np.flipud(true_phase.reshape(len(true_phase),1))
    tpcm=np.exp(1j*tpmat)
    # tpcm stands for true_phase_column_matrix
    stacked_result=np.dot(apsm,tpcm)
    mod_stack=np.abs(stacked_result)
    return mod_stack,apsm,tpcm