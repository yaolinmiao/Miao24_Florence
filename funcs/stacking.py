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
    npts=loadmax-loadmin
    nchannels=np.load(flist[0])[:,loadmin:loadmax].shape[0]
    original=np.zeros((nchannels,npts),dtype='float16')
    
    c = np.zeros((nchannels,npts), dtype=complex)
    
    for i, f in enumerate(flist):
        print(i)
        
        data=np.load(flist[i])[:,loadmin:loadmax]
#         data[:,int((loadmax-loadmin)/2)]=0
        
        ### remove average
#         for i in range(len(data)):
#             data[i,:]/=np.max(np.abs(data[i,:]))
#         for i in range(len(data)):   
#             np.mean(data,axis=0)
            
        original+=data
        
        for ch in range(nchannels):
            h = signal.hilbert(data[ch,:])
            c[ch,:] += h/np.abs(h)
        
    c = np.abs(c/nf)
    
    original[:,int((loadmax-loadmin)/2)]=np.zeros(nchannels)
    
    operator = np.ones(smoothing)/smoothing
    for ch in range(nchannels):
        c[ch,:] = np.convolve(c[ch,:], operator, 'same')
    
    pwsed = original*c**v
    pwsed[:,int((loadmax-loadmin)/2)]=np.zeros(nchannels)

    return original,pwsed,c

def locate_max_amp(data,winlen=3*10,step=1):
    
    intervals=int(len(data)-winlen)//step
    amps=[]
    
    for i in range(intervals):
        start=i*step
        amps.append(np.sum(np.abs(data[start:start+winlen])))
    return np.max(amps),np.argmax(amps)

# def ncf_snr(data,winlen=3*10):
    
#     signal=locate_max_amp(data,winlen)[0]/winlen
#     noise=np.sum(np.abs(data))/(len(data)-winlen)   
    
#     return signal/noise

def ncf_snr(data,spacing,nch,sampling_rate=10,smin=0.4,smax=3):
    
    distance=spacing*nch
    signal=data[int(distance*smin*sampling_rate):int(distance*smax*sampling_rate)]
    signal_amp=np.sum(np.abs(signal))/len(signal)
    noise=np.concatenate((data[10:int(distance*smin*sampling_rate)],data[int(distance*smax*sampling_rate):]))
    noise_amp=np.sum(np.abs(noise))/len(noise)
    
    return 10*np.log10(signal_amp/noise_amp)

def ncf_rms(data,spacing,nch,sampling_rate=10,smin=0.4,smax=3):
    
    distance=spacing*nch
    signal=data[int(distance*smin*sampling_rate):int(distance*smax*sampling_rate)]
    signal_rms=(np.sum(signal**2)/len(signal))**0.5
    noise=np.concatenate((data[10:int(distance*smin*sampling_rate)],data[int(distance*smax*sampling_rate):]))
    noise_rms=(np.sum(noise**2)/len(noise))**0.5
    
    return signal_rms/noise_rms

def linear_stacking(flist):
    mat=np.load(flist[0])
    for i in range(1,len(flist)):
        print(i)
        mat+=np.load(flist[i])
        
    return mat

def cal_snr_stacking(flist,ch,loadmin,loadmax,v,smoothing,spacing,nchannels):
    
    nf=len(flist)
    npts=loadmax-loadmin
    operator = np.ones(smoothing)/smoothing
    
    linear_snr=np.zeros(nf)
    pws_snr=np.zeros(nf)
    original=np.zeros(npts,dtype='float16')
    c=np.zeros(npts, dtype=complex)
    
    for i, f in enumerate(flist):
        print(i)
        
        data=np.load(flist[i])[ch,loadmin:loadmax]
        original+=data
        original[:10]=0
        
        h=signal.hilbert(data)
        c+=h/abs(h)
        
        tempc=abs(c)
        tempc=np.convolve(c,operator,'same')
        temppwsed=original*c**v
        temppwsed[:10]=0
        
        linear_snr[i]=ncf_snr(original,spacing,nchannels)
        pws_snr[i]=ncf_snr(temppwsed,spacing,nchannels)
        
    c=abs(c)
    c=np.convolve(c,operator,'same')
    
    pwsed=original*c**v
    original[:10]=0
    temppwsed[:10]=0

    return original,pwsed,linear_snr,pws_snr

def cal_allch_rms(matrix,spacing,start,end):
    allch_raw_rms=np.zeros(end-start)
    
    for i in range(len(allch_raw_rms)):
        allch_raw_rms[i]=ncf_rms(matrix[start+i,:],spacing,i+start)
        
    return allch_raw_rms

def binary_judge(stacked,stacked_rms,G,single_matrix,spacing=0.1,start=25,end=300):
    reformed=stacked-single_matrix
    reformed_rms=cal_allch_rms(reformed,spacing,start,end)
    filebools=np.zeros(len(reformed_rms))
    for i in range(len(reformed_rms)):
        if reformed_rms[i]<G*stacked_rms[i]:
            filebools[i]=1
        else:
            filebools[i]=0
    
    return filebools

def selective_pws(flist,bool_matrix,loadmin,loadmax,v,smoothing,zero_filling=10):
    
    nf = len(flist)
    npts=loadmax-loadmin
    nchannels=np.load(flist[0])[:,loadmin:loadmax].shape[0]
    original=np.zeros((nchannels,npts),dtype='float16')
    c=np.zeros((nchannels,npts), dtype=complex)
    
    for i, f in enumerate(flist):
        print(i)
        chbool=bool_matrix[i,:].reshape(-1,1)
        data=np.load(flist[i])[:,loadmin:loadmax]
        data[:,:zero_filling]=0
        original+=data*chbool
        
        for ch in range(nchannels):
            h = signal.hilbert(data[ch,:])
            c[ch,:] += h/abs(h)
        
    c=abs(c)
    operator=np.ones(smoothing)/smoothing
    for ch in range(nchannels):
        c[ch,:] = np.convolve(c[ch,:],operator,'same')
    pwsed=original*c**v

    return original,pwsed

def rms_stacking_main(flist,nchannel,spacing=0.1,start=25,end=300,loadmin=0,loadmax=900,v=2,smoothing=5,zero_filling=10):
    
    original=linear_stacking(flist)
    print('first round linear done')
    raw_rms=cal_allch_rms(original,spacing,start,end)
    G=1+1/len(flist)
    
    bools=np.zeros((len(flist),nchannel))
    for i in range(len(flist)):
        ch=np.load(flist[i])
        filebools=binary_judge(original,raw_rms,G,ch,start=start,end=end)
        bools[i,start:end]=filebools
        
    print('Judging done')
        
    selective_linear,selective_pwsed=selective_pws(flist,bools,loadmin,loadmax,v,smoothing,zero_filling)
    
    print('PWS done')
    
    return selective_linear,selective_pwsed

