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
import numba

def MakeDir(nameDir):
    try:
        os.makedirs(nameDir)
    except:
        pass
    return nameDir

def taper(data, taper_dur, fs ):
    hann_func = signal.hann(int(int(taper_dur*2)*fs)+1) 
    half_len = round(len(hann_func)/2-.001)
    data[:half_len] = data[:half_len]*hann_func[:half_len]
    data[-half_len:] = data[-half_len:]*hann_func[-half_len:]
    return data

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4, axis=-1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data, axis=-1)
    return y

def normalize(data,norm_method,freqmin,freqmax,fs,clip_weight=5,norm_win=10): 
    
    ### input data should be a 2D array: [channel#,sample#]
    
    data=signal.detrend(data,type='constant',axis=-1)
    data=signal.detrend(data,axis=-1)
    data=butter_bandpass_filter(data,freqmin,freqmax,fs,axis=-1)
    
    if norm_method=='clipping':
        lim=np.sqrt(np.mean(data**2,axis=-1))
        for i in range(len(data)):
            data[i,:][data[i,:]>lim[i]]=lim[i]
            data[i,:][data[i,:]<-lim[i]]=-lim[i]

    elif norm_method=='waterlevel':
        lim=np.sqrt(np.mean(data**2,axis=-1))
        for i in range(len(data)):
            data0=data[i,:]
            while len(data0[np.abs(data0)>lim[i]])>0:
                data0[data0>lim[i]]/=clip_weight
                data0[data0<-lim[i]]/=clip_weight
    
    elif norm_method=='running':
        st=0                                              
        N=norm_win*fs                                          

        while N<data.shape[-1]:
            win=data[:,st:N]
            w=np.mean(np.abs(win),axis=-1)
            data[:,st+int(norm_win*fs/2)]/=w
            st+=1
            N+=1
        
        data=data[:,int(norm_win*fs/2):(data.shape[-1]-int(norm_win*fs/2))]
        
#         data=signal.convolve2d(data,np.ones((1,N)),'valid')/N

    elif norm_method=='onebit':
        data=np.sign(data)

    elif norm_method=='nonorm':
        data=data
    
    else:
        raise Typeerror('Not a valid normalization, raw data is returned')
        
    return data,data.astype('float16')

def corr_fft(dat_s,dat_r,delta,savetime):

    fft_s=np.fft.fft(dat_s,len(dat_s)*4)
    fft_r=np.fft.fft(dat_r,len(dat_r)*4)
    dec_t2=np.real(np.fft.fftshift(np.fft.ifft((fft_r*np.conj(fft_s))/(smooth(np.abs(fft_s),30)*smooth(np.abs(fft_r),30)))))
    dec_t=dec_t2[int(len(dec_t2)/2)-savetime*delta:int(len(dec_t2)/2)+savetime*delta]
    return dec_t

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def whitening(df,factor,axis=-1):
    fft_df=np.fft.fft(df,df.shape[1]*4,axis=axis)
    for i in range(len(df)):
        fft_df[i,:]/=smooth(np.abs(fft_df[i,:]),factor)
    return fft_df

def whitening_rfft(df,factor,axis=-1):
    rfft_df=np.fft.rfft(df,df.shape[1]*4,axis=axis)
    for i in range(len(df)):
        rfft_df[i,:]/=smooth(np.abs(rfft_df[i,:]),factor)
    return rfft_df

def corr_vec(source,receiver_vec,delta,savetime):
    dec_f2=np.real(np.fft.fftshift(np.fft.ifft(receiver_vec*np.conj(source),axis=-1),axes=-1))
    dec_f=dec_f2[:,int(dec_f2.shape[1]/2)-savetime*delta:int(dec_f2.shape[1]/2)+savetime*delta]
    return dec_f

def corr_rfft(source,receiver_vec,delta,savetime):
    dec_f2=np.real(np.fft.fftshift(np.fft.irfft(receiver_vec*np.conj(source),axis=-1),axes=-1))
    dec_f=dec_f2[:,int(dec_f2.shape[1]/2)-savetime*delta:int(dec_f2.shape[1]/2)+savetime*delta]
    return dec_f


