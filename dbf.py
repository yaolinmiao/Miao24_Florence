#!/usr/bin/env python
# coding: utf-8



import os
import glob
import obspy
import numpy as np
import h5py
from obspy import UTCDateTime as UTC
import matplotlib.pyplot as plt
from scipy import signal,stats,interpolate
import sys
from obspy.signal.util import smooth

'''

In this module, we perform the full process of double beamforming with DAS using following functions:

The steps are: 1. convert the NCF matrix in to fourier domain and extract multiple matrices with different centroid frequency
               2. perform frequency pair-wise DBF either in time domain or frequency domain
               3. compute the stacked energy amplitudes with different source-receiver slowness pairs with 2-step grid search
           
There are following functions:
    multi_filter_1d: serial filter transformation of raw 1D NCF 
    multi_filter_3d: same as above, 3D NCF
    do_this_pair: time domain stream shift and stack
    do_this_pair_freq: frequency domain stream shift and stack
    (single_pair_freq): frequency domain shift and stack for a matrix of a single spectral estimate
    second_round_freq: frequency domain stream shift and stack for a second round of grid search, de-vectorized function
    main_dbf: master function of double beamforming, perform 2-step grid search
    
Initial inputs:
    NCF matrix, sampling rate, spacing, filter frequencies, window length, slowness vector
    
'''
    
def multi_filter_1d(mat,samp_rate,filter_freqs,bandwidth):

        class Gauss():
            def __init__(self, f0, a=1.0):
                self._omega0=2*np.pi*f0
                self._a=a
            def evaluate(self, freqs):
                omega=2*np.pi*freqs
                return np.exp(-((omega-self._omega0)
                                 /(self._a*self._omega0))**2)
        datalen=mat.shape[-1]
        freqs=np.fft.fftfreq(datalen,1/samp_rate)[:int(datalen/2)]
        
        coefs=np.fft.fft(mat,axis=-1)[:int(datalen/2)] ## higher dimension
        nfilt=len(filter_freqs)
        
        signal_tf=np.zeros((nfilt,datalen))

        for ifilt, f0 in enumerate(filter_freqs):
            taper=Gauss(f0,a=bandwidth)
            weights=taper.evaluate(freqs)

            nhalf=freqs.size
            
            analytic_spec=np.zeros(datalen,dtype=complex)
            analytic_spec[:nhalf]=coefs*weights
            
            if datalen%2==0:
                analytic_spec[1:nhalf-1]*=2
            else:
                analytic_spec[1:nhalf]*=2
            analytic=np.fft.ifft(analytic_spec)
            signal_tf[ifilt,:]=analytic.real

        return signal_tf
    
def multi_filter_3d(mat,samp_rate,filter_freqs,bandwidth):

        class Gauss():
            def __init__(self, f0, a=1.0):
                self._omega0=2*np.pi*f0
                self._a=a
            def evaluate(self, freqs):
                omega=2*np.pi*freqs
                return np.exp(-((omega-self._omega0)
                                 /(self._a*self._omega0))**2)
        datalen=mat.shape[-1]
        freqs=np.fft.fftfreq(datalen,1/samp_rate)[:int(datalen/2)]
        
        coefs=np.fft.fft(mat,axis=-1)[:,:,:int(datalen/2)] ## higher dimension
        nfilt=len(filter_freqs)
        
        signal_tf=np.zeros((nfilt,mat.shape[0],mat.shape[1],datalen))

        for ifilt, f0 in enumerate(filter_freqs):
            taper=Gauss(f0,a=bandwidth)
            weights=taper.evaluate(freqs)

            nhalf=freqs.size
            
            analytic_spec=np.zeros((mat.shape[0],mat.shape[1],datalen),dtype=complex)
            analytic_spec[:,:,:nhalf]=coefs*np.tile(weights.reshape(1,len(weights)),(mat.shape[0],mat.shape[1],1))
                    
            if datalen%2==0:
                analytic_spec[:,:,1:nhalf-1]*=2
            else:
                analytic_spec[:,:,1:nhalf]*=2
            analytic=np.fft.ifft(analytic_spec)
            signal_tf[ifilt,:,:,:]=analytic.real

        return signal_tf
    
def do_this_pair(ssn,rsn,mat,sampling_rate,spacing,starting,length):
    
    '''
    
    Calculate the maximum amplitude of stacked CCFs after hilbert transform
    (in the time domain)
    inputs: ssn, rsn are source slowness and receiver slowness in s/km
            mat: CCF containning matrix after gaussian filters:
                 4D array(central_freq,nsource,nreceiver,CCF)
            sampling_rate in HZ
            spacing: the distance between channels, in km
            starting: 0 index to trim the CCF
            length: len(CCF)
            
    '''
    
    ns=mat.shape[1] ### number of sources
    mids=int(0.5*ns) ### source beam midpoint
    nr=mat.shape[2] ### number of receivers
    midr=int(0.5*nr) ### receiver beam midpoint
    
    stacked=np.zeros((len(mat),length))
    
    for i in range(ns):
        for j in range(nr):
            shift=int(spacing*(mids-i)*ssn*sampling_rate)+int(spacing*(i-midr)*rsn*sampling_rate)
            pairslice=mat[:,i,j,starting:starting+length]
            stacked+=np.roll(pairslice,-1*shift,axis=-1)
            
    stacked=stacked/(ns*nr)
    envelop=np.abs(signal.hilbert(stacked,axis=-1))   
    return np.max(envelop,axis=-1),stacked,envelop

def do_this_pair_freq(ssn,rsn,fourier_df,freqs,spacing):
    
    ns=fourier_df.shape[1] ### number of sources
    mids=int(0.5*ns) ### source beam midpoint
    nr=fourier_df.shape[2] ### number of receivers
    midr=int(0.5*nr) ### receiver beam midpoint
    
    ### create a complex df containing fourier series of time-shifted traces
    stacked=np.zeros((len(fourier_df),len(freqs)),dtype=complex)
    
    for i in range(ns):
        for j in range(nr):
            shift_time=spacing*(mids-i)*ssn+spacing*(j-midr)*rsn
            stacked+=fourier_df[:,i,j,:]*np.exp(2j*freqs*np.pi*shift_time)

    stacked_time=np.fft.ifft(stacked,axis=-1).real
    envelop=np.abs(signal.hilbert(stacked_time,axis=-1))   
    return np.max(envelop,axis=-1)

def second_round_freq(ssn,rsn,fourier_df,freqs,spacing):
    
    ns=fourier_df.shape[1] ### number of sources
    mids=int(0.5*ns) ### source beam midpoint
    nr=fourier_df.shape[2] ### number of receivers
    midr=int(0.5*nr) ### receiver beam midpoint    
    
    stacked=np.zeros((len(fourier_df),len(freqs)),dtype=complex)
    
    for i in range(ns):
        for j in range(nr):
            shift_time=(spacing*(mids-i)*ssn+spacing*(j-midr)*rsn).reshape(len(fourier_df),1)
            stacked+=fourier_df[:,i,j,:]*np.exp(2j*freqs*np.pi*shift_time)

    stacked_time=np.fft.ifft(stacked,axis=-1).real
    envelop=np.abs(signal.hilbert(stacked_time,axis=-1))   
    return np.max(envelop,axis=-1)

def single_pair_freq(ssn,rsn,fourier_df,freqs,spacing):
    
    ns=fourier_df.shape[0] ### number of sources
    mids=int(0.5*ns) ### source beam midpoint
    nr=fourier_df.shape[1] ### number of receivers
    midr=int(0.5*nr) ### receiver beam midpoint
    
    ### create a complex df containing fourier series of time-shifted traces
    stacked=np.zeros(len(freqs),dtype=complex)
    
    for i in range(ns):
        for j in range(nr):
            shift_time=spacing*(mids-i)*ssn+spacing*(j-midr)*rsn
            stacked+=fourier_df[i,j,:]*np.exp(2j*freqs*np.pi*shift_time)
            
    stacked_time=np.fft.ifft(stacked).real
    envelop=np.abs(signal.hilbert(stacked_time))   
    return np.max(envelop)

def main_dbf(dbf,sampling_rate,spacing,multiplier=5,resolution=5,initial_slowness=np.arange(0.5,5,0.1)):
    
    '''
    inputs: dbf: fourier matrix after multi-filter, and after normalization
            sampling_rate: in Hz
            spacing: in km
            multiplier: ratio between first and second rounds of grid search
            resolution: resolution of second grid search around the maximum of the first grid search
            initial_slowness: an slowness array, should be somehow lower resolution
    
    '''
    
    fourier_df=np.fft.fft(dbf,axis=-1)
    freqs=np.fft.fftfreq(fourier_df.shape[-1],1/sampling_rate)

    ss=initial_slowness
    step=ss[1]-ss[0]
    grid=np.zeros((len(dbf),len(ss),len(ss)))
    for i in range(len(ss)):
        for j in range(len(ss)):
            grid[:,i,j]=do_this_pair_freq(ss[i],ss[j],fourier_df,freqs,spacing)

    mastergrid=np.zeros((len(dbf),len(ss)*multiplier,len(ss)*multiplier))
    for i in range(mastergrid.shape[1]):
        for j in range(mastergrid.shape[2]):        
            mastergrid[:,i,j]=grid[:,i//multiplier,j//multiplier]

    grid_flattened=grid.reshape(grid.shape[0],grid.shape[1]*grid.shape[2])
    dd=np.unravel_index(grid_flattened.argmax(axis=1), grid[:,:,:].shape)
    ssn_df=np.zeros((len(dbf),resolution*multiplier))
    rsn_df=np.zeros((len(dbf),resolution*multiplier))
    master_place=[]
    
    width=resolution//2
    for i in range(len(dbf)):
        if dd[1][i]-width<0:
            sstart=0
        elif dd[1][i]+width+1>len(ss)-1:
            sstart=len(ss)-1-resolution
        else:
            sstart=dd[1][i]-width

        if dd[2][i]-width<0:
            rstart=0
        elif dd[2][i]+width+1>len(ss)-1:
            rstart=len(ss)-1-resolution
        else:
            rstart=dd[2][i]-width

        ssn_df[i,:]=np.arange(ss[sstart],ss[sstart+resolution],step/multiplier)[:resolution*multiplier]
        rsn_df[i,:]=np.arange(ss[rstart],ss[rstart+resolution],step/multiplier)[:resolution*multiplier]
        master_place.append([sstart*multiplier,rstart*multiplier])

    newpatch=np.zeros((len(dbf),resolution*multiplier,resolution*multiplier))
    for i in range(newpatch.shape[1]):
        for j in range(newpatch.shape[2]):
            newpatch[:,i,j]=second_round_freq(ssn_df[:,i],rsn_df[:,j],fourier_df,freqs,spacing)

    for i in range(len(mastergrid)):
        mastergrid[i,master_place[i][0]:master_place[i][0]+resolution*multiplier,
                   master_place[i][1]:master_place[i][1]+resolution*multiplier]=newpatch[i,:,:]
        
    return mastergrid

def main_dbf_new(dbf,sampling_rate,spacing,multiplier=5,resolution=5,initial_slowness=np.arange(0.1,3,0.1)):
    
    '''
    inputs: dbf: fourier matrix after multi-filter, and after normalization
            sampling_rate: in Hz
            spacing: in km
            multiplier: ratio between first and second rounds of grid search
            resolution: resolution of second grid search around the maximum of the first grid search
            initial_slowness: an slowness array, should be somehow lower resolution
    
    '''
    
    fourier_df=np.fft.fft(dbf,axis=-1)
    freqs=np.fft.fftfreq(fourier_df.shape[-1],1/sampling_rate)

    ss=initial_slowness
    step=ss[1]-ss[0]
    grid=np.zeros((len(dbf),len(ss),len(ss)))
    for i in range(len(ss)):
        for j in range(len(ss)):
            grid[:,i,j]=do_this_pair_freq(ss[i],ss[j],fourier_df,freqs,spacing)
    
    mastergrid=np.zeros((len(dbf),len(ss)*multiplier,len(ss)*multiplier))
    for d in range(len(dbf)):
        intermodel=interpolate.interp2d(ss,ss,grid[d,:,:],kind='linear')
        newss=np.arange(ss[0],round(ss[0]+len(ss)*step,2),round(step/multiplier,2))
        mastergrid[d,:,:]=intermodel(newss,newss)
    
    grid_flattened=grid.reshape(grid.shape[0],grid.shape[1]*grid.shape[2])
    dd=np.unravel_index(grid_flattened.argmax(axis=1), grid[:,:,:].shape)
    ssn_df=np.zeros((len(dbf),resolution*multiplier))
    rsn_df=np.zeros((len(dbf),resolution*multiplier))
    master_place=[]
    
    width=resolution//2
    for i in range(len(dbf)):
        if dd[1][i]-width<0:
            sstart=0
        elif dd[1][i]+width+1>len(ss)-1:
            sstart=len(ss)-1-resolution
        else:
            sstart=dd[1][i]-width

        if dd[2][i]-width<0:
            rstart=0
        elif dd[2][i]+width+1>len(ss)-1:
            rstart=len(ss)-1-resolution
        else:
            rstart=dd[2][i]-width

        ssn_df[i,:]=np.arange(ss[sstart],ss[sstart+resolution],step/multiplier)[:resolution*multiplier]
        rsn_df[i,:]=np.arange(ss[rstart],ss[rstart+resolution],step/multiplier)[:resolution*multiplier]
        master_place.append([sstart*multiplier,rstart*multiplier])

    newpatch=np.zeros((len(dbf),resolution*multiplier,resolution*multiplier))
    for i in range(newpatch.shape[1]):
        for j in range(newpatch.shape[2]):
            newpatch[:,i,j]=second_round_freq(ssn_df[:,i],rsn_df[:,j],fourier_df,freqs,spacing)

    for i in range(len(mastergrid)):
        mastergrid[i,master_place[i][0]:master_place[i][0]+resolution*multiplier,
                   master_place[i][1]:master_place[i][1]+resolution*multiplier]=newpatch[i,:,:]
        
    return mastergrid,newpatch

def energy_ratio(matrix,small_panel):
    
    ratio=matrix.shape[0]/small_panel.shape[0]*matrix.shape[1]/small_panel.shape[1]
    
    return ratio*np.sum(small_panel)/np.sum(matrix)

def find_local_maxima(array):
    
    mx_ind=[]
    mx=[]
    
    if array[0]>array[1]:
        mx.append(array[0])
        mx_ind.append(0)
        
    for i in range(1,len(array)-1):
        if array[i]>array[i-1] and array[i]>array[i+1]:
            mx.append(array[i])
            mx_ind.append(i)
            
    if len(mx)==0:
        mx.append(array[-1])
        mx_ind.append(len(array)-1)
        
    return mx_ind,mx

def cal_max_sig(local_maxes):
    
    return np.max(local_maxes)/np.sum(local_maxes)


### define possible sr pairs

def find_pairs(source_center,half_beam_width,min_ch=25,max_ch=240,multiplier=5,ch_list=np.arange(300,2700,5)):
    
    min_center_ch=min_ch+half_beam_width*2
    max_center_ch=max_ch+half_beam_width*2
    
    possibles=np.arange(min_center_ch,max_center_ch)*multiplier+source_center
    receiver_centers=[]
    for p in possibles:
        if p in ch_list:
            receiver_centers.append(p)
    
    return receiver_centers

### build a rs matrix for a pair

def build_matrix(pair,half_beam_width=10,batch_interval=5,spacing=0.02,smin=0.4,smax=3,sampling_rate=10,
                 path='/scratch/zspica_root/zspica1/yaolinm/Florence/first_month/stacked/pws/'):
    
    nchannel_apart=pair[1]-pair[0]

    nstart=int(smin*spacing*nchannel_apart*sampling_rate)
    nend=int(smax*spacing*nchannel_apart*sampling_rate)

    beam_width=half_beam_width*2+1
    df=np.zeros((beam_width,beam_width,nend-nstart))
    sstart=pair[0]-batch_interval*half_beam_width
    sources=np.arange(sstart,sstart+batch_interval*beam_width,batch_interval)
    rstart=pair[1]-batch_interval*half_beam_width

    for i in range(len(sources)):
        dd=np.load(path+str(sources[i]).zfill(4)+'_pws.npy')
        df[i,:,:]=dd[(rstart-sources[i])//batch_interval:(rstart-sources[i])//batch_interval+beam_width,nstart:nend]
    
    return df

def cal_info(large,small,slowness=np.arange(0.4,3,0.02)):
    
    info_mat=np.zeros((len(large),4))
    
    for i in range(len(large)):
        forplot=large[i,:,:]
        small_panel=small[i,:,:]

        forplot_flattened=forplot.reshape(forplot.shape[0]*forplot.shape[1])
        dd=np.unravel_index(forplot_flattened.argmax(), forplot.shape)
        ratio=energy_ratio(forplot,small_panel)
        source_sig=ratio*cal_max_sig(find_local_maxima(forplot[dd[0],:])[1])
        receiver_sig=ratio*cal_max_sig(find_local_maxima(forplot[:,dd[1]])[1])
        info_mat[i,0]=1/slowness[dd[0]]
        info_mat[i,1]=1/slowness[dd[1]]
        info_mat[i,2]=source_sig
        info_mat[i,3]=receiver_sig
        
    return info_mat

def main(pair,half_beam_width=10,batch_interval=5,spacing=0.02,smin=0.4,smax=3,
         sampling_rate=10,frange=np.linspace(0.1,3,100),bandwidth=0.1,initial_slowness=np.arange(0.4,3,0.1),
         path='/scratch/zspica_root/zspica1/yaolinm/Florence/first_month/stacked/pws/',
         out_dir='/scratch/zspica_root/zspica1/yaolinm/Florence/first_month/master/'):
    
    df=build_matrix(pair,half_beam_width=half_beam_width,batch_interval=batch_interval,sampling_rate=sampling_rate,
                    spacing=spacing,smin=smin,smax=smax,path=path)
    
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            df[i,j,:]/=np.max(np.abs(df[i,j,:]))
            
    out=multi_filter_3d(df,sampling_rate,frange,bandwidth)
#     for i in range(len(frange)):
#         for j in range(df.shape[0]):
#             for k in range(df.shape[1]):
#                 out[i,j,k,:]=out[i,j,k,:]/np.max(np.abs(out[i,j,k,:]))
                
    large,small=main_dbf_new(out,sampling_rate,spacing*batch_interval,initial_slowness=initial_slowness)
    
    title='source_'+str(pair[0]).zfill(4)+'_receiver_'+str(pair[1]).zfill(4)+'_.npy'
    np.save(out_dir+'master_grid/'+title,large)
    np.save(out_dir+'small_panel/'+title,small)
    
    info=cal_info(large,small)
    np.save(out_dir+'info/'+title,info)