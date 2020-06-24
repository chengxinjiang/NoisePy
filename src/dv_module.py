import os
import scipy
import pycwt
import datetime
import numpy as np
import noise_module
import pyasdf
import obspy
from obspy.core.util.base import _get_function_from_entry_point
from scipy.fftpack import next_fast_len
from obspy.signal.filter import bandpass
from obspy.signal.invsim import cosine_taper
from obspy.signal.regression import linear_regression



###################### CORE FUNCTIONS ####################

'''
a compilation of all available core functions for computing phase delays based on ambient noise interferometry
quick index of dv/v methods:
1) stretching (time stretching; Weaver et al (2011))
2) dtw_dvv (Dynamic Time Warping; Mikesell et al. 2015)
3) mwcs_dvv (Moving Window Cross Spectrum; Clark et al., 2011)
4) mwcc_dvv (Moving Window Cross Correlation; Snieder et al., 2012)
5) wts_dvv (Wavelet Streching)
6) wxs_dvv (Wavelet Xross Spectrum; Mao et al., 2019)
7) wdw_dvv (Wavelet Dynamic Warping)
'''

def load_waveforms(sfile,para):
    '''
    functions to load targeted cross-correlation functions (CCFs) from ASDF file; trim and filter the CCFs
    and return for later dv-v measurements
    PARAMETERS:
    ----------------
    sfile: ASDF file for one-station pair with stacked and substacked CCFs
    para: dictionary containing all useful variables to window data
    RETURNS:
    ----------------
    ref: reference waveform
    cur: array containing all current waveforms 
    '''
    # load useful variables
    twin = para['twin']
    comp = para['ccomp']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    norm_flag = para['norm_flag']

    with pyasdf.ASDFDataSet(sfile,mode='r') as ds:
        slist = ds.auxiliary_data.list()

        # some common variables
        try:
            delta = ds.auxiliary_data[slist[0]][comp].parameters['dt']
            maxlag= ds.auxiliary_data[slist[0]][comp].parameters['maxlag']
        except Exception:
            raise ValueError('cannot open %s to read'%sfile)

        if delta != dt:
            print('dt shoud be %s at L31, not %s!'%(delta,dt))
            para['dt'] = delta

        # time axis
        tvec = np.arange(-maxlag,maxlag)*delta
        indx = np.where((tvec>=tmin) & (tvec<tmax))[0]
        if not len(indx):
            raise ValueError('Abort! time window is probably wrong')
        tt   = tvec[indx]
        npts = tt.size
        nstacks = len(slist)

        # prepare data matrix for later loading 
        data  = np.zeros((nstacks,npts),dtype=np.float32)
        stamp = np.zeros(nstacks,dytpe=np.float)
        flag  = np.zeros(nstacks,dtype=np.int16)

        # loop through each stacked segment
        for ii,dtype in enumerate(slist):
            try:
                tdata = ds.auxiliary_data[dtype][comp].data[:]
                data[ii]  = tdata[indx]
                stamp[ii] = ds.auxiliary_data[dtype][comp].parameters['time']
                flag[ii]  = 1
            except Exception:
                continue

        # remove bad ones
        indx = np.where(flag==1)
        data = data[indx]
        stamp= stamp[indx]
        del flag
    
    ref = data[0]
    data = data[1:]

    # detrend, demean and tapering
    ref  = noise_module.demean(ref)
    ref  = noise_module.detrend(ref)
    ref  = noise_module.taper(ref)    
    data = noise_module.demean(data)
    data = noise_module.detrend(data)
    data = noise_module.taper(data)
    return ref,data,stamp,para

def stretching(ref, cur, dv_range, nbtrial, para):
    
    """
    This function compares the Reference waveform to stretched/compressed current waveforms to get the relative seismic velocity variation (and associated error).
    It also computes the correlation coefficient between the Reference waveform and the current waveform.
    PARAMETERS:
    ----------------
    ref: Reference waveform (np.ndarray, size N)
    cur: Current waveform (np.ndarray, size N)
    dv_range: absolute bound for the velocity variation; example: dv=0.03 for [-3,3]% of relative velocity change ('float')
    nbtrial: number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  ('float')
    para: vector of the indices of the cur and ref windows on wich you want to do the measurements (np.ndarray, size tmin*delta:tmax*delta)
    
    For error computation, we need parameters:
        fmin: minimum frequency of the data
        fmax: maximum frequency of the data
        tmin: minimum time window where the dv/v is computed 
        tmax: maximum time window where the dv/v is computed 
    RETURNS:
    ----------------
    dv: Relative velocity change dv/v (in %)
    cc: correlation coefficient between the reference waveform and the best stretched/compressed current waveform
    cdp: correlation coefficient between the reference waveform and the initial current waveform
    error: Errors in the dv/v measurements based on Weaver et al (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3)
    Note: The code first finds the best correlation coefficient between the Reference waveform and the stretched/compressed current waveform among the "nbtrial" values. 
    A refined analysis is then performed around this value to obtain a more precise dv/v measurement .
    Originally by L. Viens 04/26/2018 (Viens et al., 2018 JGR)
    modified by Chengxin Jiang
    """ 
    # load common variables from dictionary
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    t = np.arange(tmin,tmax,dt)
    if len(t)!=len(cur):
        cur=cur[0:min([len(t),len(cur)])]
        t=t[0:min([len(t),len(cur)])]
        ref=ref[0:min([len(t),len(cur)])]

    # make useful one for measurements
    dvmin = -np.abs(dv_range)
    dvmax = np.abs(dv_range)
    Eps = 1+(np.linspace(dvmin, dvmax, nbtrial))
    cof = np.zeros(Eps.shape,dtype=np.float32)
    
    for ii in range(len(Eps)):
        tau = t*Eps[ii]
        s = np.interp(x=t, xp=tau, fp=cur)
        cof[ii] = np.corrcoef(ref,s)[0, 1]

    cdp = np.corrcoef(ref,cur)[0, 1] # correlation coefficient between the reference and initial current waveforms

    # find the maximum correlation coefficient
    imax = np.nanargmax(cof)
    if imax >= len(Eps)-2:
        imax = imax - 2
    if imax <= 2:
        imax = imax + 2

    # Proceed to the second step to get a more precise dv/v measurement
    dtfiner = np.linspace(Eps[imax-2], Eps[imax+2], 100)
    ncof    = np.zeros(dtfiner.shape,dtype=np.float32)
    for ii in range(len(dtfiner)):
        nt = t*dtfiner[ii]
        s = np.interp(x=t, xp=nt, fp=cur)
        waveform_ref = ref
        waveform_cur = s
        ncof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]

    cc = np.max(ncof) # Find maximum correlation coefficient of the refined  analysis
    if (np.argmax(ncof)==0 or (np.argmax(ncof)==len(ncof))):
        dv,error,cc,cdp=None,None,None,None
        return dv, error, cc, cdp
        
    dv = 100. * dtfiner[np.argmax(ncof)]-100 # Multiply by 100 to convert to percentage (Epsilon = -dt/t = dv/v)

    # Error computation based on Weaver et al (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3)
    T = 1 / (fmax - fmin)
    X = cc
    wc = np.pi * (fmin + fmax)
    t1 = np.min([tmin, tmax])
    t2 = np.max([tmin, tmax])
    error = 100*(np.sqrt(1-X**2)/(2*X)*np.sqrt((6* np.sqrt(np.pi/2)*T)/(wc**2*(t2**3-t1**3))))

    return dv, error, cc, cdp


def dtw_dvv(ref, cur, para, maxLag, b, direction):
    """
    Dynamic time warping for dv/v estimation.
    
    PARAMETERS:
    ----------------
    ref : reference signal (np.array, size N)
    cur : current signal (np.array, size N)
    para: dict containing useful parameters about the data window and targeted frequency
    maxLag : max number of points to search forward and backward. 
            Suggest setting it larger if window is set larger.
    b : b-value to limit strain, which is to limit the maximum velocity perturbation. 
            See equation 11 in (Mikesell et al. 2015)
    
    RETURNS:
    ------------------
    -m0 : estimated dv/v
    em0 : error of dv/v estimation
        
    """
    twin = para['twin']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    tvect = np.arange(tmin,tmax,dt)

    # setup other parameters
    npts = len(ref) # number of time samples
    
    # compute error function over lags, which is independent of strain limit 'b'.
    err = computeErrorFunction( cur, ref, npts, maxLag ) 
    
    # direction to accumulate errors (1=forward, -1=backward)
    # it is instructive to flip the sign of +/-1 here to see how the function
    # changes as we start the backtracking on different sides of the traces.
    # Also change 'b' to see how this influences the solution for stbar. You
    # want to make sure you're doing things in the proper directions in each
    # step!!!
    dist  = accumulateErrorFunction( direction, err, npts, maxLag, b )
    stbar = backtrackDistanceFunction( -1*direction, dist, err, -maxLag, b )
    stbarTime = stbar * dt   # convert from samples to time
    
    # linear regression to get dv/v
    if npts >2:

        # weights
        w = np.ones(npts)
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(tvect.flatten(), stbarTime.flatten(), w.flatten(), intercept_origin=True)

    else:
        print('not enough points to estimate dv/v for dtw')
        m0=0;em0=0
    
    return m0*100,em0*100,dist


def mwcs_dvv(ref, cur, moving_window_length, slide_step, para, smoothing_half_win=5):
    """
    Moving Window Cross Spectrum method to measure dv/v (relying on phi=2*pi*f*t in freq domain)
    PARAMETERS:
    ----------------
    ref: Reference waveform (np.ndarray, size N)
    cur: Current waveform (np.ndarray, size N)
    moving_window_length: moving window length to calculate cross-spectrum (np.float, in sec)
    slide_step: steps in time to shift the moving window (np.float, in seconds)
    para: a dict containing parameters about input data window and frequency info, including 
        delta->The sampling rate of the input timeseries (in Hz)
        window-> The target window for measuring dt/t
        freq-> The frequency bound to compute the dephasing (in Hz)
        tmin: The leftmost time lag (used to compute the "time lags array")
    smoothing_half_win: If different from 0, defines the half length of the smoothing hanning window.
    
    RETURNS:
    ------------------
    time_axis: the central times of the windows. 
    delta_t: dt
    delta_err:error 
    delta_mcoh: mean coherence
    
    Originally from MSNoise by Thomas Lecocq. (https://github.com/ROBelgium/MSNoise/tree/master/msnoise)
    Modified by Chengxin Jiang
    """
    # common variables
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    tvect = np.arange(tmin,tmax,dt)

    # parameter initialize
    delta_t = []
    delta_err = []
    delta_mcoh = []
    time_axis = []

    # info on the moving window
    window_length_samples = np.int(moving_window_length/dt)
    padd = int(2 ** (noise_module.nextpow2(window_length_samples) + 2))
    count = 0
    tp = cosine_taper(window_length_samples, 0.15)

    minind = 0
    maxind = window_length_samples

    # loop through all sub-windows
    while maxind <= len(ref):
        cci = cur[minind:maxind]
        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp

        cri = ref[minind:maxind]
        cri = scipy.signal.detrend(cri, type='linear')
        cri *= tp

        minind += int(slide_step/dt)
        maxind += int(slide_step/dt)

        # do fft
        fcur = scipy.fftpack.fft(cci, n=padd)[:padd // 2]
        fref = scipy.fftpack.fft(cri, n=padd)[:padd // 2]

        fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
        fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2

        # get cross-spectrum & do filtering
        X = fref * (fcur.conj())
        if smoothing_half_win != 0:
            dcur = np.sqrt(noise_module.smooth(fcur2, window='hanning',half_win=smoothing_half_win))
            dref = np.sqrt(noise_module.smooth(fref2, window='hanning',half_win=smoothing_half_win))
            X = noise_module.smooth(X, window='hanning',half_win=smoothing_half_win)
        else:
            dcur = np.sqrt(fcur2)
            dref = np.sqrt(fref2)

        dcs = np.abs(X)

        # Find the values the frequency range of interest
        freq_vec = scipy.fftpack.fftfreq(len(X) * 2, dt)[:padd // 2]
        index_range = np.argwhere(np.logical_and(freq_vec >= fmin,freq_vec <= fmax))

        # Get Coherence and its mean value
        coh = noise_module.getCoherence(dcs, dref, dcur)
        mcoh = np.mean(coh[index_range])

        # Get Weights
        w = 1.0 / (1.0 / (coh[index_range] ** 2) - 1.0)
        w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
        w = np.sqrt(w * np.sqrt(dcs[index_range]))
        w = np.real(w)

        # Frequency array:
        v = np.real(freq_vec[index_range]) * 2 * np.pi

        # Phase:
        phi = np.angle(X)
        phi[0] = 0.
        phi = np.unwrap(phi)
        phi = phi[index_range]

        # Calculate the slope with a weighted least square linear regression
        # forced through the origin; weights for the WLS must be the variance !
        m, em = linear_regression(v.flatten(), phi.flatten(), w.flatten())
        delta_t.append(m)

        # print phi.shape, v.shape, w.shape
        e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
        s2x2 = np.sum(v ** 2 * w ** 2)
        sx2 = np.sum(w * v ** 2)
        e = np.sqrt(e * s2x2 / sx2 ** 2)

        delta_err.append(e)
        delta_mcoh.append(np.real(mcoh))
        time_axis.append(tmin+moving_window_length/2.+count*slide_step)
        count += 1

        del fcur, fref
        del X
        del freq_vec
        del index_range
        del w, v, e, s2x2, sx2, m, em

    if maxind > len(cur) + int(slide_step/dt):
        print("The last window was too small, but was computed")

    # ensure all matrix are np array
    delta_t = np.array(delta_t)
    delta_err = np.array(delta_err)
    delta_mcoh = np.array(delta_mcoh)
    time_axis  = np.array(time_axis)

    # ready for linear regression
    delta_mincho = 0.65
    delta_maxerr = 0.1
    delta_maxdt  = 0.1
    indx1 = np.where(delta_mcoh>delta_mincho)
    indx2 = np.where(delta_err<delta_maxerr)
    indx3 = np.where(delta_t<delta_maxdt)

    #-----find good dt measurements-----
    indx = np.intersect1d(indx1,indx2)
    indx = np.intersect1d(indx,indx3)

    if len(indx) >2:

        #----estimate weight for regression----
        w = 1/delta_err[indx]
        w[~np.isfinite(w)] = 1.0

        #---------do linear regression-----------
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=True)
    
    else:
        print('not enough points to estimate dv/v')
        m0=0;em0=0

    return -m0*100,em0*100


def WCC_dvv(ref, cur, moving_window_length, slide_step, para):
    """
    Windowed cross correlation (WCC) for dt or dv/v mesurement (Snieder et al. 2012)
    Parameters:
    -----------
    ref: The "Reference" timeseries
    cur: The "Current" timeseries
    moving_window_length: The moving window length (in seconds)
    slide_step: The step to jump for the moving window (in seconds)
    para: a dict containing freq/time info of the data matrix
    Returns:
    ------------
    time_axis: central times of the moving window
    delta_t: dt
    delta_err: error
    delta_mcoh: mean coherence for each window
        
    Written by Congcong Yuan (1 July, 2019)
    """ 
    # common variables
    twin = para['twin']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)

    # parameter initialize
    delta_t = []
    delta_t_coef = []
    time_axis = []

    # info on the moving window
    window_length_samples = np.int(moving_window_length/dt)
    count = 0
    tp = cosine_taper(window_length_samples, 0.15)

    minind = 0
    maxind = window_length_samples

    # loop through all sub-windows
    while maxind <= len(ref):
        cci = cur[minind:maxind]
        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp

        cri = ref[minind:maxind]
        cri = scipy.signal.detrend(cri, type='linear')
        cri *= tp

        minind += int(slide_step/dt)
        maxind += int(slide_step/dt)

        # normalize signals before cross correlation
        cci = (cci - cci.mean()) / cci.std()
        cri = (cri - cri.mean()) / cri.std()
        
        # get maximum correlation coefficient and its index
        cc2 = np.correlate(cci, cri, mode='same')
        cc2 = cc2 / np.sqrt((cci**2).sum() * (cri**2).sum())
            
        imaxcc2 = np.where(cc2==np.max(cc2))[0]
        maxcc2 = np.max(cc2)
        
        # get the time shift
        m = (imaxcc2-((maxind-minind)//2))*dt
        delta_t.append(m)
        delta_t_coef.append(maxcc2)
       
        time_axis.append(tmin+moving_window_length/2.+count*slide_step)
        count += 1

    del cci, cri, cc2, imaxcc2, maxcc2
    del m

    if maxind > len(cur) + int(slide_step/dt):
        print("The last window was too small, but was computed")

    delta_t = np.array(delta_t)
    delta_t_coef = np.array(delta_t_coef)
    time_axis  = np.array(time_axis)

    # linear regression to get dv/v
    if count >2:
        # simple weight
        w = np.ones(count)
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(time_axis.flatten(), delta_t.flatten(), w.flatten(),intercept_origin=True)
    
    else:
        print('not enough points to estimate dv/v')
        m0=0;em0=0

    return -m0*100,em0*100


def wxs_allfreq(cur,ref,allfreq,para,dj=1/12, s0=-1, J=-1, sig=False, wvn='morlet',unwrapflag=False):
    """
    Compute dt or dv/v in time and frequency domain from wavelet cross spectrum (wxs).
    for all frequecies in an interest range
    
    Parameters
    --------------
    :type cur: :class:`~numpy.ndarray`
    :param cur: 1d array. Cross-correlation measurements.
    :type ref: :class:`~numpy.ndarray`
    :param ref: 1d array. The reference trace.
    :type t: :class:`~numpy.ndarray`
    :param t: 1d array. Cross-correlation measurements.
    :param twindow: 1d array. [earlist time, latest time] time window limit
    :param fwindow: 1d array. [lowest frequncy, highest frequency] frequency window limit
    :params, dj, s0, J, sig, wvn, refer to function 'wavelet.wct'
    :unwrapflag: True - unwrap phase delays. Default is False
    :nwindow: the times of current period/frequency, which will be time window if windowflag is False 
    :windowflag: if True, the given window 'twindow' will be used, 
                 otherwise, the current period*nwindow will be used as time window
    
    Originally written by Tim Clements (1 March, 2019)
    Modified by Congcong Yuan (30 June, 2019) based on (Mao et al. 2019).
    """
    # common variables
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)    
    tvec = np.arange(tmin,tmax,dt)
    
    # perform cross coherent analysis, modified from function 'wavelet.cwt'
    WCT, aWCT, coi, freq, sig = pycwt.wct(cur, ref, dt, dj=dj, s0=s0, J=J, sig=sig, wavelet=wvn, normalize=True)
    
    if unwrapflag:
        phase = np.unwrap(aWCT,axis=-1) # axis=0, upwrap along time; axis=-1, unwrap along frequency
    else:
        phase=aWCT
    
    # convert phase delay to time delay
    delta_t = phase / (2*np.pi*freq[:,None]) # normalize phase by (2*pi*frequency) 

    # zero out data outside frequency band
    if (fmax> np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: input frequency out of limits!')
    else:
        freq_indin = np.where((freq >= fmin) & (freq <= fmax))[0]
        
    # initialize arrays for dv/v measurements
    dvv, err = np.zeros(freq_indin.shape), np.zeros(freq_indin.shape)
         
    # loop through freq for linear regression
    for ii, ifreq in enumerate(freq_indin):
        if len(tvec)>2:
            if not np.any(delta_t[ifreq]):
                continue
            #---- use WXA as weight for regression----
            # w = 1.0 / (1.0 / (WCT[ifreq,:] ** 2) - 1.0)
            # w[WCT[ifreq,time_ind] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
            # w = np.sqrt(w * np.sqrt(WXA[ifreq,time_ind]))
            # w = np.real(w)
            w = 1/WCT[ifreq]
            w[~np.isfinite(w)] = 1.0
            
            #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
            m, em = linear_regression(tvec, delta_t[ifreq], w, intercept_origin=True)
            dvv[ii], err[ii] = -m, em
        else:
            print('not enough points to estimate dv/v')
            dvv[ii], err[ii]=np.nan, np.nan    

    del WCT, aWCT, coi, sig, phase, delta_t
    del tvec, w, m, em

    if not allfreq:
        return np.mean(dvv)*100,np.mean(err)*100
    else:        
        return freq[freq_indin], dvv*100, err*100

def wts_allfreq(ref,cur,allfreq,para,dv_range,nbtrial,dj=1/12,s0=-1,J=-1,wvn='morlet',normalize=True):
    """
    Apply stretching method to continuous wavelet transformation (CWT) of signals
    for all frequecies in an interest range
    
    Parameters
    --------------
    :type ref: :class:`~numpy.ndarray`
    :param ref: 1d array. The reference trace.
    :type cur: :class:`~numpy.ndarray`
    :param cur: 1d array. Cross-correlation measurements.
    :type ave: :class:`~numpy.ndarray`
    :param ave: flag to averaging the dv/v over a frequency range.
    
    :params, dj, s0, J, wvn, refer to function 'wavelet.cwt'
    :normalize: True - normalize signals before stretching. Default is True
    :param maxdv: Velocity relative variation range [-maxdv, maxdv](100%)
    :param ndv : Number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  ('float')
    :nwindow: the times of current period/frequency, which will be time window if windowflag is False 
    :windowflag: if True, the given window 'twindow' will be used, 
                 otherwise, the current period*nwindow will be used as time window
    
    Written by Congcong Yuan (30 Jun, 2019)  
    """
    # common variables
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)    
    tvec = np.arange(tmin,tmax,dt)
    
    # apply cwt on two traces
    cwt1, sj, freq, coi, _, _ = pycwt.cwt(cur, dt, dj, s0, J, wvn)
    cwt2, sj, freq, coi, _, _ = pycwt.cwt(ref, dt, dj, s0, J, wvn)
    
    # extract real values of cwt
    rcwt1, rcwt2 = np.real(cwt1), np.real(cwt2)
    
    # zero out data outside frequency band
    if (fmax> np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: input frequency out of limits!')
    else:
        freq_indin = np.where((freq >= fmin) & (freq <= fmax))[0]

        # initialize variable
        nfreq=len(freq_indin)
        dvv, cc, cdp, err = np.zeros(nfreq,dtype=np.float32), np.zeros(nfreq,dtype=np.float32),\
            np.zeros(nfreq,dtype=np.float32),np.zeros(nfreq,dtype=np.float32)  
        
        # loop through each freq
        for ii, ifreq in enumerate(freq_indin):

            # prepare windowed data                
            wcwt1, wcwt2 = rcwt1[ifreq], rcwt2[ifreq] 

            # Normalizes both signals, if appropriate.
            if normalize:
                ncwt1 = (wcwt1 - wcwt1.mean()) / wcwt1.std()
                ncwt2 = (wcwt2 - wcwt2.mean()) / wcwt2.std()
            else:
                ncwt1 = wcwt1
                ncwt2 = wcwt2
                
            # run stretching
            dv, error, c1, c2 = stretching(ncwt2, ncwt1, dv_range, nbtrial, para)
            dvv[ii], cc[ii], cdp[ii], err[ii]=dv, c1, c2, error     
    
    del cwt1, cwt2, rcwt1, rcwt2, ncwt1, ncwt2, wcwt1, wcwt2, coi, sj
    
    if not allfreq:
        return np.mean(dvv),np.mean(err)
    else:        
        return freq[freq_indin], dvv, err


def wtdtw_allfreq(ref,cur,allfreq,para,maxLag,b,direction,dj=1/12,s0=-1,J=-1,wvn='morlet',normalize=True):
    """
    Apply dynamic time warping method to continuous wavelet transformation (CWT) of signals
    for all frequecies in an interest range
    
    Parameters
    --------------
    :type cur: :class:`~numpy.ndarray`
    :param cur: 1d array. Cross-correlation measurements.
    :type ref: :class:`~numpy.ndarray`
    :param ref: 1d array. The reference trace.
    :type t: :class:`~numpy.ndarray`
    :param t: 1d array. Cross-correlation measurements.
    :param twindow: 1d array. [earlist time, latest time] time window limit
    :param fwindow: 1d array. [lowest frequncy, highest frequency] frequency window limit
    :params, dj, s0, J, wvn, refer to function 'wavelet.cwt'
    :normalize: True - normalize signals before stretching. Default is True
    :param maxLag : max number of points to search forward and backward. 
                Suggest setting it larger if window is set larger.
    :param b : b-value to limit strain, which is to limit the maximum velocity perturbation. 
               See equation 11 in (Mikesell et al. 2015)
    :nwindow: the times of current period/frequency, which will be time window if windowflag is False 
    :windowflag: if True, the given window 'twindow' will be used, 
                 otherwise, the current period*nwindow will be used as time window
    
    Written by Congcong Yuan (30 Jun, 2019)
    """
    # common variables
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)    
    tvec = np.arange(tmin,tmax)*dt
 
    # apply cwt on two traces
    cwt1, sj, freq, coi, _, _ = pycwt.cwt(cur, dt, dj, s0, J, wvn)
    cwt2, sj, freq, coi, _, _ = pycwt.cwt(ref, dt, dj, s0, J, wvn)
    
    # extract real values of cwt
    rcwt1, rcwt2 = np.real(cwt1), np.real(cwt2)
    
    # zero out cone of influence and data outside frequency band
    if (fmax> np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: input frequency out of limits!')
    else:
        freq_indin = np.where((freq >= fmin) & (freq <= fmax))[0]
        
        # Use DTW method to extract dvv
        nfreq=len(freq_indin)
        dvv, err = np.zeros(nfreq,dtype=np.float32), np.zeros(nfreq,dtype=np.float32)   
        
        for ii,ifreq in enumerate(freq_indin):

            # prepare windowed data 
            wcwt1, wcwt2 = rcwt1[ifreq], rcwt2[ifreq]
            # Normalizes both signals, if appropriate.
            if normalize:
                ncwt1 = (wcwt1 - wcwt1.mean()) / wcwt1.std()
                ncwt2 = (wcwt2 - wcwt2.mean()) / wcwt2.std()
            else:
                ncwt1 = wcwt1
                ncwt2 = wcwt2
            
            # run dtw
            dv, error, dist  = dtw_dvv(ncwt2, ncwt1, para, maxLag, b, direction)
            dvv[ii], err[ii] = dv, error     
    
    del cwt1, cwt2, rcwt1, rcwt2, ncwt1, ncwt2, wcwt1, wcwt2, coi, sj, dist
    
    if not allfreq:
        return np.mean(dvv),np.mean(err)
    else:        
        return freq[freq_indin], dvv, err


############## CONSIDERED UTILITY FUNCTIONS ############

def computeErrorFunction(u1, u0, nSample, lag, norm='L2'):
    """
    USAGE: err = computeErrorFunction( u1, u0, nSample, lag )
    
    INPUT:
        u1      = trace that we want to warp; size = (nsamp,1)
        u0      = reference trace to compare with: size = (nsamp,1)
        nSample = numer of points to compare in the traces
        lag     = maximum lag in sample number to search
        norm    = 'L2' or 'L1' (default is 'L2')
    OUTPUT:
        err = the 2D error function; size = (nsamp,2*lag+1)
    
    The error function is equation 1 in Hale, 2013. You could uncomment the
    L1 norm and comment the L2 norm if you want on Line 29
    
    Original by Di Yang
    Last modified by Dylan Mikesell (25 Feb. 2015)
    Translated to python by Tim Clements (17 Aug. 2018)
    """

    if lag >= nSample: 
        raise ValueError('computeErrorFunction:lagProblem','lag must be smaller than nSample')

    # Allocate error function variable
    err = np.zeros([nSample, 2 * lag + 1])

    # initial error calculation 
    # loop over lags
    for ll in np.arange(-lag,lag + 1):
        thisLag = ll + lag 

        # loop over samples 
        for ii in range(nSample):
            
            # skip corners for now, we will come back to these
            if (ii + ll >= 0) & (ii + ll < nSample):
                err[ii,thisLag] = u1[ii] - u0[ii + ll]

    if norm == 'L2':
        err = err**2
    elif norm == 'L1':
        err = np.abs(err)

    # Now fix corners with constant extrapolation
    for ll in np.arange(-lag,lag + 1):
        thisLag = ll + lag 

        for ii in range(nSample):
            if ii + ll < 0:
                err[ii, thisLag] = err[-ll, thisLag]

            elif ii + ll > nSample - 1:
                err[ii,thisLag] = err[nSample - ll - 1,thisLag]
    
    return err


def accumulateErrorFunction(dir, err, nSample, lag, b ):
    """
    USAGE: d = accumulation_diw_mod( dir, err, nSample, lag, b )
    INPUT:
        dir = accumulation direction ( dir > 0 = forward in time, dir <= 0 = backward in time)
        err = the 2D error function; size = (nsamp,2*lag+1)
        nSample = numer of points to compare in the traces
        lag = maximum lag in sample number to search
        b = strain limit (integer value >= 1)
    OUTPUT:
        d = the 2D distance function; size = (nsamp,2*lag+1)
    
    The function is equation 6 in Hale, 2013.
    Original by Di Yang
    Last modified by Dylan Mikesell (25 Feb. 2015)
    Translated to python by Tim Clements (17 Aug. 2018)
    """

    # number of lags from [ -lag : +lag ]
    nLag = ( 2 * lag ) + 1

    # allocate distance matrix
    d = np.zeros([nSample, nLag])

    # Setup indices based on forward or backward accumulation direction
    if dir > 0: # FORWARD
        iBegin, iEnd, iInc = 0, nSample - 1, 1
    else: # BACKWARD
        iBegin, iEnd, iInc = nSample - 1, 0, -1 

    # Loop through all times ii in forward or backward direction
    for ii in range(iBegin,iEnd + iInc,iInc):

        # min/max to account for the edges/boundaries
        ji = max([0, min([nSample - 1, ii - iInc])])
        jb = max([0, min([nSample - 1, ii - iInc * b])])

        # loop through all lag 
        for ll in range(nLag):

            # check limits on lag indices 
            lMinus1 = ll - 1

            # check lag index is greater than 0
            if lMinus1 < 0:
                lMinus1 = 0 # make lag = first lag

            lPlus1 = ll + 1 # lag at l+1
            
            # check lag index less than max lag
            if lPlus1 > nLag - 1: 
                lPlus1 = nLag - 1
            
            # get distance at lags (ll-1, ll, ll+1)
            distLminus1 = d[jb, lMinus1] # minus:  d[i-b, j-1]
            distL = d[ji,ll] # actual d[i-1, j]
            distLplus1 = d[jb, lPlus1] # plus d[i-b, j+1]

            if ji != jb: # equation 10 in Hale, 2013
                for kb in range(ji,jb + iInc - 1, -iInc): 
                    distLminus1 = distLminus1 + err[kb, lMinus1]
                    distLplus1 = distLplus1 + err[kb, lPlus1]
            
            # equation 6 (if b=1) or 10 (if b>1) in Hale (2013) after treating boundaries
            d[ii, ll] = err[ii,ll] + min([distLminus1, distL, distLplus1])

    return d


def backtrackDistanceFunction(dir, d, err, lmin, b):
    """
    USAGE: stbar = backtrackDistanceFunction( dir, d, err, lmin, b )
    INPUT:
        dir   = side to start minimization ( dir > 0 = front, dir <= 0 =  back)
        d     = the 2D distance function; size = (nsamp,2*lag+1)
        err   = the 2D error function; size = (nsamp,2*lag+1)
        lmin  = minimum lag to search over
        b     = strain limit (integer value >= 1)
    OUTPUT:
        stbar = vector of integer shifts subject to |u(i)-u(i-1)| <= 1/b
    The function is equation 2 in Hale, 2013.
    Original by Di Yang
    Last modified by Dylan Mikesell (19 Dec. 2014)
    Translated to python by Tim Clements (17 Aug. 2018)
    """

    nSample, nLag = d.shape
    stbar = np.zeros(nSample)

    # Setup indices based on forward or backward accumulation direction
    if dir > 0: # FORWARD
        iBegin, iEnd, iInc = 0, nSample - 1, 1
    else: # BACKWARD
        iBegin, iEnd, iInc = nSample - 1, 0, -1 

    # start from the end (front or back)
    ll = np.argmin(d[iBegin,:]) # find minimum accumulated distance at front or back depending on 'dir'
    stbar[iBegin] = ll + lmin # absolute value of integer shift

    # move through all time samples in forward or backward direction
    ii = iBegin

    while ii != iEnd: 

        # min/max for edges/boundaries
        ji = np.max([0, np.min([nSample - 1, ii + iInc])])
        jb = np.max([0, np.min([nSample - 1, ii + iInc * b])])

        # check limits on lag indices 
        lMinus1 = ll - 1

        if lMinus1 < 0: # check lag index is greater than 1
            lMinus1 = 0 # make lag = first lag

        lPlus1 = ll + 1

        if lPlus1 > nLag - 1: # check lag index less than max lag
            lPlus1 = nLag - 1

        # get distance at lags (ll-1, ll, ll+1)
        distLminus1 = d[jb, lMinus1] # minus:  d[i-b, j-1]
        distL = d[ji,ll] # actual d[i-1, j]
        distLplus1 = d[jb, lPlus1] # plus d[i-b, j+1]

        # equation 10 in Hale (2013)
        # sum errors over i-1:i-b+1
        if ji != jb:
            for kb in range(ji, jb - iInc - 1, iInc):
                distLminus1 = distLminus1 + err[kb, lMinus1]
                distLplus1  = distLplus1  + err[kb, lPlus1]
        
        # update minimum distance to previous sample
        dl = np.min([distLminus1, distL, distLplus1 ])

        if dl != distL: # then ll ~= ll and we check forward and backward
            if dl == distLminus1:
                ll = lMinus1
            else: 
                ll = lPlus1
        
        # assume ii = ii - 1
        ii += iInc 

        # absolute integer of lag
        stbar[ii] = ll + lmin 

        # now move to correct time index, if smoothing difference over many
        # time samples using 'b'
        if (ll == lMinus1) | (ll == lPlus1): # check edges to see about b values
            if ji != jb: # if b>1 then need to move more steps
                for kb in range(ji, jb - iInc - 1, iInc):
                    ii = ii + iInc # move from i-1:i-b-1
                    stbar[ii] = ll + lmin  # constant lag over that time

    return stbar


def wct_modified(y1, y2, dt, dj=1/12, s0=-1, J=-1, sig=True,
        significance_level=0.95, wavelet='morlet', normalize=True, **kwargs):
    """Wavelet coherence transform (WCT).
    The WCT finds regions in time frequency space where the two time
    series co-vary, but do not necessarily have high power.
    Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.
    dj : float, optional
        Spacing between discrete scales. Default value is 1/12.
        Smaller values will result in better scale resolution, but
        slower calculation and plot.
    s0 : float, optional
        Smallest scale of the wavelet. Default value is 2*dt.
    J : float, optional
        Number of scales less one. Scales range from s0 up to
        s0 * 2**(J * dj), which gives a total of (J + 1) scales.
        Default is J = (log2(N*dt/so))/dj.
    significance_level (float, optional) :
        Significance level to use. Default is 0.95.
    normalize (boolean, optional) :
        If set to true, normalizes CWT by the standard deviation of
        the signals.
    Returns
    -------
    TODO: Something TBA and TBC
    See also
    --------
    cwt, xwt
    """
    wavelet = pycwt.wavelet._check_parameter_wavelet(wavelet)

    # Checking some input parameters
    if s0 == -1:
        # Number of scales
        s0 = 2 * dt / wavelet.flambda()
    if J == -1:
        # Number of scales
        J = np.int(np.round(np.log2(y1.size * dt / s0) / dj))

    # Makes sure input signals are numpy arrays.
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    # Calculates the standard deviation of both input signals.
    std1 = y1.std()
    std2 = y2.std()
    # Normalizes both signals, if appropriate.
    if normalize:
        y1_normal = (y1 - y1.mean()) / std1
        y2_normal = (y2 - y2.mean()) / std2
    else:
        y1_normal = y1
        y2_normal = y2

    # Calculates the CWT of the time-series making sure the same parameters
    # are used in both calculations.
    _kwargs = dict(dj=dj, s0=s0, J=J, wavelet=wavelet)
    W1, sj, freq, coi, _, _ = pycwt.cwt(y1_normal, dt, **_kwargs)
    W2, sj, freq, coi, _, _ = pycwt.cwt(y2_normal, dt, **_kwargs)

    scales1 = np.ones([1, y1.size]) * sj[:, None]
    scales2 = np.ones([1, y2.size]) * sj[:, None]

    # Smooth the wavelet spectra before truncating.
    S1 = wavelet.smooth(np.abs(W1) ** 2 / scales1, dt, dj, sj)
    S2 = wavelet.smooth(np.abs(W2) ** 2 / scales2, dt, dj, sj)

    # Now the wavelet transform coherence
    W12 = W1 * W2.conj()
    scales = np.ones([1, y1.size]) * sj[:, None]
    S12 = wavelet.smooth(W12 / scales, dt, dj, sj)
    WCT = np.abs(S12) ** 2 / (S1 * S2)
    aWCT = np.angle(W12)
    
    # Calculate cross spectrum & its amplitude
    WXS, WXA = W12, np.abs(S12)

    # Calculates the significance using Monte Carlo simulations with 95%
    # confidence as a function of scale.

    if sig:
        a1, b1, c1 = pycwt.ar1(y1)
        a2, b2, c2 = pycwt.ar1(y2)
        sig = pycwt.wct_significance(a1, a2, dt=dt, dj=dj, s0=s0, J=J,
                               significance_level=significance_level,
                               wavelet=wavelet, **kwargs)
    else:
        sig = np.asarray([0])

    return WXS, WXA, WCT, aWCT, coi, freq, sig