"""
Detecting bursts for 1 (bipolar) channel of EEG 
recording from preterm infants

Method was developed for the following bipolar montage: 
F3-C3, F4-C4, C3-O1, C4-O2, C3-T3, 
C4-T4, Cz-C3, and C4-Cz.  

John M. O' Toole, University College Cork
Started: 22-11-2019
last update: Time-stamp: <2019-11-28 12:55:08 (otoolej)>
"""
import numpy as np
from burst_detector import feature_set, utils, bd_parameters


def eeg_bursts(eeg_data=None, Fs=256, params=None, DBplot=False):
    """burst detector for preterm EEG

    
    Parameters
    ----------
    eeg_data: ndarray
        vector (N x 1) array of EEG
    Fs: scalar
        sampling frequency (in Hz)
    params: dataclass
        parameters for the method
    DBplot: bool, optional
        plot window or not

    Returns
    -------
    burst_anno : ndarray
        mask of bursts (1 = burst, 0 = inter-bursts)
    svm_out : ndarray
        output of SVM
    """
    # test input arguments:
    if eeg_data is None:
        raise ValueError('much include EEG as function argument')
    elif eeg_data.ndim > 1:
        if eeg_data.shape[1] > 1:
            raise ValueError('EEG input should be vector not matrix')
        else:
            eeg_data = eeg_data[:, 0]
        
    if Fs < 64:
        raise ValueError('sampling frequency much be 64 Hz or greater')
    
    if params is None:
        params = bd_parameters.bdParams()

    N = len(eeg_data)

    # -------------------------------------------------------------------
    #  1. generate feature set
    # -------------------------------------------------------------------
    t_stat = feature_set.gen_feature_set(eeg_data, Fs, params)
    N_feats = t_stat.shape[0]

    # -------------------------------------------------------------------
    #  2. linear SVM
    # -------------------------------------------------------------------
    # a) scale features
    t_stat = (t_stat.T - params.zscore_shift).T
    t_stat = (t_stat.T / params.zscore_scale).T

    # b) implement: (y ~ b + ∑ᵢ wᵢ xᵢ)
    y = np.zeros(N, )
    for n in range(N_feats):
        y += t_stat[n, :] * params.lin_svm_coeff[n]
    y += params.lin_svm_bias

    # c) trim the start and end times:
    win_trim = np.ceil(params.win_trim * Fs).astype(int)
    y[0:win_trim] = np.nan
    N = len(y)
    y[(N - win_trim):N] = np.nan
    
    
    # -------------------------------------------------------------------
    #  3. threshold
    # -------------------------------------------------------------------
    burst_anno = np.copy(y)
    svm_out = y
    burst_anno[utils.find_eq_grt_than(burst_anno, 0)] = 1
    burst_anno[utils.find_less_than(burst_anno, 0)] = 0

    
    # -------------------------------------------------------------------
    #  4. post-processing
    # -------------------------------------------------------------------
    burst_anno = utils.min_ibi_burst(burst_anno, 0, params.min_ibi_dur * Fs)
    burst_anno = utils.min_ibi_burst(burst_anno, 1, params.min_burst_dur * Fs)    

    
    if DBplot:
        plt.figure(1, clear=True)
        plt.plot(svm_out)
        plt.plot(burst_anno)

    return(burst_anno, svm_out)
