"""
miscellaneous collection of tools

John M. O' Toole, University College Cork
Started: 06-09-2019
last update: <2019-09-04 13:36:01 (otoolej)>
"""
import numpy as np
from scipy import signal, sparse
from matplotlib import pyplot as plt
import os


def gen_impulsive_noise(N=1000, DBplot=False):
    """
    Impulsive noise model, idea from [1]. For testing only.

    [1] Saeed V. Vaseghi, Advanced Digital Signal Processing and 
    Noise Reduction, Second Edition, John Wiley & Sons Ltd, 2000.
    
    Parameters
    ----------
    N: scalar
        length of noise vector
    DBplot: bool, optional
        plot window or not

    Returns
    -------
    x : ndarray
        noise vector
    """
    x = np.zeros(N, )

    # a) impulse location:
    x[np.unique(np.random.randint(N, size=np.floor(N / 20).astype(int)))] = 1

    # b) amplitude modulation on the impulses:
    x *= np.random.randn(N, )

    # c) add some white Gaussian noise and low-pass filter:
    x += np.random.randn(N, ) / 20
    x = np.convolve(x, np.hamming(21), 'same')

    # d) again, add white Gaussian noise
    x += np.random.randn(N, ) / 5

    # e) scale so more similar to EEG 
    x *= 10

    # f) zero-mean:
    # x -= np.mean(x)
    
    # plot?
    if DBplot:
        plt.figure(1, clear=True)
        plt.plot(x)
    
    return(x)


def min_ibi_burst(burst_mask, burst_or_ibi=1, min_duration=256):
    """enforce minimum duration of burst or IBI

    Parameters
    ----------
    burst_mask: ndarray
        mask from burst detector of bursts (1) and interbursts (0)
    burst_or_ibi: scalar
        either bursts (1) or interbursts (0)
    min_duration: scalar
        minimum duration (in samples)

    Returns
    -------
    burst_mask : ndarray
        processed burst mask
    """
    if burst_mask.ndim > 1:
        burst_mask = burst_mask[:, 0]
    assert(np.unique(burst_mask[~np.isnan(burst_mask)]) in np.array([0, 1])), \
        'burst_mask should be binary'
    assert(burst_or_ibi in [0, 1]), 'burst_or_ibi should be [0,1]'    
        
    N = len(burst_mask)
    burst_or_ibi_opp = 1 - burst_or_ibi

    
    # extract the segments of bursts (or IBI)
    lens, istart, iend = len_cont_zeros(burst_mask, burst_or_ibi)


    # remove those below the minimum duration
    iconsider = np.argwhere(lens < min_duration)[:,0]
    for p in iconsider:
        burst_mask[istart[p]:iend[p]] = burst_or_ibi_opp

    assert(len(burst_mask) == N), 'different length signal'

    return burst_mask



def len_cont_zeros(x, const=0):
    """find runs of zeros and return length

    Parameters
    ----------
    x: ndarray
        input binary signal (1 or 0; NaNs allowed)
    const: scalar
        look for run of O's or 1's? (default=0)

    Returns
    -------
    lens : tuple
        array of lengths
    """
    # check input arguments:
    if x.ndim > 1:
        x = x[:, 0]
    assert(np.unique(x[~np.isnan(x)]) in np.array([0, 1])), 'input signal should be binary'
    assert(const in [0, 1]), 'constant should be 1 or 0'

    # check for simple case first
    if all(x == const):
        lens = np.array([len(x) - 1])
        istart = np.array([0])
        iend = lens
        
    elif all(x != const):
        lens = np.array([0])
        istart = np.array([0])
        iend = lens

    else:
        # -------------------------------------------------------------------
        #  invert signal if necessary
        # -------------------------------------------------------------------
        y = np.copy(x)    
        if const == 1:
            y[~np.isnan(x)] = 1 - x[~np.isnan(x)].astype(int)
        else:
            y[~np.isnan(x)] = x[~np.isnan(x)].astype(int)

        
        # -------------------------------------------------------------------
        #  find run of 0s or 1s:
        # -------------------------------------------------------------------
        iedge = np.diff(np.hstack((0, y == 0, 0)))
        istart = np.argwhere(iedge == 1)[:, 0]
        iend = np.argwhere(iedge == -1)[:, 0]    
        lens = iend - istart - 1;

        # -------------------------------------------------------------------
        #  test to see if extracting the correct runs:
        # -------------------------------------------------------------------
        test_arr = np.array([])
        for n in range(len(iend)):
            test_arr = np.append(test_arr, y[istart[n]:iend[n]])
            assert(np.unique(test_arr) == 0), 'not all zeros; indexing off?'

    
    return lens, istart, iend



def find_eq_grt_than(x, thres):
    """ needed when NaN in array; otherwise could use np.where """
    return(np.array([ix >= thres if ~np.isnan(ix) else False for ix in x],
                    dtype=bool))


def find_less_than(x, thres):
    """ needed when NaN in array; otherwise could use np.where """
    return(np.array([ix < thres if ~np.isnan(ix) else False for ix in x],
                    dtype=bool))



def epoch_window(P, L, win_type, Fs, DBplot=False):
    """
    calculate overlap size (in samples) and window length for overlap-and-add 
    type analysis


    Parameters
    ----------
    P: int
        length of overlap (percent)
    L: int
        epoch length (in seconds)
    win_type: string
        window type, e.g. 'hamm', 'rect'
    Fs: int
        sampling frequency
    DBplot: bool, optional
        plot window or not

    Returns
    -------
    L_lst : dict
        'L_hop'     - hop size (in samples)
        'L_epoch'   - epoch size (in samples)
        'win_epoch' - window

    """
    L_hop = (100 - P) / 100
    L = np.floor(L * Fs).astype(int)

    # check for window type to force constant-overlap add constraint
    # i.e. \sum_m w(n-mR) = 1 for all n, where R = overlap size
    win_type = win_type.lower()
    if win_type[:4] == 'hamm':
        L_hop = (L - 1) * L_hop
    elif win_type[:4] == 'hann':
        L_hop = (L + 1) * L_hop
    else:
        L_hop = L * L_hop
    L_hop = np.ceil(L_hop).astype(int)

    # get window
    win = signal.get_window(win_type, L, False)
    # and shift:
    # win = np.roll(win, np.ceil(L / 2).astype(int))

    # special case to force constant overlap-add constraint:
    # (see SPECTRAL AUDIO SIGNAL PROCESSING, JULIUS O. SMITH III)
    if win_type[:4] == 'hamm' and (L % 2) != 0:
        win[0] = win[0] / 2
        win[-1] = win[-1] / 2

    if DBplot:
        plt.figure(2, clear=True)
        plt.plot(win, '-o')
        print('L_epoch = {}; L_hop = {}'.format(L, L_hop))

    return({'L_hop': L_hop, 'L_epoch': L, 'win_epoch': win})



def gain_filt(b, a):
    """Step response from filter (used as initial conditions later)

    Parameters
    ----------
    b: ndarray
        filter coefficients
    a: ndarray
        filter coefficients

    Returns
    -------
    zi : ndarray
        step response as a vector
    """
    # max. length of coefficients:
    N_filt = max(len(b), len(a))

    # arrange entries of sparse matrix:
    rows = [*range(N_filt - 1), *range(1, N_filt - 1), *range(N_filt - 2)]
    cols = [*np.zeros(N_filt - 1).astype(int), *
            range(1, N_filt - 1), *range(1, N_filt - 1)]
    vals = [*np.hstack(((1 + a[1]), a[2:N_filt])),
            *np.ones(N_filt - 2).astype(int),
            *-np.ones(N_filt - 2).astype(int)]
    rhs = b[1:N_filt] - b[0] * a[1:N_filt]

    AA = sparse.coo_matrix((vals, (rows, cols))).tocsr()
    zi = sparse.linalg.spsolve(AA, rhs)

    return(zi)


def mat_filtfilt(b, a, x):
    """Fowards--backwards filter to match Matlab's 'filtfilt' function


    Parameters
    ----------
    b: ndarray
        filter coefficients
    a: ndarray
        filter coefficients
    x: ndarray
        input signal

    Returns
    -------
    y : ndarray
        filtered signal
    """

    # 1. pad the signal:
    L_pad = 3 * (max(len(b), len(a)) - 1)
    x_pad = np.concatenate((2 * x[0] - x[1:(L_pad + 1)][::-1],
                            x,
                            2 * x[-1] - x[len(x)-L_pad-1:-1][::-1]))

    # 2. estimate initial filter conditions:
    zi = gain_filt(b, a)

    # 3. forwards and backwards filter:
    x_pad, _ = signal.lfilter(b, a, x_pad, zi=zi * x_pad[0])
    x_pad = x_pad[::-1]
    x_pad, _ = signal.lfilter(b, a, x_pad, zi=zi * x_pad[0])
    x_pad = x_pad[::-1]

    # 4. remove the padding:
    y = x_pad[L_pad:len(x) + L_pad]

    return(y)


def do_bandpass_filter(x, Fs, LP_fc=10, HP_fc=0.5, DBplot=False):
    """bandpass filtering according to Palmu et al. (NLEO)

    Parameters
    ----------
    x: ndarray
        input signal
    Fs: scalar
        description
    LP_fc: scalar
        low-pass cut off (in Hz)
    HP_fc: scalar
        high-pass cut off (in Hz)
    DBplot: bool, optional
        to plot or not

    Returns
    -------
    y : ndarray
        filtered signal
    """
    # -------------------------------------------------------------------
    #  set parameters for the filters
    # -------------------------------------------------------------------
    HP_order = 1
    LP_order = 6

    Fs_h = Fs / 2

    # -------------------------------------------------------------------
    #  1. high-pass Butterworth filter first
    # -------------------------------------------------------------------
    b, a = signal.butter(HP_order, HP_fc / Fs_h, 'high')
    y = mat_filtfilt(b, a, x)

    # -------------------------------------------------------------------
    #  2. then low-pass elliptical filter
    # -------------------------------------------------------------------
    # hard-code filter coefficients for elliptic filter,
    # with parameters: order=6, Rp=3, Rs=50, Wn=10/(64/2), btype='low'
    if Fs in [64, 100, 200, 256, 500] and LP_fc in [10, 30]:
        b, a = filter_coeffs_ellip('ellip_filt_coeffs.npz', Fs, LP_fc)

    if len(b) == 0 and len(a) == 0:
        # otherwise redesign the filter:
        b, a = signal.ellip(LP_order, rp=3, rs=50,
                            Wn=LP_fc / Fs_h, btype='low')

    y = mat_filtfilt(b, a, y)

    if DBplot:
        plt.figure(1, clear=True)
        plt.plot(x)
        plt.plot(y)

    return(y)


def filter_coeffs_ellip(fname, Fs, LP_fc=10):
    """return stored filter coefficients to match Matlab's ellip.m function

    Filter parameters: order=6, Rp=3, Rs=50, Wn=10/(Fs/2), btype='low'

    Parameters
    ----------
    Fs: scalar
        sampling frequency; must be either 64, 100, 200, or 256 Hz


    Returns
    -------
    b, a : ndarray
        filter coefficients
    """
    b = []
    a = []
    # path of filter coefficients (in the data/ dir):
    here_path = os.path.dirname(__file__ )
    fname = os.path.join(here_path, os.pardir, 'data', fname)
    if os.path.exists(fname):
    
        all_coeffs = np.load(fname, allow_pickle=True)

        key = 'Fs{}Hz_fc{}Hz'.format(Fs, LP_fc)

        if key in all_coeffs['b'][()]:
            b = all_coeffs['b'][()][key]
            a = all_coeffs['a'][()][key]
        
        else:
            print('No coefficients for key = ' + key)
    else:
        print('cant find coefficients file: ' + coeff_fname)
        
    return([b, a])


def ma_filter(x, L_win):
    """moving average filter

    Parameters
    ----------
    x: ndarray
        input signal
    L_win: scale
        length of rect. window

    Returns
    -------
    y : ndarray
        filtered signal
    """
    N = len(x)
    y = np.zeros(N, )
    L_h = np.floor(L_win / 2).astype(int)

    y_tmp = signal.lfilter(np.ones(L_win, ), L_win, x)

    n = range(L_h, N - L_h)
    y[n] = y_tmp[n]

    return(y)


def bandpass_butter_filt(x, Fs, LP_fc=3, HP_fc=0.5, L_order=5, DBplot=False):
    """bandpass filtering using a Butterworth filter

    Parameters
    ----------
    x: ndarray
        input signal
    Fs: scalar
        description
    LP_fc: scalar
        low-pass cut off (in Hz)
    HP_fc: scalar
        high-pass cut off (in Hz)
    L_order: scalar
        filter order
    DBplot: bool, optional
        to plot or not

    Returns
    -------
    y : ndarray
        filtered signal
    """
    # -------------------------------------------------------------------
    #  set parameters for the filters
    # -------------------------------------------------------------------
    Fs_h = Fs / 2

    # -------------------------------------------------------------------    
    #  1. first the low-pass
    # -------------------------------------------------------------------
    b, a = signal.butter(L_order, LP_fc / Fs_h, 'low')
    y = mat_filtfilt(b, a, x)
    
    # -------------------------------------------------------------------
    #  2. then the high-pass filter
    # -------------------------------------------------------------------
    b, a = signal.butter(L_order, HP_fc / Fs_h, 'high')
    y = mat_filtfilt(b, a, y)


    if DBplot:
        plt.figure(1, clear=True)
        plt.plot(x)
        plt.plot(y)

    return(y)



