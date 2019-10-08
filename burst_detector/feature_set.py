"""
feature set

John M. O' Toole, University College Cork
Started: 06-09-2019
last update: Time-stamp: <2019-10-08 13:25:43 (otoolej)>
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
import bd_parameters
from burst_detector import utils


def env(x, Fs, params=None, DBplot=False):
    """estimate median of envelope (using Hilbert transform) over a 2-second window


    Parameters
    ----------
    x: array_like
        input signal
    Fs: int
        sampling frequency
    params: object, optional
        dataclass object of parameters
    DBplot: bool, optional
        plot feature vs. signal


    Returns
    -------
    t_stat : ndarray
        feature vector generated from x (same dimension)
    """
    if params is None:
        params = bd_parameters.bdParams()

    # -------------------------------------------------------------------
    #  set the epoch window parameters
    # -------------------------------------------------------------------
    idx = params.feature_set_final.index('envelope')
    epoch_p = utils.epoch_window(params.overlap, params.epoch_length[0],
                                 params.epoch_win_type[idx], Fs)

    N = len(x)
    N_epochs = np.floor(
        (N - epoch_p['L_epoch']) / epoch_p['L_hop']).astype(int)
    if N_epochs < 1:
        N_epochs = 1
    nw = np.arange(epoch_p['L_epoch'])

    # get the envelope:
    y = abs(hilbert(x))

    # -------------------------------------------------------------------
    #  iterate over all the epochs
    # -------------------------------------------------------------------
    z_all = np.zeros(N)
    win_summed = np.zeros(N)
    kshift = np.floor(epoch_p['L_epoch'] / (2 * epoch_p['L_hop'])).astype(int)

    N_epochs_plus = N_epochs + kshift
    ev = np.zeros(N_epochs_plus)

    for k in range(N_epochs):
        nf = np.remainder(nw + (k * epoch_p['L_hop']), N)

        ev[k + kshift - 1] = np.median(y[nf])

        z_all[nf] = z_all[nf] + \
            (np.ones(epoch_p['L_epoch']) * ev[k + kshift - 1])
        win_summed[nf] = win_summed[nf] + np.ones(epoch_p['L_epoch'])

    win_summed[np.where(win_summed == 0)] = np.nan
    t_stat = np.divide(z_all, win_summed)

    DBplot = True
    if DBplot:
        plt.figure(1, clear=True)
        plt.plot(x)
        plt.plot(t_stat)

    return t_stat


def feat_short_time_an(x, Fs, feat_type='envelope', **kwargs):
    """estimate median of envelope (using Hilbert transform) over a 2-second window

    Parameters
    ----------
    x: array_like
        input signal
    Fs: int
        sampling frequency
    feat_type: string
        feature type, ('envelope', 'fd-higuchi', 'edo', 'if', 'psd_r2', 'spec-power')
    feat_band: tuple (start, stop), default=(0.5, 3)
        frequency band for feature
    total_feat_band: tuple (start, stop), default=(0.5, 30)
        total frequency band for feature
    params: object, optional
        dataclass object of parameters
    DBplot: bool, optional
        plot feature vs. signal

    Returns
    -------
    t_stat : ndarray
        feature vector generated from x (same dimension)
    """
    default_args = {'freq_band': (0.5, 3),
                    'total_freq_band': (0.5, 30),
                    'params': None,
                    'DBplot': False}
    arg_list = {**default_args, **kwargs}
    # extract some arguments just for clarity:
    f_band = arg_list['freq_band']
    f_band_total = arg_list['total_freq_band']
    params = arg_list['params']

    print("f_band = {0}; total_f_band = {1}".format(f_band, f_band_total))

    DBcheck = True

    if params is None:
        params = bd_parameters.bdParams()

    # -------------------------------------------------------------------
    #  set the epoch window parameters
    # -------------------------------------------------------------------
    idx = params.feature_set_final.index(feat_type)
    epoch_p = utils.epoch_window(params.overlap, params.epoch_length[idx],
                                 params.epoch_win_type[idx], Fs)

    N = len(x)
    N_epochs = np.floor(
        (N - epoch_p['L_epoch']) / epoch_p['L_hop']).astype(int)
    if N_epochs < 1:
        N_epochs = 1
    nw = np.arange(epoch_p['L_epoch'])

    # -------------------------------------------------------------------
    #  different operations for different features
    #  (and anything else that should be defined outside of the
    #   short-time analysis)
    # -------------------------------------------------------------------
    if feat_type == 'envelope':
        # -------------------------------------------------------------------
        #  envelope of the signal using the Hilbert transform
        # -------------------------------------------------------------------
        x = abs(hilbert(x))

    elif feat_type == 'psd_r2':
        # -------------------------------------------------------------------
        #  goodness-of-fit of line to log-log PSD
        # -------------------------------------------------------------------
        
        # define the frequency range and conver to log-scale
        freq = np.linspace(0, Fs/2, params.N_freq)
        irange = np.where((freq > f_band[0]) & (freq < f_band[1]))
        freq_limit = freq[irange]
        freq_db = 10 * np.log10(freq_limit)
        if DBcheck:
            print("frequencies between {0:g} and {1:g} Hz".format(
                freq_limit[0], freq_limit[-1]))

    elif feat_type == 'rel_spectral_power':
        # -------------------------------------------------------------------
        #  relative spectral power
        # -------------------------------------------------------------------
        
        # define the frequency range and conver to log-scale
        freq = np.linspace(0, Fs/2, np.ceil(params.N_freq / 2).astype(int))
        irange = np.where((freq > f_band[0]) & (freq <= f_band[1]))
        irange_total = np.where(
            (freq > f_band_total[0]) & (freq <= f_band_total[1]))
        freq_limit = freq[irange]
        freq_limit_total = freq[irange_total]
        if DBcheck:
            print("frequencies between {0:g} and {1:g} Hz".format(
                freq_limit[0], freq_limit[-1]))
            print("Total frequencies between {0:g} and {1:g} Hz".format(
                freq_limit_total[0], freq_limit_total[-1]))
            print("i_BP =({0}, {1}); i_BP =({2}, {3});"
                  .format(irange[0][0], irange[0][-1],
                          irange_total[0][0], irange_total[0][-1]))

    elif feat_type == 'if':
        # -------------------------------------------------------------------
        #  instantaneous frequency estimate
        # -------------------------------------------------------------------
        
        # estimate instantaneous frequency (IF):
        est_IF = estimate_IF(x, Fs)

        # bind within frequency bands:
        est_IF[est_IF > f_band[1]] = f_band[1]
        est_IF[est_IF < f_band[0]] = f_band[0]

        # normalized between 0 and 1 (need when combining features):
        est_IF = (est_IF - f_band[0]) / (f_band[1] - f_band[0])

        # invert:
        x = 1 - est_IF

    # -------------------------------------------------------------------
    #  iterate over all the epochs
    # -------------------------------------------------------------------
    z_all = np.zeros(N)
    win_summed = np.zeros(N)
    kshift = np.floor(epoch_p['L_epoch'] / (2 * epoch_p['L_hop'])).astype(int)

    N_epochs_plus = N_epochs + kshift
    # ev = np.zeros(N_epochs_plus)

    for k in range(N_epochs):
        nf = np.remainder(nw + (k * epoch_p['L_hop']), N)
        x_epoch = x[nf] * epoch_p['win_epoch']

        # different actions for different features:
        if feat_type == 'envelope':
            # median value over the epoch:
            feat_x = np.median(x_epoch)

        elif feat_type == 'psd_r2':
            # generate the log-log spectrum and fit a line:
            pxx = abs(np.fft.fft(x_epoch, params.N_freq))
            pxx_db = 20 * np.log10(pxx[irange])
            feat_x = ls_fit_params(freq_db, pxx_db, DBplot=True)

        elif feat_type == 'rel_spectral_power':
            # generate the log-log spectrum and fit a line:
            pxx = abs(np.fft.fft(x_epoch, params.N_freq)) ** 2
            feat_x = sum(pxx[irange]) / sum(pxx[irange_total])

        elif feat_type == 'if':
            feat_x = np.median(x_epoch)

        # upsample to EEG sampling rate:
        z_all[nf] = z_all[nf] + (np.ones(epoch_p['L_epoch']) * feat_x)
        win_summed[nf] = win_summed[nf] + np.ones(epoch_p['L_epoch'])

    # remove the effect of the windowed approach:
    win_summed[np.where(win_summed == 0)] = np.nan
    t_stat = np.divide(z_all, win_summed)

    if arg_list['DBplot']:
        plt.figure(1, clear=True)
        plt.plot(x)
        plt.plot(t_stat)

    return t_stat


def ls_fit_params(x, y, DBplot=True):
    """least squares line fit

    Parameters
    ----------
    x: array_type
        input vector of line
    y: array_type
        output vector of line

    Returns
    -------
    r2 : scalar
        fit of regression line
    """
    # Setup matrices:
    m = np.shape(x)[0]
    X = np.matrix([np.ones(m), x]).T
    Y = np.matrix(y).T

    # Solve for projection matrix
    params = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    # print(params)

    # estimate goodness of fit:
    y_fit = params[0] + params[1] * x
    y_residuals = np.asarray(y - y_fit)
    r2 = 1 - (((y_residuals ** 2).sum()) / (m * np.var(y)))

    # Plot data, regression line
    if DBplot:
        # Find regression line
        xx = np.linspace(min(x), max(x), 2)
        yy = np.array(params[0] + params[1] * xx)

        plt.figure(10, clear=True)
        plt.plot(xx, yy.T, color='b')
        plt.scatter(x, y, color='r')
        plt.show()

    return(r2)


def estimate_IF(x, Fs=1):
    """instantaneous frequency estimate from angle
        of analytic signal

    Parameters
    ----------
    x: ndarray
        input signal
    Fs: scalar
        sampling frequency

    Returns
    -------
    if_a : ndarray
        IF array
    """
    if np.all(np.isreal(x)):
        z = hilbert(x)
    else:
        z = x
    N = len(z)

    MF = 2 * np.pi
    SCALE = Fs / (4 * np.pi)

    if_a = np.zeros(N,)
    n = np.arange(0, N - 2)

    # central finite difference for IF:
    z_diff = np.angle(z[n+2]) - np.angle(z[n])
    if_a[n + 1] = np.mod(MF + z_diff, MF) * SCALE

    return(if_a)
