"""
feature set

John M. O' Toole, University College Cork
Started: 06-09-2019
last update: Time-stamp: <2019-11-28 12:56:41 (otoolej)>
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert, resample_poly
from burst_detector import utils, bd_parameters



def edo_feat(x, Fs, params=None, DBplot=False):
    """generate Envelope Derivative Operator (EDO). See [1] for details:

    [1] J.M. O' Toole and N.J. Stevenson, “Assessing instantaneous energy in the EEG: a
    non-negative, frequency-weighted energy operator”, In Engineering in Medicine and
    Biology Society (EMBC), 2014 36th Annual International Conference of the IEEE,
    pp. 3288-3291. IEEE, 2014.


    Parameters
    ----------
    x: ndarray
        input signal
    Fs: scale
        sampling frequency
    params: object, optional
        dataclass object of parameters
    DBplot: bool, optional
        plot feature vs. signal

    Returns
    -------
    edo : ndarray
        EDO, same size as input signal x

    """
    if params is None:
        params = bd_parameters.bdParams()

    N_x = len(x)
    if Fs != params.Fs_edo:
        Fs_orig = Fs
        x = resample_poly(x, params.Fs_edo, Fs)
        Fs = params.Fs_edo
    else:
        Fs_orig = []

    # -------------------------------------------------------------------
    #  1. bandpass filter the signal from 0.5 to 10 Hz
    # -------------------------------------------------------------------
    x = utils.do_bandpass_filter(x, Fs, params.band_pass_edo[1],
                                 params.band_pass_edo[0])

    # -------------------------------------------------------------------
    #  2. envelope-derivated operator
    # -------------------------------------------------------------------
    N_start = len(x)
    if (N_start % 2) != 0:
        x = np.hstack((x, 0))

    N = len(x)
    Nh = np.ceil(N / 2).astype(int)
    nl = np.arange(1, N - 1)
    xx = np.zeros(N)

    # calculate the Hilbert transform build the Hilbert transform in
    # the frequency domain:
    k = np.arange(N)
    H = -1j * np.sign(Nh - k) * np.sign(k)
    h = np.fft.ifft(np.fft.fft(x) * H)
    h = np.real(h)

    # implement with the central finite difference equation
    xx[nl] = ((x[nl+1] ** 2) + (x[nl-1] ** 2) +
              (h[nl+1] ** 2) + (h[nl-1] ** 2)) / 4 - ((x[nl+1] * x[nl-1] +
                                                       h[nl+1] * h[nl-1]) / 2)

    # trim and zero-pad and the ends:
    x_edo = np.pad(xx[2:(len(xx) - 2)], (2, 2),
                   'constant', constant_values=(0, 0))

    x_edo = x_edo[0:N_start]

    # ---------------------------------------------------------------------
    #  3. smooth with window
    # ---------------------------------------------------------------------
    x_filt = utils.ma_filter(x_edo, Fs)

    # zero pad the end:
    L = len(x_filt)
    x_edo[0:L] = x_filt
    x_edo[L:-1] = 0

    # ---------------------------------------------------------------------
    #  4. downsample
    # ---------------------------------------------------------------------
    if Fs_orig:
        x_edo = resample_poly(x_edo, Fs_orig, Fs)
        # resampling may introduce very small negative values:
        x_edo[x_edo < 0] = 0

    if N_x != len(x_edo):
        x_edo = x_edo[:N_x]

    if DBplot:
        plt.figure(1, clear=True)
        plt.plot(x)
        plt.plot(x_edo)

    return(x_edo)


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
    freq_band: tuple (start, stop), default=(0.5, 3)
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

    DBcheck = False

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

        # define the frequency range (to keep the same as per Matlab code):
        f_scale = params.N_freq / Fs
        irange = np.arange(np.ceil(f_band[0] * f_scale).astype(int),
                           np.floor(f_band[1] * f_scale).astype(int) + 1)
        irange_total = np.arange(np.ceil(f_band_total[0] * f_scale).astype(int),
                                 np.floor(f_band_total[1] * f_scale).astype(int) + 1)
        irange = irange - 1
        irange_total = irange_total - 1

        if DBcheck:
            freq = np.linspace(
                0, Fs/2, np.round(params.N_freq / 2).astype(int))
            # irange = np.where((freq > f_band[0]) & (freq <= f_band[1]))
            # irange_total = np.where(
            #     (freq > f_band_total[0]) & (freq <= f_band_total[1]))
            freq_limit = freq[irange]
            freq_limit_total = freq[irange_total]

            print("frequencies between {0:g} and {1:g} Hz".format(
                freq_limit[0], freq_limit[-1]))
            print("Total frequencies between {0:g} and {1:g} Hz".format(
                freq_limit_total[0], freq_limit_total[-1]))
            print("i_BP =({0}, {1}); i_BP =({2}, {3});"
                  .format(irange[0], irange[-1],
                          irange_total[0], irange_total[-1]))

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

    elif feat_type == 'fd_higuchi':
        # -------------------------------------------------------------------
        #  Fractal Dimension estimate: band-pass filter from 0.5 to 30 Hz
        # -------------------------------------------------------------------
        x = utils.do_bandpass_filter(x, Fs, params.band_pass_fd[1],
                                     params.band_pass_fd[0])

        
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

        # -------------------------------------------------------------------
        #  different actions for different features:
        # -------------------------------------------------------------------
        if feat_type == 'envelope':
            # median value over the epoch:
            feat_x = np.median(x_epoch)

        elif feat_type == 'psd_r2':
            # generate the log-log spectrum and fit a line:
            pxx = abs(np.fft.fft(x_epoch, params.N_freq))
            pxx_db = 20 * np.log10(pxx[irange])
            feat_x = ls_fit_params(freq_db, pxx_db, DBplot=False)

        elif feat_type == 'rel_spectral_power':
            # generate the log-log spectrum and fit a line:
            pxx = abs(np.fft.fft(x_epoch, params.N_freq)) ** 2
            feat_x = sum(pxx[irange]) / sum(pxx[irange_total])

        elif feat_type == 'if':
            feat_x = np.median(x_epoch)

        elif feat_type == 'fd_higuchi':
            feat_x = fd_hi(x_epoch, params.k_max)
            feat_x = -feat_x

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


def ls_fit_params(x, y, DBplot=False):
    """goodness of fit (r^2) for least-squares line fit

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


def fit_line(x, y, DBplot=False):
    """least squares line fit

    Parameters
    ----------
    x: array_type
        input vector of line
    y: array_type
        output vector of line
    DBplot: bool, optional
        plot feature vs. signal

    Returns
    -------
    params : array_type
        2 coefficients (c,m) from line fit
    """
    # Setup matrices:
    m = np.shape(x)[0]
    X = np.matrix([np.ones(m), x]).T
    Y = np.matrix(y).T

    # Solve for projection matrix
    params = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    # Plot data, regression line
    if DBplot:
        # Find regression line
        xx = np.linspace(min(x), max(x), 2)
        yy = np.array(params[0] + params[1] * xx)

        plt.figure(10, clear=True)
        plt.plot(xx, yy.T, color='b')
        plt.scatter(x, y, color='r')

    return(params)


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


def fd_hi(x, kmax=[], DBplot=False):
    """fractal dimension estimate using the Higuchi approach [1]

    [1] T. Higuchi, “Approach to an irregular time series on the basis of 
    the fractal theory,” Phys. D Nonlinear Phenom., vol. 31, pp. 277–283, 
    1988.


    Parameters
    ----------
    x: ndarray
        input signal
    kmax: scalar
        maximum scale value (default = N/10)
    DBplot: bool, optional
        plot feature vs. signal

    Returns
    -------
    fd : scalar
        fractal dimension estimate
    """
    N = len(x)
    if not kmax:
        kmax = np.floor(N / 10).astype(int)

    # what values of k to compute?
    ik = 1
    k_all = []
    knew = 0
    while knew < kmax:
        if ik <= 4:
            knew = ik
        else:
            knew = np.floor(2 ** ((ik+5)/4)).astype(int)
        if knew <= kmax:
            k_all.append(knew)
        ik = ik + 1

    # ---------------------------------------------------------------------
    #  curve length for each vector:
    # ---------------------------------------------------------------------
    inext = 0
    L_avg = np.zeros(len(k_all), )

    for k in k_all:
        L = np.zeros(k, )

        for m in range(1, k + 1):
            ik = np.arange(1, np.floor((N - m) / k).astype(int) + 1)
            scale_factor = (N - 1) / (np.floor((N - m) / k) * k)

            L[m - 1] = sum(abs(x[m + ik * k - 1] - x[m + (ik-1) * k - 1])) * \
                (scale_factor / k)

        L_avg[inext] = np.nanmean(L)
        inext = inext + 1

    # -------------------------------------------------------------------
    #  form log-log plot of scale v. curve length
    # -------------------------------------------------------------------
    x1 = np.log2(k_all)
    y1 = np.log2(L_avg)

    c = fit_line(x1, y1, DBplot)
    fd = -c[1, 0]

    return(fd)



def gen_feature_set(x, Fs=None, params=None, DBplot=False):
    """generate feature set for signal x

    Parameters
    ----------
    x: ndarray
        input signal
    Fs: scalar
        sampling frequency
    params: object, optional
        dataclass object of parameters
    DBplot: bool, optional
        plot feature vs. signal

    Returns
    -------
    t_stat : ndarray
        feature set
    """
    if params is None:
        params = bd_parameters.bdParams()
    assert(Fs is not None), 'need to specify sampling frequency'
        
    feat_set = params.feature_set_final
    N = len(x)

    # declare the empty feature matrix:
    N_feats = len(feat_set)
    t_stat = np.zeros((N_feats, N), dtype=float)
    t_stat.fill(np.nan)

    # for missing data, insert 0's when generating the features
    inans = np.argwhere(np.isnan(x))
    if inans.size > 0:
        x[inans] = 0


    #---------------------------------------------------------------------
    # 1. do band-pass filtering first 
    #%---------------------------------------------------------------------
    filter_bands=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    x_filt = {}
    for n in range(len(filter_bands)):

        # is this band needed?
        match_fb = [p for p in range(len(params.feature_set_freqbands))
                    if params.feature_set_freqbands[p] == filter_bands[n]]
        # print("for freq. band = {}; match={}".format(filter_bands[n], np.any(match_fb)))
        if np.any(match_fb):
            # print("from {} to {} Hz".format(params.freq_bands[n][1],params.freq_bands[n][0]))
            x_f = utils.bandpass_butter_filt(x, Fs, params.freq_bands[n][1],
                                                params.freq_bands[n][0], params.L_filt)
            x_filt[filter_bands[n]] = x_f


    total_freq_bands = (params.freq_bands[0][0], params.freq_bands[-1][-1])
    # -------------------------------------------------------------------
    #  do for all features
    # -------------------------------------------------------------------
    for n in range(N_feats):

        # find filter band:
        if params.feature_set_freqbands[n] in x_filt.keys():
            y = x_filt[params.feature_set_freqbands[n]]
            freq_band = params.freq_bands[params.feature_set_freqbands[n].index(1)]
        else:
            y = x
            freq_band = None
            
        # special case for RSP:
        if feat_set[n] == 'rel_spectral_power':
            y = x

            
        if feat_set[n] in ['envelope', 'psd_r2', 'if', 'rel_spectral_power', 'fd_higuchi']:
            # -------------------------------------------------------------------
            #  short-time analysis for these features
            # -------------------------------------------------------------------
            t_stat[n, :] = feat_short_time_an(y, Fs, feat_type=feat_set[n],
                                              freq_band=freq_band,
                                              total_freq_band=total_freq_bands, 
                                              params=params)

        elif feat_set[n] == 'edo':
            # -------------------------------------------------------------------
            #  special case for the envelope-derivative operater as is
            #  calculate on the whole signal
            # -------------------------------------------------------------------
            t_stat[n, :] = edo_feat(y, Fs, params=params)

        else:
            raise ValueError('feature _ ' + feat_set[n] + ' _ not implemented')

        
        # -------------------------------------------------------------------
        #  log transform any of the variables
        # -------------------------------------------------------------------
        if feat_set[n] in params.log_feats:
            t_stat[n, :] = np.log( t_stat[n, :] + np.spacing(1))

    # -------------------------------------------------------------------
    #  re-fill the NaNs
    # -------------------------------------------------------------------
    if inans.size > 0:
        t_stat[:, inans] = np.nan

    # -------------------------------------------------------------------
    #  plot
    # -------------------------------------------------------------------
    if DBplot:
        plt.figure(1, clear=True)
        for p in range(N_feats):
            plt.plot(np.arange(N) / Fs, t_stat[p, :], label=feat_set[p])
        plt.legend()

    return(t_stat)
