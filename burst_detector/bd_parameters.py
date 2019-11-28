"""
Parameters for the detector -- KEEP AS IS

John M. O' Toole, University College Cork
Started: 05-09-2019
last update: <2019-09-04 13:36:01 (otoolej)>
"""
from dataclasses import dataclass

# TODO: use as params = bdParams()



@dataclass(frozen=True)
class bdParams:
    # -------------------------------------------------------------------
    #  feature set (which features to use and what frequency band)
    # -------------------------------------------------------------------
    # order important here as other variables rely on it
    feature_set_final: tuple = ('envelope', 'fd_higuchi', 'edo', 'if', 'psd_r2',
                                'envelope', 'envelope', 'rel_spectral_power')
    feature_set_freqbands: tuple = (
        (1, 0, 0, 0), # envelope
        None, # FD
        None, # EDO
        (0, 0, 0, 1), # IF
        (1, 0, 0, 0), # PSD r2
        (0, 0, 0, 1), # envelope
        (0, 0, 1, 0), # envelope
        (0, 0, 0, 1)) # relative PSD

    # epoch size is slightly different for each feature
    # (follows feature order from above); (in seconds):
    epoch_length: tuple = (1, 1, 1, 2, 2, 1, 1, 2)
    # 75% overlap in feature epoch
    overlap: int = 75
    # type of window used:
    epoch_win_type: tuple = ('rect', 'rect', 'rect',
                             'rect', 'hamm', 'rect', 'rect', 'hamm')
    # frequency resolution for PSD estimates:
    N_freq: int = 2048

    log_feats: tuple = ('edo', 'envelope', 'rel_spectral_power')

    # -------------------------------------------------------------------
    #  EDO parameters
    # -------------------------------------------------------------------
    Fs_edo: int = 256
    band_pass_edo: tuple = (0.5, 10)

    # -------------------------------------------------------------------
    #  fractal dimension parameters
    # -------------------------------------------------------------------
    k_max: int = 6
    band_pass_fd: tuple = (0.5, 30)    
    

    # -------------------------------------------------------------------
    #  parameters for SVM (DO NOT EDIT)
    # -------------------------------------------------------------------
    lin_svm_coeff: tuple = (1.7231474419918471,
                            1.0944716356730551,
                            0.7363681616540538,
                            0.2108197752929180,
                            0.1599842614703928,
                            1.2797596630663119,
                            0.1359372077622752,
                            -0.4639574087961215)
    lin_svm_bias: float = 1.9547538993812108

    # z-score parameters for all the features:
    zscore_shift: tuple = (2.2236308408108432,
                           -1.4936074444300351,
                           -2.3027523468322415,
                           0.6397162426448688,
                           0.8560832002385120,
                           -0.4875872867038523,
                           -0.1620584673697459,
                           -5.5624983363377645)

    zscore_scale: tuple = (1.1129990354895545,
                           0.1944487737255839,
                           1.8773426455696265,
                           0.0742902276773938,
                           0.0454168957546715,
                           0.6524400617931475,
                           0.8130292037078876,
                           1.1720334471495857)

    # trim off start and end of detector output (as features use
    # short-time windowing approach):
    win_trim: int = 1  # in seconds

    # -------------------------------------------------------------------
    #  filter details (IIR Butterworth filter)
    # -------------------------------------------------------------------
    L_filt: int = 5
    # band-pass filter in this band
    freq_bands: tuple = ((0.5, 3), (3, 8), (8, 15), (15, 30))

    # -------------------------------------------------------------------
    #  use either static (True) or adaptive (False) threshold
    # -------------------------------------------------------------------
    static_thres: bool = True

    # -------------------------------------------------------------------
    #  post-processing to force minimum duration IBI and bursts:
    # -------------------------------------------------------------------
    # set to 0 to turn off: (in seconds)
    min_ibi_dur: int = 1.1
    min_burst_dur: int = 0.9
