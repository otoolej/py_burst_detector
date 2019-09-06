"""
miscellaneous collection of tools

John M. O' Toole, University College Cork
Started: 06-09-2019
last update: <2019-09-04 13:36:01 (otoolej)>
"""
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def epoch_window(P, L, win_type, Fs, DBplot=False):
    """
    calculate overlap size (in samples) and window length for overlap-and-add type analysis


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
        print(f'L_epoch = {L}; L_hop = {L_hop}')

    return({'L_hop': L_hop, 'L_epoch': L, 'win_epoch': win})
