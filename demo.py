"""
demonstration of the EEG burst detector

John M. O' Toole, University College Cork
Started: 28-11-2019
last update: Time-stamp: <2019-11-28 16:01:02 (otoolej)>
"""
from burst_detector import eeg_burst_detector, utils
from matplotlib import pyplot as plt


# 1. generate a test signal with impulsive noise
N = 5000
Fs = 64
x = utils.gen_impulsive_noise(N)


# 2. run the burst detector on the test signal:
burst_anno, svm_out = eeg_burst_detector.eeg_bursts(x, Fs)

# 3. plot:
ttime = np.arange(N) / Fs
fig, ax = plt.subplots(nrows=2, ncols=1, num=1, clear=True, sharex=True)
ax[0].plot(ttime, x, label='test signal')
ax[1].plot(ttime, burst_anno, label='burst annotation')
ax[1].plot(ttime, svm_out, label='SVM output')
ax[1].legend(loc='upper right')
ax[0].legend(loc='upper left')
plt.xlabel('time (seconds)')

