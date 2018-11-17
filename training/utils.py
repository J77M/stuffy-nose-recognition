import numpy as np
from scipy import signal
import os



def bandpass(start, stop, data, fs):
    bp_Hz = np.array([start, stop])
    b, a = signal.butter(5, bp_Hz / (fs / 2.0), btype='bandpass')
    return signal.lfilter(b, a, data, axis=0)


def fft(data, fs):
    '''fast fourier transform '''
    L = len(data)
    freq = np.linspace(0.0, 1.0 / (2.0 * fs **-1), L // 2)
    yi = np.fft.fft(data)[1:]
    y = yi[range(int(L / 2))]
    return freq, abs(y)


def reduce_resolution(data, mn = 12):
    '''mean of every chunk of data with lenght of mn'''
    reduced = [np.mean(data[i:i+mn]) for i in range(0, len(data), mn)]
    return np.asarray(reduced)


def reload_data(path):
    '''reading npy data saved by data_record.py and preparing them for training model'''
    xs = []
    ys = []
    for i in os.listdir(path):
        dat = np.load(os.path.join(path, i))
        xs.append(dat[0])
        ys += [dat[1] for _ in range(len(dat[0]))]

    data = np.concatenate(np.asarray(xs), axis=0)
    ys = np.array(ys)

    indices = np.arange(ys.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    y = ys[indices]
    return data, y

