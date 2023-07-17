from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import mne
from pandas import Series

def BandPassECG(file,Fs):
    '''
    This function takes in a "path" and imports the ECG signal in .mat format
    '''
    # Import the signal
    data = mne.io.read_raw_edf(file)
    raw_data = data.get_data()
    ECG    = raw_data[7]
    del data
    del raw_data
    # Implementing the Butterworth BP filter
    W1     = 5*2/Fs                                    # --> 5 Hz cutt-off (high-pass) and Normalize by Sample Rate
    W2     = 15*2/Fs                                   # --> 15 Hz cutt-off (low-pass) and Normalize by Sample Rate
    b, a   = signal.butter(4, [W1,W2], 'bandpass')     # --> create b,a coefficients , since this is IIR we need both b and a coefficients
    ECG    = np.asarray(ECG)                           # --> let's convert the ECG to a numpy array, this makes it possible to perform vector operations 
    ECG    = np.squeeze(ECG)                           # --> squeeze
    ECG_BP = signal.filtfilt(b,a,ECG)    # --> filtering: note we use a filtfilt that compensates for the delay
    return ECG_BP,ECG

def Differentiate(ECG):
    '''
    Compute single difference of the signal ECG
    '''
    ECG_df  = np.diff(ECG)
    ECG_sq  = np.power(ECG_df,2)
    return np.insert(ECG_sq,0, ECG_sq[0])


def MovingAverage(ECG,N=30):
    '''
    Compute moving average of signal ECG with a rectangular window of N
    '''
    window  = np.ones((1,N))/N
    ECG_ma  = np.convolve(np.squeeze(ECG),np.squeeze(window))
    return ECG_ma

def QRSpeaks(ECG,Fs):
    '''
    Finds peaks in a smoothed signal ECG and sampling freq Fs.
    '''
    peaks, _  = signal.find_peaks(ECG, height=np.mean(ECG), distance=round(Fs*0.200))
    return peaks

