###Script by Yaman Özakin.


import pickle
from scipy.signal import butter, lfilter

with open('ssavg-noica.npy','rb') as f:
    ssavg=pickle.load(f)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

for k in ssavg:
    print(k)
    for i in range(len(ssavg[k])):
        for j in range(len(ssavg[k][i]._data)):
            ssavg[k][i]._data[j]=butter_bandpass_filter(ssavg[k][i]._data[j],.1,20,100,order=4)

with open('ssavg-noica_filtered.npy','wb') as f:
    pickle.dump(ssavg,f)