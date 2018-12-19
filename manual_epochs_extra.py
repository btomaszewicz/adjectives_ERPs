###Script by Barbara Tomaszewicz and Yaman Özakin.

#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser(description='Plot and edit epochs. Run it in IPython shell.')
parser.add_argument('epochs_file', help='epochs file name')
args = parser.parse_args()

import os.path as op
import numpy as np
import mne
from mne.viz import plot_evoked_topo


try:
    epochs = mne.read_epochs(args.epochs_file)
except:
    raise IOError("Couldn't read the epoch file {}".format(args.epochs_file))

codes = {'sem-yes-x': 203,
    'sem-no-x': 208,
    'world-yes-x': 213,
    'world-no-x': 218,
    'rel-yes-x': 223,
    'rel-no-x': 228,
    'abs—min-yes-x': 233,
    'abs—min-no-x': 238,
    'abs—max-yes-x': 243,
    'abs—max-no-x': 248
    } 

print(epochs)

#(1) look the participants' averages to see in which channels there are the most artifacts:
ssavg = {}
for c in codes:
    ssavg[c] = list()

for c in codes:
        evoked = epochs[c].average()
        ssavg[c].append(evoked)

evoked_avg = [] #creates a LIST of evoked objects (dict)

for key in codes:
    # create a copy of the first evoked object in ssavg[key]:
    evoked_avg.append(ssavg[key][0].copy())
    evoked_avg[-1].data = evoked_avg[-1].data * 0

    n_ssavg = len(ssavg[key])
    for i in range(n_ssavg):
        if len(ssavg[key][i].data) == len(evoked_avg[-1].data):
            # average the data accross participants:
            evoked_avg[-1].data =  evoked_avg[-1].data + ssavg[key][i].data / n_ssavg

print("Click on the thing to show zoomed-in electrode")
plot_evoked_topo(evoked_avg) #first argument is a list of evoked instances



#(2) manually remove bad epochs:
#epochs.drop_bad()
#epochs.plot_drop_log()
epochs.plot()

#(3) remember to save
#out_file = args.epochs_file.split('.')[0]+'-epo.fif'
print("If you want to save the file, do this in python shell:")
print("epochs.save(args.epochs_file)")
#epochs.save('Obja0004-epo.fif')
#epochs.save(args.epochs_file)



