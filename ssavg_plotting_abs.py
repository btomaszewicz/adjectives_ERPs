###Script by Barbara Tomaszewicz and Yaman Özakin.

import os.path as op
import numpy as np
import mne
import pickle
from mne.viz import plot_evoked_topo

filehandler = open('ssavg-noica_filtered.npy', 'rb')
#filehandler = open('ssavg-noica.npy', 'rb')
ssavg = pickle.load(filehandler)
filehandler.close()

# Only these codes will show on the evoked topo plot:
selected_codes = [
    # 'sem-yes-x',
    # 'world-yes-x',
    # 'rel-yes-x',
    # 'abs-yes-x',
    'sem-no-x',
    'world-no-x',
    'rel-no-x',
    'abs-no-x',
]

evoked_avg = [] #creates a LIST of evoked objects (dict)

for key in selected_codes:
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

#Cz
mne.viz.plot_compare_evokeds(evoked_avg,picks=[12],invert_y=False,ci=.83)
#mne.viz.plot_compare_evokeds(ssavg,picks=[12],invert_y=True,ci=None)


# # # #Global Field Power and TOPOS 


# WORLD as baseline
# semworldno = mne.combine_evoked([semnox,worldnox],[-1, 1])
# semworldno.plot_joint(title='Sem vs. World NO Conditions', times=[.45],ts_args = dict(gfp=True))
# relworldno = mne.combine_evoked([relnox,worldnox],[-1, 1])
# relworldno.plot_joint(title='Rel vs. World NO Conditions', times=[.45],ts_args = dict(gfp=True))
# absworldno = mne.combine_evoked([absnox,worldnox],[-1, 1])
# absworldno.plot_joint(title='Abs vs. World NO Conditions', times=[.45],ts_args = dict(gfp=True))

