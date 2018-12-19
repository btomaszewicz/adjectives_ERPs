###Script by Ingmar Brilmayer. Edits by Barbara Tomaszewicz and Yaman Özakin.

import sys

orig_stdout = sys.stdout
sys.stdout = open('dropped_epochs.txt', 'w')

import mne
from mne.preprocessing import ICA, create_eog_epochs
from philistine.mne import abs_threshold, retrieve
import numpy as np
from numpy import *
import matplotlib as mpl
import pandas as pd
#import operator
import pathlib


inpath = '/Users/administrator/Dropbox/OBJEKT-results/OBJEKT/eeg/'
outpath = '/Users/administrator/Dropbox/OBJEKT-results/OBJEKT/erp_blink/absmanual_CUNY/'

force_overwrite = False # if true, it calculates and overwrites epoch files even if they exist

#choose your channel layout (there are several, just google it. Ours is EEG1005.lay with a standard 10:20 montage
layout = mne.channels.read_layout("EEG1005.lay") 
montage = mne.channels.read_montage(kind="standard_1020") 

codes = {'sem-yes-x': 203,
    'sem-no-x': 208,
    'world-yes-x': 213,
    'world-no-x': 218,
    'rel-yes-x': 223,
    'rel-no-x': 228,
    'abs-min-yes-x': 233,
    'abs-min-no-x': 238,
    'abs-max-yes-x': 243,
    'abs-max-no-x': 248
    } 

newcodes = {'sem-yes-x': 203,
    'sem-no-x': 208,
    'world-yes-x': 213,
    'world-no-x': 218,
    'rel-yes-x': 223,
    'rel-no-x': 228,
    'abs-yes-x': 233243,
    'abs-no-x': 238248
    }

#!!!!!!!!! do you want to filter the raw data before calculating means?
#no difference for mean amplitude (Luck 2014: because high-frequency noise is canceled out in measurements of mean amplitude (assuming the measurement window is wide enough), filtering out the high frequencies prior to measuring mean amplitude doesn ’ t help anything.)
#you can just smooth the plots (Luck 2014: we do apply a low-pass filter before plotting the data in our figures. This simply makes it easier for the reader to see the differences between the waveforms without being distracted by noise. You will see that we explicitly state that the waveforms have been filtered in our figure captions.)
#if you do, below run: raw.filter(.1,45, method='fir', n_jobs=4,fir_design='firwin')


##Rejection criteria for automatic rection; measures in V; 1e-6 = 0.000001 V = 1uV
peak_to_peak = dict(eeg=150e-6)#,eog=180e-6) # peak-to-peak rejection criteria (you can play around with this)
#you can go up to 180 in eeg, and 200 in the eyes, but this shouldn't be a problem here, because we don't use EOG in epoching, and eyes are corrected using ICA - in Philip's class we had 250 EOG
flatline = dict(eeg=5e-6) # flatline rejection criteria (i.e. when a channel is near zero at a certain time point

ssavg = {} 
ssavg_out = outpath+'ssavg.npy'

subjects=[1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,32,33]#[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24]

wins = dict(baseline=(-200,0),
            n400classic=(300, 500),
            n400narrow=(350, 450),
            n400early=(300, 400),
            n400late=(400, 500),
            p600early=(500, 700),
            p600=(600, 800),
            p600narrow1=(600,700),
            p600narrow2=(600,750),
            win1=(100, 200),
            win2=(200, 300),
            win3=(300, 400),
            win4=(400, 500),
            win5=(500, 600),
            win6=(600, 700),
            win7=(700, 800),
            win8=(800, 900))


for c in newcodes:
    ssavg[c] = list() 

for i in subjects:
    print('Working on subject {:02d}'.format(i+1))
    subjfile = inpath + 'Obj00{:02d}.vhdr'.format(i+1) #.format(i+1) adds 1 to i, since python starts counting at 0, so that subject 0 = subject 1
    csvout = outpath+'Obj00{:02d}_wins.csv'.format(i+1) 
    evoked_out = outpath+'Obj00{:02d}-epo.fif'.format(i+1)
    icafile = outpath+'Obj00{:02d}-ica.fif'.format(i+1) 
    
    #loading saved epochs files
    if (not pathlib.Path(evoked_out).exists()) or force_overwrite:    
    #if (not (pathlib.Path(evoked_out).exists() and pathlib.Path(csvout).exists())) or force_overwrite:    
        print('creating epoch file from scratch')
        #load the raw data
        raw = mne.io.read_raw_brainvision(subjfile, preload=True)

        if subjfile.split('/')[-1]=='Obj0021.vhdr':
            subjfile2 = inpath + 'Obj0021b.vhdr'
            raw2 = mne.io.read_raw_brainvision(subjfile2, preload=True)
            raw.append(raw2, preload=True) #method, so changing the object in place

        if subjfile.split('/')[-1]=='Obj0022.vhdr':
            subjfile2 = inpath + 'Obj0022b.vhdr'
            raw2 = mne.io.read_raw_brainvision(subjfile2, preload=True)
            raw.append(raw2, preload=True)

        if subjfile.split('/')[-1]=='Obj0024.vhdr':
            subjfile2 = inpath + 'Obj0024b.vhdr'
            raw2 = mne.io.read_raw_brainvision(subjfile2, preload=True)
            raw.append(raw2, preload=True)


        # for fixing participants' 1&2 trigger codes in OBJEKT:
        if subjfile.split('/')[-1]=='Obj0001.vhdr' or subjfile.split('/')[-1]=='Obj0002.vhdr':
            codes_list=[]
            data = raw._data[32]
            for i in range(len(data)):
                if data[i] != 0:
                    codes_list.append(data[i])
                    if codes_list[-1] == 248 and codes_list[-2] != 247:
                        data[i] = 244

        #import ipdb;ipdb.set_trace()

        #rename non-standard channels (=EOGli etc.) to international standard names
        raw.rename_channels({'EOGli':'LO1','EOGre': 'LO2', 'EOGobre':'SO2','EOGunre':'IO2','ReRef':'A1'})
        #which channels are EOG channels?
        raw.set_channel_types({'IO2':'eog','SO2':'eog','LO1':'eog', 'LO2':'eog'})
        #add missing reference channel A2
        raw = mne.add_reference_channels(raw,'A2',copy=True) 
        #set the channel montage to the file named above
        raw.set_montage(montage)
        #re-reference to linked mastoids
        raw.set_eeg_reference(['A1','A2'])

        #extract events from raw file [lists all events in the data]
        events = mne.find_events(raw) #extract events from raw, this returns an array of the type [time, 0, trigger]
        #make a copy of the events array, just in case, then...
        old_events = events.copy()
        #...loop through the events
        for ev in range(len(events[:,2])): 
            if events[ev,2] in codes.values(): 
                events[ev,1] = events[ev-5,2] #for OBJEKT -5 because the item code preceds ROI by 5
        #triggers in OBJEKT: item, block, picture/condition, das, ist, ROI, display, correct, 2(?)

        raw, events = raw.resample(100,npad='auto',events=events)  #resample the raw data to 100 Hz, including events to account for timing issues with events that appear at a point no longer included after resampling (leads to max. timing imprecisions of +-9 ms; if we would do speech analysis, this would be a lot, but in language research this is ok, i guess)
        

        ##### ICA part #############################
        if not pathlib.Path(icafile).exists(): 
            print('ICA file doesn\'t exist so calculating ICA componenst from scratch')
            raw2ica = raw.copy().filter(1,None,method='fir', n_jobs=4,fir_design='firwin') # make copy of raw, and apply 1Hz high-pass filter (ICA likes "flat", stationary signals). This is the absolut standard in ICA literature. One could also apply 45 Hz low pass here, to account for line noise, if wanted (but filtering is bad...so leave it out if the decomposition turns out ok)

            raw.filter(.1,45, method='fir', n_jobs=4,fir_design='firwin') #filter the unfiltered raw file for erps later on (this is also done when loading the ICA file below (the "else..."par), so don't worry, data is never filtered twice
            #define picks, i.e channels you want to calculate ICA with
            #BT debate whether to include EOG channels, because we have so few, we keep them because the more data for ICA the better
            ch_picks_ica = mne.pick_types(raw2ica.info, # set parameters for ICA
                                            meg=False,
                                            eeg=True,
                                            eog=True, # with or without eog channels? BT see my comment above
                                            stim=False,
                                            exclude='bads') #bads are excluded if there is any bad channel (obsolete in the present script)
            ica = ICA(method='extended-infomax',max_iter=2000,verbose='INFO').fit(raw2ica,
                        picks=ch_picks_ica) #calculate ica, if not for demonstration purposes (i.e. if you have time), use "extended-infomax" instead of "fastica" as method; 2000 iterations should be enough (can be increased though, yet, in matlab, the fit increase above 2000 iterations was always only marginal, so 2000 should suffice.

            # EYE ARTIFACT DETECTION in raw data
            eog_epochs = create_eog_epochs(raw) #create epochs with eog activity in raw data
            eog_average = create_eog_epochs(raw).average(picks=ch_picks_ica) #create average of eog_epochs
            blinks, blink_scores = ica.find_bads_eog(eog_epochs,ch_name='IO2',reject_by_annotation=False, threshold = 3) #which one is a blink? Store the IC-index in "blinks", store the scores in the correlation test in "blink_scores"
            #this doesn't work all to well with saccades, but still it works [sometimes], so do the same for the lateral channels, to possibly get some saccade IC
            saccades, sacc_scores = ica.find_bads_eog(eog_epochs,ch_name='LO1',reject_by_annotation=False, threshold = 3)
            artifs = blinks+saccades #summarize artifactual components in the list artifs

            #DO PLOTTING. SANITY CHECK!!! maybe save figures to file to be able to loop without popups
            #   title_scores = '%s scores of components' #title for scores
            #   title_overlay = 'Overlay of raws, with and without %s components' #title for overlay
            #   title_props = 'Properties of component %s' #title for ic properties
            #   title_source = 'Eye component %s' #title for eog component plot
            #   ica.plot_scores(blink_scores, exclude=blinks,title=title_scores) # %artif_types) # plot the r scores of components and blink events
            #BT: 'artif_types' is not defined
            #   ica.plot_sources(eog_average, exclude=blinks)#,title=title_source %artif_types) #plot eog_component(s)
            #   ica.plot_overlay(raw, exclude=blinks, show=True,title = title_overlay)#% artif_types) #mean of raw with and without artifact components
            #   ica.plot_properties(raw, picks=blinks) #plot properties of artifact ics

            ica.exclude.extend(artifs) #mark ICs for exclusion; uncomment to exclude ICs saved in list artifs, blinks or saccades
            #uncomment this for reading and writing
            ica.save(icafile) #save ica to file
            del raw2ica #delete raw2ica
        else: #else load ica from file
            print('Loading ICA from file {}'.format(icafile))
            ica = mne.preprocessing.read_ica(icafile)
            raw.filter(.1,45, method='fir', n_jobs=4,fir_design='firwin') #filter the unfiltered raw file for erps later on
        #end of ICA loop

        ica.apply(raw) #apply ICA to raw file

        # #BT: mark blinks as bads
        # eog_events = mne.preprocessing.find_eog_events(raw)
        # n_blinks = len(eog_events)
        # # Center to cover the whole blink with full duration of 0.5s:
        # onset_bl = eog_events[:, 0] / raw.info['sfreq'] - 0.25
        # duration_bl = np.repeat(0.5, n_blinks)
        # raw.annotations = mne.Annotations(onset_bl, duration_bl, ['bad blink'] * n_blinks,
        #                           orig_time=raw.info['meas_date'])
        
        #which channel types for epochs? Here, EOG channels are recorded
        ch_picks_epoch = mne.pick_types(raw.info, # set parameters for ICA
                                        meg=False,
                                        eeg=True,
                                        eog=True, # with or without eog channels?
                                        stim=False,
                                        exclude='bads') #bads are excluded if there is any bad channel (obsolete in the present script)
        #create epochs
        epochs = mne.Epochs(raw,
                            events=events,
                            event_id=codes,
                            tmin=-.2,
                            tmax=1,
                            detrend=1,
                            baseline=None,
                            preload=True,
                            flat=flatline,
                            reject=peak_to_peak,
                            reject_by_annotation=False, #If True (default), epochs overlapping with segments whose description begins with 'bad' are rejected.
                            picks=ch_picks_epoch
                            )
        epochs = mne.epochs.combine_event_ids(epochs, ['abs-min-yes-x', 'abs-max-yes-x'], {'abs-yes-x': 233243})
        epochs = mne.epochs.combine_event_ids(epochs, ['abs-min-no-x', 'abs-max-no-x'], {'abs-no-x': 238248})


        #save epochs to file
        epochs.save(evoked_out)
        #retrieve time windows of interest
        df = retrieve(epochs, wins,items=epochs.events[:,1])
        df.to_csv(csvout)
    else:
        print('epoch file "{}" exists, reading...'.format(evoked_out))
        try:
            epochs = mne.read_epochs(evoked_out)
            print('Successfully read the file, yay!')
        except FileNotFoundError:
            raise FileNotFoundError("I can't find this damn file {}".format(evoked_out))
        except:
            raise IOError("Problem reading the file: {}".format(evoked_out))
    print(epochs)

 

    for c in newcodes:
        erp = epochs[c].average()
        ssavg[c].append(erp)
#this is the end of the loop, all done for each participant

import pickle
filehandler = open(ssavg_out, 'wb')
pickle.dump(ssavg, filehandler)
filehandler.close()

sys.stdout.close()
sys.stdout=orig_stdout #to change the output back to the console you need to keep a reference to the original stdout




