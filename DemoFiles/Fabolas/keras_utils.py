import numpy as np
import pandas as pd
import keras as ks
from random import randint
from keras.optimizers import Optimizer
from keras import backend as K

#pc = 0: home pc
#pc = 1: titania

pc = 1

if pc == 0:
    path = '/home/henningj/Documents/MRTDaten/'
    sshpath = '/home/henningj/'
    pathaal = path
if pc == 1:
    path = '/home/hjanssen/Schreibtisch/Scra'
    sshpath = '/spm-data/Scratch/spielwiese_henning/MRTDaten/'
    pathaal = path

def get_rois(atlas='aal', rois=[]):
    if atlas.lower() == 'aal':
        # need to use image file
        # need to check voxel size, orientation... and reshape, re-orient masks if necessary

        aal = np.load(pathaal + 'aal-79-95-69.npy')
        aal_names = pd.read_table(pathaal + 'aal.txt', header=None)
        for i in range(aal_names.shape[0]):
            aal_names.iloc[i] = aal_names.values[i][0].replace(" ", "_")

        masks = {}
        if rois:
            # check if all roi_ids are positive
            if np.min(rois) < 1:
                raise ValueError('ROI IDs must not be <1!')
            for roi_id in rois:
                roi = np.zeros(aal.shape)
                roi[aal==roi_id] = 1
                masks[aal_names.values[roi_id-1][0]]  = roi

            roi = np.zeros(aal.shape)
            roi[aal!=0] = 1
            masks['whole_brain'] = roi

        else:   # if no roi_ids are supplied, use all rois
            all_rois = list(range(1,aal_names.shape[0]+1))
            masks = get_rois(atlas=atlas, rois = all_rois )
    else:
        raise ValueError('Currently AAL is the only valid Atlas!')
        masks = []

    return masks


class Callback(ks.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        model = self.model
        weights = model.get_weights()
        for i in weights[0]:
            i[0] = randint(0,9)
        model.set_weights(weights)
        #print('epoch start')

    def on_epoch_end(self, epoch, logs={}):
        print('\nepoch end')

    def on_train_end(self, train, logs={}):
        print('Done')
