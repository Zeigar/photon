import nibabel as nib
import numpy as np
from keras_utils import get_rois

class NiiLoader(object):

    def __call__(self, filepaths, vectorize=False, **kwargs):
        # loading all .nii-files in one folder is no longer supported

        if isinstance(filepaths, str):
            raise TypeError('Filepaths must be passed as list.')

        elif isinstance(filepaths, list):
            # iterate over and load every .nii file
            # this requires one nifti per subject
            img_data = []
            print('[INFO] Load nii files...')
            for ind_sub, val in enumerate(filepaths):
                try:
                    img = nib.load(filepaths[ind_sub], mmap=False)
                    img_data.append(img.get_data())

                    if ind_sub > 0 and ind_sub % 100 == 0:
                        print("[INFO] processed {}/{}".format(ind_sub, len(filepaths)))


                except FileNotFoundError:
                    print('File: ', filepaths[ind_sub], ' not found')
        else:
            raise TypeError('Filepaths must be passed as list.')


        # stack list elements to matrix
        print('[INFO] stack list elements to matrix...')
        data = np.stack(img_data, axis=0)


        #data from marburg scanner are 567 nii files with shape (79, 95, 69)
        if vectorize:
            print('[INFO] vectorize starts...')
            data = np.reshape(data, (data.shape[0], data.shape[1] *
                                     data.shape[2] * data.shape[3]))
        else:
            #test for small trainingset
            masks = get_rois(atlas='aal', rois=[])
            data_masked = data * np.tile(masks['whole_brain'], (data.shape[0], 1, 1, 1))

            data_vec = np.zeros((int(data_masked.shape[0]),int(np.sum(masks['whole_brain']))))

            for i in range(data_masked.shape[0]):
               data_vec[i,:] = data_masked[i,:,:,:][data_masked[i,:,:,:]!=0]
            data = data_vec

        return data

