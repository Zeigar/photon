import pandas as pd
import nibabel as nib
import os
import numpy as np
import scipy.io as spio
import gzip


#Todo: make sure that each class is returning an pandas DataFrame Object

class MatLoader(object):

    def __call__(self, filename, **kwargs):
        mat_data = self.load_mat(filename)
        if 'var_name' in kwargs:
            var_name = kwargs.get('var_name')
            mat_data = mat_data[var_name]
        return pd.DataFrame(data=mat_data)

    def load_mat(self, filename):
        '''
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        '''
        data = spio.loadmat(filename, struct_as_record=False,
                            squeeze_me=True)
        return self._check_keys(data)

    def _check_keys(self, item_dict):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in item_dict:
            if isinstance(item_dict[key],
                          spio.matlab.mio5_params.mat_struct):
                item_dict[key] = self._to_dict(item_dict[key])
        return item_dict

    def _to_dict(self, matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        return_dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                return_dict[strg] = self._to_dict(elem)
            else:
                return_dict[strg] = elem
        return return_dict


class CsvLoader(object):

    def __call__(self, filename, **kwargs):
        csv_data = pd.read_csv(filename, **kwargs)
        return csv_data


class XlsxLoader(object):

    def __call__(self, filename, **kwargs):
        return pd.read_excel(filename)


class NiiLoader(object):

    def __call__(self, filepaths, vectorize=False, **kwargs):
        # loading all .nii-files in one folder is no longer supported

        if isinstance(filepaths, str):
            raise TypeError('Filepaths must be passed as list.')

        elif isinstance(filepaths, list):
            # iterate over and load every .nii file
            # this requires one nifti per subject
            img_data = []
            for ind_sub in range(len(filepaths)):
                img = nib.load(filepaths[ind_sub], mmap=True)
                img_data.append(img.get_data())

        else:
            # Q for Ramona: This error is handled in the
            # DataContainer class. Handle it here anyway? Maybe to
            # ensure proper functionality even when DataContainer
            # changes?
            raise TypeError('Filepaths must be passed as list.')


        # stack list elements to matrix
        data = np.stack(img_data, axis=0)
        if vectorize:
            data = np.reshape(data, (data.shape[0], data.shape[1] *
                                 data.shape[2] * data.shape[3]))
        return data


    def get_filenames(self, directory):
        filenames = []
        for file in os.listdir(directory):
            if file.endswith(".nii"):
                filenames.append(file)
        # check if files have been found
        if len(filenames) == 0:
            raise ValueError('There are no .nii-files in the '
                             'specified folder!')
        else:
            return filenames


class MNISTLoader(object):

    def __call__(
        self,
        datapath='./',
        traindataf='train-images-idx3-ubyte.gz',
        trainlabelf='train-labels-idx1-ubyte.gz',
        testdataf='t10k-images-idx3-ubyte.gz',
        testlabelf='t10k-labels-idx1-ubyte.gz',
        **kwargs
    ):
        '''
        Loads the MNIST-Dataset and downloads it if necessary

        :param datapath: Paths containing the MNIST-files
        :param traindataf: File containing the training-data
        :param trainlabelf: File containing the training-labels
        :param testdataf: File containing the test-data
        :param testlabelf: File containing the test-labels
        :param kwargs: runaway-args
        :return: pandas data-frame of
            train_data, train_labels, test_data, test_labels
        '''
        # ensure all needed files exist and download them if necessary
        self._ensure_file_existance(
            [
                (traindataf, 'train-images-idx3-ubyte.gz'),
                (trainlabelf, 'train-labels-idx1-ubyte.gz'),
                (testdataf, 't10k-images-idx3-ubyte.gz'),
                (testlabelf, 't10k-labels-idx1-ubyte.gz')
            ],
            datapath
        )
        train_data = self._load_images(datapath+traindataf)
        train_labels = self._load_labels(datapath+trainlabelf)
        test_data = self._load_images(datapath+testdataf)
        test_labels = self._load_labels(datapath+testlabelf)

        return pd.DataFrame(train_data), train_labels,\
                pd.DataFrame(test_data), test_labels

    def _load_images(self, file):
        '''
        Loads the images from the dataset

        :param file: path to datafile
        :return: vector of shaped images (shape: (examples, channels, rows, columns))
        '''
        with gzip.open(file, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # Reshape datavectors to monochrome images
        # shape: (examples, channels, rows, columns)
        img_pixels = 28**2
        data = data.reshape(int(len(data)/img_pixels), img_pixels)
        # Convert byte-values [0,255] to [0,1] (actually [0, 255/256])
        return data/np.float32(256)

    def _load_labels(self, file):
        '''
        Loads the labels

        :param file: file containing labels
        :return: vector of labels
        '''
        # Load labels as vector of ints
        with gzip.open(file, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    def _ensure_file_existance(
        self,
        filenames,
        localpath='./',
        remotepath='http://yann.lecun.com/exdb/mnist/'
    ):
        '''
        Checks if the local files exist and downloads them otherwise

        :param filenames: dict(localfilename:remotefilename)
            or list of filenames if they are the same not including the full path
        :param localpath: local path to store files
        :param remotepath: base-url containing remote files
        '''
        from urllib.request import urlretrieve

        for l,r in filenames:
            if not os.path.exists(localpath+l):
                urlretrieve(remotepath+l, localpath+r)
