import pandas as pd
from pymongo import MongoClient
from NiiLoader import NiiLoader
import numpy as np
from keras_utils import path
from keras_utils import pc
from keras_utils import sshpath

class photonDB(object):


    def __init__(self):
        self.data = []
        self.gender = []

        if pc < 3:
            from sshtunnel import SSHTunnelForwarder
            MONGO_HOST = '141.2.203.64'
            MONGO_PORT = '22'
            MONGO_USER = 'christoph'
            MONGO_SSHPATH = sshpath + '.ssh/id_rsa'

            #Open SSH Tunnel for Mongodb
            server = SSHTunnelForwarder(
                MONGO_HOST,
                ssh_username=MONGO_USER,
                ssh_pkey=MONGO_SSHPATH,
                remote_bind_address=('127.0.0.1', 27017)
            )
            server.start()
            client = MongoClient('127.0.0.1', server.local_bind_port)
        else:
            client = MongoClient('localhost', 27017)

        photondb = client.photon
        data = photondb.data

        ids = []
        group = []
        images = []
        scanner = []
        studyId = []
        age = []
        self.gender = []
        for sub in data.find({ '$and': [{ 'session.0.images.VBM': {"$exists": True}},
                                        {'session.0.images.VBM.scanner': 'siemens-marburg'},
                                        {'$or': [
                                            {'session.0.disorder': 'HC'},
                                            {'session.0.disorder': 'MDD'}]}
                                        ]}):

            ids.append(sub['ID'])
            if sub['session'][0]['disorder'] == 'HC':
                group.append(0)
            else:
                group.append(1)
            image = path + 'AllStudies_GM_VBM8_renamed/r2' + sub['session'][0]['images']['VBM']['filename']
            images.append(image)
            scanner.append(sub['session'][0]['images']['VBM']['scanner'])
            studyId.append(sub['StudyID'])
            age.append(sub['session'][0]['age'])
            self.gender.append(sub['session'][0]['gender'])

        df = pd.DataFrame([ids, group, studyId, scanner, images]).transpose()
        filepaths = df[4].tolist()
        self.gender = np.array(self.gender)
        #Nur die ersten 1000 Testpersonen
        #filepaths = filepaths[:1000]
        #gender = gender[:1000]

        loader = NiiLoader()
        self.data = loader(filepaths, vectorize=False)

        print('[INFO]', len(self.data), 'files loaded')

        if pc < 3:
            server.stop()
