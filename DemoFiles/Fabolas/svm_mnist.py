#!/usr/bin/env python3

# Add project_root to path s.t. we can run this script from terminal
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from Framework.PhotonBase import Hyperpipe, PipelineElement
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

mnist = fetch_mldata('MNIST original')
train_data, test_data, train_labels, test_labels = train_test_split(
    mnist.data/255.,
    mnist.target,
    test_size=1/7
)

job_id = sys.argv[1] if len(sys.argv) > 1 else 0

cv = KFold(n_splits=3, shuffle=True, random_state=0)

n_train_data = len(train_data)
pipe = Hyperpipe(
    'mnistsvm',
    cv,
    optimizer='fabolas',
    optimizer_params={
        'n_min_train_data': int(n_train_data/9000), 'n_train_data': n_train_data,
        'log': {'id': job_id, 'path': 'logs/', 'name': 'mnistsvm', 'incumbents': True}
    },
    metrics=['accuracy'],
    verbose=2
)
pipe += PipelineElement.create('svc', {'C': [-10, 10, float], 'gamma': [-10, 10, float]})

pipe.fit(train_data, train_labels)
