
from HPOFramework.HPOBaseClasses import Hyperpipe, PipelineElement
from DataLoading.DataLoader import MNISTLoader
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import classification_report

packed = 1
train_data, train_labels, test_data, test_labels = MNISTLoader()()

cv = KFold(n_splits=3, shuffle=True, random_state=0)
lower = np.array([-10, -10])
upper = np.array([10, 10])

pipe = Hyperpipe(
    'god',
    cv,
    optimizer='fabolas',
    optimizer_params={
        'n_init': 8, 's_min': 100, 's_max': 5000,
        'num_iterations': 10, 'subsets': [2048, 4096, 8192]
    })
pipe.add(PipelineElement.create('svc', {'C': [-10, 10, 1.0], 'gamma': [-10, 10, 0.0]}))

pipe.fit(train_data, train_labels)
predicted_labels = pipe.predict(test_data)
print(classification_report(predicted_labels, test_labels))
