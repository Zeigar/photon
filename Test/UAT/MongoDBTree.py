import sys
sys.path.append("..")
import numpy as np
from Framework.PhotonBase import Hyperpipe, PipelineElement
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.datasets import load_breast_cancer

#  -----------> calculate something ------------------- #

# LOAD DATA
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
print(np.sum(y)/len(y))

# BUILD PIPELINE
manager = Hyperpipe('test_manager',
                    optimizer='timeboxed_random_grid_search', optimizer_params={'limit_in_minutes': 1},
                    outer_cv=ShuffleSplit(test_size=0.2, n_splits=1),
                    inner_cv=KFold(n_splits=10, shuffle=True), best_config_metric='accuracy',
                    metrics=['accuracy', 'precision', 'recall', "f1_score"], logging=False, eval_final_performance=True, verbose=2)

manager.add(PipelineElement.create('standard_scaler', test_disabled=True))
svm = PipelineElement.create('svc', hyperparameters={'C': [0.5, 1], 'kernel': ['linear']})
manager.add(svm)
manager.fit(X, y)

#  -----------> Result Tree generated ------------------- #
result_tree = manager.result_tree

result_tree.write_to_db()

# THE END
debugging = True