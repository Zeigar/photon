# Checking if finding best model after hp search (i.e. eval finals performance) works as intended

import random
import unittest

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Framework.PhotonBase import PipelineElement, Hyperpipe


class CVTestsLocalSearchTrue(unittest.TestCase):
    __X = None
    __y = None

    def setUp(self):
        from sklearn.datasets import load_breast_cancer
        dataset = load_breast_cancer()
        self.__X = dataset.data
        self.__y = dataset.target
        random.seed(42)

    def testCaseA(self):
        pca_n_components = [2, 5]
        svc_c = [.1, 1]
        #svc_kernel = ['rbf']
        svc_kernel = ['rbf','linear']

        # SET UP HYPERPIPE
        my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search',
                            metrics=['accuracy', 'precision', 'f1_score'],
                            hyperparameter_specific_config_cv_object=KFold(n_splits=2, random_state=3),
                            hyperparameter_search_cv_object=KFold(n_splits=3, random_state=3),
                            eval_final_performance=True)

        my_pipe.add(PipelineElement.create('standard_scaler'))
        my_pipe.add(PipelineElement.create('pca', {'n_components': pca_n_components}))
        my_pipe.add(PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel}))

        # START HYPERPARAMETER SEARCH
        my_pipe.fit(self.__X, self.__y)

        # print(my_pipe.test_performances)
        # print(my_pipe.test_performances['accuracy'])

        pipe_results = {'train': [], 'test': []}
        for i in range(len(my_pipe.performance_history_list)):
            pipe_results['train'].extend(
                my_pipe.performance_history_list[i]['accuracy_folds']['train'])
            pipe_results['test'].extend(
                my_pipe.performance_history_list[i]['accuracy_folds']['test'])


        print('\n\n')
        print('Running sklearn version...')
        cv_outer = KFold(n_splits=3, random_state=3)
        cv_inner_1 = KFold(n_splits=2, random_state=3)

        opt_tr_acc = []
        opt_test_acc = []

        for train_1, test in cv_outer.split(self.__X):
            data_train_1 = self.__X[train_1]
            data_test = self.__X[test]
            y_train_1 = self.__y[train_1]
            y_test = self.__y[test]
            sk_results = {'train': [], 'test': [], 'train_mean': [], 'test_mean': [], 'test_std': [], 'train_std': []}
            config = {'C': [], 'n_comp': [], 'kernel': []}
            for n_comp in pca_n_components:
                for c in svc_c:
                    for current_kernel in svc_kernel:
                        config['C'].extend([c])
                        config['n_comp'].extend([n_comp])
                        config['kernel'].extend([current_kernel])

                        tr_acc = []
                        val_acc = []

                        for train_2, val_1 in cv_inner_1.split(
                                data_train_1):
                            data_train_2 = data_train_1[train_2]
                            data_val_1 = data_train_1[val_1]
                            y_train_2 = y_train_1[train_2]
                            y_val_1 = y_train_1[val_1]

                            my_scaler = StandardScaler()
                            my_scaler.fit(data_train_2)
                            data_train_2 = my_scaler.transform(data_train_2)
                            data_val_1 = my_scaler.transform(data_val_1)

                            # Run PCA
                            my_pca = PCA(n_components=n_comp)
                            my_pca.fit(data_train_2)
                            data_tr_2_pca = my_pca.transform(data_train_2)
                            data_val_1_pca = my_pca.transform(data_val_1)

                            # Run SVC
                            my_svc = SVC(kernel=current_kernel, C=c)
                            my_svc.fit(data_tr_2_pca, y_train_2)

                            tr_acc.append(my_svc.score(data_tr_2_pca, y_train_2))
                            val_acc.append(my_svc.score(data_val_1_pca, y_val_1))
                            print('n_components: ', n_comp, 'kernel:',
                                  current_kernel, 'c:', c)
                            print('Training 2:', tr_acc[-1],
                                  'Validation 1:', val_acc[-1])

                        sk_results['train'].extend(tr_acc)
                        sk_results['test'].extend(val_acc)
                        sk_results['train_mean'].extend([np.mean(tr_acc)])
                        sk_results['test_mean'].extend([np.mean(val_acc)])
                        sk_results['test_std'].extend([np.std(val_acc)])
                        sk_results['train_std'].extend([np.std(tr_acc)])

            # find best config
            combined_metric = np.add(sk_results['test_mean'], np.subtract(1, sk_results['test_std']))
            best_config_id = np.argmax(combined_metric)

            # fit optimum pipe
            my_scaler = StandardScaler()
            my_scaler.fit(data_train_1)
            data_train_1 = my_scaler.transform(data_train_1)
            data_test = my_scaler.transform(data_test)

            # Run PCA
            best_n_components = config['n_comp'][best_config_id]
            my_pca = PCA(n_components=best_n_components)

            my_pca.fit(data_train_1)
            data_tr_1_pca = my_pca.transform(data_train_1)
            data_test_pca = my_pca.transform(data_test)

            best_svc_kernel = config['kernel'][best_config_id]
            best_svc_C = config['C'][best_config_id]
            # SKLEARN finds C=0.1 and kernel = linear while
            # PIPE finds C=1 and kernel = rbf
            # my_svc = SVC(kernel='rbf', C=1)
            my_svc = SVC(kernel=best_svc_kernel, C=best_svc_C)
            my_svc.fit(data_tr_1_pca, y_train_1)

            opt_train_predictions = my_svc.predict(data_tr_1_pca)
            opt_test_predictions = my_svc.predict(data_test_pca)
            opt_tr_acc.append(accuracy_score(y_train_1, opt_train_predictions))
            opt_test_acc.append(accuracy_score(y_test, opt_test_predictions))

        print('Best Sklearn config:')
        print('n_components: ', best_n_components)
        print('svc_kernel: ', best_svc_kernel)
        print('svc_C: ', best_svc_C)
        print('\nCompare results of last iteration (outer cv)...')
        print('SkL  Train:', sk_results['train'])
        print('Pipe Train:', pipe_results['train'])
        print('SkL  Test: ', sk_results['test'])
        print('Pipe Test: ', pipe_results['test'])
        print('\nEval final performance:')
        print('Pipe final perf:', my_pipe.test_performances['accuracy'])
        print('Sklearn final perf:', opt_test_acc)

        self.assertEqual(sk_results['test'], pipe_results['test'])
        self.assertEqual(sk_results['train'], pipe_results['train'])
        self.assertEqual(opt_test_acc, my_pipe.test_performances['accuracy'])


if __name__ == '__main__':
    unittest.main()