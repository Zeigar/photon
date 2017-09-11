import numpy as np
import george
import numbers

import os
import json

from functools import partial

from Framework.fabolas.GPMCMC import FabolasGPMCMC
from Framework.fabolas.Priors import EnvPrior
from Framework.fabolas.Maximizer import InformationGainPerUnitCost, Direct, MarginalizationGPMCMC
from Logging.Logger import Logger

def _quadratic_bf(x):
    return (1 - x) ** 2
def _linear_bf(x):
    return x

class Fabolas:
    def __init__(
            self,
            n_min_train_data,
            n_train_data,
            pipeline_elements,
            n_init=40,
            num_iterations=100,
            subsets=[256, 128, 64],
            burnin=100,
            chain_length=100,
            n_hypers=12,
            model_pool_size=-1, # if -1 then min(n_hypers, cpu_count)
            acquisition_pool_size=-1,
            rng=None,
            verbose_maximizer=False,
            log=None,
            **_
    ):
        assert n_init <= num_iterations, "Number of initial design point has to be <= than the number of iterations"

        if log is not None:
            if not isinstance(log, dict):
                raise ValueError("log must be a dict with keys id, path, name")
            log['id'] = int(log['id']) if 'id' in log else 0
            log['name'] = str(log['name']) if 'name' in log else 'fabolas'
            log['incumbents'] = bool(log['incumbents']) if 'incumbents' in log else False
            if 'path' not in log:
                raise ValueError("log must contain the key path")
            log['bn'] = '{name}_{id}'.format(name=log['name'], id=log['id'])
            log['path'] = os.path.realpath(os.path.join(str(log['path']), log['bn']))
            if not os.path.exists(log['path']):
                os.makedirs(log['path'])
            Logger().info("Fabolas: writing logs to "+log['path'])

        self._log = log
        self._verbose_maximizer = verbose_maximizer
        self._lower = []
        self._upper = []
        self._number_param_keys = []
        param_types = []
        self._param_dict = {}
        for key, val in pipeline_elements[0].hyperparameters.items():
            key = pipeline_elements[0].name + '__' + key
            if isinstance(val, list):
                if not len(val) >= 3 \
                        or not isinstance(val[0], numbers.Number)\
                        or not isinstance(val[1], numbers.Number)\
                        or not (val[2] is int or val[2] is float):
                    raise ValueError(
                        "Hyperparam '"+key+"' is not a list with [lowerbound, upperbound, int/float]"
                    )

                if val[0] > val[1]:
                    raise ValueError(
                        "Error for param '"+key+"'."
                        "First value must be lower bound, second value must be upper bound."
                    )
                self._number_param_keys.append(key)
                self._lower.append(val[0])
                self._upper.append(val[1])
                param_types.append(val[2])
            self._param_dict.update({key: val})

        self._param_int_indices = np.where(np.array(param_types) == int)[0]

        n_dims = len(self._lower)
        self._lower = np.array(self._lower)
        self._upper = np.array(self._upper)

        self._s_min = n_min_train_data
        self._s_max = n_train_data
        self._n_init = n_init
        self._num_iterations = num_iterations
        self._subsets = subsets
        self._rng = np.random.RandomState() if rng is None else rng

        self._X = []
        self._Y = []
        self._cost = []
        self._it = 0
        self._model_objective = None
        self._model_cost = None

        kernel = 1  # 1 = covariance amplitude

        # ARD Kernel for the configuration space
        degree = 1
        for d in range(n_dims):
            kernel *= george.kernels.Matern52Kernel(
                np.ones([1]) * 0.01,
                ndim=n_dims+1,
                dim=d
            )

        env_kernel = george.kernels.BayesianLinearRegressionKernel(
            n_dims+1,
            dim=n_dims,
            degree=degree
        )
        env_kernel[:] = np.ones([degree + 1]) * 0.1

        kernel *= env_kernel

        # Take 3 times more samples than we have hyperparameters
        if n_hypers < 2*len(kernel):
            n_hypers = 3 * len(kernel)
            if n_hypers % 2 == 1:
                n_hypers += 1

        prior = EnvPrior(
            len(kernel) + 1,
            n_ls=n_dims,
            n_lr=(degree + 1),
            rng=self._rng
        )

        self._model_objective = FabolasGPMCMC(
            kernel,
            prior=prior,
            burnin_steps=burnin,
            chain_length=chain_length,
            n_hypers=n_hypers,
            normalize_output=False,
            basis_func=_quadratic_bf,
            lower=self._lower,
            upper=self._upper,
            rng=self._rng,
            pool_size=model_pool_size
        )

        cost_degree = 1
        cost_env_kernel = george.kernels.BayesianLinearRegressionKernel(
            n_dims+1,
            dim=n_dims,
            degree=cost_degree
        )
        cost_kernel = 1  # 1 = covariance amplitude

        # ARD Kernel for the configuration space
        for d in range(n_dims):
            cost_kernel *= george.kernels.Matern52Kernel(
                np.ones([1]) * 0.01,
                ndim=n_dims+1,
                dim=d
            )

        cost_env_kernel[:] = np.ones([cost_degree + 1]) * 0.1

        cost_kernel *= cost_env_kernel

        cost_prior = EnvPrior(
            len(cost_kernel) + 1,
            n_ls=n_dims,
            n_lr=(cost_degree + 1),
            rng=rng
        )

        self._model_cost = FabolasGPMCMC(
            cost_kernel,
            prior=cost_prior,
            burnin_steps=burnin,
            chain_length=chain_length,
            n_hypers=n_hypers,
            basis_func=_linear_bf,
            normalize_output=False,
            lower=self._lower,
            upper=self._upper,
            rng=self._rng,
            pool_size=model_pool_size
        )

        # Extend input space by task variable
        extend_lower = np.append(self._lower, 0)
        extend_upper = np.append(self._upper, 1)
        is_env = np.zeros(extend_lower.shape[0])
        is_env[-1] = 1

        # Define acquisition function and maximizer
        ig = InformationGainPerUnitCost(
            self._model_objective,
            self._model_cost,
            extend_lower,
            extend_upper,
            is_env_variable=is_env,
            n_representer=50
        )
        self._acquisition_func = MarginalizationGPMCMC(ig, pool_size=acquisition_pool_size)

        direct_logfile = os.devnull if self._log is None\
            else os.path.join(self._log['path'], "maximizer_results.txt")

        self._maximizer = Direct(
            self._acquisition_func,
            extend_lower,
            extend_upper,
            verbose=self._verbose_maximizer,
            logfilename=direct_logfile,
            n_func_evals=200
        )

    def calc_config(self):
        Logger().debug('Fabolas: Starting initialization')
        for self._it in range(0, self._n_init):
            Logger().debug('Fabolas: step ' + str(self._it) + ' (init)')
            yield self._create_param_dict(self._init_models())

        self._X = np.array(self._X)
        self._Y = np.array(self._Y)
        self._cost = np.array(self._cost)

        Logger().debug('Fabolas: Starting optimization')
        for self._it in range(self._n_init, self._num_iterations):
            Logger().debug('Fabolas: step ' + str(self._it) + ' (opt)')
            yield self._create_param_dict(self._optimize_config())

        Logger().debug('Fabolas: Final config')
        self._model_objective.train(self._X, self._Y, do_optimize=True)
        yield self._create_param_dict(self.get_incumbent())

    def process_result(self, config, subset_frac, score, cost):
        # We're done
        if self._it >= self._num_iterations:
            return

        score = 1-score

        config_dict = config # preserve for logging

        # init-loop
        if self._it < self._n_init:
            config = self._get_params_from_dict(config)
            self._X.append(np.append(config, self._transform(self._s_max/subset_frac)))
            self._Y.append(np.log(score))  # Model the target function on a logarithmic scale
            self._cost.append(np.log(cost))  # Model the cost on a logarithmic scale

        # opt-loop
        else:
            config = np.array(self._get_params_from_dict(config))
            config = np.append(config, subset_frac)
            self._X = np.concatenate((self._X, config[None, :]), axis=0)
            self._Y = np.concatenate((self._Y, np.log(np.array([score]))), axis=0)  # Model the target function on a logarithmic scale
            self._cost = np.concatenate((self._cost, np.log(np.array([cost]))), axis=0)  # Model the cost function on a logarithmic scale

        self._generate_log(config_dict, subset_frac, score, cost)

    def get_incumbent(self):
        # This final configuration should be the best one
        final_config, _ = self._projected_incumbent_estimation(
            self._model_objective, self._X[:, :-1], proj_value=1
        )
        return final_config[:-1].tolist(), 1  # subset is the whole data-set

    def _adjust_param_types(self, params):
        for i in self._param_int_indices:
            params[i] = int(np.round(params[i]))
        return params

    def _create_param_dict(self, params):
        params, s = params
        params = self._adjust_param_types(np.exp(params))
        self._param_dict.update(
            dict(zip(
                self._number_param_keys,
                params
            ))
        )
        return self._param_dict, s

    def _get_params_from_dict(self, pdict):
        params = []
        for key in self._number_param_keys:
            params.append(pdict[key])
        return np.log(params)


    def _init_models(self):
        s = self._subsets[self._it % len(self._subsets)]
        x = self._init_random_uniform(self._lower, self._upper, 1)[0]
        return x, s

    def _optimize_config(self):
        # Train models
        Logger().debug("Fabolas: Train model_objective")
        self._model_objective.train(self._X, self._Y, do_optimize=True)
        Logger().debug("Fabolas: Train model_cost")
        self._model_cost.train(self._X, self._cost, do_optimize=True)

        # Maximize acquisition function
        Logger().debug("Fabolas: Update acquisition func")
        self._acquisition_func.update(self._model_objective, self._model_cost)
        Logger().debug("Fabolas: Generate new config by maximizing")
        new_x = self._maximizer.maximize()

        s = self._s_max/self._retransform(new_x[-1])
        Logger().debug("Fabolas: config generation done for this step")

        return new_x[:-1], int(s)

    def _projected_incumbent_estimation(self, model, X, proj_value=1):
        projection = np.ones([X.shape[0], 1]) * proj_value
        X_projected = np.concatenate((X, projection), axis=1)

        m, _ = model.predict(X_projected)

        best = np.argmin(m)
        incumbent = X_projected[best]
        incumbent_value = m[best]

        return incumbent, incumbent_value

    def _generate_log(self, conf, subset, result, cost):
        if self._log is None:
            return

        Logger().debug("Fabolas: generating log")
        l = {
            'config': conf,
            'subset_frac': subset,
            'config_result': result,
            'config_cost': cost,
            'iteration': self._it,
            'operation': 'init' if self._it < self._n_init else 'opt'
        }
        if self._log['incumbents']:
            if self._it < self._n_init:
                best_i = np.argmin(self._Y)
                l['incumbents'], _ = self._create_param_dict((self._X[best_i][:-1], 1))
                l['incumbents_estimated_performance'] = -1
            else:
                inc, inc_val = self._projected_incumbent_estimation(self._model_objective, self._X[:, :-1])
                l['incumbents'], _ = self._create_param_dict(inc[:-1], 1)
                l['incumbents_estimated_performance'] = inc_val

        with open(os.path.join(
                self._log['path'],
                self._log['bn']+'_it{it}.json'.format(it=self._it)
        ), 'w') as f:
            json.dump(l, f)

    def _transform(self, s):
        s_transform = (np.log2(s) - np.log2(self._s_min)) \
                      / (np.log2(self._s_max) - np.log2(self._s_min))
        return s_transform

    def _retransform(self, s_transform):
        s = np.rint(2**(s_transform * (np.log2(self._s_max) \
                                       - np.log2(self._s_min)) \
                        + np.log2(self._s_min)))
        return int(s)

    def _init_random_uniform(self, lower, upper, n_points):
        """
        Samples N data points uniformly.

        Parameters
        ----------
        lower: np.ndarray (D)
            Lower bounds of the input space
        upper: np.ndarray (D)
            Upper bounds of the input space
        n_points: int
            The number of initial data points
        rng: np.random.RandomState
                Random number generator
        Returns
        -------
        np.ndarray(N,D)
            The initial design data points
        """

        n_dims = lower.shape[0]

        return np.array([self._rng.uniform(lower, upper, n_dims) for _ in range(n_points)])
