
from itertools import product, tee
import numpy as np
import datetime
import george

from .fabolas.GPMCMC import FabolasGPMCMC, MarginalizationGPMCMC
from .fabolas.Priors import EnvPrior
from .fabolas.Maximizer import InformationGainPerUnitCost, Direct


class GridSearchOptimizer(object):
    def __init__(self):
        self.param_grid = []
        self.pipeline_elements = None
        self.parameter_iterable = None
        self.next_config = self.next_config_generator()

    def prepare(self, pipeline_elements):
        self.pipeline_elements = pipeline_elements
        self.next_config = self.next_config_generator()
        possible_configurations = []
        for p_element in self.pipeline_elements:
            if p_element.config_grid:
                possible_configurations.append(p_element.config_grid)
        if len(possible_configurations) == 1:
            self.param_grid = [[i] for i in possible_configurations[0]]
        else:
            self.param_grid = product(*possible_configurations)

    def next_config_generator(self):
        for parameters in self.param_grid:
            param_dict = {}
            for item in parameters:
                param_dict.update(item)
            yield param_dict

    def evaluate_recent_performance(self, config, performance):
        # influence return value of next_config
        pass


class RandomGridSearchOptimizer(GridSearchOptimizer):

    def __init__(self, k=None):
        super(RandomGridSearchOptimizer, self).__init__()
        self.k = k

    def prepare(self, pipeline_elements):
        super(RandomGridSearchOptimizer, self).prepare(pipeline_elements)
        self.param_grid = list(self.param_grid)
        # create random chaos in list
        np.random.shuffle(self.param_grid)
        if self.k is not None:
            self.param_grid = self.param_grid[0:self.k]


class TimeBoxedRandomGridSearchOptimizer(RandomGridSearchOptimizer):

    def __init__(self, limit_in_minutes=60):
        super(TimeBoxedRandomGridSearchOptimizer, self).__init__()
        self.limit_in_minutes = limit_in_minutes
        self.start_time = None
        self.end_time = None

    def next_config_generator(self):
        if self.start_time is None:
            self.start_time = datetime.datetime.now()
            self.end_time = self.start_time + datetime.timedelta(minutes=self.limit_in_minutes)
        for parameters in super(TimeBoxedRandomGridSearchOptimizer, self).next_config_generator():
            if datetime.datetime.now() < self.end_time:
                yield parameters


class FabolasOptimizer(object):
    def __init__(self, lower, upper, s_min, s_max,
            n_init=40, num_iterations=100, subsets=[256, 128, 64],
            burnin=100, chain_length=100, n_hypers=12, rng=None,
            score='accuracy'):

        assert n_init <= num_iterations, "Number of initial design point has to be <= than the number of iterations"
        assert lower.shape[0] == upper.shape[0], "Dimension miss match between upper and lower bound"

        self._lower = lower
        self._upper = upper
        self._s_min = s_min
        self._s_max = s_max
        self._n_init = n_init
        self._num_iterations = num_iterations
        self._subsets = subsets
        self._burnin = burnin
        self._chain_length = chain_length
        self._n_hypers = n_hypers
        self._rng = np.random.RandomState() if rng is None else rng

        self._X = []
        self._Y = []
        self._cost = []
        self._it_counter = 0
        self._model_objective = None
        self._model_cost = None

        self._kernel = 1  # 1 = covariance amplitude

        n_dims = lower.shape[0]

        # ARD Kernel for the configuration space
        degree = 1
        for d in range(n_dims):
            self._kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                        ndim=n_dims+1, dim=d)

        self._prior = EnvPrior(len(self._kernel) + 1, n_ls=n_dims,
                                 n_lr=(degree + 1), rng=self._rng)

        cost_degree = 1
        cost_env_kernel = george.kernels.BayesianLinearRegressionKernel(n_dims+1,
                                                                     dim=n_dims,
                                                                     degree=cost_degree)
        cost_env_kernel[:] = np.ones([cost_degree + 1]) * 0.1

        self._cost_kernel *= cost_env_kernel

        self._cost_prior = EnvPrior(len(self._cost_kernel) + 1,
                                    n_ls=n_dims,
                                    n_lr=(self._cost_degree + 1),
                                    rng=rng)

        self.next_config = self.next_config_generator()

    def prepare(self, pipeline_elements):
        self._X = []
        self._Y = []
        self._cost = []

        quadratic_bf = lambda x: (1 - x) ** 2
        linear_bf = lambda x: x

        self._model_objective = FabolasGPMCMC(self._kernel,
                                        prior=self._prior,
                                        burnin_steps=self._burnin,
                                        chain_length=self._chain_length,
                                        n_hypers=self._n_hypers,
                                        normalize_output=False,
                                        basis_func=quadratic_bf,
                                        lower=self._lower,
                                        upper=self._upper,
                                        rng=self._rng)

        self._model_cost = FabolasGPMCMC(self._cost_kernel,
                                   prior=self._cost_prior,
                                   burnin_steps=self._burnin,
                                   chain_length=self._chain_length,
                                   n_hypers=self._n_hypers,
                                   basis_func=self._linear_bf,
                                   normalize_output=False,
                                   lower=self._lower,
                                   upper=self._upper,
                                   rng=self._rng)

        # Extend input space by task variable
        extend_lower = np.append(self._lower, 0)
        extend_upper = np.append(self._upper, 1)
        is_env = np.zeros(extend_lower.shape[0])
        is_env[-1] = 1

        # Define acquisition function and maximizer
        ig = InformationGainPerUnitCost(self._model_objective,
                                        self._model_cost,
                                        extend_lower,
                                        extend_upper,
                                        is_env_variable=is_env,
                                        n_representer=50)
        self._acquisition_func = MarginalizationGPMCMC(ig)
        self._maximizer = Direct(self._acquisition_func, extend_lower, extend_upper, verbose=True, n_func_evals=200)

    def next_config_generator(self):
        for self._it in range(self._n_init):
            x = self._init_random_uniform(self._lower, self._upper, 1, self._rng)[0]
            self._X.append(x)
            yield x

        self._X = np.array(self._X)
        self._Y = np.array(self._Y)
        self._cost = np.array(self._cost)

        for self._it in range(self._n_init, self._num_iterations):
            # Train models
            self._model_objective.train(self._X, self._Y, do_optimize=True)
            self._model_cost.train(self._X, self._cost, do_optimize=True)

            # Estimate incumbent by projecting all observed points to the task of interest and
            # pick the point with the lowest mean prediction
            incumbent, incumbent_value = self._projected_incumbent_estimation(
                self._model_objective, self._X[:, :-1], proj_value=1
            )

            # Maximize acquisition function
            self._acquisition_func.update(self._model_objective, self._model_cost)
            new_x = self._maximizer.maximize()
            self._X = np.concatenate((self._sX, new_x[None, :]), axis=0)

            yield new_x[:-1]

    def evaluate_recent_performance(self, config, performance, times):
        score = performance[1]
        cost = performance[2]+performance[3]

        if self._it < self._n_init:
            #init-stuff
            self._Y.append(np.log(score))  # Model the target function on a logarithmic scale
            self._cost.append(np.log(cost))  # Model the cost on a logarithmic scale
            return

        self._Y = np.concatenate((self._Y, np.log(np.array([score]))), axis=0)  # Model the target function on a logarithmic scale
        self._cost = np.concatenate((self._cost, np.log(np.array([cost]))), axis=0)  # Model the cost function on a logarithmic scale

    def _projected_incumbent_estimation(model, X, proj_value=1):
        projection = np.ones([X.shape[0], 1]) * proj_value
        X_projected = np.concatenate((X, projection), axis=1)

        m, _ = model.predict(X_projected)

        best = np.argmin(m)
        incumbent = X_projected[best]
        incumbent_value = m[best]

        return incumbent, incumbent_value

    def _transform(s, s_min, s_max):
        s_transform = (np.log2(s) - np.log2(s_min)) / (np.log2(s_max) - np.log2(s_min))
        return s_transform


    def _retransform(s_transform, s_min, s_max):
        s = np.rint(2**(s_transform * (np.log2(s_max) - np.log2(s_min)) + np.log2(s_min)))
        return int(s)

    def _init_random_uniform(lower, upper, n_points, rng=None):
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

        if rng is None:
            rng = np.random.RandomState(np.random.randint(0, 10000))

        n_dims = lower.shape[0]

        return np.array([rng.uniform(lower, upper, n_dims) for _ in range(n_points)])


# class AnyHyperparamOptimizer(object):
#     def __init__(self, params_to_optimize):
#         self.params_to_optimize = params_to_optimize
#         self.next_config = self.next_config_generator()
#         self.next_config_to_try = 1
#     def prepare(self, pipeline_elements):
#         pass
#
#     def next_config_generator(self):
#         yield self.next_config_to_try
#
#     def evaluate_recent_performance(self, config, performance):
#         # according to the last performance for the given config,
#         # the next item should be chosen wisely
#         self.next_config_to_try = self.params_to_optimize(2)
