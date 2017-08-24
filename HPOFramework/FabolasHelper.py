import numpy as np
import george
import numbers

from .fabolas.GPMCMC import FabolasGPMCMC
from .fabolas.Priors import EnvPrior
from .fabolas.Maximizer import InformationGainPerUnitCost, Direct, MarginalizationGPMCMC

class FabolasHelper:
    def __init__(self, s_min, s_max, pipeline_elements,
                n_init=40, num_iterations=100, subsets=[256, 128, 64],
                burnin=100, chain_length=100, n_hypers=12, rng=None, **_):
        assert n_init <= num_iterations, "Number of initial design point has to be <= than the number of iterations"

        self._lower = []
        self._upper = []
        self._number_param_keys = []
        self._param_dict = {}
        for key, val in pipeline_elements[0].hyperparameters.items():
            key = pipeline_elements[0].name + '__' + key
            if isinstance(val, list):
                if not len(val) >= 2 \
                        or not isinstance(val[0], numbers.Number)\
                        or not isinstance(val[1], numbers.Number):
                    raise ValueError(
                        "Hyperparam '"+key+"' is not a list with [lowerbound, upperbound, *]"
                    )

                if val[0] > val[1]:
                    raise ValueError(
                        "Error for param '"+key+"'."
                        "First value must be lower bound, second value must be upper bound."
                    )
                self._number_param_keys.append(key)
                self._lower.append(val[0])
                self._upper.append(val[1])
            self._param_dict.update({key: val})

        n_dims = len(self._lower)
        self._lower = np.array(self._lower)
        self._upper = np.array(self._upper)

        self._s_min = s_min
        self._s_max = s_max
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

        quadratic_bf = lambda x: (1 - x) ** 2
        linear_bf = lambda x: x

        # ARD Kernel for the configuration space
        degree = 1
        for d in range(n_dims):
            kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                        ndim=n_dims+1, dim=d)

        env_kernel = george.kernels.BayesianLinearRegressionKernel(n_dims+1,
                                                                   dim=n_dims,
                                                                   degree=degree)
        env_kernel[:] = np.ones([degree + 1]) * 0.1

        kernel *= env_kernel

        prior = EnvPrior(len(kernel) + 1, n_ls=n_dims,
                                 n_lr=(degree + 1), rng=self._rng)

        self._model_objective = FabolasGPMCMC(kernel,
                                        prior=prior,
                                        burnin_steps=burnin,
                                        chain_length=chain_length,
                                        n_hypers=n_hypers,
                                        normalize_output=False,
                                        basis_func=quadratic_bf,
                                        lower=self._lower,
                                        upper=self._upper,
                                        rng=self._rng)

        cost_degree = 1
        cost_env_kernel = george.kernels.BayesianLinearRegressionKernel(n_dims+1,
                                                                     dim=n_dims,
                                                                     degree=cost_degree)
        cost_kernel = 1  # 1 = covariance amplitude

        # ARD Kernel for the configuration space
        for d in range(n_dims):
            cost_kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                         ndim=n_dims+1, dim=d)

        cost_env_kernel[:] = np.ones([cost_degree + 1]) * 0.1

        cost_kernel *= cost_env_kernel

        cost_prior = EnvPrior(len(cost_kernel) + 1,
                                    n_ls=n_dims,
                                    n_lr=(cost_degree + 1),
                                    rng=rng)

        self._model_cost = FabolasGPMCMC(cost_kernel,
                                   prior=cost_prior,
                                   burnin_steps=burnin,
                                   chain_length=chain_length,
                                   n_hypers=n_hypers,
                                   basis_func=linear_bf,
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

    def calc_config(self):
        print('Fabolas: Starting initialization')
        for self._it in range(0, self._n_init):
            print('Fabolas: step ' + str(self._it) + ' (init)')
            yield self._create_param_dict(*self._init_models())

        self._X = np.array(self._X)
        self._Y = np.array(self._Y)
        self._cost = np.array(self._cost)

        print('Fabolas: Starting optimization')
        for self._it in range(self._n_init, self._num_iterations):
            print('Fabolas: step ' + str(self._it) + ' (opt)')
            yield self._create_param_dict(*self._optimize_config())

        print('Fabolas: Final config')
        yield self._create_param_dict(*self.get_incumbent())

    def process_result(self, score, cost):
        # We're done
        if self._it >= self._num_iterations:
            return

        score = 1-score
        # init-loop
        if self._it < self._n_init:
            self._Y.append(np.log(score))  # Model the target function on a logarithmic scale
            self._cost.append(np.log(cost))  # Model the cost on a logarithmic scale

        # opt-loop
        else:
            self._Y = np.concatenate((self._Y, np.log(np.array([score]))), axis=0)  # Model the target function on a logarithmic scale
            self._cost = np.concatenate((self._cost, np.log(np.array([cost]))), axis=0)  # Model the cost function on a logarithmic scale

    def get_incumbent(self):
        # This final configuration should be the best one
        final_config, _ = self._projected_incumbent_estimation(
            self._model_objective, self._X[:, :-1], proj_value=1
        )
        return final_config[:-1].toList(), 1  # subset is the whole data-set

    def _create_param_dict(self, params, s):
        self._param_dict.update(
            dict(zip(
                self._number_param_keys,
                np.exp(params)
            ))
        )
        return self._param_dict, s

    def _init_models(self):
        s = self._subsets[self._it % len(self._subsets)]
        x = self._init_random_uniform(self._lower, self._upper, 1, self._rng)[0]
        self._X.append(np.append(x, self._transform(self._s_max/s)))
        return x, s

    def _optimize_config(self):
        # Train models
        self._model_objective.train(self._X, self._Y, do_optimize=True)
        self._model_cost.train(self._X, self._cost, do_optimize=True)

        # Maximize acquisition function
        self._acquisition_func.update(self._model_objective, self._model_cost)
        new_x = self._maximizer.maximize()
        s = self._s_max/self._retransform(new_x[-1])
        self._X = np.concatenate((self._X, new_x[None, :]), axis=0)

        return new_x[:-1], s

    def _projected_incumbent_estimation(self, model, X, proj_value=1):
        projection = np.ones([X.shape[0], 1]) * proj_value
        X_projected = np.concatenate((X, projection), axis=1)

        m, _ = model.predict(X_projected)

        best = np.argmin(m)
        incumbent = X_projected[best]
        incumbent_value = m[best]

        return incumbent, incumbent_value

    def _transform(self, s):
        s_transform = (np.log2(s) - np.log2(self._s_min)) \
                      / (np.log2(self._s_max) - np.log2(self._s_min))
        return s_transform

    def _retransform(self, s_transform):
        s = np.rint(2**(s_transform * (np.log2(self._s_max) \
                                       - np.log2(self._s_min)) \
                        + np.log2(self._s_min)))
        return int(s)

    def _init_random_uniform(self, lower, upper, n_points, rng=None):
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