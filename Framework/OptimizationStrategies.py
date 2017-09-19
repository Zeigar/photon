import datetime
from itertools import product

import numpy as np
from time import time
import os
import json

from Framework.fabolas.Fabolas import Fabolas


class GridSearchOptimizer(object):
    def __init__(self, log=None):
        self.param_grid = []
        self.pipeline_elements = None
        self.parameter_iterable = None
        self.next_config = self.next_config_generator()
        if log is not None:
            log['name'] = str(log['name']) if 'name' in log else self.__class__.__name__
            if 'path' not in log:
                raise ValueError("log must contain the key path")
            log['path'] = os.path.realpath(os.path.join(str(log['path']), log['name']))
            if not os.path.exists(log['path']):
                os.makedirs(log['path'])
        self.log = log

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
        it = 0
        for parameters in self.param_grid:
            start = time()
            param_dict = {}
            for item in parameters:
                param_dict.update(item)
            track = {
                'overhead_time': time()-start,
                'iteration': it
            }
            it += 1
            yield param_dict, 1, track

    def evaluate_recent_performance(self, config, performance, _, track):
        # influence return value of next_config
        if self.log is not None:
            l = {
                'config': config,
                'performance': performance,
            }
            l.update(track)
            with open(os.path.join(
                self.log['path'],
                self.log['name'] + '_it{it}.json'.format(it=l['iteration'])
            ), 'w') as f:
                json.dump(l, f)


class RandomGridSearchOptimizer(GridSearchOptimizer):

    def __init__(self, k=None, **kwargs):
        super(RandomGridSearchOptimizer, self).__init__(**kwargs)
        self.k = k

    def prepare(self, pipeline_elements):
        super(RandomGridSearchOptimizer, self).prepare(pipeline_elements)
        self.param_grid = list(self.param_grid)
        # create random chaos in list
        np.random.shuffle(self.param_grid)
        if self.k is not None:
            self.param_grid = self.param_grid[0:self.k]


class TimeBoxedRandomGridSearchOptimizer(RandomGridSearchOptimizer):

    def __init__(self, limit_in_minutes=60, **kwargs):
        super(TimeBoxedRandomGridSearchOptimizer, self).__init__(**kwargs)
        self.limit_in_minutes = limit_in_minutes
        self.start_time = None
        self.end_time = None

    def prepare(self, pipeline_elements):
        super(TimeBoxedRandomGridSearchOptimizer, self).prepare(pipeline_elements)
        self.start_time = None

    def next_config_generator(self):
        if self.start_time is None:
            self.start_time = datetime.datetime.now()
            self.end_time = self.start_time + datetime.timedelta(minutes=self.limit_in_minutes)
        for parameters in super(TimeBoxedRandomGridSearchOptimizer, self).next_config_generator():
            if datetime.datetime.now() < self.end_time:
                yield parameters, 1, None


class FabolasOptimizer(object):

    def __init__(self, **fabolas_params):
        self._pipeline_elements = None
        self._param_grid = []
        self._fabolas_params = fabolas_params
        self._fabolas = None
        self.next_config = self.next_config_generator()

    def prepare(self, pipeline_elements):
        self._fabolas_params.update({'pipeline_elements': pipeline_elements})
        self._fabolas = Fabolas(**self._fabolas_params)

    def next_config_generator(self):
        yield from self._fabolas.calc_config()

    def evaluate_recent_performance(self, config, performance, subset_frac, tracking):
        score = performance[1]
        cost = performance[2]+performance[3]+performance[4]
        self._fabolas.process_result(config, int(subset_frac), score, cost, tracking)


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
