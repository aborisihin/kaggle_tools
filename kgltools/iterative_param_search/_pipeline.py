""" iterative_param_search module.
Contains tools for iterative search for best parameters of model
"""

import math
from collections.abc import Iterable
from abc import ABCMeta, abstractmethod
from typing import Generic, Union, List, Tuple

import pandas as pd
from sklearn.model_selection import GridSearchCV

from ..context import KglToolsContext, KglToolsContextChild
from ..logger import Logger


__all__ = ['IPSPipeline', 'IPSStageBase', 'IPSConstantSetter', 'IPSGridSearcher', 'IPSGridFineSearcher']


class IPSPipeline(KglToolsContextChild):

    def __init__(self,
                 context: KglToolsContext,
                 estimator_class: Generic,
                 X: pd.DataFrame,
                 y: Union[pd.DataFrame, pd.Series],
                 metrics: str,
                 n_folds: int = 5,
                 stratified: bool = True,
                 shuffle: bool = True,
                 base_params: dict = {},
                 verbose: bool = True) -> None:
        super().__init__(context)
        self.estimator_class = estimator_class
        self.X = X
        self.y = y
        self.metrics = metrics
        self.n_folds = n_folds
        self.stratified = stratified
        self.shuffle = shuffle
        self.base_params = base_params
        self.random_state = self.context.random_state
        self.n_jobs = self.context.n_jobs
        self.verbose = verbose

        self.stages = []
        self.fitted_params = None
        self.logger = Logger(nesting_level=0, verbose=self.verbose)

        self.metrics_mapping = self.context.extra_dicts['metrics_mapping']
        self.estimator_parameter_limits = self.context.extra_dicts['estimator_parameter_limits']

        self.estimator_classname = '{}.{}'.format(estimator_class.__module__, estimator_class.__name__)
        self.estimator_metrics = self.metrics_mapping[metrics][self.estimator_classname]

        self.base_params['random_state'] = self.random_state
        self.base_params['n_jobs'] = self.n_jobs

    def fit(self) -> None:
        self.logger.log('IPSPipeline ({})'.format(self.estimator_classname))
        self.logger.start_timer()

        self.fitted_params = dict()

        for stage_name, stage in self.stages:
            self.logger.log('Stage <{}>'.format(stage_name))
            stage_params = stage.process({**self.base_params, **self.fitted_params})
            self.fitted_params = {**self.fitted_params, **stage_params}

        self.logger.log_timer()

    def add_stages(self, stage_descriptors: List[Tuple[object, dict]]) -> None:
        for stage_object, stage_grid in stage_descriptors:
            self.add_stage(stage_object, stage_grid)

    def add_stage(self, stage_object: object, stage_grid: dict) -> None:
        stage_name = '{}({})'.format(type(stage_object).__name__, ','.join(list(stage_grid.keys())))

        if not isinstance(stage_object, IPSStageBase):
            self.logger.log('Can\'t add stage: {}. Must be an instance of IPSStageBase.'.format(stage_name))
            return

        if not isinstance(stage_object, IPSGridFineSearcher):
            grid_errors = self.check_parameter_list(stage_grid)
            if len(grid_errors) > 0:
                self.logger.log('Can\'t add stage: <{}>. Wrong values for {}.'.format(stage_name, grid_errors))
                return

        stage_object.set_parent_pipeline(self)
        stage_object.set_param_grid(stage_grid)
        self.stages.append((stage_name, stage_object))

    def check_parameter_list(self, param_grid: dict) -> List[str]:
        if self.estimator_classname not in self.estimator_parameter_limits:
            return []  # nothing to check
        errors = []
        for param, values_list in param_grid.items():
            if not isinstance(values_list, Iterable):
                values_list = [values_list]
            for val in values_list:
                if not self.check_parameter_value(param, val):
                    errors.append(param)
        return errors

    def check_parameter_value(self, param_name: str, param_value: float) -> bool:
        limits = self.estimator_parameter_limits[self.estimator_classname]
        if param_name not in limits:
            return True
        param_min, param_max = limits[param_name]
        if (param_min and (param_value < param_min)) or (param_max and (param_value > param_max)):
            return False
        else:
            return True


class IPSStageBase(object):

    __metaclass__ = ABCMeta

    def __init__(self) -> None:
        self.logger = Logger(nesting_level=1)

    def set_parent_pipeline(self, parent_pipeline: IPSPipeline) -> None:
        self.parent = parent_pipeline
        self.logger.verbose = self.parent.verbose

    def set_param_grid(self, param_grid: dict) -> None:
        self.param_grid = param_grid

    @abstractmethod
    def process(self, params: dict = {}) -> dict:
        pass


class IPSConstantSetter(IPSStageBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def process(self, params: dict = {}) -> dict:
        self.logger.log(self.param_grid)
        return self.param_grid


class IPSGridSearcher(IPSStageBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def process(self, params: dict = {}) -> dict:
        self.logger.start_timer()

        train_mean, train_std, test_mean, test_std = 0.0, 0.0, 0.0, 0.0

        g_search = GridSearchCV(
            estimator=self.parent.estimator_class(**params),
            param_grid=self.param_grid,
            return_train_score=True,
            cv=self.parent.n_folds,
            scoring=self.parent.metrics,
            n_jobs=self.parent.n_jobs,
            verbose=False)
        g_search.fit(self.parent.X, self.parent.y)

        train_mean = g_search.cv_results_['mean_train_score'][g_search.best_index_]
        train_std = g_search.cv_results_['std_train_score'][g_search.best_index_]
        test_mean = g_search.cv_results_['mean_test_score'][g_search.best_index_]
        test_std = g_search.cv_results_['std_test_score'][g_search.best_index_]

        self.logger.log_timer()
        self.logger.log('train: {:0.5f} (std={:0.5f})'.format(train_mean, train_std))
        self.logger.log('test: {:0.5f} (std={:0.5f})'.format(test_mean, test_std))
        self.logger.log(g_search.best_params_)

        return g_search.best_params_


class IPSGridFineSearcher(IPSStageBase):

    # {'max_depth': {'steps': 2, 'scale': '1_exp'}}
    # {'max_depth': {'steps': 3, 'scale': '2'}}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def parse_param_grid(self, params: dict) -> None:
        parsed_param_grid = dict()
        for param_name, search_dict in self.param_grid.items():
            if param_name not in params:
                self.logger.log('Parameter {} should be set before fine tuning!'.format(param_name))
                continue
            param_value = params[param_name]
            param_exp = math.floor(math.log(param_value, 10))
            if isinstance(search_dict['scale'], str):
                if '_exp' in search_dict['scale']:
                    step = float(search_dict['scale'].split('_')[0]) * (10**param_exp)
                else:
                    step = float(search_dict['scale'])
            else:
                step = search_dict['scale']
            grid = [param_value]
            grid += [param_value + (s * step) for s in range(1, search_dict['steps'] + 1)]
            grid += [param_value - (s * step) for s in range(1, search_dict['steps'] + 1)]
            grid = sorted(grid)
            if param_exp < 0.0:
                grid = [round(s, abs(param_exp)) for s in grid]
            grid = [s for s in grid if self.parent.check_parameter_value(param_name, s)]
            parsed_param_grid[param_name] = grid
        return parsed_param_grid

    def process(self, params: dict = {}) -> dict:
        self.param_grid = self.parse_param_grid(params)
        self.logger.log('Parsed params: {}'.format(self.param_grid))

        gs = IPSGridSearcher()
        gs.set_parent_pipeline(self.parent)
        gs.set_param_grid(self.param_grid)
        return gs.process(params)
