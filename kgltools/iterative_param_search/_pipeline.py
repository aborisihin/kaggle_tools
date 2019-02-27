""" iterative_param_search module.
Contains tools for iterative search for best parameters of model
"""

import os
import json
from collections.abc import Iterable
from abc import ABCMeta, abstractmethod
from typing import Generic, Union, List, Tuple

import pandas as pd
from sklearn.model_selection import GridSearchCV

from ..context import KglToolsContext
from ..logger import Logger


__all__ = ['IPSPipeline', 'IPSStageBase', 'IPSConstantSetter', 'IPSGridSearcher', 'IPSGridFineSearcher']


class IPSPipeline(object):
    # from context:
    # data, random_state, n_jobs, metrics_mapping, estimator_parameters_limits

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
                 random_state: int = 0,
                 n_jobs: int = -1,
                 verbose: bool = True) -> None:
        self.estimator_class = estimator_class
        self.X = X
        self.y = y
        self.metrics = metrics
        self.n_folds = n_folds
        self.stratified = stratified
        self.shuffle = shuffle
        self.base_params = base_params
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.stages = []
        self.best_params = None
        self.logger = Logger(nesting_level=0)

        ###############################################
        cur_path = os.path.dirname(os.path.abspath(__file__))
        settings_dir = os.path.abspath(os.path.join(cur_path, '../../settings'))

        # metrics_mapping
        mm_path = os.path.join(settings_dir, 'metrics_mapping.json')
        if not os.path.exists(mm_path):
            print('Settings file metrics_mapping.json is not exist!')
            self.metrics_mapping = None
        else:
            with open(mm_path, 'r') as mm_file:
                self.metrics_mapping = json.load(mm_file)

        # estimator_parameter_limits
        epl_path = os.path.join(settings_dir, 'estimator_parameter_limits.json')
        if not os.path.exists(epl_path):
            print('Settings file estimator_parameter_limits.json is not exist!')
            self.estimator_parameter_limits = None
        else:
            with open(epl_path, 'r') as epl_file:
                self.estimator_parameter_limits = json.load(epl_file)
        ###############################################

        self.estimator_classname = '{}.{}'.format(estimator_class.__module__, estimator_class.__name__)
        self.estimator_metrics = self.metrics_mapping[metrics][self.estimator_classname]

    def fit(self) -> None:
        if self.verbose:
            self.logger.log('IPSPipeline ({})'.format(self.estimator_classname))
            self.logger.start_timer()

        self.best_params = self.base_params

        for stage_name, stage in self.stages:
            if self.verbose:
                self.logger.log('Stage <{}>'.format(stage_name))
            stage_params = stage.process(self.best_params)
            self.best_params = {**self.best_params, **stage_params}

        if self.verbose:
            self.logger.log_timer()

    def add_stages(self, stage_descriptors: List[Tuple[object, dict]]) -> None:
        for stage_object, stage_grid in stage_descriptors:
            self.add_stage(stage_object, stage_grid)

    def add_stage(self, stage_object: object, stage_grid: dict) -> None:
        stage_name = '{}({})'.format(type(stage_object).__name__, ','.join(list(stage_grid.keys())))

        if not isinstance(stage_object, IPSStageBase):
            self.logger.log('Can\'t add stage: {}. Must be an instance of IPSStageBase.'.format(stage_name))
            return

        grid_errors = self.check_parameter_limits(stage_grid)
        if len(grid_errors) > 0:
            self.logger.log('Can\'t add stage: <{}>. Wrong values for {}.'.format(stage_name, grid_errors))
            return

        stage_object.set_parent_pipeline(self)
        stage_object.set_param_grid(stage_grid)
        self.stages.append((stage_name, stage_object))

    def check_parameter_limits(self, param_grid: dict) -> List[str]:
        if self.estimator_classname not in self.estimator_parameter_limits:
            return [] # nothing to check
        limits = self.estimator_parameter_limits[self.estimator_classname]
        errors = []
        for param, values_list in param_grid.items():
            if param in limits:
                if not isinstance(values_list, list):
                    if not isinstance(values_list, Iterable):
                        values_list = [values_list]
                    else:
                        values_list = list(values_list)
                param_min, param_max = limits[param]
                for val in values_list:
                    if (param_min and (val < param_min)) or (param_max and (val > param_max)):
                        errors.append(param)
        return errors


class IPSStageBase(object):

    __metaclass__ = ABCMeta

    def __init__(self) -> None:
        self.logger = Logger(nesting_level=1)

    def set_parent_pipeline(self, parent_pipeline: IPSPipeline) -> None:
        self.parent = parent_pipeline

    def set_param_grid(self, param_grid: dict) -> None:
        self.param_grid = param_grid

    @abstractmethod
    def process(self, params: dict = {}) -> dict:
        pass


class IPSConstantSetter(IPSStageBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)   

    def process(self, params: dict = {}) -> dict:
        if self.parent.verbose:
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

        if self.parent.verbose:
            self.logger.log_timer()
            self.logger.log('train: {:0.5f} (std={:0.5f})'.format(train_mean, train_std))
            self.logger.log('test: {:0.5f} (std={:0.5f})'.format(test_mean, test_std))
            self.logger.log(g_search.best_params_)

        return g_search.best_params_


class IPSGridFineSearcher(IPSStageBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def process(self, params: dict = {}) -> dict:
        # import math
        # math.floor(math.log(1, 10))
        pass
