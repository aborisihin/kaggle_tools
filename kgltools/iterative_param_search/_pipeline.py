""" iterative_param_search module.
Contains tools for iterative search for best parameters of model
"""

from typing import Generic, Union, List, Tuple
from abc import ABCMeta, abstractmethod

import pandas as pd

from ..logger import Logger


__all__ = ['IPSPipeline', 'IPSStageBase']


class IPSPipeline(object):

    def __init__(self,
                 estimator_class: Generic,
                 X: pd.DataFrame,
                 y: Union[pd.DataFrame, pd.Series],
                 metrics: str,
                 n_folds: int = 3,
                 stratified: bool = True,
                 shuffle: bool = True,
                 base_params: dict = {},
                 random_state: int = 0,
                 verbose: bool = True) -> None:
        self.estimator_class = estimator_class
        self.X = X
        self.y = y
        self.metrics = metrics
        self.n_folds = n_folds
        self.metrics = metrics
        self.stratified = stratified
        self.shuffle = shuffle
        self.base_params = base_params
        self.random_state = random_state
        self.verbose = verbose

        self.stages = []
        self.best_params = base_params
        self.logger = Logger(nesting_level=0)

    def add_stages(self, stage_descriptors: List[Tuple[str, object, dict]]) -> None:
        for stage_name, stage_object, stage_grid in stage_descriptors:
            self.add_stage(stage_name, stage_object, stage_grid)

    def add_stage(self, stage_name: str, stage_object: object, stage_grid: dict) -> None:
        # if IPSStageBase not in inspect.getmro(stage_type):
        if not isinstance(stage_object, IPSStageBase):
            self.logger.log('Can\'t add stage: {}. Must be an instance of IPSStageBase.'.format(stage_name))
            return

        stage_object.set_parent_pipeline(self)
        stage_object.set_param_grid(stage_grid)
        self.stages.append((stage_name, stage_object))

    def fit(self) -> None:
        if self.verbose:
            self.logger.log('IPSPipeline ({})'.format(type(self.estimator_class()).__name__))
            self.logger.start_timer()

        for stage_name, stage in self.stages:
            if self.verbose:
                self.logger.log('Stage <{}>'.format(stage_name))
            stage_params = stage.process(self.best_params)
            self.best_params = {**self.best_params, **stage_params}

        if self.verbose:
            self.logger.log_timer()


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


class IPSGridSearcher(IPSStageBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def process(self, params: dict = {}) -> dict:
        pass


class IPSGridFineSearcher(IPSStageBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def process(self, params: dict = {}) -> dict:
        pass
