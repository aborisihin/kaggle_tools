""" iterative_param_search module.
Contains tools for iterative search for best parameters of model
"""

from typing import Generic, Type, Union

import pandas as pd

from kgltools.logger import Logger

__all__ = ['IPSPipeline', 'IPSStage']


class IPSPipeline():

    def __init__(self, estimator_class: Generic, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series], metrics: str, stages: list = [], n_folds: int = 3, stratified: bool = True, shuffle: bool = True, base_params: dict = {}, random_state: int = 0, verbose: bool = True) -> None:
        self.estimator_class = estimator_class
        self.X = X
        self.y = y
        self.metrics = metrics
        self.stages = stages
        self.n_folds = n_folds
        self.metrics = metrics
        self.stratified = stratified
        self.shuffle = shuffle
        self.base_params = base_params
        self.random_state = random_state
        self.processed_params = dict()
        self.verbose = verbose
        self.logger = Logger(stage_name='IPSPipeline ({})'.format(type(estimator_class()).__name__), nesting_level=0)

    def fit(self) -> bool:
        if self.verbose:
            self.logger.init()
        return True


class IPSStageBase():

    def __init__(self, parent_pipeline: IPSPipeline, stage_name: str, param_grid: dict = {}) -> None:
        self.parent = parent_pipeline
        self.param_grid = param_grid
        self.verbose = self.parent.verbose
        self.logger = Logger(stage_name=stage_name, nesting_level=1)

    def process(self) -> dict:
        if self.verbose:
            self.logger.init()
        return self.parent.base_params
