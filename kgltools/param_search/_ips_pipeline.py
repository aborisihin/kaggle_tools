""" ips_pipeline module.
Contains tools for iterative search for best parameters of model
"""

import math
from itertools import product
from collections.abc import Iterable
from abc import ABCMeta, abstractmethod
from typing import Union, List, Tuple, Optional, Callable

import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import StratifiedKFold, KFold

from ..context import KglToolsContext
from ..logger import Logger
from ._param_searcher import ParamSearcher


__all__ = ['IPSPipeline',
           'IPSStageBase',
           'ConstantSetter',
           'GridFineSearcherBase',
           'GridSearcher',
           'ManualGridSearcher',
           'GridFineSearcher',
           'ManualGridFineSearcher']


class IPSPipeline(ParamSearcher):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stages = []
        self.estimator_parameter_limits = self.context.extra_dicts['estimator_parameter_limits']

    def fit(self) -> None:
        self.fit_next(len(self.stages))

    def fit_next(self, num_stages: Optional[int] = None) -> None:
        self.logger.log('IPSPipeline ({})'.format(self.estimator_class_name))
        self.logger.start_timer()

        stages_to_fit = [(s_name, s) for s_name, s in self.stages if s.fitted_params is None]
        if num_stages is not None:
            stages_to_fit = stages_to_fit[:num_stages]

        if len(stages_to_fit) == 0:
            self.logger.log('No stages to fit!')

        for stage_name, stage in stages_to_fit:
            self.logger.log('Stage <{}>'.format(stage_name))
            stage.fit({**self.base_params, **self.fitted_params})
            self.fitted_params.update(stage.fitted_params)

        self.logger.log_timer()

    def add_stages_list(self, stage_descriptors: List[Tuple[object, dict]]) -> None:
        for stage_object, stage_grid in stage_descriptors:
            self.add_stage(stage_object, stage_grid)

    def add_stage(self, stage_object: object, stage_grid: dict) -> None:
        stage_name = '{}({})'.format(type(stage_object).__name__, ','.join(list(stage_grid.keys())))

        if not isinstance(stage_object, IPSStageBase):
            self.logger.log('Can\'t add stage: {}. Must be an instance of IPSStageBase.'.format(stage_name))
            return

        if not isinstance(stage_object, GridFineSearcherBase):
            grid_errors = self.check_parameter_list(stage_grid)
            if len(grid_errors) > 0:
                self.logger.log('Can\'t add stage: <{}>. Wrong values for {}.'.format(stage_name, grid_errors))
                return

        stage_object.set_parent_pipeline(self)
        stage_object.set_param_grid(stage_grid)
        self.stages.append((stage_name, stage_object))

    def print_stages(self) -> None:
        stages_summary = list()
        for stage_name, stage_object in self.stages:
            stages_summary.append((stage_name, str(stage_object.fitted_params)))
        stages_summary_df = pd.DataFrame(data=stages_summary,
                                         index=range(1, len(self.stages) + 1),
                                         columns=['stage_name', 'fit_result'])
        self.logger.log('IPSPipeline ({})'.format(self.estimator_class_name))
        self.logger.log(str(stages_summary_df))

    def check_parameter_list(self, param_grid: dict) -> List[str]:
        if self.estimator_class_name not in self.estimator_parameter_limits:
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
        limits = self.estimator_parameter_limits[self.estimator_class_name]
        if param_name not in limits:
            return True
        param_min, param_max = limits[param_name]
        check_min = (param_min is None) or (param_value >= param_min)
        check_max = (param_max is None) or (param_value <= param_max)
        if check_min and check_max:
            return True
        else:
            return False


class IPSStageBase(object):

    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs) -> None:
        self.logger = Logger(nesting_level=1)
        self.fitted_params = None
        self.train_score_mean = None
        self.train_score_std = None
        self.test_score_mean = None
        self.test_score_std = None
        self.parent = None
        self.param_grid = None

    def set_parent_pipeline(self, parent_pipeline: IPSPipeline) -> None:
        self.parent = parent_pipeline
        self.logger.verbose = self.parent.verbose

    def set_param_grid(self, param_grid: dict) -> None:
        self.param_grid = param_grid

    @abstractmethod
    def fit(self, params: dict) -> None:
        pass


class ConstantSetter(IPSStageBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def fit(self, params: dict) -> None:
        self.logger.log(self.param_grid)
        self.fitted_params = self.param_grid


class GridSearcher(IPSStageBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def fit(self, params: dict) -> None:
        self.logger.start_timer()

        g_search = GridSearchCV(
            estimator=self.parent.estimator_class(**params),
            param_grid=self.param_grid,
            return_train_score=True,
            cv=self.parent.n_folds,
            scoring=self.parent.metrics,
            n_jobs=self.parent.n_jobs,
            verbose=False)
        g_search.fit(self.parent.X, self.parent.y)

        self.train_score_mean = g_search.cv_results_['mean_train_score'][g_search.best_index_]
        self.train_score_std = g_search.cv_results_['std_train_score'][g_search.best_index_]
        self.test_score_mean = g_search.cv_results_['mean_test_score'][g_search.best_index_]
        self.test_score_std = g_search.cv_results_['std_test_score'][g_search.best_index_]

        self.logger.log_timer()
        self.logger.log('train: {:0.5f} (std={:0.5f})'.format(self.train_score_mean, self.train_score_std))
        self.logger.log('test: {:0.5f} (std={:0.5f})'.format(self.test_score_mean, self.test_score_std))
        self.logger.log(g_search.best_params_)

        self.fitted_params = g_search.best_params_


class ManualGridSearcher(IPSStageBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def fit(self, params: dict) -> None:
        self.logger.start_timer()

        pg_items = sorted(self.param_grid.items())
        pg_keys, pg_values = zip(*pg_items)
        pg_list = [dict(zip(pg_keys, v)) for v in product(*pg_values)]

        if self.parent.stratified:
            cv_folds = StratifiedKFold(n_splits=self.parent.n_folds,
                                       shuffle=self.parent.shuffle,
                                       random_state=self.parent.random_state)
        else:
            cv_folds = KFold(n_splits=self.parent.n_folds,
                             shuffle=self.parent.shuffle,
                             random_state=self.parent.random_state)

        scores_list = list()
        for current_params in pg_list:
            estimator = self.parent.estimator_class(**{**params, **current_params})
            cv_result = cross_validate(estimator=estimator,
                                       X=self.parent.X,
                                       y=self.parent.y,
                                       scoring=self.parent.metrics,
                                       cv=cv_folds,
                                       n_jobs=self.parent.n_jobs,
                                       verbose=False,
                                       return_train_score=True)
            scores_list.append((cv_result['train_score'].mean(), cv_result['train_score'].std(),
                                cv_result['test_score'].mean(), cv_result['test_score'].std()))

            self.logger.log('Check {}'.format(current_params))
            self.logger.increase_level()
            self.logger.log('train: {:0.5f} (std={:0.5f})'.format(scores_list[-1][0], scores_list[-1][1]))
            self.logger.log('test: {:0.5f} (std={:0.5f})'.format(scores_list[-1][2], scores_list[-1][3]))
            self.logger.decrease_level()

        test_scores_list = [s[2] for s in scores_list]
        best_idx = test_scores_list.index(max(test_scores_list))

        self.train_score_mean = scores_list[best_idx][0]
        self.train_score_std = scores_list[best_idx][1]
        self.test_score_mean = scores_list[best_idx][2]
        self.test_score_std = scores_list[best_idx][3]
        self.fitted_params = pg_list[best_idx]

        self.logger.log_timer()
        self.logger.log('train: {:0.5f} (std={:0.5f})'.format(self.train_score_mean, self.train_score_std))
        self.logger.log('test: {:0.5f} (std={:0.5f})'.format(self.test_score_mean, self.test_score_std))
        self.logger.log(str(self.fitted_params))


class GridFineSearcherBase(IPSStageBase):

    # {'max_depth': {'steps': 2, 'scale': '1_exp'}}
    # {'max_depth': {'steps': 3, 'scale': '2'}}
    # {'max_depth': {'steps': 3, 'scale': 3}}

    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parsed_param_grid = None

    def parse_param_grid(self, params: dict) -> dict:
        parsed_param_grid = dict()
        for param_name, search_dict in self.param_grid.items():
            if param_name not in params:
                print(params)
                self.logger.log('Parameter {} should be set before fine tuning!'.format(param_name))
                continue
            param_value = params[param_name]
            param_exp = int(math.floor(math.log(param_value, 10)))
            grid_exp = 0
            if isinstance(search_dict['scale'], str):
                if '_exp' in search_dict['scale']:
                    step_val = float(search_dict['scale'].split('_')[0])
                    step_exp = int(math.floor(math.log(step_val, 10)))
                    grid_exp = min(param_exp + step_exp, param_exp)
                    step = float(search_dict['scale'].split('_')[0]) * (10**param_exp)
                else:
                    step = float(search_dict['scale'])
            else:
                step = search_dict['scale']
            grid = [param_value]
            grid += [param_value + (s * step) for s in range(1, search_dict['steps'] + 1)]
            grid += [param_value - (s * step) for s in range(1, search_dict['steps'] + 1)]
            grid = sorted(grid)
            if grid_exp < 0.0:
                grid = [round(s, abs(int(grid_exp))) for s in grid]
            grid = [s for s in grid if self.parent.check_parameter_value(param_name, s)]
            parsed_param_grid[param_name] = grid
        return parsed_param_grid

    def fit_base(self, params: dict, grid_searcher: IPSStageBase) -> None:
        self.parsed_param_grid = self.parse_param_grid(params)
        self.logger.log('Parsed params: {}'.format(self.parsed_param_grid))

        gs = grid_searcher()
        gs.set_parent_pipeline(self.parent)
        gs.set_param_grid(self.parsed_param_grid)
        gs.fit(params)

        self.fitted_params = gs.fitted_params
        self.train_score_mean = gs.train_score_mean
        self.train_score_std = gs.train_score_std
        self.test_score_mean = gs.test_score_mean
        self.test_score_std = gs.test_score_std

    @abstractmethod
    def fit(self, params: dict) -> None:
        pass


class GridFineSearcher(GridFineSearcherBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def fit(self, params: dict) -> None:
        self.fit_base(params, GridSearcher)


class ManualGridFineSearcher(GridFineSearcherBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def fit(self, params: dict) -> None:
        self.fit_base(params, ManualGridSearcher)
