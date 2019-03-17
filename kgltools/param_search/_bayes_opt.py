""" bayes_opt module.
Contains bayesian hyper parameters search class
"""

from typing import Optional, List

from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization
from bayes_opt.event import Events

from ._param_searcher import ParamSearcher

__all__ = ['BayesianOptimizer']


class BayesianOptimizer(ParamSearcher):

    def __init__(self,
                 init_points: int,
                 n_iter: int,
                 verbose_level: Optional[int] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.init_points = init_points
        self.n_iter = n_iter
        self.verbose_level = verbose_level
        self.param_grid = None
        self.probe_param_grid = None
        self.int_params = set()
        self.optimization_log = None
        self.callback_counter = 0

    def set_grid(self, param_grid: dict) -> None:
        self.param_grid = param_grid
        for param_name, param_tuple in self.param_grid.items():
            if isinstance(param_tuple[0], int) and isinstance(param_tuple[1], int):
                self.int_params.add(param_name)

    def set_probe(self, probe_param_grid: dict) -> None:
        self.probe_param_grid = probe_param_grid

    def get_optimization_log(self) -> List[dict]:
        return self.optimization_log

    def opt_function(self, **kwargs) -> float:
        checked_kwargs = dict()
        for param_name, param_value in kwargs.items():
            if param_name in self.int_params:
                checked_kwargs[param_name] = int(round(param_value))
            else:
                checked_kwargs[param_name] = param_value

        est_params = {**self.base_params, **checked_kwargs}
        cv_scores = cross_val_score(estimator=self.estimator_class(**est_params),
                                    X=self.X,
                                    y=self.y,
                                    scoring=self.metrics,
                                    cv=self.n_folds,
                                    n_jobs=self.n_jobs,
                                    verbose=False)
        return cv_scores.mean()

    def fit(self) -> None:
        if self.param_grid is None:
            self.logger.log('No parameters grid found!')
            return
        self.logger.log('BayesianOptimizer: fit {} steps'.format(self.init_points + self.n_iter), tg_send=True)
        for param in self.param_grid:
            self.logger.log('{} = {}'.format(param, self.param_grid[param]))
        self.logger.increase_level()
        self.logger.start_timer()

        self.optimization_log = None
        self.callback_counter = 0

        optimizer = BayesianOptimization(f=self.opt_function,
                                         pbounds=self.param_grid,
                                         random_state=self.context.random_state,
                                         verbose=0)
        if self.probe_param_grid is not None:
            optimizer.probe(params=self.probe_param_grid, lazy=True)
        optimizer.subscribe(event=Events.OPTMIZATION_STEP, subscriber=self, callback=self.optimizer_callback)
        optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)

        self.logger.decrease_level()
        self.logger.log_timer(tg_send=True)

        self.optimization_log = optimizer.res
        self.fitted_params = {k: int(round(v)) if k in self.int_params else v
                              for k, v in optimizer.max['params'].items()}

    def optimizer_callback(self, event: str, instance: BayesianOptimization) -> None:
        if 'optmization:step' in event:
            self.callback_counter += 1
            if (self.verbose_level is not None) and (self.callback_counter % self.verbose_level == 0):
                self.logger.log('Steps done: {}. Max target: {}'.format(self.callback_counter,
                                                                        instance.max['target']), tg_send=True)
