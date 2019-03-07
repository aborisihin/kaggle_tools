""" param_searcher module.
Contains base class for estrimator parameters search
"""

from abc import ABCMeta, abstractmethod

from ..context import KglToolsContext, KglToolsContextChild

__all__ = ['ParamSearcher']


class ParamSearcher(KglToolsContextChild):

	__metaclass__ = ABCMeta

	def __init__(self,
                 context: KglToolsContext,
                 estimator_class: Callable,
                 X: pd.DataFrame,
                 y: Union[pd.DataFrame, pd.Series],
                 metrics: str,
                 n_folds: int = 5,
                 stratified: bool = True,
                 shuffle: bool = True,
                 base_params: dict = None,
                 verbose: bool = True) -> None:
		super().__init__(context)
		self.estimator_class = estimator_class
        self.X = X
        self.y = y
        self.metrics = metrics
        self.n_folds = n_folds
        self.stratified = stratified
        self.shuffle = shuffle
        self.base_params = base_params if base_params is not None else dict()
        self.verbose = verbose

        self.random_state = self.context.random_state
        self.n_jobs = self.context.n_jobs

        self.logger = Logger(nesting_level=0, verbose=self.verbose)
        self.estimator_class_name = '{}.{}'.format(estimator_class.__module__, estimator_class.__name__)
        self.estimator_metrics = self.context.extra_dicts['metrics_mapping'][metrics][self.estimator_class_name]

        self.base_params['random_state'] = self.random_state
        self.fitted_params = dict()

	@abstractmethod
    def fit(self) -> None:
        pass

    def get_best_params(self) -> dict:
        return self.best_params

    def get_best_model(self) -> object:
        return self.estimator_class({**self.base_params, **self.fitted_params})

    def update_base_params(self, params: dict):
        self.base_params.update(params)
