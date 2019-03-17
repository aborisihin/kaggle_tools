""" stacker module.
Contains models stacking class
"""

from typing import Union, List, Optional
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold

from ..context import KglToolsContext, KglToolsContextChild
from ..logger import Logger
from ..data_tools import DataTools

__all__ = ['Stacker']


class Stacker(KglToolsContextChild):

    def __init__(self,
                 context: KglToolsContext,
                 estimators: List[object],
                 metrics: str,
                 predict_proba: bool = False,
                 n_folds: int = 5,
                 stratified: bool = True,
                 shuffle: bool = True,
                 verbose: bool = True) -> None:
        super().__init__(context)
        self.estimators = estimators
        self.metrics = metrics
        self.predict_method = 'predict_proba' if predict_proba else 'predict'
        self.n_folds = n_folds
        self.stratified = stratified
        self.shuffle = shuffle
        self.verbose = verbose

        self.random_state = self.context.random_state
        self.n_jobs = self.context.n_jobs

        self.logger = Logger(self.context, self.__class__, nesting_level=0, verbose=self.verbose)

        # make folds
        if self.stratified:
            self.folds_maker = StratifiedKFold(n_splits=self.n_folds,
                                               shuffle=self.shuffle,
                                               random_state=self.random_state)
        else:
            self.folds_maker = KFold(n_splits=self.n_folds,
                                     shuffle=self.shuffle,
                                     random_state=self.random_state)

        self.fitted_estimators = [[None for _ in range(self.n_folds)] for _ in range(len(estimators))]

    def fit(self,
            X: pd.DataFrame,
            y: Union[pd.DataFrame, pd.Series],
            dump_file: Optional[str] = None) -> Optional[pd.DataFrame]:
        # check estimators
        for est in self.estimators:
            if (not hasattr(est, 'fit')) or (not hasattr(est, self.predict_method)):
                log_mes = 'Estimator {} does\'t have methods "fit" or "{}"'
                self.logger.log(log_mes.format(type(est), self.predict_method))
                return None

        self.logger.log('Stacker: fit {} estimators on {} folds'.format(len(self.estimators),
                                                                        self.n_folds), tg_send=True)

        meta_values = np.zeros((X.shape[0], len(self.estimators)))
        for fold_idx, (train_idx, test_idx) in enumerate(self.folds_maker.split(X, y)):
            self.logger.log('Fold {}:'.format(fold_idx + 1), tg_send=True)
            self.logger.increase_level()
            self.logger.start_timer()

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test = X.iloc[test_idx]

            for est_idx, est in enumerate(self.estimators):
                self.logger.start_timer()

                fold_estimator = deepcopy(est)
                fold_estimator.fit(X_train, y_train)
                self.fitted_estimators[est_idx][fold_idx] = fold_estimator
                fold_est_predict_method = getattr(fold_estimator, self.predict_method)
                predictions = fold_est_predict_method(X_test)
                if self.predict_method == 'predict_proba':
                    predictions = predictions[:, 1]
                meta_values[test_idx, est_idx] = predictions

                self.logger.log_timer(prefix_mes=type(est).__name__, tg_send=True)

            self.logger.log_timer(tg_send=True)
            self.logger.decrease_level()

        meta_df = pd.DataFrame(data=meta_values,
                               index=X.index,
                               columns=[type(est).__name__ for est in self.estimators])

        if dump_file is not None:
            self.save_dump(meta_df, dump_file)

        return meta_df

    def transform(self,
                  X_test: pd.DataFrame,
                  dump_file: Optional[str] = None) -> Optional[pd.DataFrame]:
        # check estimators
        for folds in self.fitted_estimators:
            if None in folds:
                self.logger.log('Have unfitted estimators, can\'t make transform!')
                return None

        meta_values = np.zeros((len(X_test), len(self.fitted_estimators)))
        for est_idx, folds in enumerate(self.fitted_estimators):
            est_predictions = np.zeros((len(X_test), len(folds)))
            for fold_idx, fold_estimator in enumerate(folds):
                fold_est_predict_method = getattr(fold_estimator, self.predict_method)
                predictions = fold_est_predict_method(X_test)
                if self.predict_method == 'predict_proba':
                    predictions = predictions[:, 1]
                est_predictions[:, fold_idx] = predictions
            meta_values[:, est_idx] = est_predictions.mean(axis=1)
        meta_df = pd.DataFrame(data=meta_values,
                               index=X_test.index,
                               columns=[type(est[0]).__name__ for est in self.fitted_estimators])

        if dump_file is not None:
            self.save_dump(meta_df, dump_file)

        return meta_df

    def save_dump(self, df: pd.DataFrame, filename: str) -> None:
        data_tools = self.context.get_child(DataTools)
        if data_tools:
            data_tools.write_metaset(df, filename)
