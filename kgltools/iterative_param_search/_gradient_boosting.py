""" iterative_param_search module.
Contains tools for iterative search for best parameters of model
"""

from ._pipeline import IPSStageBase

import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor
# import lightgbm as lgb
# import catboost as cb

__all__ = ['GBoostNTreesSearcher']


class GBoostNTreesSearcher(IPSStageBase):
    """ Класс этапа подбора количества деревьев в градиентном бустинге.
    Наследник базового класса IPSStage.
    Формат param_grid: {<n_estimators_param_name>: max_n_estimators}

    Args:

    Attributes:

    """

    def __init__(self, early_stopping_rounds: int = 25, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.early_stopping_rounds = early_stopping_rounds

    def process(self, params: dict = {}) -> dict:
        if len(self.param_grid) != 1:
            self.logger.log('Wrong param_grid argument!')
            return dict()

        max_n_estimators = list(self.param_grid.values())[0]
        n_estimators_param_name = list(self.param_grid.keys())[0]
        fitted_params = {n_estimators_param_name: 0}

        if max_n_estimators <= 0:
            self.logger.log('Wrong value in param_grid!')
            return fitted_params

        self.logger.start_timer()

        train_mean, train_std, test_mean, test_std = 0.0, 0.0, 0.0, 0.0

        if self.parent.estimator_class in (XGBClassifier, XGBRegressor):
            xgb_params = self.parent.estimator_class(**params).get_xgb_params()
            xgd_data = xgb.DMatrix(self.parent.X.values, label=self.parent.y.values)
            cv_result = xgb.cv(params=xgb_params,
                               dtrain=xgd_data,
                               num_boost_round=max_n_estimators,
                               nfold=self.parent.n_folds,
                               metrics=self.parent.metrics,
                               early_stopping_rounds=self.early_stopping_rounds,
                               stratified=self.parent.stratified,
                               seed=self.parent.random_state,
                               shuffle=self.parent.shuffle,
                               verbose_eval=False)
            train_mean = cv_result.iloc[len(cv_result) - 1, 0]
            train_std = cv_result.iloc[len(cv_result) - 1, 1]
            test_mean = cv_result.iloc[len(cv_result) - 1, 2]
            test_std = cv_result.iloc[len(cv_result) - 1, 3]
            fitted_params[n_estimators_param_name] = len(cv_result)

        if self.parent.verbose:
            self.logger.log_timer()
            self.logger.log('train: {:0.5f} (std={:0.5f})'.format(train_mean, train_std))
            self.logger.log('test: {:0.5f} (std={:0.5f})'.format(test_mean, test_std))
            self.logger.log('{} = {}'.format(n_estimators_param_name, fitted_params[n_estimators_param_name]))

        return fitted_params
