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

    def fit(self, params: dict = {}) -> None:
        if len(self.param_grid) != 1:
            self.logger.log('Wrong param_grid argument!')
            return dict()

        n_estimators_param_name = list(self.param_grid.keys())[0]
        max_n_estimators = list(self.param_grid.values())[0]
        self.fitted_params = {n_estimators_param_name: 0}

        self.logger.start_timer()

        train_mean, train_std, test_mean, test_std = 0.0, 0.0, 0.0, 0.0

        # xgboost
        if self.parent.estimator_class in (XGBClassifier, XGBRegressor):
            xgb_params = self.parent.estimator_class(**params).get_xgb_params()
            xgd_data = xgb.DMatrix(self.parent.X.values, label=self.parent.y.values)
            cv_result = xgb.cv(params=xgb_params,
                               dtrain=xgd_data,
                               metrics=self.parent.estimator_metrics,
                               num_boost_round=max_n_estimators,
                               nfold=self.parent.n_folds,
                               early_stopping_rounds=self.early_stopping_rounds,
                               stratified=self.parent.stratified,
                               seed=self.parent.random_state,
                               shuffle=self.parent.shuffle,
                               verbose_eval=False)
            self.train_score_mean = cv_result.iloc[len(cv_result) - 1, 0]
            self.train_score_std = cv_result.iloc[len(cv_result) - 1, 1]
            self.test_score_mean = cv_result.iloc[len(cv_result) - 1, 2]
            self.test_score_std = cv_result.iloc[len(cv_result) - 1, 3]
            self.fitted_params[n_estimators_param_name] = len(cv_result)

        self.logger.log_timer()
        self.logger.log('train: {:0.5f} (std={:0.5f})'.format(self.train_score_mean, self.train_score_std))
        self.logger.log('test: {:0.5f} (std={:0.5f})'.format(self.test_score_mean, self.test_score_std))
        self.logger.log(self.fitted_params)
