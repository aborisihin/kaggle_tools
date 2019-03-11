""" gradient_boosting module.
Contains stages for using in IPSPipeline
"""

from ._ips_pipeline import IPSStageBase

import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor

import catboost as cb
from catboost import CatBoostClassifier, CatBoostRegressor

import warnings

warnings.filterwarnings("ignore")
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
warnings.simplefilter('always')


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

    def fit(self, params: dict) -> None:
        if len(self.param_grid) != 1:
            self.logger.log('Wrong param_grid argument!')
            return

        n_estimators_param_name = list(self.param_grid.keys())[0]
        max_n_estimators = list(self.param_grid.values())[0]

        # will set through 'num_boost_round' param of cv method
        modified_params = params
        modified_params.pop(n_estimators_param_name, None)

        self.logger.start_timer()

        if self.parent.estimator_class in (XGBClassifier, XGBRegressor):
            xgb_params = self.parent.estimator_class(**modified_params).get_xgb_params()
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
            self.fitted_params = {n_estimators_param_name: len(cv_result)}
        elif self.parent.estimator_class in (LGBMClassifier, LGBMRegressor):
            lgb_data = lgb.Dataset(self.parent.X.values, label=self.parent.y.values)
            cv_result = lgb.cv(params=modified_params,
                               train_set=lgb_data,
                               num_boost_round=max_n_estimators,
                               nfold=self.parent.n_folds,
                               stratified=self.parent.stratified,
                               shuffle=self.parent.shuffle,
                               metrics=self.parent.estimator_metrics,
                               early_stopping_rounds=self.early_stopping_rounds,
                               seed=self.parent.random_state,
                               verbose_eval=False)
            means = cv_result['{}-mean'.format(self.parent.estimator_metrics)]
            stds = cv_result['{}-stdv'.format(self.parent.estimator_metrics)]
            self.train_score_mean = 0.0
            self.train_score_std = 0.0
            self.test_score_mean = means[-1]
            self.test_score_std = stds[-1]
            self.fitted_params = {n_estimators_param_name: len(means)}
        elif self.parent.estimator_class in (CatBoostClassifier, CatBoostRegressor):
            modified_params['eval_metric'] = self.parent.estimator_metrics
            # no cat features!
            cb_data = cb.Pool(self.parent.X.values, label=self.parent.y.values, cat_features=None)
            cv_result = cb.cv(params=modified_params,
                              pool=cb_data,
                              num_boost_round=max_n_estimators,
                              nfold=self.parent.n_folds,
                              stratified=self.parent.stratified,
                              shuffle=self.parent.shuffle,
                              early_stopping_rounds=self.early_stopping_rounds,
                              seed=self.parent.random_state,
                              plot=False,
                              verbose_eval=False)
            self.train_score_mean = cv_result['train-{}-mean'.format(self.parent.estimator_metrics)].iloc[-1]
            self.train_score_std = cv_result['train-{}-std'.format(self.parent.estimator_metrics)].iloc[-1]
            self.test_score_mean = cv_result['test-{}-mean'.format(self.parent.estimator_metrics)].iloc[-1]
            self.test_score_std = cv_result['test-{}-std'.format(self.parent.estimator_metrics)].iloc[-1]
            self.fitted_params = {n_estimators_param_name: len(cv_result)}

        self.logger.log_timer()
        self.logger.log('train: {:0.5f} (std={:0.5f})'.format(self.train_score_mean,
                                                              self.train_score_std), tg_send=True)
        self.logger.log('test: {:0.5f} (std={:0.5f})'.format(self.test_score_mean,
                                                             self.test_score_std), tg_send=True)
        self.logger.log(self.fitted_params, tg_send=True)
