""" iterative_param_search module.
Contains tools for iterative search for best parameters of model
"""

import numpy as np
import pandas as pd

from kgltools.iterative_param_search.pipeline import IPSPipeline

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

__all__ = ['GBoostNTreesSearcher']


class GBoostNTreesSearcher():

    def __init__(self, parent_pipeline: IPSPipeline):
        self.parent = parent_pipeline
