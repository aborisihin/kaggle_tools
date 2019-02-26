""" iterative_param_search module.
Contains tools for iterative search for best parameters of model
"""

from typing import Generic, Type

import numpy as np
import pandas as pd

__all__ = ['IPSPipeline', 'IPSStage']


class IPSPipeline():

    def __init__(self, estimator_class: Generic, stages: list = []) -> None:
        self.estimator_class = estimator_class
        self.stages = stages


class IPSStage():

    def __init__(self):
        pass
