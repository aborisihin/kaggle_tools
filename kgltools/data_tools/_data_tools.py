""" data_tools module.
Contains DataTools class for data-oriented tasks
"""

import os
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..context import KglToolsContext, KglToolsContextChild

__all__ = ['DataTools']


class DataTools(KglToolsContextChild):
    """ Класс для работы с данными (загрузка, сохранение, разбивка и т.д.)

    Args:
        context: Контекст окружения

    Attributes:
        context (KglToolsContext): Контекст окружения
        settings (dict): Словарь с настройками
        random_state (int): Инициализирующее random значение
        X_train, y_train (pd.DataFrame): Обучающая выборка данных
        X_validate, y_validate (pd.DataFrame): Валидационная выборка данных
    """

    def __init__(self, context: KglToolsContext) -> None:
        super().__init__(context)
        self.settings = context.settings.get('data_tools', dict())
        self.random_state = context.random_state
        self.X_train = None
        self.y_train = None
        self.X_validate = None
        self.y_validate = None

    def get_validate_split(self,
                           X: pd.DataFrame,
                           y: Optional[pd.DataFrame] = None,
                           validation_size: float = 0.2) -> Tuple[pd.DataFrame]:
        """
        Разбивка датасета на обучающую и валидационную выборки

        Args:
            X: Исходный датасет
            y: Датасет (вектор) с истинными ответами
            validation_size: Пропорция разбиения
        """
        if y is None:
            X_t, X_v = train_test_split(X, test_size=validation_size, shuffle=True,
                                        random_state=self.random_state)
            self.X_train = X_t
            self.X_validate = X_v
            return (X_t, X_v)
        else:
            X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=validation_size, shuffle=True,
                                                  stratify=y, random_state=self.random_state)
            self.X_train = X_t
            self.X_validate = X_v
            self.y_train = y_t
            self.y_validate = y_v
            return (X_t, X_v, y_t, y_v)

    def write_submission(self, predictions: np.ndarray) -> bool:
        """
        Запись файла с предсказаниями в заданном формате

        Args:
            predictions: Вектор или матрица с предсказаниями
        """
        submission_settings = self.settings['submission_params']

        sample_submission_path = os.path.join(self.settings['path'], submission_settings['sample_file'])

        if not os.path.exists(sample_submission_path):
            print('DataTools::write_submission(): sample submission file is not exist!')
            return False

        sample_sbm = pd.read_csv(sample_submission_path, **submission_settings['pd_read_csv_params'])
        sample_sbm[submission_settings['target_fields']] = predictions

        sbm_filename = '{}_sbm.csv'.format(datetime.now().strftime("%Y-%m-%d_%H-%M"))
        smb_filepath = os.path.join(submission_settings['submissions_dir'], sbm_filename)

        sample_sbm.to_csv(smb_filepath, **submission_settings['pd_write_csv_params'])

        print('save submission:\n{}'.format(sbm_filename))
