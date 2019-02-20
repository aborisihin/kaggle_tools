""" data_tools module.
Contains DataTools class for data-oriented tasks
"""

import os
from datetime import datetime

from typing import Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

__all__ = ['DataTools']


class DataTools():
    """ Класс для работы с данными (загрузка, сохранение, разбивка и т.д.)

    Args:
        settings: Словарь с настройками окружения задачи
        random_state: Инициализирующее random значение

    Attributes:
        settings (dict): Словарь с настройками окружения задачи
        random_state (int): Инициализирующее random значение
    """

    def __init__(self, settings: dict, random_state: int) -> None:
        self.settings = settings
        self.random_state = random_state

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
            return (X_t, X_v)
        else:
            X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=validation_size, shuffle=True,
                                                  stratify=y, random_state=self.random_state)
            return (X_t, X_v, y_t, y_v)
