""" context module.
Contains class for work with common task context
"""

import os
import json

from typing import Optional

from kgltools.data_tools import DataTools

__all__ = ['KglToolsContext']


class KglToolsContext(object):
    """ Класс контекста задачи
    Используется для поддержки общей структуры задач через трансляцию пользовательских настроек

    Args:
        settings_path: Путь к файлу настроек

    Attributes:
        settings (dict): Словарь с настройками окружения задачи
    """

    def __init__(self, settings_path: str) -> None:
        if not os.path.exists(settings_path):
            print('Settings file {} is not exist!'.format(settings_path))
            self.settings = None
            return

        with open(settings_path, 'r') as settings_file:
            self.settings = json.load(settings_file)

        if 'random_state' in self.settings:
            self.random_state = self.settings['random_state']
        else:
            self.random_state = 0

    def getDataTools(self) -> Optional[DataTools]:
        """
        Получить объект класса DataTools (data_tools.py)
        """
        if self.settings is None:
            print('No settings found!')
            return None
        if 'data_tools' not in self.settings:
            print('No "data_tools" settings found!')
            return None
        return DataTools(self.settings['data_tools'], self.random_state)
