""" context module.
Contains class for work with common task context
"""

import os
import json

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

        self.random_state = self.settings.get('random_state', 0)
        self.n_jobs = self.settings.get('n_jobs', -1)


class KglToolsContextChild(object):
    def __init__(self):
        pass
