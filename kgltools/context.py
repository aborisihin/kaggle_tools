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
        self.childs = list()

        # load extra dicts
        cur_path = os.path.dirname(os.path.abspath(__file__))
        settings_dir = os.path.abspath(os.path.join(cur_path, '../settings'))

        self.extra_dicts = dict()
        extra_dicts_names = ['metrics_mapping', 'estimator_parameter_limits']

        for edn in extra_dicts_names:
            path = os.path.join(settings_dir, '{}.json'.format(edn))
            if not os.path.exists(path):
                print('Settings file {}.json is not exist!'.format(edn))
                self.extra_dicts[edn] = None
            else:
                with open(path, 'r') as ed_file:
                    self.extra_dicts[edn] = json.load(ed_file)

    def add_child(self, child: object) -> None:
        self.childs.append(child)


class KglToolsContextChild(object):

    def __init__(self, context: KglToolsContext) -> None:
        self.context = context
        context.add_child(self)
