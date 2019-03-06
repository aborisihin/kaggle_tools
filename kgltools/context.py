""" context module.
Contains class for work with common task context
"""

import os
import json

from typing import Optional

__all__ = ['KglToolsContext', 'KglToolsContextChild']


class KglToolsContext(object):
    """ Common tasks context
    Класс контекста задачи
    Используется для поддержки общей структуры задач через трансляцию пользовательских настроек

    Args:
        settings_path: Путь к файлу настроек

    Attributes:
        settings (dict): Словарь с настройками окружения задачи
        random_state (int): Инициализирующее random значение
        n_jobs (int): Количество параллельных процессов выполнения задач; (-1 - использование всех процессов)
        child_list: List[KglToolsContextChild]: Список дочерних объектов
        extra_dicts (dict): Словарь вспомогательных словарей.
            Грузятся из каталога settings. Используемые словари:
            - metrics_mapping (маппинг метрик из sklearn на другие библиотеки)
            - estimator_parameter_limits (пределы допустимых значений параметров моделей)
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
        self.child_list = list()

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

    def add_child(self, child: 'KglToolsContextChild') -> None:
        """Add child object
        Добавление дочернего объекта

        Args:
            child - дочерний объект
        """
        self.child_list.append(child)

    def get_child(self, child_type: type) -> Optional['KglToolsContextChild']:
        """Get child object
        Получение объекта заданного типа из списка дочерних

        Args:
            child_type - тип объекта

        Returns:
            Дочерний объект или None в случае его отсутствия
        """
        found_child: Optional['KglToolsContextChild'] = None
        for child in self.child_list:
            if isinstance(child, child_type):
                found_child = child
        return found_child


class KglToolsContextChild(object):
    """ Context's child base class
    Базовый класс дочернего контексту объекта

    Args:
        context: Объект контекста

    Attributes:
        context (KglToolsContext): Объект контекста
    """
    def __init__(self, context: KglToolsContext) -> None:
        self.context = context
        context.add_child(self)
