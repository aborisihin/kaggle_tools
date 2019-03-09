""" logger module.
Contains class for logging
"""

import time
import math

from typing import Type, Any

from telegram import Bot

from .context import KglToolsContext, KglToolsContextChild

__all__ = ['Logger']


class Logger(KglToolsContextChild):
    """ Common logger class
    Класс общего логгера
    Используется для однотипного вывода сообщений в лог с поддержкой подсчета используемого времени и отступами

    Args:
        context: Контекст окружения
        owner: Класс-владелец
        nesting_level: Значение отступа сообщений
        verbose: Флаг доступности вывода сообщений в лог

    Attributes:
        context (KglToolsContext): Контекст окружения
        settings (dict): Словарь с настройками
        owner_name (str): Имя класса-владельца
        nesting_level (int): Значение отступа сообщений
        verbose (bool): Флаг доступности вывода сообщений в лог
        start_time (float): Значение стартовой отсечки для подсчета используемого времени
    """
    def __init__(self,
                 context: KglToolsContext,
                 owner: Type[Any],
                 nesting_level: int = 0,
                 verbose: bool = True) -> None:
        super().__init__(context)
        self.settings = context.settings.get('logger', dict())
        self.owner_name = '{}'.format(owner.__name__)
        self.nesting_level = nesting_level
        self.verbose = verbose
        self.start_time = 0

        self.telegram_chat_id = None
        self.telegram_bot = None

        if 'telegram_notifications' in self.settings:
            tg_settings = self.settings['telegram_notifications']
            if tg_settings.get('enabled', False):
                if 'telegram_bot_token' in self.settings:
                    self.telegram_bot = Bot(token=self.settings['telegram_bot_token'])
                self.telegram_chat_id = self.settings.get('chat_id', '')

    def log(self, text: str, tg_send: bool = False) -> None:
        """Print message in log
        Печать сообщения в лог

        Args:
            text - текст сообщения
        """
        if not self.verbose:
            return
        space = " " * (4 * self.nesting_level)
        print("{}{}".format(space, text))

        if tg_send and self.telegram_bot:
            tg_message = '<b>{}</b>\n{}'.format(self.owner_name, text)
            try:
                self.telegram_bot.send_message(text=tg_message, chat_id=self.telegram_chat_id, parse_mode='html')
            except Exception as ex:
                self.logger.log('telegram bot exception: {}'.format(str(ex)))

    def increase_level(self) -> None:
        """Increase messages indent
        Увеличение отступа сообщений логгера
        """
        self.nesting_level += 1

    def decrease_level(self) -> None:
        """Decrease messages indent
        Уменьшение отступа сообщений логгера
        """
        self.nesting_level = max(0, self.nesting_level - 1)

    def start_timer(self) -> None:
        """Start timing
        Запуск отсчета используемого времени
        """
        self.start_time = time.time()

    def log_timer(self, tg_send: bool = False) -> None:
        """Log tmer value
        Вывод в лог значение используемого времени
        """
        if not self.verbose:
            return

        time_spent = float(time.time() - self.start_time)

        if time_spent < 60.0:
            time_str = '{:0.2f}s'.format(time_spent)
        elif time_spent < 3600.0:
            time_m = int(math.floor(time_spent / 60))
            time_s = int(round(time_spent - (time_m * 60)))
            time_str = '{:d}m {:d}s'.format(time_m, time_s)
        else:
            time_h = int(math.floor(time_spent / 3600))
            time_m = int(math.floor((time_spent - (time_h * 3600)) / 60))
            time_s = int(round(time_spent - (time_h * 3600) - (time_m * 60)))
            time_str = '{:d}h {:d}m {:d}s'.format(time_h, time_m, time_s)

        self.log("Time spent: {}".format(time_str), tg_send)
