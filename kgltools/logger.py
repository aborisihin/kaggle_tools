""" logger module.
Contains class for logging
"""

import time
import math

from typing import Type, Any, Union

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
        start_time_stack (List[float]): Стек значений стартовых отсечек для подсчета используемого времени
        telegram_chat_id (str): Идентификатор чата для отправки сообщений в telegram
        telegram_bot (telegram.Bot): Объект telegram бота
    """

    def __init__(self,
                 context: KglToolsContext,
                 owner: Union[Type[Any], str],
                 nesting_level: int = 0,
                 verbose: bool = True) -> None:
        super().__init__(context)
        self.settings = context.settings.get('logger', dict())
        self.owner_name = owner if isinstance(owner, str) else '{}'.format(owner.__name__)
        self.nesting_level = nesting_level
        self.verbose = verbose
        self.start_time_stack = list()

        self.telegram_chat_id = None
        self.telegram_bot = None

        tg_settings = self.settings.get('telegram_notifications', None)
        if tg_settings and tg_settings.get('enabled', False):
            if 'bot_token' in tg_settings:
                self.telegram_bot = Bot(token=tg_settings['bot_token'])
            self.telegram_chat_id = tg_settings.get('chat_id', '')

    def log(self, text: str, tg_send: bool = False) -> None:
        """Print message in log
        Печать сообщения в лог

        Args:
            text - текст сообщения
            tg_send - флаг отправки сообщения в telegram
        """
        if not self.verbose:
            return
        space = " " * (4 * self.nesting_level)
        print("{}{}".format(space, text))

        if tg_send:
            self.telegram_log(text)

    def telegram_log(self, text: str) -> None:
        """Send message via telegram bot
        Отправка сообщения в telegram

        Args:
            text - текст сообщения
        """
        if self.telegram_bot:
            tg_message = '<b>{}</b>\n{}'.format(self.owner_name, text)
            for _ in range(5):  # 5 attempts
                try:
                    self.telegram_bot.send_message(text=tg_message,
                                                   chat_id=self.telegram_chat_id,
                                                   parse_mode='html')
                except Exception as ex:
                    self.log('Exception caught while sending telegram message: {}'.format(type(ex).__name__))
                    time.sleep(1)
                else:
                    break

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
        self.start_time_stack.append(time.time())

    def log_timer(self, prefix_mes: str = '', tg_send: bool = False) -> None:
        """Log tmer value
        Вывод в лог значение используемого времени

        Args:
            prefix mes - текст сообщения
            tg_send - флаг отправки сообщения в telegram
        """
        if len(self.start_time_stack) == 0:
            return
        if not self.verbose:
            self.start_time_stack.pop()
            return

        time_spent = float(time.time() - self.start_time_stack.pop())

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

        if len(prefix_mes):
            self.log("{}. Time spent: {}".format(prefix_mes, time_str), tg_send)
        else:
            self.log("Time spent: {}".format(time_str), tg_send)
