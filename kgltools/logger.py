""" logger module.
Contains class for logging
"""

__all__ = ['Logger']


class Logger():
    def __init__(self, stage_name: str = '', nesting_level: int = 0) -> None:
        self.nesting_level = nesting_level
        self.stage_name = stage_name

    def init(self) -> None:
        self.log('{}'.format(self.stage_name))

    def log(self, text: str) -> None:
        space = " " * (4 * self.nesting_level)
        print("{}{}".format(space, text))

    def increase_level(self) -> None:
        self.nesting_level += 1

    def decrease_level(self) -> None:
        self.nesting_level = max(0, self.nesting_level - 1)
