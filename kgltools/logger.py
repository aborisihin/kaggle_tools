""" logger module.
Contains class for logging
"""

import time

__all__ = ['Logger']


class Logger(object):
    def __init__(self, nesting_level: int = 0, verbose: bool = True) -> None:
        self.nesting_level = nesting_level
        self.verbose = verbose
        self.start_time = 0

    def log(self, text: str) -> None:
        if not self.verbose:
            return
        space = " " * (4 * self.nesting_level)
        print("{}{}".format(space, text))

    def increase_level(self) -> None:
        self.nesting_level += 1

    def decrease_level(self) -> None:
        self.nesting_level = max(0, self.nesting_level - 1)

    def start_timer(self):
        self.start_time = time.time()

    def log_timer(self):
        if not self.verbose:
            return
        self.log("Time spent: {:0.2f} sec".format(time.time() - self.start_time))
