""" logger module.
Contains class for logging
"""

import time
import math

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

        self.log("Time spent: {}".format(time_str))
