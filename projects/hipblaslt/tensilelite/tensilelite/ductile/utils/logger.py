# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import logging
import math
import numbers
import os
import numpy as np

LEVELS = {0: logging.WARNING,
          1: logging.INFO,
          2: logging.DEBUG}


def setup(name: str,
          fmt: str,
          level: int = logging.INFO,
          log_file: str = None,
          mode: str = "w"):

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    fmt = logging.Formatter(fmt)
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file = os.fspath(log_file)
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode=mode))
    for hdlr in handlers:
        hdlr.setLevel(level)
        hdlr.setFormatter(fmt)
        logger.addHandler(hdlr)
    return logger


class Logger:
    def __init__(self,
                 name: str,
                 verbose: int = 1,
                 log_file: str = None):

        if log_file and os.path.isfile(log_file):
            try:
                os.remove(log_file)
            except FileNotFoundError:
                pass

        self.level = LEVELS[verbose]

        self.msg_logger = setup(f"name_msg",
                                "%(message)s",
                                level=self.level,
                                log_file=log_file,
                                mode="a")
        self.app_logger = setup(name,
                                "%(name)s:%(levelname)s %(message)s",
                                level=self.level,
                                log_file=log_file,
                                mode="a")

        self.log = self.app_logger.log
        self.info = self.app_logger.info
        self.debug = self.app_logger.debug
        self.warn = self.app_logger.warning
        self.warning = self.app_logger.warning
        self.error = self.app_logger.error
        self.critical = self.app_logger.critical
        self.exception = self.app_logger.exception
        self.fatal = self.app_logger.fatal

    def print(self, msg: str):
        self.msg_logger.info(msg)

    def log_lines(self, msg: str, level: int):
        for line in f"{msg}".split("\n"):
            self.log(level, line)

    def print_stats(self, **kwargs):
        header = " "
        row = " "
        for i, (k, v) in enumerate(kwargs.items(), 1):
            if isinstance(v, numbers.Number):
                width = int(math.log10(v)) + 3 if v > 0 else 3
                if isinstance(v, (float, np.floating)):
                    v = round(float(v), 3)
                    width += 2
                width = max(width, len(k))
            else:
                width = max(len(k), len(v))
            header += f"{k:^{width}}"
            row += f"{v:^{width}}"
            if i < len(kwargs):
                header += " | "
                row += " | "

        n = len(header) + 2
        self.print("=" * n)
        self.print(header)
        self.print("-" * n)
        self.print(row)
        self.print("=" * n)
