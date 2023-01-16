"""Create logger"""
import logging
import sys

import colorlog

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# basic logging setup
stream_formatter = colorlog.ColoredFormatter(
    "[%(asctime)s] %(log_color)s%(levelname)-8s%(reset)s (%(filename)17s:%(lineno)-4s)"
    " %(blue)4s%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    },
)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(stream_formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
