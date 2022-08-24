import logging
from logging.config import fileConfig

fileConfig('logger_config.ini')
logger = logging.getLogger()