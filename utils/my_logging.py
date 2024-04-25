import os
import pytz
import logging

from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from logging import StreamHandler

def set_colorful_log(log_name: str, log_path: str, log_level: int = logging.DEBUG) -> logging.Logger:
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return create_logger(log_name, log_path, log_level)

class ISO8601Formatter(logging.Formatter):
    converter = datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%dT%H:%M:%S")
            s = "%s.%03dZ" % (t, record.msecs)
        return s

def create_logger(log_name: str, log_path: str, log_level: int = logging.DEBUG) -> logging.Logger:
    # create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create console handler and set level to debug
    ch = StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create file handler and set level to debug
    fh = TimedRotatingFileHandler(log_path, when='midnight', interval=1, backupCount=7)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def get_logger(log_name: str, log_path: str, log_level: int = logging.DEBUG) -> logging.Logger:
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return create_logger(log_name, log_path, log_level)

def get_utc_time() -> str:
    return datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    pass