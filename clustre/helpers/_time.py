import warnings
from datetime import datetime


def get_time():
    t = datetime.now()
    return format_time(t)


def format_time(t):
    return t.strftime("%m/%d/%Y, %H:%M:%S.%f")
