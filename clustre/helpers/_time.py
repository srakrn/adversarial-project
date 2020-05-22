import warnings
from datetime import datetime


def get_time_string():
    t = datetime.now()
    return format_time(t)


def get_time():
    warnings.warn(
        "This method is deprecated. Use get_time_string() instead.",
        warnings.DeprecationWarning,
    )
    return get_time()


def format_time(t):
    return t.strftime("%m/%d/%Y, %H:%M:%S.%f")
