from datetime import datetime

from dateutil.relativedelta import relativedelta


def get_time():
    t = datetime.now()
    return format_time(t)


def format_time(t):
    return t.strftime("%m/%d/%Y, %H:%M:%S.%f")


def delta_time_string(b, a):
    delta = relativedelta(b, a)
    return f"{delta.hours:02d}:{delta.minutes:02d}:{delta.seconds:02d}.{delta.microseconds:06d}"
