from datetime import datetime


def get_time():
    t = datetime.now()
    return t.strftime("%m/%d/%Y, %H:%M:%S")
