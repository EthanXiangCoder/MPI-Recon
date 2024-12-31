import time

class Time():
    def __init__(self):
        self.time = []
        self.begintime = time.time()

    def begin_time(self):
        self.begintime = time.time()
    
    def cal_time(self):
        self.time.append(time.time() - self.begintime)

    def reset(self):
        self.time.clear()