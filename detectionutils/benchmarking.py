import time

def now():
    return time.time() * 1000


class Benchmark:
    def __init__(self, step_name):
        self.start = None
        self.end = None
        self.step_name = step_name

    def __enter__(self): 
        print(now(), f"started {self.step_name}")
        self.start = now()
        
    def __exit__(self, ext, exv, trb):
        self.end = now()
        duration = self.end - self.start
        print(now(), f"{self.step_name} took {duration} ms")
