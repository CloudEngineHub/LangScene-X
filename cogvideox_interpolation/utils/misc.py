import torch, time
class Timer:
    def __init__(self, name, log=False):
        self.name = name
        self.start_time = None
        self.end_time = None

    @property
    def elapsed_time(self):
        return self.end_time - self.start_time

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.end_time = time.time()