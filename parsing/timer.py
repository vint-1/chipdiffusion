import time

class Timer():
    def __init__(self, start = False):
        self.elapsed = 0
        self.t_0 = None
        if start:
            self.start()

    def start(self):
        if self.t_0 is None:
            self.t_0 = time.time()
    
    def stop(self):
        t_segment = time.time() - self.t_0
        self.elapsed += t_segment
        self.t_0 = None

    def read(self):
        t_extra = (time.time() - self.t_0) if self.t_0 is not None else 0
        return self.elapsed - t_extra

    def reset(self):
        """
        Stops timer and resets elapsed time
        """
        self.stop()
        elapsed = self.elapsed
        self.elapsed = 0
        return elapsed