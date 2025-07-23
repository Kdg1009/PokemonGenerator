import torch.profiler as profiler

class Profiler:
    def __init__(self, enabled=True, log_dir="./log_dir", wait=1, warmup=1, active=3, repeat=1):
        self.enabled = enabled
        self.profiler = None
        if self.enabled:
            self.profiler = profiler.profile(
                schedule=profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
                on_trace_ready=profiler.tensorboard_trace_handler(log_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True
            )

    def __enter__(self):
        if self.enabled:
            self.profiler.__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            self.profiler.__exit__(exc_type, exc_value, traceback)

    def step(self):
        if self.enabled:
            self.profiler.step()