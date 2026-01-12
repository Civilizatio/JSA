import sys
import re
from lightning.pytorch.callbacks import Callback


class StdRedirector:
    """Utility class to redirect stdout and stderr to a file while keeping console output."""
    
    
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, data):
        # Write to console
        self.stdout.write(data)
        self.stdout.flush()
        
        
        # Write to file
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()
        
    


class RedirectStdCallback(Callback):
    """
    Redirect stdout/stderr to a log file during training.
    Saves both console output and logs to file.
    """
    def __init__(self, filename="training_output.log"):
        super().__init__()
        self.filename = filename
        self.redirector = None

    def on_fit_start(self, trainer, pl_module):
        log_dir = trainer.log_dir or "./"
        full_path = f"{log_dir}/{self.filename}"
        print(f"[RedirectStdCallback] Logging stdout/stderr to: {full_path}")
        self.redirector = StdRedirector(full_path)

    def on_fit_end(self, trainer, pl_module):
        if self.redirector:
            self.redirector.close()
