import builtins
import sys
from lightning.pytorch.callbacks import Callback


class PrintRedirector:
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, "a")
        self.original_print = builtins.print

    def write(self, msg: str):
        # 只记录非空消息
        if msg.strip():
            self.file.write(msg + "\n")
            self.file.flush()

    def redirect(self):
        def custom_print(*args, **kwargs):
            msg = " ".join(map(str, args))
            self.write(msg)

        builtins.print = custom_print

    def restore(self):
        builtins.print = self.original_print
        self.file.close()


class RedirectPrintCallback(Callback):
    def __init__(self, filename: str = "train.log"):
        super().__init__()
        self.filename = filename
        self.redirector = None

    def on_fit_start(self, trainer, pl_module):
        # 创建重定向器
        full_path = f"{trainer.log_dir}/{self.filename}" if trainer.log_dir else self.filename
        self.redirector = PrintRedirector(full_path)
        self.redirector.redirect()

        print(f"[RedirectPrintCallback] Logging prints to: {full_path}")
    def on_fit_end(self, trainer, pl_module):
        if self.redirector:
            self.redirector.restore()
            self.redirector = None

        print("[RedirectPrintCallback] Restored normal print.")
