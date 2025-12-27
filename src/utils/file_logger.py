# src/utils/file_logger.py
import logging
import os

def get_file_logger(log_path: str, name: str = "train_logger",rank:int=0) -> logging.Logger:
    """Get a file logger that logs messages to a specified file.
    
    Only rank zero will actually log messages.
    """
    

    logger = logging.getLogger(name)
    
    logger.propagate = False  # not to propagate to root logger
    
    if rank == 0:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger.setLevel(logging.INFO)
        # Avoid adding duplicate handlers (DDP / multiple inits)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            fh = logging.FileHandler(log_path, mode="a")
            formatter = logging.Formatter(
                "[%(asctime)s][%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
    else:
        logger.setLevel(logging.WARNING)  # Only log warnings and above for non-zero ranks
        # Ensure no handlers are added for non-zero ranks
        logger.handlers = []

    return logger
