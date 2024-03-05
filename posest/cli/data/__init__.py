from .app import data_app

# Import submodules to register the commands
from .smp2idx import *
from .split import *

__all__ = [
    "data_app",
]
