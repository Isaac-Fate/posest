from .app import app

# Import submodules to register the commands
from .train import *


__all__ = [
    "app",
]
