from .app import data_app

# Import submodules to register the commands
from .smp2idx import *
from .split import *

# Add sub apps
from .jta import jta_app

data_app.add_typer(jta_app, name="jta")

__all__ = [
    "data_app",
]
