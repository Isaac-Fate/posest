import typer

# Import submodules to register the commands
from .data import data_app

# Main app
app = typer.Typer(
    help="A tool for pose estimation.",
)

# Sub-apps
app.add_typer(data_app, name="data")


@app.command()
def hello():
    print("Welcome to PosEst!")
