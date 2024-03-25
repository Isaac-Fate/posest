import typer

# Import submodules to register the commands
from .jta import jta_app


# Main app
app = typer.Typer(
    help="Mobile Human Pose.",
)

# Sub-apps
app.add_typer(jta_app, name="jta")
