from rich.console import Console
from rich.logging import RichHandler
import logging


_console = Console()

def get_logger(name: str = __name__):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=_console, rich_tracebacks=True)],
    )
    logger = logging.getLogger(name)
    return logger