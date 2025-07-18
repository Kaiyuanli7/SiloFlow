from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("granarypredict")
except PackageNotFoundError:
    __version__ = "0.1.0"

from . import config, ingestion, cleaning, features, evaluate, model, optuna_cache  # noqa: F401 