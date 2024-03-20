from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stadion")
except PackageNotFoundError:
    __version__ = ""