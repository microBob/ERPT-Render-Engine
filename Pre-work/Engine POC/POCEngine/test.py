import sys
import importlib

try:
    import engine
except ModuleNotFoundError:
    print("Could not find 'engine' module")
    exit(-1)

