
import importlib
import sys
from .common import Data, MyTask, loader
from .general import utils
from . import datatool


def reload():
    dic = {module_name: module for module_name, module in sys.modules.items()}
    for module_name, module in dic.items():
        if module_name.startswith('unified_ar'):
            # print(module)
            importlib.reload(module)
