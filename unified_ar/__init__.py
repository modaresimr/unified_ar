
import importlib
import sys
from .common import Data, MyTask, loader
from .general import utils
from . import datatool
from . import feature_extraction
from . import general
from . import activity_fetcher
from . import classifier
from . import combiner
from . import common
from . import metric
from . import ml_strategy
from . import optimizer
from . import preprocessing
from . import result_analyse
from . import SED
from . import segmentation
from . import wardmetrics
from . import test
from . import evaluation


def reload():
    dic = {module_name: module for module_name, module in sys.modules.items()}
    for module_name, module in dic.items():
        if module_name.startswith('unified_ar'):
            # print(module)
            importlib.reload(module)
