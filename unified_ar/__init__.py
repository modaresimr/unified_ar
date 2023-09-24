
import importlib
import sys
from . import common
from .common import Data, MyTask, loader
from .general import utils
from . import datatool
from . import feature_extraction
from . import general
from . import activity_fetcher
from . import classifier
from . import combiner

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
        if not module_name.startswith('unified_ar'):
            continue
            # print(module)
        if "data" in f'{module}'.lower():
            continue
        importlib.reload(module)
