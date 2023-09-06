from . import TatbulMetric
from . import MyMetric
from . import MyClassical
from . import Metrics
from . import GEM_NEW
from . import EventBasedMetric
from . import event_confusion_matrix2
from . import event_confusion_matrix
from . import CMbasedMetric
from . import classical
__import__('pkg_resources').declare_namespace(__name__)

# from os.path import dirname, basename, isfile, join
# import glob
# modules = glob.glob(join(dirname(__file__), "*.py"))
# __all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

# print(basename(dirname(__file__)),' ==> ',__all__)
# import importlib
# for pack in __all__:
#     globals().update(importlib.import_module(basename(dirname(__file__))+'.'+pack).__dict__)
