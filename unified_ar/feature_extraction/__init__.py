# import importlib
# import glob
# from os.path import dirname, basename, isfile, join
from . import Context
from . import Cook
from . import DeepLearningFeatureExtraction
from . import features
from . import feature_abstract
from . import KHistory
from . import ModifiedCook
from . import NLP
# from . import PAL_Features
from . import Raw
from . import Recent
from . import SequenceFeatureExtraction
from . import SequenceRaw
from . import Simple
__import__('pkg_resources').declare_namespace(__name__)

# modules = glob.glob(join(dirname(__file__), "*.py"))
# __all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
# print(__all__)
# # print(basename(dirname(__file__)),' ==> ',__all__)
# for pack in __all__:
#     globals().update(importlib.import_module(basename(dirname(__file__))+'.'+pack).__dict__)
