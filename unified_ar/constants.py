import pandas as pd
import classifier.Keras
import classifier.libsvm
import datatool.a4h_handeler
import datatool.casas_handeler
import datatool.vankasteren_handeler
import activity_fetcher.CookActivityFetcher
import activity_fetcher.MaxActivityFetcher
import combiner.SimpleCombiner
import evaluation.SplitEval
import evaluation.KFoldEval
import feature_extraction.Simple
import feature_extraction.KHistory
import feature_extraction.DeepLearningFeatureExtraction
import feature_extraction.Cook
import feature_extraction.NLP
import feature_extraction.Recent
import feature_extraction.Context
import feature_extraction.Raw
from general.utils import Data
import metric.classical
import ml_strategy.Simple
import ml_strategy.FastFinder
import ml_strategy.SeperateGroup
import ml_strategy.WeightedGroup
import ml_strategy.WeightedGroup2
import preprocessing.SimplePreprocessing
import segmentation.Probabilistic
import segmentation.FixedEventWindow
import segmentation.FixedSlidingWindow
import segmentation.FixedTimeWindow
import segmentation.MetaDecomposition
import segmentation.ActivityWindow
import classifier.MySKLearn
no_memory_limit = True


# import classifier.PyActLearn


# import feature_extraction.PAL_Features
# from general.libimport import *

methods = Data('methods')
methods.run_names = {'out': 'temp'}
methods.meta_segmentation_sub_tasks = [

    {'method': lambda: segmentation.FixedEventWindow.FixedEventWindow(), 'params': [
        {'var': 'size', 'min': 2, 'max': 30, 'type': 'int', 'init': 5, 'range': [2, 5, 8, 11, 15, 20, 30]},
        {'var': 'shift', 'min': 2, 'max': 20, 'type': 'int', 'init': 2, 'range': [2, 5, 8, 11, 15, 20]}
    ], 'findopt': True},
    {'method': lambda: segmentation.FixedSlidingWindow.FixedSlidingWindow(), 'params': [
        {'var': 'size', 'min': 60, 'max': 15*60, 'type': 'float', 'init': 120/4, 'range': list(range(15, 120, 15))},
        {'var': 'shift', 'min': 10, 'max': 7*60, 'type': 'float', 'init': 60/2, 'range': list(range(15, 120, 15))}
    ], 'findopt': True},
    {'method': lambda: segmentation.Probabilistic.Probabilistic(), 'params': [], 'findopt':False},
    # {'method': lambda:segmentation.FixedTimeWindow.FixedTimeWindow(), 'params':[
    #                  {'var':'size','min':pd.Timedelta(1, unit='s').total_seconds(), 'max': pd.Timedelta(30, unit='m').total_seconds(), 'type':'float','init':pd.Timedelta(15, unit='s').total_seconds()},
    #                  {'var':'shift','min':pd.Timedelta(1, unit='s').total_seconds(), 'max': pd.Timedelta(30, unit='m').total_seconds(), 'type':'float','init':pd.Timedelta(1, unit='s').total_seconds()}
    # ],'findopt':False},
]


methods.segmentation = [
    # {'method': lambda: segmentation.FixedEventWindow.FixedEventWindow(), 'params': [
    #     {'size': 25},
    #     {'shift': 1}
    # ], 'findopt': False},

    {'method': lambda: segmentation.ActivityWindow.SlidingEventActivityWindow(), 'params': [
        {'size': 25},
        {'shift': 1}
    ], 'findopt': False},

    {'method': lambda: segmentation.MetaDecomposition.SWMeta(), 'params': [
        {'meta_size': '24h'},
        {'meta_overlap_rate': 1},
        {'meta_mode': 'keras'}
    ], 'findopt': False
    },
    {'method': lambda: segmentation.FixedEventWindow.FixedEventWindow(), 'params': [
        {'var': 'size', 'min': 10, 'max': 30, 'type': 'int', 'init': 10, 'range': list(range(10, 26, 5))},
        {'var': 'shift', 'min': 2, 'max': 20, 'type': 'int', 'init': 10, 'range': list(range(10, 16, 5))}
    ], 'findopt': False},
    {'method': lambda: segmentation.FixedSlidingWindow.FixedSlidingWindow(), 'params': [
        {'var': 'size', 'min': 60, 'max': 15*60, 'type': 'float', 'init': 120/4, 'range': list(range(15, 76, 15))},
        {'var': 'shift', 'min': 10, 'max': 7*60, 'type': 'float', 'init': 60/2, 'range': list(range(15, 45, 15))}
    ], 'findopt': False},
    {'method': lambda: segmentation.Probabilistic.Probabilistic(), 'params': [], 'findopt':False},
    # {'method': lambda:segmentation.FixedTimeWindow.FixedTimeWindow(), 'params':[
    #                  {'var':'size','min':pd.Timedelta(1, unit='s').total_seconds(), 'max': pd.Timedelta(30, unit='m').total_seconds(), 'type':'float','init':pd.Timedelta(15, unit='s').total_seconds()},
    #                  {'var':'shift','min':pd.Timedelta(1, unit='s').total_seconds(), 'max': pd.Timedelta(30, unit='m').total_seconds(), 'type':'float','init':pd.Timedelta(1, unit='s').total_seconds()}
    # ],'findopt':False},
]

methods.preprocessing = [
    {'method': lambda: preprocessing.SimplePreprocessing.SimplePreprocessing()},
]
methods.classifier = [
    {'method': lambda: classifier.Keras.FCN(), 'params': [
        {'epochs': 400}
    ]},

    {'method': lambda: classifier.MySKLearn.UAR_RandomForest(), 'params': [
        {'n_estimators': 20},
        {'random_state': 0},
        {'max_depth': 12},
        {'max_features_rate': .5}
    ]},
    {'method': lambda: classifier.Keras.SimpleKeras(), 'params': [
        {'epochs': 10}
    ]},
    {'method': lambda: classifier.Keras.LSTMTest(), 'params': [
        {'epochs': 10}
    ]},
    {'method': lambda: classifier.Keras.LSTMAE(), 'params': [
        {'epochs': 10}
    ]},
    {'method': lambda: classifier.libsvm.LibSVM()},

    # {'method': lambda: classifier.PyActLearn.PAL_LSTM_Legacy(), 'params': [
    #     {'var': 'epochs', 'init': 3}
    # ]},
    {'method': lambda: classifier.MySKLearn.UAR_KNN(), 'params': [
        {'k': 5},
    ]},
    {'method': lambda: classifier.MySKLearn.UAR_SVM(), 'params': [
        {'kernel': 'rbf'},
        {'gamma': 1},
        {'C': 100.},
        {'decision_function_shape': 'ovr'}
    ]},
    {'method': lambda: classifier.MySKLearn.UAR_SVM2(), 'params': [
        {'kernel': 'linear'},
        {'gamma': 1},
        {'C': 100.},
        {'decision_function_shape': 'ovr'}
    ]},
    {'method': lambda: classifier.MySKLearn.UAR_DecisionTree(), 'params': []},
]


methods.classifier_metric = [
    {'method': lambda: metric.classical.Accuracy()},
    #{'method': lambda: Accuracy()},
]

methods.event_metric = [
    # {'method': lambda: metric.Accuracy.Accuracy()},
    #{'method': lambda: Accuracy()},
]

methods.activity_fetcher = [
    # {'method': lambda: activity_fetcher.MaxActivityFetcher.MaxActivityFetcher()},
    {'method': lambda: activity_fetcher.CookActivityFetcher.CookActivityFetcher()}
]
methods.combiner = [
    {'method': lambda: combiner.SimpleCombiner.EmptyCombiner2()},
    # {'method':lambda: combiner.SimpleCombiner.SimpleCombiner()},
    # {'method':lambda: combiner.SimpleCombiner.EmptyCombiner()},
]
methods.evaluation = [
    {'method': lambda: evaluation.SplitEval.SplitEval()},
    # {'method': lambda: evaluation.KFoldEval.KFoldEval(5)},
    #  {'method': lambda: evaluation.KFoldEval.PKFoldEval(5)},

]


methods.feature_extraction = [
    {'method': lambda: feature_extraction.NLP.SensorWord(), 'params': [],     'findopt':False},
    {'method': lambda: feature_extraction.Recent.Recent(), 'params': [{'lastState': False}], 'findopt': False},
    {'method': lambda: feature_extraction.KHistory.KHistory(), 'params': [{'k': 4}, {'method': feature_extraction.Simple.Simple()}], 'findopt': False},
    {'method': lambda: feature_extraction.Simple.Simple(), 'params': [], 'findopt':False},
    {'method': lambda: feature_extraction.Cook.Cook1(), 'params': [],     'findopt':False},
    {'method': lambda: feature_extraction.Context.Diff(), 'params': [],     'findopt':False},
    {'method': lambda: feature_extraction.KHistory.KHistory(), 'params': [{'k': 2}, {'method': feature_extraction.Simple.Simple()}], 'findopt': False},
    {'method': lambda: feature_extraction.KHistory.KHistory(), 'params': [{'k': 3}, {'method': feature_extraction.Simple.Simple()}], 'findopt': False},

    {'method': lambda: feature_extraction.KHistory.KHistory(), 'params': [{'k': 1}, {'method': feature_extraction.Cook.Cook1()}], 'findopt': False},
    {'method': lambda: feature_extraction.KHistory.KHistory(), 'params': [{'k': 1}, {'method': feature_extraction.Simple.Simple()}], 'findopt': False},
    {'method': lambda: feature_extraction.DeepLearningFeatureExtraction.DeepLearningFeatureExtraction(), 'params': [
        {'var': 'size', 'min': 10, 'max': 20, 'type': 'int', 'init': 50},
        {'var': 'layers', 'min': 1, 'max': 3, 'type': 'int', 'init': pd.Timedelta(20, unit='s').total_seconds()}
    ],
        'findopt': False},

    #  {'method': lambda: feature_extraction.PAL_Features.PAL_Features(), 'params': [],  'findopt':False},
    {'method': lambda: feature_extraction.Raw.Classic(), 'params': [{'normalized': True}]},
    {'method': lambda: feature_extraction.Raw.Sequence(), 'params': [{'normalized': True}, {'per_sensor': True}]},
]


methods.dataset = [
    {'method': lambda: datatool.casas_handeler.CASAS('datasetfiles/CASAS/Home1/', 'Home1')},
    {'method': lambda: datatool.casas_handeler.CASAS('datasetfiles/CASAS/Home2/', 'Home2')},
    {'method': lambda: datatool.casas_handeler.CASAS('datasetfiles/CASAS/Aruba/', 'Aruba')},
    {'method': lambda: datatool.casas_handeler.CASAS('datasetfiles/CASAS/KaryoAdlNormal/', 'KaryoAdlNormal')},
    {'method': lambda: datatool.a4h_handeler.A4H('datasetfiles/A4H/', 'A4H')},
    {'method': lambda: datatool.vankasteren_handeler.VanKasteren('datasetfiles/VanKasteren/oldformat/', 'VanKasteren')},
    {'method': lambda: datatool.casas_handeler.CASAS('datasetfiles/VanKasteren/A/', 'Kasteren_A')},
    {'method': lambda: datatool.casas_handeler.CASAS('datasetfiles/VanKasteren/B/', 'Kasteren_B')},
    {'method': lambda: datatool.casas_handeler.CASAS('datasetfiles/VanKasteren/C/', 'Kasteren_C')},
]

methods.mlstrategy = [
    {'method': lambda: ml_strategy.Simple.NormalStrategy()},

    {'method': lambda: ml_strategy.WeightedGroup2.WeightedGroup2Strategy(alpha=20, mode=1)},
    {'method': lambda: ml_strategy.WeightedGroup2.WeightedGroup2Strategy(alpha=20, mode=2)},
    {'method': lambda: ml_strategy.WeightedGroup2.WeightedGroup2Strategy(alpha=20, mode=3)},
    {'method': lambda: ml_strategy.WeightedGroup.WeightedGroupStrategy(alpha=20)},
    {'method': lambda: ml_strategy.SeperateGroup.SeperateGroupStrategy()},
    {'method': lambda: ml_strategy.FastFinder.FastFinder(days=5)},


]

methods.optimizer = [

]
