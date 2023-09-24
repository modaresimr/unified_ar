import pandas as pd
from unified_ar.general.utils import Data
import unified_ar as ar

no_memory_limit = True


# import unified_ar.classifier.PyActLearn


# import ar.feature_extraction.PAL_Features
# from unified_ar.general.libimport import *

methods = Data('methods')
methods.run_names = {'out': 'temp'}
methods.meta_segmentation_sub_tasks = [

    {'method': lambda: ar.segmentation.FixedEventWindow.FixedEventWindow(), 'params': [
        {'var': 'size', 'min': 2, 'max': 30, 'type': 'int', 'init': 5, 'range': [2, 5, 8, 11, 15, 20, 30]},
        {'var': 'shift', 'min': 2, 'max': 20, 'type': 'int', 'init': 2, 'range': [2, 5, 8, 11, 15, 20]}
    ], 'findopt': True},
    {'method': lambda: ar.segmentation.FixedSlidingWindow.FixedSlidingWindow(), 'params': [
        {'var': 'size', 'min': 60, 'max': 15 * 60, 'type': 'float', 'init': 120 / 4, 'range': list(range(15, 120, 15))},
        {'var': 'shift', 'min': 10, 'max': 7 * 60, 'type': 'float', 'init': 60 / 2, 'range': list(range(15, 120, 15))}
    ], 'findopt': True},
    {'method': lambda: ar.segmentation.Probabilistic.Probabilistic(), 'params': [], 'findopt': False},
    # {'method': lambda:segmentation.FixedTimeWindow.FixedTimeWindow(), 'params':[
    #                  {'var':'size','min':pd.Timedelta(1, unit='s').total_seconds(), 'max': pd.Timedelta(30, unit='m').total_seconds(), 'type':'float','init':pd.Timedelta(15, unit='s').total_seconds()},
    #                  {'var':'shift','min':pd.Timedelta(1, unit='s').total_seconds(), 'max': pd.Timedelta(30, unit='m').total_seconds(), 'type':'float','init':pd.Timedelta(1, unit='s').total_seconds()}
    # ],'findopt':False},
]


methods.segmentation = [
    {'method': lambda: ar.segmentation.FixedEventWindow.FixedEventWindow(), 'params': [
        {'size': 15, 'range': list(range(10, 26, 5))},
        # {'size':10, 'min': 10, 'max': 30},
        {'shift': 5, 'range': list(range(10, 16, 5))}
    ], 'findopt': False},
    {'method': lambda: ar.segmentation.FixedSlidingWindow.FixedSlidingWindow(), 'params': [
        {'size': 30, 'range': list(range(15, 76, 15))},
        {'shift': 10, 'range': list(range(15, 45, 15))}
        # {'var': 'size', 'min': 60, 'max': 15 * 60, 'type': 'float', 'init': 120 / 4, 'range': list(range(15, 76, 15))},
        # {'var': 'shift', 'min': 10, 'max': 7 * 60, 'type': 'float', 'init': 60 / 2, 'range': list(range(15, 45, 15))}
    ], 'findopt': False},
    {'method': lambda: ar.segmentation.Probabilistic.Probabilistic(), 'params': [], 'findopt': False},

    {'method': lambda: ar.segmentation.FixedEventWindow.FixedEventWindow(), 'params': [
        {'size': 120},
        {'shift': 20}
    ], 'findopt': False},

    {'method': lambda: ar.segmentation.ActivityWindow.SlidingEventActivityWindow(), 'params': [
        {'size': 250},
        {'shift': 120}
    ], 'findopt': False},

    {'method': lambda: ar.segmentation.MetaDecomposition.SWMeta(), 'params': [
        {'meta_size': '24h'},
        {'meta_overlap_rate': 1},
        {'meta_mode': 'keras'}
    ], 'findopt': False
    },
    
    # {'method': lambda:segmentation.FixedTimeWindow.FixedTimeWindow(), 'params':[
    #                  {'var':'size','min':pd.Timedelta(1, unit='s').total_seconds(), 'max': pd.Timedelta(30, unit='m').total_seconds(), 'type':'float','init':pd.Timedelta(15, unit='s').total_seconds()},
    #                  {'var':'shift','min':pd.Timedelta(1, unit='s').total_seconds(), 'max': pd.Timedelta(30, unit='m').total_seconds(), 'type':'float','init':pd.Timedelta(1, unit='s').total_seconds()}
    # ],'findopt':False},
]

methods.preprocessing = [
    {'method': lambda: ar.preprocessing.SimplePreprocessing.SimplePreprocessing()},
]
methods.classifier = [
    
    {'method': lambda: ar.classifier.DeepW2V.FCN(), 'params': [
        {'epochs': 400},
        {'batch_size': 64}
    ]},
    {'method': lambda: ar.classifier.DeepW2V.FCNEmbedded(), 'params': [
        {'epochs': 400},
        {'batch_size': 64},
        {'vocab_size': 1000}
    ]},
    {'method': lambda: ar.classifier.DeepW2V.LiciottiBiLSTM(), 'params': [
        {'epochs': 400},
        {'batch_size': 64},
        {'vocab_size': 1000},
        {"emb_dim": 64},
        {"nb_units": 64}
    ]},
    {'method': lambda: ar.classifier.Keras.SimpleKeras(), 'params': [
        {'epochs': 400},
        {'batch_size': 64},
    ]},
    {'method': lambda: ar.classifier.Keras.NormalKeras(), 'params': [
        {'epochs': 400},
        {'batch_size': 64},
    ]},
    {'method': lambda: ar.classifier.Keras.LSTMTest(), 'params': [
        {'epochs': 400},
        {'batch_size': 64},
    ]},
{'method': lambda: ar.classifier.CNN_LSTM.CNN_LSTM(), 'params': [
        {'epochs': 400},
        {'batch_size': 64}
    ]},
    {'method': lambda: ar.classifier.MySKLearn.UAR_RandomForest(), 'params': [
        {'n_estimators': 20},
        {'random_state': 0},
        {'max_depth': 12},
        {'max_features_rate': .5}
    ]},
    
    {'method': lambda: ar.classifier.Keras.LSTMAE(), 'params': [
        {'epochs': 400},
        {'batch_size': 64},
    ]},
    {'method': lambda: ar.classifier.libsvm.LibSVM()},

    # {'method': lambda: ar.classifier.PyActLearn.PAL_LSTM_Legacy(), 'params': [
    #     {'var': 'epochs', 'init': 3}
    # ]},
    {'method': lambda: ar.classifier.MySKLearn.UAR_KNN(), 'params': [
        {'k': 5},
    ]},
    {'method': lambda: ar.classifier.MySKLearn.UAR_SVM(), 'params': [
        {'kernel': 'rbf'},
        {'gamma': 1},
        {'C': 100.},
        {'decision_function_shape': 'ovr'}
    ]},
    {'method': lambda: ar.classifier.MySKLearn.UAR_SVM2(), 'params': [
        {'kernel': 'linear'},
        {'gamma': 1},
        {'C': 100.},
        {'decision_function_shape': 'ovr'}
    ]},
    {'method': lambda: ar.classifier.MySKLearn.UAR_DecisionTree(), 'params': []},
]


methods.classifier_metric = [
    {'method': lambda: ar.metric.classical.Accuracy()},
    # {'method': lambda: Accuracy()},
]

methods.event_metric = [
    # {'method': lambda: ar.metric.Accuracy.Accuracy()},
    # {'method': lambda: Accuracy()},
]

methods.activity_fetcher = [
    # {'method': lambda: ar.activity_fetcher.MaxActivityFetcher.MaxActivityFetcher()},
    {'method': lambda: ar.activity_fetcher.CookActivityFetcher.CookActivityFetcher()}
]
methods.combiner = [
    {'method': lambda: ar.combiner.SimpleCombiner.EmptyCombiner2()},
    # {'method':lambda: ar.combiner.SimpleCombiner.SimpleCombiner()},
    # {'method':lambda: ar.combiner.SimpleCombiner.EmptyCombiner()},
]
methods.evaluation = [
    {'method': lambda: ar.evaluation.SplitEval.SplitEval()},
    # {'method': lambda: ar.evaluation.KFoldEval.KFoldEval(5)},
    #  {'method': lambda: ar.evaluation.KFoldEval.PKFoldEval(5)},

]


methods.feature_extraction = [
    {'method': lambda: ar.feature_extraction.NLP.SensorWord(), 'params': [{'vocab_size': 1000}], 'findopt': False},
    {'method': lambda: ar.feature_extraction.Recent.Recent(), 'params': [{'lastState': False}], 'findopt': False},
    {'method': lambda: ar.feature_extraction.KHistory.KHistory(), 'params': [{'k': 4}, {'method': ar.feature_extraction.Simple.Simple()}], 'findopt': False},
    {'method': lambda: ar.feature_extraction.Simple.Simple(), 'params': [], 'findopt': False},
    {'method': lambda: ar.feature_extraction.Cook.Cook1(), 'params': [], 'findopt': False},
    {'method': lambda: ar.feature_extraction.Context.Diff(), 'params': [], 'findopt': False},
    {'method': lambda: ar.feature_extraction.KHistory.KHistory(), 'params': [{'k': 2}, {'method': ar.feature_extraction.Simple.Simple()}], 'findopt': False},
    {'method': lambda: ar.feature_extraction.KHistory.KHistory(), 'params': [{'k': 3}, {'method': ar.feature_extraction.Simple.Simple()}], 'findopt': False},

    {'method': lambda: ar.feature_extraction.KHistory.KHistory(), 'params': [{'k': 1}, {'method': ar.feature_extraction.Cook.Cook1()}], 'findopt': False},
    {'method': lambda: ar.feature_extraction.KHistory.KHistory(), 'params': [{'k': 1}, {'method': ar.feature_extraction.Simple.Simple()}], 'findopt': False},
    {'method': lambda: ar.feature_extraction.DeepLearningFeatureExtraction.DeepLearningFeatureExtraction(), 'params': [
        {'var': 'size', 'min': 10, 'max': 20, 'type': 'int', 'init': 50},
        {'var': 'layers', 'min': 1, 'max': 3, 'type': 'int', 'init': pd.Timedelta(20, unit='s').total_seconds()}
    ],
        'findopt': False},

    #  {'method': lambda: ar.feature_extraction.PAL_Features.PAL_Features(), 'params': [],  'findopt':False},
    {'method': lambda: ar.feature_extraction.Raw.Classic(), 'params': [{'normalized': True}]},
    {'method': lambda: ar.feature_extraction.Raw.Sequence(), 'params': [{'normalized': True}, {'per_sensor': True}]},
]


methods.dataset = [
    {'method': lambda: ar.datatool.casas_handeler.CASAS('datasets/casas/home1/', 'Home1')},
    {'method': lambda: ar.datatool.casas_handeler.CASAS('datasets/casas/home2/', 'Home2')},
    {'method': lambda: ar.datatool.casas_handeler.CASAS('datasets/casas/aruba/', 'Aruba')},
    {'method': lambda: ar.datatool.casas_handeler.CASAS('datasets/casas/karyo_adl_normal/', 'KaryoAdlNormal')},
    {'method': lambda: ar.datatool.a4h_handeler.A4H('datasets/orange4home/', 'A4H')},
    # {'method': lambda: ar.datatool.vankasteren_handeler.VanKasteren('datasets/VanKasteren/oldformat/', 'VanKasteren')},
    {'method': lambda: ar.datatool.casas_handeler.CASAS('datasets/van_kasteren/A/', 'Kasteren_A')},
    {'method': lambda: ar.datatool.casas_handeler.CASAS('datasets/van_kasteren/B/', 'Kasteren_B')},
    {'method': lambda: ar.datatool.casas_handeler.CASAS('datasets/van_kasteren/C/', 'Kasteren_C')},
]

methods.mlstrategy = [
    {'method': lambda: ar.ml_strategy.Simple.NormalStrategy()},

    {'method': lambda: ar.ml_strategy.WeightedGroup2.WeightedGroup2Strategy(alpha=20, mode=1)},
    {'method': lambda: ar.ml_strategy.WeightedGroup2.WeightedGroup2Strategy(alpha=20, mode=2)},
    {'method': lambda: ar.ml_strategy.WeightedGroup2.WeightedGroup2Strategy(alpha=20, mode=3)},
    {'method': lambda: ar.ml_strategy.WeightedGroup.WeightedGroupStrategy(alpha=20)},
    {'method': lambda: ar.ml_strategy.SeperateGroup.SeperateGroupStrategy()},
    {'method': lambda: ar.ml_strategy.FastFinder.FastFinder(days=5)},


]

methods.optimizer = [

]
