import auto_profiler
import logging

from sklearn.metrics import confusion_matrix

from unified_ar.feature_extraction.feature_abstract import featureExtraction
from unified_ar import Data, MyTask
import unified_ar.general.utils
from unified_ar.metric.CMbasedMetric import CMbasedMetric
from unified_ar.metric.event_confusion_matrix import event_confusion_matrix
# from unified_ar.metric.EventBasedMetric import EventBasedMetric
from unified_ar.ml_strategy.abstract import MLStrategy
from unified_ar.optimizer.BruteForce import method_param_selector
from unified_ar.optimizer.OptLearn import OptLearn, ParamMaker
from unified_ar.segmentation.segmentation_abstract import prepare_segment2

logger = logging.getLogger(__file__)


class NormalStrategy(MLStrategy):
    def train(self, datasetdscr, data, acts, weight=None, update_model=True):
        self.datasetdscr = datasetdscr
        self.acts = acts
        self.update_model = update_model
        self.weight = weight
        self.traindata = self.justifySet(self.acts, data)
        uniqueKey = {'strategy': 'simple', 'acts': acts, 'weight': weight, 'dataset': datasetdscr.shortname()}
        bestOpt = method_param_selector(self.learning, uniqueKey)
        self.functions = bestOpt.functions
        self.bestOpt = bestOpt

        if ('result' in bestOpt.result):
            result = bestOpt.result['result']
        else:
            result = test(data)
        return result

    def learning(self, func):
        result = self.pipeline(func, self.traindata, train=True, update_model=self.update_model)
        if result is None:
            return 100000, None
        return result.quality['f1'], result

    def get_info(self):
        func = self.functions
        result = Data('Result')
        result.shortrunname = func.shortrunname
        result.functions = {}
        for f in func.__dict__:
            obj = func.__dict__[f]
            if isinstance(obj, MyTask):
                result.functions[f] = (obj.shortname(), obj.params)
        return result

    def pipeline(self, func, data, train, update_model=False):
        import os
        os.system("taskset -a -pc 0-1000 %d >>/dev/null" % os.getpid())

        func.acts = self.acts
        logger.debug('Starting .... %s' % (func.shortrunname))
        Tdata = func.preprocessor.process(self.datasetdscr, data)
        logger.debug('Preprocessing Finished %s' % (func.preprocessor.shortname()))
        if hasattr(self, 'ui_debug'):
            func.ui_debug = self.ui_debug
        Sdata = prepare_segment2(func, Tdata, self.datasetdscr, train)
        logger.debug('Segmentation Finished %d segment created %s' % (len(Sdata.set_window), func.segmentor.shortname()))
        Sdata.set = featureExtraction(func.featureExtractor, self.datasetdscr, Sdata, True)
        logger.debug('FeatureExtraction Finished shape %s , %s' % (str(Sdata.set.shape), func.featureExtractor.shortname()))
        if len(Sdata.set) == 0:
            return None
        if (train):
            func.classifier.createmodel(Sdata.set[0].shape, len(self.acts), update_model=update_model)
            func.classifier.setWeight(self.weight)
            logger.debug('Classifier model created  %s' % (func.classifier.shortname()))
            func.classifier.train(Sdata.set, Sdata.label)
            logger.debug('Classifier model trained  %s' % (func.classifier.shortname()))

        # logger.debug("Evaluating....")
        result = Data('Result')
        result.shortrunname = func.shortrunname
        result.Sdata = Sdata
        result.functions = {}
        for f in func.__dict__:
            obj = func.__dict__[f]
            if isinstance(obj, MyTask):
                result.functions[f] = (obj.shortname(), obj.params)

        result.predicted = func.classifier.predict(Sdata.set)
        result.predicted_classes = func.classifier.predict_classes(Sdata.set)
        pred_events = func.combiner.combine(Sdata.s_event_list, Sdata.set_window, result.predicted)
        logger.debug('events merged  %s' % (func.combiner.shortname()))

        result.pred_events = pred_events
        result.real_events = data.a_events
        import sklearn
        result.cm = sklearn.metrics.confusion_matrix(result.Sdata.label, result.predicted_classes, labels=self.acts)
        result.event_cm = event_confusion_matrix(data.a_events, pred_events, self.acts)
        result.quality = CMbasedMetric(result.event_cm, 'macro', self.weight)
        # eventeval=EventBasedMetric(Sdata.a_events,pred_events,self.acts)

        logger.debug('Evalution quality is %s' % result.quality)
        return result

    def test(self, data):
        func = self.functions
        data = self.justifySet(self.acts, data)
        func.acts = self.acts
        result = self.pipeline(func, data, train=False)
        return result
