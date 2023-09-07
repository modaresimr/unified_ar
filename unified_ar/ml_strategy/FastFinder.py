import auto_profiler
import logging

from sklearn.metrics import confusion_matrix

from unified_ar import Data, MyTask
import unified_ar.general.utils
from unified_ar.metric.CMbasedMetric import CMbasedMetric
from unified_ar.metric.event_confusion_matrix import event_confusion_matrix
# from unified_ar.metric.EventBasedMetric import EventBasedMetric
from .abstract import MLStrategy
from unified_ar.optimizer.BruteForce import method_param_selector
from unified_ar.optimizer.OptLearn import OptLearn, ParamMaker
from unified_ar.segmentation.segmentation_abstract import prepare_segment2

logger = logging.getLogger(__file__)


class FastFinder(MLStrategy):
    def __init__(self, days=10):
        self.days = days

    def shortname(self):
        return super().shortname() + f' days={self.days}'

    def train(self, datasetdscr, data, acts, weight=None):
        self.datasetdscr = datasetdscr
        self.acts = acts
        self.weight = weight
        self.traindata = self.justifySet(self.acts, data)

        import copy
        from unified_ar.constants import methods

        import unified_ar.general.Cache as Cache
        Cache.GlobalDisable = True
        data2 = self.fewDaysSplit(data, 10)
        fast_strategy = ml_strategy.Simple.NormalStrategy()
        fast_strategy.train(datasetdscr, data2, acts, weight)
        logger.debug(fast_strategy.get_info().functions)
        result = fast_strategy.pipeline(fast_strategy.functions, self.traindata, train=True)
        self.strategy = fast_strategy
        return result

    def test(self, data):
        return self.strategy.test(data)

    def fewDaysSplit(self, data, count):
        sensor_events = data.s_events
        activity_events = data.a_events
        sdate = sensor_events.time.apply(lambda x: x.date())
        adate = activity_events.StartTime.apply(lambda x: x.date())
        days = adate.unique()
        import random
        selecteddays = random.sample(list(days), count)

        Train0 = Data('train_random_days' + str(selecteddays))
        Train0.s_events = sensor_events.loc[sdate.isin(selecteddays)]
        Train0.a_events = activity_events.loc[adate.isin(selecteddays)]
        Train0.s_event_list = Train0.s_events.values

        return Train0
