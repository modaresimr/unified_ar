import pandas as pd

from .evaluation_abstract import Evaluation
from unified_ar.general.utils import Data
from unified_ar.general.utils import saveState
from unified_ar.general.utils import saveFunctions


class SplitEval(Evaluation):

    def evaluate(self, dataset, strategy):
        self.dataset = dataset
        from constants import methods
        methods.run_names['fold'] = 0

        Train, Test = self.makeTrainTest(dataset.sensor_events, dataset.activity_events)
        acts = [a for a in dataset.activities_map]
        trainres = strategy.train(dataset, Train, acts)
        testres = strategy.test(Test)
        return {0: {'test': testres, 'train': trainres}}

    def makeTrainTest(self, sensor_events, activity_events):
        # dataset_split = min(activity_events.StartTime) + ((max(activity_events.EndTime)-min(activity_events.StartTime))*2/10)
        # dataset_split = min(activity_events.StartTime) + pd.Timedelta('1491684s')

        test_start = min(activity_events.StartTime)
        test_end = test_start+((max(activity_events.EndTime)-min(activity_events.StartTime))*3/10)
        if (self.dataset.data_dscr == 'Home2'):  # for using HHMM data
            test_start = pd.to_datetime('1253551880', unit='s')
            test_end = pd.to_datetime('1256157814', unit='s')
        elif (self.dataset.data_dscr == 'Home1'):  # for using HHMM data
            # test_start = pd.to_datetime('1247845945', unit='s')
            test_end = pd.to_datetime('1249337628', unit='s')
        # dataset_split = pd.to_datetime(dataset_split.date())  # day
        Train = Data('train')
        Test = Data('test')
        Train.s_events = sensor_events[(sensor_events.time < test_start) | (sensor_events.time > test_end)]
        Train.a_events = activity_events[(activity_events.StartTime < test_start) | (activity_events.EndTime > test_end)]
        Train.s_event_list = Train.s_events.values

        Test.s_events = sensor_events[(sensor_events.time >= test_start) & (sensor_events.time <= test_end)]
        Test.a_events = activity_events[(activity_events.EndTime >= test_start) & (activity_events.StartTime <= test_end)]
        Test.s_event_list = Test.s_events.values
        return Train, Test
