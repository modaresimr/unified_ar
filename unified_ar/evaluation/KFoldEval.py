from .evaluation_abstract import Evaluation
from unified_ar.general.utils import Data
from sklearn.model_selection import KFold
import logging
logger = logging.getLogger(__file__)


class KFoldEval(Evaluation):
    def __init__(self, fold):
        self.fold = fold

    def precompute(self, dataset):
        pass

    def process(self, dataset, strategy, fold, d):
        Train, Test = d
        logger.debug(f'=========Fold{fold} ============')
        acts = [a for a in dataset.activities_map]
        trainres = strategy.train(dataset, Train, acts)
        testres = strategy.test(Test)
        return {'test': testres, 'train': trainres}

    def evaluate(self, dataset, strategy):
        ttmaker = self.makeFoldTrainTest(
            dataset.sensor_events, dataset.activity_events, self.fold)
        models = {}
        for f, d in enumerate(ttmaker):
            from constants import methods
            methods.run_names['fold'] = f
            models[f] = self.process(dataset, strategy, f, d)

        return models

    def makeFoldTrainTest(self, sensor_events, activity_events, fold):
        sdate = sensor_events.time.apply(lambda x: x.date())
        adate = activity_events.StartTime.apply(lambda x: x.date())
        days = adate.unique()
        kf = KFold(n_splits=fold)
        kf.get_n_splits(days)

        for j, (train_index, test_index) in enumerate(kf.split(days)):
            Train0 = Data('train_fold_'+str(j))
            Train0.s_events = sensor_events.loc[sdate.isin(days[train_index])]
            Train0.a_events = activity_events.loc[adate.isin(days[train_index])]
            Train0.s_event_list = Train0.s_events.values
            Test0 = Data('test_fold_'+str(j))
            Test0.s_events = sensor_events.loc[sdate.isin(days[test_index])]
            Test0.a_events = activity_events.loc[adate.isin(days[test_index])]
            Test0.s_event_list = Test0.s_events.values

            yield Train0, Test0


class PKFoldEval(KFoldEval):
    def __init__(self, fold):
        super().__init__(fold)

    def evaluate(self, dataset, strategy):
        ttmaker = list(self.makeFoldTrainTest(
            dataset.sensor_events, dataset.activity_events, self.fold))
        models = {}

        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(lambda x: self.process(dataset, strategy, -1, x), ttmaker)
            return {i: r for i, r in result}

        return models
