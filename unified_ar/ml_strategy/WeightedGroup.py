from unified_ar.metric.event_confusion_matrix import event_confusion_matrix
from unified_ar.metric.CMbasedMetric import CMbasedMetric
from collections import defaultdict
import pandas as pd
from unified_ar.combiner.SimpleCombiner import EmptyCombiner
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from .abstract import MLStrategy
from .Simple import NormalStrategy
import unified_ar.result_analyse.visualisation as vs
import logging
import numpy as np
from intervaltree.intervaltree import IntervalTree
from unified_ar.general.utils import Data
from unified_ar.general import utils
logger = logging.getLogger(__file__)


class WeightedGroupStrategy(MLStrategy):
    def __init__(self, alpha):
        self.alpha = alpha

    def groupize(self, datasetdscr, acts):
        # gacts=[[a] for a in datasetdscr.activities_map]
        # gacts.append([a for a in datasetdscr.activities_map])
        gacts = [[a] for a in acts[1:]]
        gacts.append([a for a in acts])
        return gacts

    def train(self, datasetdscr, data, acts):
        self.gacts = self.groupize(datasetdscr, acts)
        self.acts = acts
        self.strategies = {}
        self.acts_name = {}
        train_results = {}
        self.train_quality = {}

        intree = IntervalTree()

        # intree = IntervalTree()
        for indx, tacts in enumerate(self.gacts):
            logger.info("=======================working on activties "+tacts.__str__()+"=========")
            weight = np.ones(len(acts))
            for a in tacts:
                weight[a] = self.alpha
            datasetdscr.indx = indx
            self.strategies[indx] = NormalStrategy()
            self.strategies[indx].train(datasetdscr, data, acts, weight)
            if ('result' in self.strategies[indx].bestOpt.result):
                result = self.strategies[indx].bestOpt.result['result']
            else:
                result = self.strategies[indx].test(data)

            utils.saveState(self.strategies[indx].get_info(), 'wgroupact', 'n-%d' % (indx))

            train_results[indx] = result
        utils.saveState([self.strategies[indx].get_info() for indx in self.strategies], 'wgroupact', 'n-all')
        return self.fusion(train_results, data.a_events, True)

    def test(self, data):
        test_results = {}
        for indx, tacts in enumerate(self.gacts):
            logger.info("=======================working on activties "+tacts.__str__()+"=========")
            result = self.strategies[indx].test(data)
            test_results[indx] = result

        return self.fusion(test_results, data.a_events, False)

    def fusion(self, results, real_events, isTrain):
        intree = IntervalTree()
        logger . info("\n=======================fusion activties ========")
        # intree = IntervalTree()
        # Segmentaion ###########################
        for indx, tacts in enumerate(self.gacts):
            result = results[indx]

            for i in range(0, len(result.Sdata.set_window)):
                idx = result.Sdata.set_window[i]
                start = result.Sdata.s_event_list[idx[0], 1]
                end = result.Sdata.s_event_list[idx[-1], 1]
                rcls = result.Sdata.label[i]
                pcls = result.predicted_classes[i]
                fullprob = result.predicted[i]
                if (end == start):
                    continue
                d = Data(str(i))
                d.real = rcls
                d.pred = pcls
                d.pred_prob = fullprob
                if (isTrain):
                    self.train_quality[indx] = result.quality

                d.gindx = indx
                # {'real':rcls,'pred':pcls,'pred_prob':fullprob,'train_q':result.quality}
                intree[start:end] = d

        intree.split_overlaps()
        segments = defaultdict(dict)
        for item in intree.items():
            segments[item.begin.value << 64 | item.end.value]['begin'] = item.begin
            segments[item.begin.value << 64 | item.end.value]['end'] = item.end
            segments[item.begin.value << 64 | item.end.value][item.data.gindx] = item.data

        # Feature Extraction ###########################
        f = np.zeros((len(segments), len(self.gacts)*len(self.acts)))
        label = np.zeros(len(segments))
        times = []
        iseg = 0
        for timeseg in segments:
            seg = segments[timeseg]
            b = seg['begin']
            e = seg['end']
            times.append({'begin': b, 'end': e})
            for indx in range(len(self.gacts)):
                if (indx in seg):
                    label[iseg] = seg[indx].real
                    start = indx*len(self.acts)
                    end = (indx+1)*len(self.acts)
                    if (self.train_quality[indx]['f1'] < 0.1):
                        continue
                    f[iseg, start:end] = seg[indx].pred_prob
            iseg += 1

        # TRAIN #######################

        if (isTrain):
            inputsize = (len(f[0]),)
            outputsize = len(self.acts)
            self.fusion_model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(128, input_shape=inputsize),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(outputsize, activation=tf.nn.softmax)
            ], name='fusion')
            if (np.max(label) == 0):
                # self.trained=False
                cw = np.ones(len(self.acts))
            else:
                cw = compute_class_weight("balanced", self.acts, label)

            self.fusion_model.summary()
            self.fusion_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')])
            self.fusion_model.fit(f, label, epochs=10, class_weight=cw)

        # EVALUATE #######################
        result = Data('result')
        result.results = results
        result.predicted = self.fusion_model.predict(f)
        result.predicted_classes = self.fusion_model.predict_classes(f)
        # predicted   = np.argmax(model.predict(f), axis=1)
        pred_events = []
        ptree = {}
        epsilon = pd.to_timedelta('1s')

        for i in range(len(f)):
            start = times[i]['begin']
            end = times[i]['end']
            pclass = result.predicted_classes[i]
            pred_events.append({'Activity': pclass, 'StartTime': start, 'EndTime': end})

        pred_events = pd.DataFrame(pred_events)
        pred_events = pred_events.sort_values(['StartTime'])
        pred_events = pred_events.reset_index()
        pred_events = pred_events.drop(['index'], axis=1)

        result.shortrunname = "fusion model" + str({r: results[r].shortrunname for r in results})
        result.times = times
        result.pred_events = pred_events
        result.real_events = real_events

        result.event_cm = event_confusion_matrix(result.real_events, result.pred_events, self.acts)
        result.quality = CMbasedMetric(result.event_cm, 'macro')
        result.functions = {r: results[r].functions for r in results}
        logger.debug('Evalution quality is %s' % result.quality)

        return result
