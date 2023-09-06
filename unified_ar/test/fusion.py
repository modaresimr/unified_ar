
import pandas as pd
from combiner.SimpleCombiner import EmptyCombiner
from sklearn.utils.class_weight import compute_class_weight
import tensorflow_addons as tfa
import tensorflow as tf
from unified_ar.metric.event_confusion_matrix import event_confusion_matrix
from unified_ar.metric.CMbasedMetric import CMbasedMetric
from collections import defaultdict
import unified_ar.result_analyse.visualisation as vs
import logging
import numpy as np
from intervaltree.intervaltree import IntervalTree
from unified_ar.general.utils import Data
from unified_ar.general import utils
logger = logging.getLogger(__file__)

savedata = utils.loadState('sepg1c')
intree = savedata.intree

segments = defaultdict(dict)
for item in intree.items():
    segments[item.begin.value << 64 | item.end.value]['begin'] = item.begin
    segments[item.begin.value << 64 | item.end.value]['end'] = item.end
    segments[item.begin.value << 64 | item.end.value][item.data.gindx] = item.data
    # segments[str(item.begin)+'-'+str(item.end)][item.data.gindx]=item.data

print('finsihed')
acts = savedata.gacts[-1]  # savedata.acts
f = np.zeros((len(segments), len(savedata.gacts)*len(acts)))
label = np.zeros(len(segments))
times = []
iseg = 0
for timeseg in segments:
    seg = segments[timeseg]
    # b=timeseg>>64
    # e=((1<<64)-1)&timeseg
    b = seg['begin']  # .value
    e = seg['end']  # .value
    times.append((b, e))
    for g in range(len(savedata.gacts)):
        if (g in seg):
            label[iseg] = seg[g].real
            start = g*len(acts)
            end = (g+1)*len(acts)
            if (np.isnan(seg[g].train_q['f1'])):
                continue
            f[iseg, start:end] = seg[g].pred_prob
    iseg += 1


inputsize = (len(f[0]),)
outputsize = len(acts)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_shape=inputsize),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(outputsize, activation=tf.nn.softmax)
], name='test')
if (np.max(label) == 0):
    # self.trained=False
    cw = np.ones(len(acts))
else:
    cw = compute_class_weight("balanced", acts, label)

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')])
model.fit(f, label, epochs=3, class_weight=cw)

model.evaluate(f, label)


combiner = EmptyCombiner()
predicted = model.predict_classes(f)
# predicted   = np.argmax(model.predict(f), axis=1)
pred_events = []
ptree = {}
epsilon = pd.to_timedelta('1s')

for i in range(len(f)):

    start = times[i][0]
    end = times[i][1]
    pclass = predicted[i]

    pred_events.append({'Activity': pclass, 'StartTime': start, 'EndTime': end})

pred_events = pd.DataFrame(pred_events)
pred_events = pred_events.sort_values(['StartTime'])
pred_events = pred_events.reset_index()
pred_events = pred_events.drop(['index'], axis=1)

result = Data('result')
result.pred_events = pred_events
result.real_events = savedata.train_results[11].real_events

result.event_cm = event_confusion_matrix(result.real_events, result.pred_events, savedata.acts)
result.quality = CMbasedMetric(result.event_cm, 'macro')

print(result.quality)
