import logging

import numpy as np
import pandas as pd
from intervaltree.intervaltree import IntervalTree

from .combiner_abstract import Combiner

logger = logging.getLogger(__file__)

# print([p['start'] for p in sw[label==7]])
# pev.loc[pev.Activity==7]



class KerasCombiner(Combiner):
    def precompute2(self, times, act_data, labels):
        trainX,trainY=self.create_XY(times,act_data,labels)
        from unified_ar.classifier.Keras import SimpleKeras
        keras=SimpleKeras()
        keras.applyParams({'epochs':100,'batch_size':32,'verbose':1})
        keras.createmodel(trainX[0].shape,max(trainY))
        keras.train(trainX,trainY)
        self.keras=keras

        
    def create_XY(self,times,act_data,labels=None):
        predicted = np.argmax(act_data, axis=1)
        events = []
        ptree = {}
        trainX=[]
        trainY=[]
        epsilon = pd.to_timedelta('1s')
        for i in range(1,len(times)):
            start = times[i]['begin']
            end = times[i]['end']
            # pclass = np.argmax(predicted[i])
            
            trainX.append([predicted[i-1],predicted[i],*act_data[i]])
            if labels is not None:
                trainY.append(labels[i])
        
        
        return trainX,trainY

    def combine2(self, times, act_data):
        testX,_=self.create_XY(times,act_data,None)
        predicted= self.keras.predict_classes(testX)
        # predicted = np.argmax(act_data, axis=1)
        events = []
        ptree = {}
        epsilon = pd.to_timedelta('1s')

        for i in range(len(times)):
            start = times[i]['begin']
            end = times[i]['end']
            self.keras.predict_classes()
            # pclass = np.argmax(predicted[i])
            pclass = predicted[i]
            if (pclass == 0):
                continue
            if len(events) > 0:
                priority_new=False
                if not priority_new:
                    start=max(events[-1]['EndTime'],start)
                    if start >= end:
                        start=end-epsilon

                events[-1]['EndTime'] = min(events[-1]['EndTime'], start)
                if events[-1]['StartTime'] >= events[-1]['EndTime']:
                    events.pop()
                
                    
            newe = {'Activity': pclass, 'StartTime': start, 'EndTime': end}
            if (len(events) > 0 and events[-1]['Activity'] == newe['Activity'] and events[-1]['EndTime'] < newe['StartTime']):
                events.append({'Activity': pclass, 'StartTime': events[-1]['EndTime'], 'EndTime': newe['StartTime']})
            events.append(newe)

        events = pd.DataFrame(events)
        if (len(events) > 0):
            events = events.sort_values(['StartTime'])
        events = events.reset_index()
        events = events.drop(['index'], axis=1)
        return events

