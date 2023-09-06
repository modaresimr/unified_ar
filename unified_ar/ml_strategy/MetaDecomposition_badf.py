import auto_profiler
import logging
import pandas as pd
from sklearn.metrics import confusion_matrix

from feature_extraction.feature_abstract import featureExtraction
from general.utils import Data, MyTask
import general.utils
from metric.CMbasedMetric import CMbasedMetric
from metric.event_confusion_matrix import event_confusion_matrix
# from metric.EventBasedMetric import EventBasedMetric
import ml_strategy.abstract
from optimizer.BruteForce import method_param_selector
from optimizer.OptLearn import OptLearn, ParamMaker
from segmentation.segmentation_abstract import prepare_segment2

logger = logging.getLogger(__file__)

class SimpleMeta(ml_strategy.abstract.MLStrategy):
    def __init__(self):
        pass
    def shortname(self):
        return super().shortname()
        
    def train(self, datasetdscr, data, acts,weight=None):
        self.datasetdscr=datasetdscr
        self.acts=acts
        self.weight=weight
        self.traindata=self.justifySet(self.acts,data)
        
        import copy
        from constants import methods
        
        import general.Cache as Cache
        Cache.GlobalDisable=True

        
        ssize=pd.to_timedelta('1d')
        overlap=ssize*1.0
        
        starts=data.s_events.time.dt.floor(overlap).unique()
        ends=starts+ssize
        meta_features=[]
        meta_targets=[]
        fast_strategy=ml_strategy.Simple.NormalStrategy()
        for s,e in zip(starts,ends):
            data2=self.customSplit(data,s,e)
            print(f's={s} : {e}============= #sevent={len(data2.s_events)} #aevents={len(data2.a_events)}')
            if len(data2.s_events)==0 or len(data2.a_events)==0:
                continue
            
            result=fast_strategy.train(datasetdscr,data2,acts,weight,update_model=True)
            logger.debug(fast_strategy.get_info().functions)
            # result=fast_strategy.pipeline(fast_strategy.functions,self.traindata,train=True)
            aggr=data2.s_events.groupby('SID').count()
            fea={k: aggr.loc[k]['value'] if k in aggr.index else 0 for k in datasetdscr.sensor_id_map_inverse}
            fea['time']=s
            meta_features.append(fea)
            seg=result.functions['segmentor']
            meta_targets.append({'method':seg[0], **{p:seg[1][p] for p in seg[1]}})
            # self.strategy=fast_strategy

        d={'meta_features':meta_features, 'meta_target':meta_targets }
        # print(d)
        import general.utils
        general.utils.saveState(d,'temp')

        finalFeatureDf=pd.DataFrame(meta_features)
        # print('resssssssssssssssssssssssssssssss===========')
        # print(finalFeatureDf)
        finaltargetDf=pd.DataFrame(meta_targets)
        # print(finaltargetDf)

        return result
    
    def test(self, data):
        return self.strategy.test(data)

    def customSplit(self, data,start,end):
        sensor_events=data.s_events
        activity_events=data.a_events
        
        Train0 = Data(f'train_random_days {start}-{end}')
        print(f's={start} e={end}')

        Train0.s_events = sensor_events.loc[(sensor_events.time>=start)& (sensor_events.time<end)]
        Train0.a_events = activity_events.loc[(activity_events.StartTime>=start)&(activity_events.StartTime<end)]
        Train0.s_event_list=Train0.s_events.values

        return  Train0