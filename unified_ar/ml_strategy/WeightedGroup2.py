import ml_strategy.abstract
import ml_strategy.Simple
import result_analyse.visualisation as vs
import logging
import numpy as np
from intervaltree.intervaltree import IntervalTree
from general.utils import Data 
from general import utils
logger = logging.getLogger(__file__)
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from combiner.SimpleCombiner import EmptyCombiner
import pandas as pd

from collections import defaultdict
from metric.CMbasedMetric import CMbasedMetric
from metric.event_confusion_matrix import event_confusion_matrix


class WeightedGroup2Strategy(ml_strategy.abstract.MLStrategy):
    def __init__(self,alpha,mode):
        self.alpha=alpha
        self.mode=mode
    def groupize(self,datasetdscr,acts):
        #gacts=[[a] for a in datasetdscr.activities_map]
        #gacts.append([a for a in datasetdscr.activities_map])
        gacts=[[a] for a in acts[1:]]
        gacts.append([a for a in acts])
        return gacts

    def train(self,datasetdscr,data,acts):
        self.gacts=self.groupize(datasetdscr,acts)
        self.acts=acts
        self.strategies ={}
        self.acts_name  ={}
        train_results   ={}
        self.train_quality={}

        intree=IntervalTree()
      	

        # intree = IntervalTree()
        for indx,tacts in enumerate(self.gacts):
            logger.info("=======================working on activties "+tacts.__str__()+"=========")
            weight=np.ones(len(acts))
            for a in tacts:weight[a]=self.alpha
            datasetdscr.indx=indx
            self.strategies[indx]=ml_strategy.Simple.SimpleStrategy()
            self.strategies[indx].train(datasetdscr,data,acts,weight)
            if('result' in self.strategies[indx].bestOpt.result):
                result=self.strategies[indx].bestOpt.result['result']
            else:
                result=self.strategies[indx].test(data)
            
            utils.saveState(self.strategies[indx].get_info(),'wgroupact','%d-%d'%(self.mode,indx))
            
            
            train_results[indx]=result
        utils.saveState([self.strategies[indx].get_info() for indx in self.strategies],'wgroupact','%d-all'%self.mode)
        return self.fusion(train_results,data.a_events,True)
        


    def test(self,data):
        test_results   ={}
        for indx,tacts in enumerate(self.gacts):
            logger.info("=======================working on activties "+tacts.__str__()+"=========")
            result=self.strategies[indx].test(data)
            test_results[indx]=result
            
        return self.fusion(test_results,data.a_events,False)



    def fusion(self,results,real_events,isTrain):                
        intree = IntervalTree()
        logger . info("\n=======================fusion activties ========")
        # intree = IntervalTree()
        # Segmentaion ###########################
        for indx, tacts in enumerate(self.gacts):
            result = results[indx]
            
            for i in range(0, len(result.Sdata.set_window)):
                idx         = result.Sdata.set_window[i]
                start       = result.Sdata.s_event_list[idx[0],1]
                end         = result.Sdata.s_event_list[idx[-1],1]
                rcls        = result.Sdata.label[i]
                pcls        = result.predicted_classes[i]
                fullprob    = result.predicted[i]
                if(end==start):
                    continue
                d            = Data(str(i))
                d.real       = rcls
                d.pred       = pcls
                d.pred_prob  = fullprob
                if(isTrain):
                    self.train_quality[indx]=result.quality
                
                d.gindx=indx
                # {'real':rcls,'pred':pcls,'pred_prob':fullprob,'train_q':result.quality}
                intree[start:end]=d

        intree.split_overlaps()
        segments = defaultdict(dict)
        for item in intree.items():
            segments[item.begin.value<<64|item.end.value]['begin']=item.begin
            segments[item.begin.value<<64|item.end.value]['end']=item.end
            segments[item.begin.value<<64|item.end.value][item.data.gindx]=item.data

        probs=np.zeros((len(segments),len(self.acts)))

        # Feature Extraction ###########################
        
        label=np.zeros(len(segments))
        times=[]
        iseg=0
        for timeseg in segments:
            seg=segments[timeseg]
            b=seg['begin']
            e=seg['end']
            times.append({'begin':b,'end':e})
            for indx in range(len(self.gacts)):
                if(indx in seg):
                    label[iseg]=seg[indx].real
                    if(self.mode==1):
                        probs[iseg,:]+=np.array(seg[indx].pred_prob) *self.train_quality[indx]['f1']/len(self.gacts)
                    elif self.mode==2:
                        p=np.zeros(len(self.acts))
                        p[np.argmax(seg[indx].pred_prob)]=1
                        probs[iseg,:]+=p
                    else:
                        p=np.zeros(len(self.acts))
                        p[np.argmax(seg[indx].pred_prob)]=self.train_quality[indx]['f1']
                        probs[iseg,:]+=p
            iseg+=1
        plabel=np.argmax(probs,1)
        

        

        #EVALUATE #######################
        result=Data('result')
        result.results=results
        result.predicted        =probs
        result.predicted_classes=plabel
        # predicted   = np.argmax(model.predict(f), axis=1) 
        pred_events      = []
        ptree       = {}
        epsilon=pd.to_timedelta('1s')

        for i in range(len(segments)): 
            start   = times[i]['begin']
            end     = times[i]['end']
            pclass  = result.predicted_classes[i]
            pred_events.append({'Activity': pclass, 'StartTime': start, 'EndTime': end})

        pred_events = pd.DataFrame(pred_events)
        pred_events = pred_events.sort_values(['StartTime'])
        pred_events = pred_events.reset_index()
        pred_events = pred_events.drop(['index'], axis=1)

        
        result.shortrunname = "fusion model" + str({r: results[r].shortrunname for r in results})
        result.times        = times
        result.pred_events  = pred_events
        result.real_events  = real_events

        result.event_cm     = event_confusion_matrix(result.real_events,result.pred_events,self.acts)
        result.quality      = CMbasedMetric(result.event_cm,'macro')
        result.functions    = {r: results[r].functions for r in results}
        logger.debug('Evalution quality is %s'%result.quality)

        return result

    def shortname(self):
        return f'{super().shortname()} mode={self.mode}'
    