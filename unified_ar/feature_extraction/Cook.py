from feature_extraction.feature_abstract import FeatureExtraction
import pandas as pd
import numpy as np
class Cook1(FeatureExtraction):
    def getShape(self):
        scount=sum(1 for x in self.datasetdscr.sensor_id_map);
        return (scount+3,)
        
    sec_in_day=(60*60*24)
    
    def featureExtract(self,win):
        window=win['window']
        scount=sum(1 for x in self.datasetdscr.sensor_id_map);
        f=np.ones(scount+3)*-1
        for i in range(0,window.shape[0]):
            f[self.datasetdscr.sensor_id_map_inverse[window.iat[i,0]]]=window.iat[i,2]   #f[sensor_id_map_inverse[x.SID]]=1
        stime=window.iat[0,1]#startdatetime
        etime=window.iat[-1,1]#enddatetime
        ts=(stime-pd.to_datetime(stime.date())).total_seconds()
        f[scount+0]=ts/self.sec_in_day
        f[scount+1]=(etime-stime).total_seconds()/self.sec_in_day
        f[scount+2]=len(window)
        return f        

    def featureExtract2(self,s_event_list,idx):
        window=s_event_list
        scount=sum(1 for x in self.datasetdscr.sensor_id_map);
        f=np.ones(scount+3)*-1
        for i in idx:
            f[self.datasetdscr.sensor_id_map_inverse[window[i,0]]]=window[i,2]   #f[sensor_id_map_inverse[x.SID]]=1
        stime=window[idx[0] ,1]#startdatetime
        etime=window[idx[-1],1]#enddatetime
        ts=(stime-pd.to_datetime(stime.date())).total_seconds()
        f[scount+0]=ts/self.sec_in_day
        f[scount+1]=(etime-stime).total_seconds()/self.sec_in_day
        f[scount+2]=len(idx)
        return f        