from .dataset_abstract import Dataset
import os
import wget
import pandas as pd
from intervaltree.intervaltree import IntervalTree
import json
from .. import Data,utils
import numpy as np


class SED(Dataset):
    def __init__(self, datadir, title, srcDataset, **kwargs):
        super().__init__(datadir, title)
        self.srcDataset = srcDataset

    def _load(self):
        # rootfolder='datasetfiles/CASAS/'+name+'/'
        pred = pd.read_csv(self.data_path, delimiter='\t')
        if(self.srcDataset==None):
            self.files={p:i for (i,p) in enumerate(pred['filename'].unique())}
            self.max_len=pred['offset'].max()*1.5
            activities=None
            #activities = pred['event_label'].unique()
        else:
            self.files=self.srcDataset.files
            self.max_len=self.srcDataset.max_len
            activities=self.srcDataset.activities[1:]
            

        all=[]
        for i,x in pred.iterrows():
        #    if(i>10):continue
            p={}
            if not(x['filename'] in self.files):
                self.files[x['filename']]=max(self.files.values())+1
            
            t=self.files[x['filename']] * self.max_len 
            p['StartTime'] = pd.to_datetime(x['onset'] + t, unit='s')
            p['EndTime'] = pd.to_datetime(x['offset'] + t, unit='s')
            p['Activity'] = x['event_label']
            if(np.isnan(x['onset'])):continue
            all.append(p)
        
        activity_events=pd.DataFrame(all)
        activity_events = activity_events.sort_values(['StartTime', 'EndTime'])
        activity_events['Duration'] = activity_events['EndTime']-activity_events['StartTime']
        if(activities is None):activities = activity_events['Activity'].unique()

        # 3
        # sensor_events=sensor_events.drop(columns=['activity_hint'])

        sensor_events = pd.DataFrame(columns=["SID", "time", "value"])
        sens = {
            1:	'Microwave',
        }

        #activities = ['alarm', 'crying baby', 'crash', 'barking dog', 'running engine', 'female scream',
        #              'female speech', 'burning fire', 'footsteps', 'knocking on door', 'male scream', 'male speech',
        #              'ringing phone', 'piano']

        sensor_desc = pd.DataFrame(columns=['ItemId', 'ItemName', 'Cumulative',
                                            'Nominal', 'OnChange', 'ItemRange', 'Location', 'Object', 'SensorName'])
        tmp_sensors = sensor_events['SID'].unique()
        for k in sens:
            item = {'ItemId': sens[k], 'ItemName': sens[k], 'Cumulative': 0, 'Nominal': 1, 'OnChange': 1, 'ItemRange': {
                'range': ['0', '1']}, 'Location': 'None', 'Object': 'None', 'SensorName': 'None'}
            sensor_desc = sensor_desc.append(item, ignore_index=True)

        return activity_events, activities, sensor_events, sensor_desc

