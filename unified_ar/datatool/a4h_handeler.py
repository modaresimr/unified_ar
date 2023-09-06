from datatool.dataset_abstract import Dataset
import os
import wget
import pandas as pd
from intervaltree.intervaltree import IntervalTree
import json
from general.utils import Data

class A4H(Dataset):
    def __init__(self,data_path,data_dscr):
        super().__init__(data_path,data_dscr);
        
    def _load(self):
        rootfolder=self.data_path

        actfile = rootfolder+'activities.csv'
        sensfile = rootfolder+'sensor_events.csv'
        descfile = rootfolder+'sensor_description.csv'
        

        sensor_events = pd.read_csv(sensfile,)
        t = pd.to_datetime(sensor_events['time'], format='%Y-%m-%d %H:%M:%S')
        sensor_events.loc[:, 'time'] = t
        sensor_events = sensor_events.sort_values(['time'])
        sensor_events = sensor_events.reset_index()
        sensor_events = sensor_events.drop(columns=['index'])

        activity_events = pd.read_csv(actfile, index_col='Id')

        activity_events = activity_events.sort_values(['StartTime', 'EndTime'])
        st = pd.to_datetime(
            activity_events['StartTime'], format='%Y-%m-%d %H:%M:%S')
        et = pd.to_datetime(
            activity_events['EndTime'], format='%Y-%m-%d %H:%M:%S')
        activity_events['StartTime'] = st
        activity_events['EndTime'] = et
        # activity_events['Interval']=pd.IntervalIndex.from_arrays(activity_events['StartTime'],activity_events['EndTime'],closed='both')

        activities = activity_events['Activity'].unique()
        
        sensor_desc = pd.read_csv(descfile)
        sensor_desc.ItemRange = sensor_desc.ItemRange.apply(
            lambda x: json.loads(x))

        return activity_events, activities, sensor_events, sensor_desc
        
