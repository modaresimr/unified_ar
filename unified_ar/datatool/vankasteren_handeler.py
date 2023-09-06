from datatool.dataset_abstract import Dataset
import os
import wget
import pandas as pd
from intervaltree.intervaltree import IntervalTree
import json
from general.utils import Data


class VanKasteren(Dataset):
    def __init__(self,data_path,data_dscr):
        super().__init__(data_path,data_dscr);
        
    def _load(self):
        rootfolder = self.data_path
        sensefile = rootfolder + "sensedata.txt"
        actfile = rootfolder   + "actdata.txt"

        all = pd.read_csv(sensefile, '\t', None, header=0, names=[
            "StartTime", "EndTime", "SID", "value"])

        all.StartTime = pd.to_datetime(all.StartTime, format='%d-%b-%Y %H:%M:%S')
        all.EndTime = pd.to_datetime(all.EndTime, format='%d-%b-%Y %H:%M:%S')
        acts = {
            0: 'None',
            1: 'leave house',
            4: 'use toilet',
            5: 'take shower',
            10: 'go to bed',
            13: 'prepare Breakfast',
            15: 'prepare Dinner',
            17: 'get drink'}

        sens = {
            1:	'Microwave',
            5:	'Hall-Toilet door',
            6:	'Hall-Bathroom door',
            7:	'Cups cupboard',
            8:	'Fridge',
            9:	'Plates cupboard',
            12:	'Frontdoor',
            13:	'Dishwasher',
            14:	'ToiletFlush',
            17:	'Freezer',
            18:	'Pans Cupboard',
            20:	'Washingmachine',
            23:	'Groceries Cupboard',
            24:	'Hall-Bedroom door'}

        sensor_events = pd.DataFrame(columns=["SID", "time", "value"])
        for i, s in all.iterrows():
            sensor_events = sensor_events.append({'SID': sens[s.SID], 'time': s.StartTime, 'value': s.value}, ignore_index=True)
            sensor_events = sensor_events.append({'SID': sens[s.SID], 'time': s.EndTime, 'value': 0}, ignore_index=True)

        activity_events = pd.read_csv(actfile, '\t', None, header=0, names=["StartTime", "EndTime", "Activity"])
        activity_events.StartTime = pd.to_datetime(
            activity_events.StartTime, format='%d-%b-%Y %H:%M:%S')
        activity_events.EndTime  = pd.to_datetime(activity_events.EndTime, format='%d-%b-%Y %H:%M:%S')
        activity_events.Activity = activity_events.Activity.apply(lambda x: acts[x])
        sensor_events = sensor_events.sort_values(['time'])
        activity_events = activity_events.sort_values(['StartTime', 'EndTime'])
        # print('finish downloading files')
        

        activities = [acts[k] for k in acts]

        sensor_desc = pd.DataFrame(columns=['ItemId', 'ItemName', 'Cumulative',
                                            'Nominal', 'OnChange', 'ItemRange', 'Location', 'Object', 'SensorName'])
        tmp_sensors = sensor_events['SID'].unique()
        for k in sens:
            item = {'ItemId': sens[k], 'ItemName': sens[k], 'Cumulative': 0, 'Nominal': 1, 'OnChange': 1, 'ItemRange': {
                'range': ['0', '1']}, 'Location': 'None', 'Object': 'None', 'SensorName': 'None'}
            sensor_desc = sensor_desc.append(item, ignore_index=True)

        return activity_events, activities, sensor_events, sensor_desc
    # loadVanKasterenDataset()
