import compress_pickle
import os
from .. import MyTask,utils
import numpy as np
from intervaltree.intervaltree import IntervalTree

import pandas as pd
import logging
logger = logging.getLogger(__file__)


class Dataset(MyTask):
    def __init__(self, data_path, data_dscr):
        self.data_path = data_path
        self.data_dscr = data_dscr
        self.cache_path = f'{data_path}/cache.pkl.lz4'
        self.load()

    def shortname(self):
        return self.data_dscr

    def _load(self):
        pass

    def _load_from_cache(self):
        try:
            if not os.path.exists(self.cache_path):
                return False
            attrs = compress_pickle.load(self.cache_path)
            for k in attrs:
                setattr(self, k, attrs[k])
        except Exception as e:
            logger.debug(f'cache file is broken {e}')
            return False

        return True

    def _save_to_cache(self):
        attrs = self.__dict__
        compress_pickle.dump(attrs, self.cache_path)

    def load(self):
        if self._load_from_cache():
            return self
        self.activity_events, self.activities, self.sensor_events, self.sensor_desc = self._load()
        logger.debug('database file loaded... now convert it ')
        self._calculate_activity()
        logger.debug('activities converted...')
        self._caclculate_sensor()
        logger.debug('sensors converted...')
        self._save_to_cache()
        return self

    def _calculate_activity(self):
        self.activity_events = self.activity_events.sort_values(['StartTime', 'EndTime'])
        print(self.activities)
        self.activities.sort()
        self.activities = np.insert(self.activities, 0, 'None')
        self.activities_map_inverse = {k: v for v, k in enumerate(self.activities)}
        self.activities_map = {v: k for v, k in enumerate(self.activities)}
        self.activity_events.Activity = self.activity_events.Activity.apply(
            lambda x: self.activities_map_inverse[x])
        self.activity_events['Duration'] = self.activity_events.EndTime - \
            self.activity_events.StartTime
        self.activity_events_tree = IntervalTree()
        for i, act in self.activity_events.iterrows():
            if (act.StartTime.value == act.EndTime.value):
                self.activity_events_tree[act.StartTime.value:act.StartTime.value +
                                          1] = {'StartTime': act.StartTime, 'EndTime': act.EndTime, 'Activity': act.Activity}
            else:
                self.activity_events_tree[act.StartTime.value:act.EndTime.value] = {
                    'StartTime': act.StartTime, 'EndTime': act.EndTime, 'Activity': act.Activity}

    def _caclculate_sensor(self):
        self.sensor_events = self.sensor_events.sort_values(['time'])
        self.sensor_desc = self.sensor_desc.sort_values(by=['ItemName'])
        self.sensor_desc = self.sensor_desc.set_index('ItemId')

        self.sensor_desc_map_inverse = {}
        self.sensor_desc_map = {}

        for i, sd in self.sensor_desc[self.sensor_desc.Nominal == 1].iterrows():
            self.sensor_desc_map_inverse[i] = {k: v for v, k in enumerate(sd.ItemRange['range'])}
            self.sensor_desc_map[i] = {v: k for v, k in enumerate(sd.ItemRange['range'])}

        def _convertVal3(x):
            return _convertVal(x.SID, x.Value)

        def _convertVal2(sid, val):
            try:
                valf = float(val)
                return valf
            except:
                return self.sensor_desc_map_inverse[sid][val]

        def _convertVal(sid, val):

            if sid in self.sensor_desc_map_inverse:
                # if type(x.value) is float :
                #         return self.sensor_desc_map_inverse[x.SID][str(int(x.value))]
                return self.sensor_desc_map_inverse[sid][val]
            else:
                return float(val)
        # for i,x in self.sensor_events.iterrows() :

        import time
        s = time.time()
        # for i in range(0,len(self.sensor_events)):
        #     self.sensor_events.iat[i,2] = _convertVal(self.sensor_events.iat[i,0],self.sensor_events.iat[i,2])
        for p in self.sensor_desc_map_inverse:
            for v in self.sensor_desc_map_inverse[p]:
                self.sensor_events = self.sensor_events.replace({'SID': p, 'value': v}, {'value': self.sensor_desc_map_inverse[p][v]})
        self.sensor_events.value = pd.to_numeric(self.sensor_events.value)
        # print(time.time()-s)
        # print(self.sensor_events)
        self.sensor_id_map = {v: k for v, k in enumerate(self.sensor_desc.index)}
        self.sensor_id_map_inverse = {k: v for v, k in enumerate(self.sensor_desc.index)}

    # region PublicActivityRoutines

    def get_activities_by_indices(self, activity_ids):
        """Get a group of activities by their corresponding indices

        Args:
            activity_ids (:obj:`list` of :obj:`int`): A list of activity indices

        Returns:
            :obj:`list` of :obj:`str`: A list of activity labels in the same order
        """
        return [self.get_activity_by_index(cur_id) for cur_id in activity_ids]

    def get_activity_by_index(self, activity_id):
        """Get Activity name by their index

        Args:
            activity_id (:obj:`int`): Activity index

        Returns:
            :obj:`str`: Activity label
        """
        if activity_id in self.activities_map:
            return self.activities_map[activity_id]
        logger.error('Failed to find activity with index %d' % activity_id)
        return ""

    def get_activity_index(self, activity_label):
        """Get Index of an activity

        Args:
            activity_label (:obj:`str`): Activity label

        Returns:
            :obj:`int`: Activity index (-1 if not found or not enabled)
        """
        if activity_label in self.activity_map_inverse:
            return self.activity_map_inverse[activity_label]
        else:
            return -1

    def remove_activities(self, acts):
        """Get label list of all enabled activities

        """
        raise NotImplementedError

    def get_activity_color(self, activity_label):
        """Find the color string for the activity.

        Args:
            activity_label (:obj:`str`): activity label

        Returns:
            :obj:`str`: RGB color string
        """

        # Pick the color from color list based on the activity index
        activity_index = self.get_activity_index(activity_label)
        if activity_index >= 0:
            return self._COLORS[activity_index % len(self._COLORS)]
        else:
            return '#C8C8C8'   # returns grey

    # region PublicSensorRoutines

    def remove_sensor(self, sensor_name):
        """Enable a sensor

        Args:
            sensor_name (:obj:`str`): Sensor Name

        Returns
            :obj:`int`: The index of the enabled sensor
        """
        raise NotImplementedError

    def get_sensor_by_index(self, sensor_id):
        """Get the name of sensor by index

        Args:
            sensor_id (:obj:`int`): Sensor index

        Returns:
            :obj:`str`: Sensor name
        """
        if sensor_id in self.sensor_id_map:
            return self.sensor_id_map[sensor_id]
        logger.error('Failed to find sensor with index %d' % sensor_id)
        return ""

    def get_sensor_index(self, sensor_name):
        """Get Sensor Index

        Args:
            sensor_name (:obj:`str`): Sensor Name

        Returns:
            :obj:`int`: Sensor index (-1 if not found or not enabled)
        """
        if sensor_name in self.sensor_id_map_inverse:
            return self.sensor_id_map_inverse[sensor_name]
        else:
            return -1

    # endregion

    # # region PickleState

    # def __getstate__(self):
    #     """Save x as sparse matrix if the density of x is smaller than 0.5
    #     """
    #     # state = self.__dict__.copy()
    #     # if self.x is not None:
    #     #     density_count = np.count_nonzero(self.x)
    #     #     density = float(density_count) / self.x.size
    #     #     if density < 0.5:
    #     #         state['x'] = sp.csr_matrix(state['x'])
    #     return self.__dict__

    # def __setstate__(self, state):
    #     """Set state from pickled file
    #     """
    #     # if sp.issparse(state['x']):
    #     #     state['x'] = state['x'].todense()
    #     # self.__dict__.update(state)
    # # endregion

    # region Summary
    def summary(self):
        """Print summary of loaded datasets
        """
        logger.debug('Dataset Path: %s' % self.data_path)
        logger.debug('Sensors: %d' % len(self.sensor_desc))
        logger.debug('Activities: %d' % len(self.activities))
        logger.debug('loaded events: %d' % len(self.sensor_events))
    # endregion

    _COLORS = ('#b20000, #56592d, #acdae6, #cc00be, #591616, #d5d9a3, '
               '#007ae6, #4d0047, #a67c7c, #2f3326, #00294d, #b35995, '
               '#ff9180, #1c330d, #73b0e6, #f2b6de, #592400, #6b994d, '
               '#1d2873, #ff0088, #cc7033, #50e639, #0000ff, #7f0033, '
               '#e6c3ac, #00d991, #c8bfff, #592d3e, #8c5e00, #80ffe5, '
               '#646080, #d9003a, #332200, #397367, #6930bf, #33000e, '
               '#ffbf40, #3dcef2, #1c0d33, #8c8300, #23778c, #ba79f2, '
               '#e6f23d, #203940, #302633').split(',')
