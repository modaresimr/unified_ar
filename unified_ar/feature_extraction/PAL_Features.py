from feature_extraction.feature_abstract import FeatureExtraction
import pyActLearn.CASAS.stat_features as sf
import numpy as np
import logging
logger = logging.getLogger(__file__)


class PAL_Features(FeatureExtraction):
    feature_list = {}
    routines={}

    def precompute(self, datasetdscr, windows):
        self.max_windowsize = max([len(w) for w in windows])
        self.datasetdscr=datasetdscr
        normalized = self.normalized
        per_sensor = self.per_sensor
        self.scount = sum(1 for x in self.datasetdscr.sensor_id_map)
        self._add_feature(sf.EventHour(normalized=normalized))
        self._add_feature(sf.EventSeconds(normalized=normalized))
        self._add_feature(sf.LastSensor(per_sensor=per_sensor))
        self._add_feature(sf.WindowDuration(normalized=normalized))
        self._add_feature(sf.SensorCount(normalized=normalized))
        self._add_feature(sf.DominantSensor(per_sensor=per_sensor))
        self._add_feature(sf.SensorElapseTime(normalized=normalized))
        self.fcount = self._count_feature_columns()
        self.sensor_list = {}
        for i,s in datasetdscr.sensor_desc.iterrows():
            name = s['ItemName']
            self.sensor_list[name] = {'name': name}
            self.sensor_list[name]['index'] = datasetdscr.sensor_id_map_inverse[name]
            self.sensor_list[name]['enable'] = True
            self.sensor_list[name]['lastFireTime'] = None
        return super().precompute(datasetdscr, windows)

    def featureExtract(self, win):
        window = win['window']
        f = self._calculate_stat_features(window)
        self.lastwindow = window
        return f
    def featureExtract2(self,s_event_list,idx):
        window=s_event_list
        f = self._calculate_stat_features(window,idx)
        self.lastwindow = idx
        return f


    # region FeatureCalculation
    def _calculate_stat_features(self, window,idx):
        """Populate the feature vector with statistical features using sliding window
        """
        num_feature_columns = self._count_feature_columns()

        # Execute feature update routine
        for (key, routine) in self.routines.items():
            if routine.enabled:
                routine.clear()

        return self._calculate_window_feature(window,idx)

    def _conver_window2eventlist(self, window,idx):
        event_list = []
        for i in range(0, idx.shape[0]):
            cur_data_dict = {
                'datetime':     window[idx[i], 1],
                'sensor_id':    window[idx[i], 0],
                'sensor_status':window[idx[i], 2],
                # 'activity':lastACT
            }
            event_list.append(cur_data_dict)
        return event_list

    def _calculate_window_feature(selsuf, window,idx):
        """Calculate feature vector for current window specified by cur_row_id

        Args:
                cur_row_id (:obj:`int`): Row index of current window (last row)
                cur_sample_id (:obj:`int`): Row index of current sample in self.x

        Returns:
                :obj:`int`: number of feature vector added
        """
        # Default Window Size to 30

        num_enabled_sensors = self.scount
        event_list = self._conver_window2eventlist(window,idx)
        window_size = min(self.max_windowsize, len(event_list))-1
        
        cur_row_id = window_size
        x = np.zeros(self.fcount, dtype=np.float)
        if(window_size==0):return x

        # Execute feature update routine
        for (key, routine) in self.routines.items():
            if routine.enabled:
                routine.update(data_list=event_list, cur_index=cur_row_id,
                               window_size=window_size, sensor_info=self.sensor_list)

        # Get Feature Data and Put into arFeature array
        for (key, feature) in self.feature_list.items():
            if feature.enabled:
                # If it is per Sensor index, we need to iterate through all sensors to calculate
                if feature.per_sensor:
                    for sensor_name in self.sensor_list.keys():
                        if self.sensor_list[sensor_name]['enable']:
                            column_index = self.num_static_features + \
                                feature.index * num_enabled_sensors + \
                                self.sensor_list[sensor_name]['index']
                            x[column_index] = feature.get_feature_value(data_list=event_list,
                                                                        cur_index=cur_row_id,
                                                                        window_size=window_size,
                                                                        sensor_info=self.sensor_list,
                                                                        sensor_name=sensor_name)
                else:
                    x[feature.index] = feature.get_feature_value(data_list=event_list,
                                              cur_index=cur_row_id,
                                              window_size=window_size,
                                              sensor_info=self.sensor_list,
                                              sensor_name=None)
        return x
    # endregion

    # region Stat Feature Management

    def _add_feature(self, feature):
        """Add Feature to feature list [ref. PyActLearn]

        Args:
                                                                                                                                        feature (:class:`pyActlearn.CASAS.stat_features`): FeatureTemplate Object
        """
        if feature.name in self.feature_list.keys():
            logger.warning(
                'feature: %s already existed. Add Feature Function ignored.' % feature.name)
        else:
            logger.debug('Add Feature %s: %s' %
                         (feature.name, feature.description))
            self.feature_list[feature.name] = feature
            if feature.routine is not None:
                self._add_routine(feature.routine)
            self._assign_feature_indexes()

    # region Stat Feature Routine Update Management
    def _add_routine(self, routine):
        """Add routine to feature update routine list

        Args:
                routine (:class:`pyActLearn.CASAS.stat_features.FeatureRoutineTemplate`): routine to be added
        """
        if routine.name in self.routines.keys():
            logger.debug('feature routine %s already existed.' % routine.name)
        else:
            logger.debug('Add feature routine %s: %s' %
                         (routine.name, routine.description))
            self.routines[routine.name] = routine

    # region PublicFeatureRoutines
    def get_feature_by_index(self, index):
        """Get Feature Name by Index

        Args:
                        index (:obj:`int`): column index of feature

        Returns:
                        :obj:`tuple` of :obj:`str`: (feature name, sensor name) tuple.
                                        If it is not per-sensor feature, the sensor name is None.
        """
        max_id = self.num_feature_columns
        num_enabled_sensors = len(self.get_enabled_sensors())
        if index > max_id:
            logger.error('index %d is greater than the number of feature columns %d' %
                         (index, max_id))
        if index >= self.num_static_features:
            # It is per_sensor Feature
            sensor_id = (
                index - self.num_static_features) % num_enabled_sensors
            feature_id = math.floor(
                (index - self.num_static_features) / num_enabled_sensors)
            per_sensor = True
        else:
            # It is a generic feature
            sensor_id = -1
            feature_id = index
            per_sensor = False
        # Find Corresponding feature name and sensor label
        feature_name = None
        for featureLabel in self.feature_list.keys():
            feature = self.feature_list[featureLabel]
            if feature.index == feature_id and feature.per_sensor == per_sensor:
                feature_name = featureLabel
                break
        sensor_name = 'Window'
        if sensor_id >= 0:
            for sensor_label in self.sensor_list.keys():
                sensor = self.sensor_list[sensor_label]
                if sensor['index'] == sensor_id:
                    sensor_name = sensor_label
                    break
        return feature_name, sensor_name

    def _assign_feature_indexes(self):
        """Assign index to features
        """
        static_id = 0
        per_sensor_id = 0
        for featureLabel in self.feature_list.keys():
            feature = self.feature_list[featureLabel]
            if feature.enabled:
                if feature.per_sensor:
                    feature.index = per_sensor_id
                    per_sensor_id += 1
                else:
                    feature.index = static_id
                    static_id += 1
            else:
                feature.index = -1
        self.num_static_features = static_id
        self.num_per_sensor_features = per_sensor_id
        logger.debug('Finished assigning index to features. %d Static Features, %d Per Sensor Features' %
                     (static_id, per_sensor_id))

    def _update_feature_count(self):
        """Update feature count values
        """
        self.num_enabled_features = 0
        self.num_static_features = 0
        self.num_per_sensor_features = 0
        for name, feature in self.feature_list.items():
            if feature.enabled:
                self.num_enabled_features += 1
                if feature.per_sensor:
                    self.num_per_sensor_features += 1
                else:
                    self.num_static_features += 1

    def _count_feature_columns(self):
        """Count the size of feature columns

        Returns:
                :obj:`int`: size of feature columns
        """
        self.num_feature_columns = 0
        num_enabled_sensors = self.scount
        for feature_name in self.feature_list.keys():
            if self.feature_list[feature_name].enabled:
                if self.feature_list[feature_name].per_sensor:
                    self.num_feature_columns += num_enabled_sensors
                else:
                    self.num_feature_columns += 1
        return self.num_feature_columns * self.max_windowsize
    # endregion
