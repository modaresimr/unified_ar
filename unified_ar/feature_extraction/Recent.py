from feature_extraction.feature_abstract import FeatureExtraction
import pandas as pd
import numpy as np


class Recent(FeatureExtraction):
    def getShape(self):
        scount = len(self.datasetdscr.sensor_id_map)
        self.scount = scount

        self.total_feat = (2 * scount if self.lastState else scount)+4
        return (self.total_feat,)

    sec_in_day = (60*60*24)

    def featureExtract2(self, s_event_list, idx):
        window = s_event_list
        scount = self.scount
        f = np.zeros(self.total_feat)
        for i in idx:
            f[self.datasetdscr.sensor_id_map_inverse[window[i, 0]]] += 1  # window[i, 2]  # f[sensor_id_map_inverse[x.SID]]=1
            if self.lastState:
                f[scount+self.datasetdscr.sensor_id_map_inverse[window[i, 0]]] = window[i, 2]  # f[sensor_id_map_inverse[x.SID]]=1
        stime = window[idx[0], 1]  # startdatetime
        etime = window[idx[-1], 1]  # enddatetime
        ts = (etime-pd.to_datetime(etime.date())).total_seconds()
        f[self.total_feat-1] = ts/self.sec_in_day
        f[self.total_feat-2] = (etime-stime).total_seconds()/self.sec_in_day
        f[self.total_feat-3] = len(idx)
        f[self.total_feat-4] = etime.dayofweek
        # f[scount+4] = etime.dayofweek
        return f
