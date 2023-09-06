from .segmentation_abstract import Segmentation
import pandas as pd


class FixedTimeWindow(Segmentation):
    def applyParams(self, params):
        if not super().applyParams(params):
            return False
        shift = pd.Timedelta(params['shift'], unit='s')
        size = pd.Timedelta(params['size'], unit='s')
        return shift <= size

    def segment(self, w_history, buffer):
        idx = self.segment2(w_history, buffer)
        window = buffer.data.iloc[idx]
        # buffer.removeTop(sindex)
        window.iat[0, 1].value
        return {'window': window, 'start': stime, 'end': etime}

    def segment2(self, w_history, buffer):
        params = self.params
        shift = pd.Timedelta(params['shift'], unit='s')
        size = pd.Timedelta(params['size'], unit='s')

        if len(w_history) == 0:
            lastStart = pd.to_datetime(0)
        else:
            lastStart = w_history[len(w_history)-1]['start']

        lastStartshift = lastStart+shift
        sindex = buffer.searchTime(lastStartshift, -1)

        if (sindex is None):
            return None
        # try:
        stime = lastStart+shift

        eindex = buffer.searchTime(stime+size, +1)
        if (eindex is None):
            eindex = sindex
        etime = buffer.times[eindex]

        idx = range(sindex, eindex + 1)
        return (idx, None)
