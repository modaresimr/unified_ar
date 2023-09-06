from segmentation.segmentation_abstract import Segmentation
import pandas as pd


class FixedEventWindow(Segmentation):

    def applyParams(self, params):
        shift = params['shift']
        size = params['size']
        if not (shift > 0):
            return False
        if not (size > 0):
            return False
        if (shift > size):
            return False
        try:
            shift = int(shift)
            size = int(size)
        except:
            return False
        return super().applyParams(params)

    def segment(self, w_history, buffer):
        params = self.params
        shift = int(params['shift'])
        size = int(params['size'])

        if len(w_history) == 0:
            lastStart = pd.to_datetime(0)
        else:
            lastStart = w_history[len(w_history) - 1]['start']

        sindex = buffer.searchTime(lastStart, -1)

        if (sindex is None):
            return None
        sindex = sindex + shift
        if (len(buffer.times) <= sindex):
            return None

        eindex = min(len(buffer.times) - 1, sindex + size)
        if (eindex - sindex < size):
            return None
        etime = buffer.times[eindex]
        stime = buffer.times[sindex]
        window = buffer.data.iloc[sindex:eindex + 1]
        buffer.removeTop(sindex)
        window.iat[0, 1].value
        return {'window': window, 'start': stime, 'end': etime}

    def segment2(self, w_history, buffer):
        shift = int(self.shift)
        size = int(self.size)

        if len(w_history) == 0:
            sindex = 0
        else:
            # lastStart = buffer.times[w_history[len(w_history)-1][0]]
            sindex = w_history[-1][0]

        sindex = sindex + shift
        if (len(buffer.times) <= sindex):
            return None
        stime = buffer.times[sindex]

        eindex = min(len(buffer.times) - 1, sindex + size)

        etime = buffer.times[eindex]
        if etime-stime>pd.Timedelta('12h'):
            filteridx = buffer.searchTime(stime + pd.Timedelta('12h'), +1)
            eindex = min(eindex, filteridx)
        # if (eindex - sindex < size):
        #     return None
        try:
            # etime = buffer.times[eindex]
            # stime = buffer.times[sindex]
            idx = range(sindex, eindex + 1)
        # buffer.removeTop(sindex)

        except:
            print('eindex', eindex)
            print('sindex', sindex)
            print('size', size)
            print('len', len(buffer.times))

        return (idx,None)
