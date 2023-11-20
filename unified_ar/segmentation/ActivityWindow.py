from .segmentation_abstract import Segmentation
import pandas as pd


class ActivityWindow(Segmentation):

    def applyParams(self, params):

        res = super().applyParams(params)
        return res

    def segment3(self, buffer):
        for i, row in self.a_events.iterrows():
            act = row.Activity
            sindex = buffer.searchTime(row.StartTime, -1)

            if (sindex is None):
                continue
            # try:

            eindex = buffer.searchTime(row.EndTime, +1)
            # etime=buffer.times[eindex]
            if eindex == None:
                eindex = sindex
            idx = range(sindex, eindex + 1)

            yield idx, act


class SlidingEventActivityWindow(Segmentation):

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
            params['shift'] = int(shift)
            params['size'] = int(size)
        except:
            return False
        return super().applyParams(params)

    # def _create_segments(self, sindex, eindex, act):
    #     olds = 0
    #     olde = 0

    #     for i in range(sindex-self.size+1, eindex+1, self.shift):
    #         s = max(i, sindex)
    #         e = min(i+self.size, eindex+1)
    #         if s == olds and e == olde:
    #             continue
    #         olds, olde = s, e

    #         yield range(s, e), act

    def _create_segments(self, sindex, eindex, act):
        
        for i in range(sindex, max(sindex, eindex-self.size+1)+1, self.shift):
            s = max(i, sindex)
            e = min(i+self.size, eindex+1)

            yield range(s, e), act

    def segment3(self, buffer):
        olde = 0
        for i, row in self.a_events.iterrows():

            act = row.Activity
            sindex = buffer.searchTime(row.StartTime, -1)

            if (sindex is None):
                continue
            if olde < sindex-1:
                yield from self._create_segments(olde, sindex-1, 0)

            eindex = buffer.searchTime(row.EndTime, +1)

            # etime=buffer.times[eindex]
            if eindex == None:
                eindex = sindex
            olde = eindex+1
            if sindex < eindex:
                yield from self._create_segments(sindex, eindex, act)

        last = len(buffer.times)-1
        if last > olde:
            # print(f'creating segments from {olde} to {last} for act={0}')
            yield from self._create_segments(olde, last, 0)
