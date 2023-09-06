from .segmentation_abstract import Segmentation
import pandas as pd


class FixedSlidingWindow(Segmentation):

    def applyParams(self, params):

        res = super().applyParams(params)
        if (params['size'] <= 1) or params['shift'] <= 1:
            return False
        try:
            shift = int(params['shift'])
            size = int(params['size'])
        except:
            return False

        self.shift = pd.Timedelta(params['shift'], unit='s')
        self.size = pd.Timedelta(params['size'], unit='s')
        if (self.shift > self.size):
            return False

        return res

#     def segment(self,w_history,buffer):
#         params=self.params
#         shift=self.shift
#         size=pd.Timedelta(params['size'],unit='s')

#         if len(w_history)==0 :
#           lastStart=pd.to_datetime(0)
#         else :
#           lastStart=w_history[len(w_history)-1]['start']

#         lastStartshift=lastStart+shift;
#         sindex=buffer.searchTime(lastStartshift,-1)

#         if(sindex is None):
#             return None
#         #try:
#         stime=buffer.times[sindex]

#         etime=stime+size
#         eindex=buffer.searchTime(stime+size,+1)
#         #etime=buffer.times[eindex]

#         window=buffer.data.iloc[sindex:eindex+1];
#         buffer.removeTop(sindex)
# #        print(window.iat[0,1])
#         window.iat[0,1].value
#         return {'window':window,'start':stime, 'end':etime}

    def segment2(self, w_history, buffer):
        shift = self.shift
        size = self.size

        if len(w_history) == 0:
            lastStart = pd.to_datetime(0)
        else:
            lastStart = buffer.times[w_history[len(w_history)-1][0]]

        lastStartshift = lastStart+shift
        sindex = buffer.searchTime(lastStartshift, -1)

        if (sindex is None):
            return None
        # try:
        stime = buffer.times[sindex]

        etime = stime+size
        eindex = buffer.searchTime(stime+size, +1)
        # etime=buffer.times[eindex]
        if eindex == None:
            eindex = sindex
        idx = range(sindex, eindex + 1)
#         window=buffer.data.iloc[sindex:eindex+1];
#         buffer.removeTop(sindex)
# #        print(window.iat[0,1])
#         window.iat[0,1].value
        return (idx, None)
