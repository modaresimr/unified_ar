from .feature_abstract import FeatureExtraction
import numpy as np


class KHistory(FeatureExtraction):
    def precompute(self, datasetdscr, windows):
        self.method.precompute(datasetdscr, windows)
        mshape = self.method.getShape()
        if (len(mshape) == 2):
            raise ValueError('only 1D array supported')
        self.shape = (self.k, mshape[0])
        super().precompute(datasetdscr, windows)

    def featureExtract2(self, s_event_list, idx):
        f = np.zeros(self.shape)
        l = len(idx)
        start = idx[0]
        for k in range(self.k):
            newstart = start-l*(self.k-k-1)
            idx = range(newstart, newstart+l)
            f[k, :] = self.method.featureExtract2(s_event_list, idx)

        return f
    # def shortname(self):
    #     return f'{super().shortname()} k={self.k} {self.method}'
