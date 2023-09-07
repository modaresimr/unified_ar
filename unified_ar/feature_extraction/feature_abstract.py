from unified_ar import MyTask
import numpy as np
from unified_ar.general.sparray import sparray
import tempfile
from tqdm.auto import tqdm


def featureExtraction(feat, datasetdscr, Sdata, istrain):

    if (istrain):
        feat.precompute(datasetdscr, Sdata.set_window)
    else:
        feat.prepare4test()
    shape = feat.getShape()

    # filename=tempfile.mktemp('featureExtract')
    if (len(shape) == 1):
        # result= np.memmap(filename,shape=(len(Sdata.set_window),shape[0]),mode='w+',dtype=np.float32)
        result = np.zeros(shape=(len(Sdata.set_window), shape[0]), dtype=np.float32)
    else:
        # result= np.memmap(filename,shape=(len(Sdata.set_window),shape[0],shape[1]),mode='w+', dtype=np.float32)
        result = np.zeros(shape=(len(Sdata.set_window), shape[0], shape[1]), dtype=np.float32)
        # result= np.zeros((len(windows),shape[0],shape[1]), dtype=np.float32)
        # result=np.zeros((len(windows),fw.shape[0],fw.shape[1]))
    feat.prepareData(Sdata.s_event_list)
    for i in tqdm(range(len(Sdata.set_window)), desc=f'{feat.shortname()} {"train" if istrain else "test"} {feat.params}', bar_format='{l_bar}{bar}count={n_fmt} time={elapsed} speed={rate_fmt}{postfix}'):
        result[i] = feat.featureExtract2(Sdata.s_event_list, Sdata.set_window[i])
        # return result[i]

    result = feat.normalize(result, istrain)

    return result


def featureExtraction2(feat, datasetdscr, Sdata, istrain):
    if (istrain):
        feat.precompute(datasetdscr, Sdata.set_window)
    else:
        feat.prepare4test()
    for i in range(len(Sdata.set_window)):
        yield (feat.featureExtract2(Sdata.s_event_list, Sdata.set_window[i]), Sdata.label[i])
    return 1


class FeatureExtraction(MyTask):
    def getShape(self):
        return self.shape

    def precompute(self, datasetdscr, windows):
        self.datasetdscr = datasetdscr
        pass

    def featureExtract(self, window):
        pass

    def prepareData(self, s_event_list):
        pass

    def normalize(self, windows, istrain):
        return windows

    def prepare4test(self):
        pass

    def __str__(self):
        return self.shortname()
