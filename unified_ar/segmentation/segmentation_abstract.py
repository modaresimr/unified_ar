from general.utils import Data, MyTask
import numpy as np
from general.utils import Buffer
import logging
from tqdm.auto import tqdm
# Define segmentation


def segment(dtype, datasetdscr, segment_method):
    buffer = Buffer(dtype.s_events, 0, 0)
    w_history = []
    segment_method.reset()
    segment_method.precompute(datasetdscr, dtype.s_events, dtype.a_events, dtype.acts)
    while (1):
        window = segment_method.segment(w_history, buffer)
        if window is None:
            return
        w_history.append(window)
        yield window


class Segmentation(MyTask):
    def precompute(self, datasetdscr, s_events, a_events, acts):
        pass

    def set_activity_info(self, a_events, acts):
        self.a_events = a_events
        self.acts = acts

    def segment(self, w_history, buffer):
        pass

    def segment2(self, w_history, buffer):
        raise NotImplementedError

    def segment3(self, buffer):
        w_history = []
        while (1):
            window = self.segment2(w_history, buffer)
            if window is None:
                # print(None)
                return
            w_history.append(window[0])
            # print(window)
            yield window


# def prepare_segment(func, dtype, datasetdscr):
#     segmentor = func.segmentor

#     func.activityFetcher.precompute(dtype)

#     procdata = Data(segmentor.__str__())
#     procdata.generator = segment(dtype, datasetdscr, segmentor)
#     procdata.set = []
#     procdata.label = []
#     procdata.set_window = []
#     procdata.acts = func.acts
#     procdata.s_events = dtype.s_events
#     procdata.a_events = dtype.a_events

#     i = 0
#     for x, act in tqdm(procdata.generator):
#         # if i % 10000 == 0:
#         #     logger.debug(segmentor.shortname(), i)
#         i += 1
#         procdata.set_window.append(x)
#         act = act if act != None else func.activityFetcher.getActivity(dtype, x['window'])
#         procdata.label.append(act)
#     del procdata.generator
#     procdata.label = np.array(procdata.label)

#     return procdata


def prepare_segment2(func, dtype, datasetdscr, train):
    segmentor = func.segmentor

    func.activityFetcher.precompute(dtype)

    procdata = Data(segmentor.__str__())
    procdata.generator = segment2(dtype, datasetdscr, segmentor, train)
    procdata.set = []
    procdata.label = []
    procdata.set_window = []
    procdata.acts = func.acts
    procdata.s_events = dtype.s_events
    procdata.s_event_list = dtype.s_event_list
    procdata.a_events = dtype.a_events

    i = 0
    itrs = procdata.generator
    if not hasattr(func, 'ui_debug') or func.ui_debug.get('seg', 1):
        itrs = tqdm(itrs, desc=f'{segmentor.shortname()} {segmentor.params}', bar_format='{l_bar}{bar}count={n_fmt} time={elapsed} speed={rate_fmt}{postfix}')
    for x, act in itrs:
        # print(x, act)
        # if i % 10000 == 0:
        #     print(segmentor.shortname(), i)
        i += 1
        procdata.set_window.append(x)
        act = act if act != None else func.activityFetcher.getActivity2(dtype.s_event_list, x)
        procdata.label.append(act)

    del procdata.generator
    procdata.label = np.array(procdata.label)

    return procdata

# Define segmentation


def segment2(dtype, datasetdscr, segment_method, train):
    buffer = Buffer(dtype.s_events, 0, 0)

    segment_method.reset()
    if train:
        segment_method.precompute(datasetdscr, dtype.s_events, dtype.a_events, dtype.acts)
    segment_method.set_activity_info(dtype.a_events, dtype.acts)
    return segment_method.segment3(buffer)
