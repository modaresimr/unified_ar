
import numpy as np
import pandas as pd
from intervaltree.intervaltree import IntervalTree
from unified_ar.general.utils import Data


def event_confusion_matrix2(r_activities, p_activities, labels):
    cm = np.zeros((len(labels), len(labels)))

    predicted = [p for i, p in p_activities.iterrows()]
    real = [p for i, p in r_activities.iterrows()]

    events = merge_split_overlap_IntervalTree(real, predicted)

    for eobj in events:
        e = eobj.data
        pact = e.P.Activity if not (e.P is None) else 0
        ract = e.R.Activity if not (e.R is None) else 0
        cm[ract][pact] += max((eobj.end-eobj.begin)/pd.to_timedelta('60s').value, 0.01)

    return cm


def merge_split_overlap_IntervalTree(r_acts, p_acts):
    tree = IntervalTree()

    for act in p_acts:
        if (act['Activity'] == 0):
            continue
        start = act['StartTime'].value
        end = act['EndTime'].value
        if (start == end):
            start = start-1
        # tree[start:end]={'P':{'Activitiy':act.Activity,'Type':'P','Data':act}]
        d = Data('P-act')
        d.P = act
        d.R = None
        tree[start:end] = d  # {'P':act,'PActivitiy':act.Activity}

    for act in r_acts:
        start = act['StartTime'].value
        end = act['EndTime'].value
        if (start == end):
            start = start-1
        # tree[start:end]=[{'Activitiy':act.Activity,'Type':'R','Data':act}]
        d = Data('P-act')
        d.P = None
        d.R = act
        tree[start:end] = d  # {'R':act,'RActivitiy':act.Activity}

    tree.split_overlaps()

    def data_reducer(x, y):
        res = x
        if not (y.P is None):
            if (res.P is None) or y.P['EndTime'] < res.P['EndTime']:
                res.P = y.P
        if not (y.R is None):
            if (res.R is None) or y.R['EndTime'] < res.R['EndTime']:
                res.R = y.R
        return res

    tree.merge_equals(data_reducer=data_reducer)

    return tree
