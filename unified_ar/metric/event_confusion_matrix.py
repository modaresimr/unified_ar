import numpy as np
import pandas as pd
from intervaltree.intervaltree import IntervalTree
from general.utils import Data


def event_confusion_matrix(r_activities, p_activities, labels):
    cm = np.zeros((len(labels), len(labels)))
    # begin=real0.StartTime.min()
    # end=real0.EndTime.max()

    #   predicted.append({'StartTime':begin,'EndTime':end,'Activity':0})
    #   real.append({'StartTime':begin,'EndTime':end,'Activity':0})
    events = merge_split_overlap_IntervalTree(r_activities, p_activities)
    # predictedtree=makeIntervalTree(labels)

    for eobj in events:
        e = eobj.data
        pact = e.P['Activity'] if not (e.P is None) else 0
        ract = e.R['Activity'] if not (e.R is None) else 0
        cm[ract][pact] += max((eobj.end-eobj.begin)/pd.to_timedelta('60s').value, 0.01)

    # for p in predicted:
    #  for q in realtree[p['StartTime'].value:p['EndTime'].value]:
    #      timeconfusion_matrix[p['Activity']][q.data['Activity']]+=findOverlap(p,q.data);

    return cm


def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]


def merge_split_overlap_IntervalTree(r_acts, p_acts):
    tree = IntervalTree()
    from result_analyse.visualisation import plotJoinTree
    if len(p_acts) > 0:
        PACT = column_index(p_acts, 'Activity')
        PSTIME = column_index(p_acts, 'StartTime')
        PETIME = column_index(p_acts, 'EndTime')

        for i, row in enumerate(p_acts.values):
            if (row[PACT] == 0):
                continue
            start = row[PSTIME]
            end = row[PETIME]
            startv = start.value
            endv = end.value
            if (startv == endv):
                # startv = startv-1
                continue
            # tree[start:end]={'P':{'Activitiy':act.Activity,'Type':'P','Data':act}]
            d = Data('P-act')
            d.P = {'Activity': row[PACT], 'StartTime': start, 'EndTime': end}
            d.R = None
            try:
                tree[startv:endv] = d
            except:
                print(i, startv - endv, row)
                from IPython.display import display
                display(p_acts.iloc[i])
                display(p_acts.values[i, :])
                raise

    RACT = column_index(r_acts, 'Activity')
    RSTIME = column_index(r_acts, 'StartTime')
    RETIME = column_index(r_acts, 'EndTime')

    for row in r_acts.values:
        if (row[RACT] == 0):
            continue
        start = row[RSTIME]
        end = row[RETIME]
        startv = start.value
        endv = end.value
        if (startv == endv):
            # startv = startv-1
            continue
        # tree[start:end]=[{'Activitiy':act.Activity,'Type':'R','Data':act}]
        d = Data('R-act')
        d.P = None
        d.R = {'Activity': row[RACT], 'StartTime': start, 'EndTime': end}
        tree[startv:endv] = d
    # cmTreePlot(tree)
    tree.split_overlaps()
    # cmTreePlot(tree)

    def data_reducer(x, y):
        res = Data('merge')
        res.R = x.R
        res.P = x.P
        if not (y.P is None):
            if (res.P is None) or y.P['EndTime'] < res.P['EndTime']:
                res.P = y.P
        if not (y.R is None):
            if (res.R is None) or y.R['EndTime'] < res.R['EndTime']:
                res.R = y.R
        return res

    tree.merge_equals(data_reducer=data_reducer)

    return tree


def cmTreePlot(tree):
    ptree = IntervalTree()
    rtree = IntervalTree()
    for item in tree:
        if not (item.data.R is None):
            rtree[item.begin:item.end] = item
        if not (item.data.P is None):
            ptree[item.begin:item.end] = item
    from result_analyse.visualisation import plotJoinTree
    plotJoinTree(rtree, ptree)


if __name__ == '__main__':
    import result_analyse.visualisation as vs
    from metric.CMbasedMetric import CMbasedMetric
    r = vs.convert2event(np.array([(65, 75), (157, 187)]))
    p = vs.convert2event(np.array([(66, 73), (78, 126)]))

    r = vs.convert2event(np.array([(20, 70), (100, 200)]))
    p = vs.convert2event(np.array([(10, 80), (150, 250), (100, 200), (0, 500)]))

    import general.utils as utils
    # r, p = utils.loadState('ali')
    cm = event_confusion_matrix(r, p, range(11))

    print(cm)
    print(CMbasedMetric(cm, average='macro'))
    print(CMbasedMetric(cm))
