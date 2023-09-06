import math

import pandas as pd
from intervaltree import intervaltree
from matplotlib.pylab import plt
from pandas.core.frame import DataFrame
from prompt_toolkit.shortcuts import set_title
import unified_ar.metric.classical
import sklearn


def __str__():
    return 'MyClsasical Metric'


def eval(rlabel, plabel, acts):
    result = {}
    if (len(acts) == 1):
        act = acts[0]
        rlabel = [0 if r == act else 1 for r in rlabel]
        plabel = [0 if p == act else 1 for p in plabel]
        acts = [0, 1]

    # p, r, f, s = sklearn.metrics.{'EventCM':(rlabel, plabel, beta=1, labels=acts, average=None)
    p, r, f, s = sklearn.metrics.precision_recall_fscore_support(rlabel, plabel, beta=1, labels=acts, average=None)

    return {'classical': {'precision': p[0], 'recall': r[0], 'f1': f[0]}}
    # for act in acts:
    #     # if debug :print(act,"======================")
    #     result[act] = {'precision':p[act],'recall':r[act],'f1':f[act]}
    # if(len(acts)==1):
    #     return result[act]
    # return result
