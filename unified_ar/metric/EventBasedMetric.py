import pandas as pd
from wardmetrics.core_methods import eval_segments
from wardmetrics.utils import print_detailed_segment_results, print_twoset_segment_metrics
import logging
logger = logging.getLogger(__file__)

def time2int(t):
    return t.value/pd.to_timedelta('60s').value

def eval(real_a_event, pred_a_event, acts):
    revent = {}
    pevent = {}
    for act in acts:
        revent[act]=[]
        pevent[act]=[]
        
    for i,e in real_a_event.iterrows():
        if not(e.Activity in acts):continue
        revent[e.Activity].append((time2int(e.StartTime), time2int(e.EndTime)))
    for i,e in pred_a_event.iterrows():
        if not(e.Activity in acts):continue
        pevent[e.Activity].append((time2int(e.StartTime), time2int(e.EndTime)))

    result={}
    for act in acts:

        try:
            twoset_results, segments_with_scores, segment_counts, normed_segment_counts = eval_segments(revent[act], pevent[act])
        except:
            twoset_results=[]
            segment_counts=[]
            normed_segment_counts=[]
        #'segments_with_scores':segments_with_scores,
        result[act]={'twoset_results':twoset_results,  'segment_counts':segment_counts, 'normed_segment_counts':normed_segment_counts}
        print_detailed_segment_results(segment_counts)
        print_detailed_segment_results(normed_segment_counts)
        print_twoset_segment_metrics(twoset_results)
    if(len(acts)==1):
        return result[act]
    return result