# -*- coding: utf-8 -*-
import math

import pandas as pd
from intervaltree import intervaltree
from matplotlib.pylab import plt
from pandas.core.frame import DataFrame
from prompt_toolkit.shortcuts import set_title
from wardmetrics.core_methods import eval_events, merge_events_if_necessary

import result_analyse.SpiderChart as spiderchart


def eval(real_a_event, pred_a_event, acts,debug=0,calcne=0):
    revent = {}
    pevent = {}
    for act in acts:
        revent[act]=open(f"/tmp/tat-{act}.realn", "w")
        pevent[act]=open(f"/tmp/tat-{act}.predn", "w")
        
    sec1=pd.to_timedelta('1s').value
    for i,e in real_a_event.iterrows():
        if not (e.Activity in acts):
            continue
        revent[e.Activity].write(f'{int(e.StartTime.value/sec1)-1247911961} {int(e.EndTime.value/sec1)-1247911961}\n')
    for i,e in pred_a_event.iterrows():
        if not (e.Activity in acts):
            continue
        pevent[e.Activity].write(f'{int(e.StartTime.value/sec1)-1247911961} {int(e.EndTime.value/sec1) -1247911961}\n')    
    for act in acts:
        revent[act].close()
        pevent[act].close()
    
    result={}
    for act in acts:
        if debug :print(act,"======================")
        
        result[act] = eval_my_metric(f"/tmp/tat-{act}.realn", f"/tmp/tat-{act}.predn",debug=debug)
    if(len(acts)==1):
        return result[act]
    return result

def call(real, pred, debug, beta, alpha, gamma, delta):
    import os
    import subprocess
    proc = subprocess.run(['/workspace/TSAD-Evaluator/src/evaluate', '-v', '-tn', f'{real}', f'{pred}', str(beta),str(alpha), gamma ,delta, delta],  stdout=subprocess.PIPE)
    a=proc.stdout.decode('utf-8')
    terms={'Precision':'precision','Recall':'recall','F-Score':'f1'}
    result={}
    for x in a.split('\n'):
        for term in terms:
            if(term in x):
                result[terms[term]]=float(x[len(term)+3:])
       
    return result

def eval_my_metric(real,pred,debug=0):
    result={}
    result['Tatbul(a=0)']=call(real,pred,0,beta=1,alpha=0,gamma='one',delta='flat')
    result['Existence(a=1)']=call(real,pred,0,beta=1,alpha=1,gamma='one',delta='udf_delta')
    result['Cardinality(γ=reci)']=call(real,pred,0,beta=1,alpha=0,gamma='reciprocal',delta='udf_delta')
    result['Positional(δ=flat)']=call(real,pred,0,beta=1,alpha=0,gamma='one',delta='flat')
    
    return result
    # print(a)
    # # os.system(f'/workspace/TSAD-Evaluator/src/evaluate -v -tn {real} {pred} 1 0 one flat flat')
    # print(f'/workspace/TSAD-Evaluator/src/evaluate -v -tn {real} {pred} 1 0 one flat flat');

if __name__ == "__main__":
    import result_analyse.resultloader
    import result_analyse.kfold_analyse as an
    # import metric.MyMetric as mymetric
    import metric.TatbulMetric as mymetric
    import general.utils as utils
    
    run_info,dataset,evalres=utils.loadState('201104_19-48-50-Home1')
    #             print(evalres[t])
    for i in evalres:
        evalres[i]['test'].Sdata=None
    dataset.sensor_events=None
    res=an.mergeEvals(dataset,evalres,mymetric)
    print(res)
