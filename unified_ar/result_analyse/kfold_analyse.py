import numpy as np
from IPython.display import display
from tqdm.auto import tqdm
import unified_ar.metric.Metrics


def get_metric(metricname):
    metrics = {'GEM': unified_ar.metric.Metrics.GEM(),
               'GEM_NEW': unified_ar.metric.Metrics.GEM_NEW(),
               'EventCM': unified_ar.metric.Metrics.EventCM(),
               'Tatbul': unified_ar.metric.Metrics.Tatbul(),
               'Classical': unified_ar.metric.Metrics.Classical(),
               }
    return metrics[metricname]


def _evaluateAct(compact_item):
    evalres, metricname, act = compact_item
    evalobj = get_metric(metricname)
    res = {}
    res[act] = {'avg': {}}

    for fold in tqdm(evalres, desc='fold', leave=False):
        # print('eval for act=%d fold=%d' % (act, fold), end="\r")
        real_events = evalres[fold]['test'].real_events
        pred_events = evalres[fold]['test'].pred_events
        metr = evalobj.eval(real_events, pred_events, [act])

        # print('.',end='')
        res[act]['avg'] = add2Avg(res[act]['avg'], metr, len(evalres))
        res[act][fold] = metr
        # print(act)
        # display(res[act])
    return act, res[act]


def mergeEvals(dataset, evalres, metricname):
    evalobj = get_metric(metricname)
    if (evalobj.classical):
        return mergeEvalsClassic(dataset, evalres, evalobj)

    import unified_ar.general.utils as utils
    acts = range(1, len(dataset.activities_map))
    items = [(evalres, metricname, act) for act in acts]
    parallelRes = utils.parallelRunner(True, _evaluateAct, items)
    weights = dataset.activity_events['Activity'].value_counts()
    res = {'avg': {}}
    for act, act_res in parallelRes:
        res[act] = act_res
        res['avg'] = add2Avg(res['avg'], res[act]['avg'], len(acts))
        res['avg_weighted'] = add2Avg(res['avg_weighted'], res[act]['avg_weighted'], None, weights[act])
    return res


def mergeEvals_old(dataset, evalres, evalobj):
    if (evalobj.classical):
        return mergeEvalsClassic(dataset, evalres, evalobj)

    res = {'avg': {}}
    # if len(evalres) > 4:
    #     f=3
    #     evalres = {f: evalres[f]}
    #     print('fold',f)
    for act in tqdm(range(1, len(dataset.activities_map)), desc='act'):

        res[act] = {'avg': {}}

        for fold in tqdm(evalres, desc='fold', leave=False):
            # print('eval for act=%d fold=%d' % (act, fold), end="\r")
            real_events = evalres[fold]['test'].real_events
            pred_events = evalres[fold]['test'].pred_events
            metr = evalobj.eval(real_events, pred_events, [act])

            # print('.',end='')
            res[act]['avg'] = add2Avg(res[act]['avg'], metr, len(evalres))
            res[act][fold] = metr
            # print(act)
            # display(res[act])

        res['avg'] = add2Avg(res['avg'], res[act]['avg'], len(dataset.activities_map))

        # print('.')
        # print(res);
    # display(res)
    return res


def mergeEvalsClassic(dataset, evalres, evalobj):
    res = {'avg': {}}

    for act in range(1, len(dataset.activities_map)):

        res[act] = {'avg': {}}

        for fold in evalres:
            print('eval for act=%d fold=%d' % (act, fold), end="\r")
            real_events = evalres[fold]['test'].Sdata.label
            pred_events = evalres[fold]['test'].predicted_classes
            metr = evalobj.eval(real_events, pred_events, [act])
            # print(metr)
            # print('.',end='')
            res[act]['avg'] = add2Avg(res[act]['avg'], metr, len(evalres))
            res[act][fold] = metr

        weights = dataset.activity_events['Activity'].value_counts()

        res['avg'] = add2Avg(res['avg'], res[act]['avg'], len(dataset.activities_map))
        res['avg_weighted'] = add2Avg(res['avg_weighted'], res[act]['avg_weighted'], None, weights[act])

        # print('.')
        # print(res);
    return res


def add2Avg(oldd, newd, count, weights=None):

    for item in newd:
        if type(newd[item]) == type({}):
            oldd[item] = add2Avg(oldd[item] if item in oldd else {}, newd[item], count, weights)
        else:
            if not (item in oldd):
                oldd[item] = 0
            if weights is None:
                oldd[item] += np.array(newd[item]) / count
            else:
                oldd[item] += np.array(newd[item]) * weights[item] / sum(weights.values())

    # if 'f1' in newd and 'precision' in newd and 'recall' in newd:
    #     oldd['f1']=2*(oldd['precision']*oldd['recall'])/(oldd['precision']+oldd['recall']+.000000001)
    return oldd


if __name__ == "__main__":
    import unified_ar.general.utils as utils
    import unified_ar.metric.Metrics
    import unified_ar.result_analyse.kfold_analyse as an

    files = ['200515_13-42-24-VanKasteren']
    titles = 'a,b,c'
    metric = unified_ar.metric.Metrics.Tatbul()
    run_info = {}
    dataset = {}
    evalres = {}
    res = {}
    titles = titles.split(',')
    if (len(titles) != len(files)):
        print('Titles are not correct. use files names instead')
        titles = files
    print(files)
    for i, file in enumerate(files):
        print(i, file)
        t = titles[i]
        run_info[t], dataset[t], evalres[t] = utils.loadState(file)
    #             print(evalres[t])
#                 for i in evalres[t]:
#                     evalres[t][i]['test'].Sdata=None

        dataset[t].sensor_events = None
        res[t] = an.mergeEvals(dataset[t], evalres[t], metric)
    res = {t: res[t] for t in sorted(res.keys())}
    import pandas as pd

    actres = {}
    for k in dataset[t].activities_map:
        if (k == 0):
            continue
        actres[k] = {m: res[m][k]['avg'] for m in res}
        print('act=', k, '==============================')
        print(actres[k])
        if (len(actres[k]) == 0):
            print('No Eval')
        else:
            df2 = pd.DataFrame([actres[k]])
            print(df2)
