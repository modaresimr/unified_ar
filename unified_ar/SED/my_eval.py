import numpy as np
import pandas as pd
import os
import glob
import unified_ar.datatool.seddata
import unified_ar.general.utils
import unified_ar.result_analyse.kfold_analyse as an
import unified_ar.metric.Metrics
import unified_ar.result_analyse.visualisation as vs
from SED.evaluation_measures import psds_score, compute_psds_from_operating_points, compute_metrics
import SED.evaluation_measures
import psds_eval
import warnings

warnings.filterwarnings('error')


def psds_metric(dtc_threshold, gtc_threshold, cttc_threshold, ground_truth, metadata, predictions):
    psds = psds_eval.PSDSEval(dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold, cttc_threshold=cttc_threshold,
                              ground_truth=ground_truth, metadata=metadata.reset_index().rename(columns={'index': 'filename'}))
    # from psds_macro_f1, psds_f1_classes = psds.compute_macro_f_score(predictions)
    det_t = psds._init_det_table(predictions)
    counts, tp_ratios, _, _ = psds._evaluate_detections(det_t)
    per_class_tp = np.diag(counts)[:-1]

    num_gts = np.divide(per_class_tp, tp_ratios, out=np.zeros_like(per_class_tp), where=tp_ratios != 0)
    try:
        num_gts_old = per_class_tp / tp_ratios
        if (num_gts != num_gts_old):
            print(f'not equal {per_class_tp} / {tp_ratios}')
            print(f'num_gts={num_gts} num_gtsold={num_gts_old}')
    except:
        # print(f'devide by zero {per_class_tp} / {tp_ratios}')
        # print(f'num_gts={num_gts}')
        pass

    per_class_fp = counts[:-1, -1]
    per_class_fn = num_gts - per_class_tp
    classes = sorted(set(psds.class_names).difference([psds_eval.psds.WORLD]))

    dic = {c: {'Ntp': tp, 'Nfp': fp, 'Nfn': fn} for c, tp, fp, fn in zip(classes, per_class_tp, per_class_fp, per_class_fn)}

    return dic


def computeGem3(ground_truth, metadata, predictions, debug=0):
    import unified_ar.metric.GEM_NEW as m
    ev = m.eval(ground_truth, predictions, metadata, debug=debug)
    mm = ev[list(ev.keys())[0]].keys()
    out = {m: {c: {'Ntp': ev[c][m]['tp'], 'Nfp': ev[c][m]['fp'], 'Nfn': ev[c][m]['fn'], 'Ntn': ev[c][m]['tn']} for c in ev} for m in mm}
    return out


def computeGem2(gtf, pef, debug=1):
    groundtruth_dataset = datatool.seddata.SED(gtf, 'gt', None)
    groundtruth_dataset.load()
    ######
    pred = datatool.seddata.SED(pef, '', groundtruth_dataset)
    pred.load()
    ########
    evalres = {0: {'test': general.utils.Data('SED')}}
    evalres[0]['test'].real_events = groundtruth_dataset.activity_events
    evalres[0]['test'].pred_events = pred.activity_events
    #######
    res = an.mergeEvals(groundtruth_dataset, evalres, metric.Metrics.GEM())

    import pandas as pd
    from IPython.display import display, HTML

    compact = pd.DataFrame(columns=['avg'])
    actres = {}
    for k in res:
        if (k == 0):
            continue
        if (k == 'avg'):
            a = 'avg'
            actres[k] = {e: res['avg'][e] for e in res['avg']}
        else:
            a = groundtruth_dataset.activities[k]
            actres[k] = {e: res[k]['avg'][e] for e in res[k]['avg']}

        print('act=', a, '==============================')
    #                 print(actres[k])
        compact.loc[a] = None
        if (len(actres[k]) == 0):
            print('No Eval')
        else:
            df2 = pd.DataFrame(actres[k]).round(2)
            for c in df2.columns:
                if not (c in compact.columns):
                    compact[c] = None
                compact.loc[a][c] = df2.loc['f1'][c]
            if (debug):
                display(HTML(df2.to_html()))

    # df2=pd.DataFrame(res['avg']).round(2)
    # for c in df2.columns:
    #     if not(c in compact.columns):compact[c]=None
    #     compact.loc[a][c]=df2.loc['f1'][c]
    compact['avg'] = compact.mean(axis=1)
    if (debug):
        display(HTML(df2.to_html()))
        display(HTML(compact.to_html()))
        vs.plotJoinMetric({'eval': res}, [k for k in res], groundtruth_dataset.activities_map)
    return compact


def get_single_result(gtf, pef, metaf=None, psdsf=None, debug=0):
    res = {'macro_avg', 'micro_avg', 'class'}

    # gem=computeGem(gtf,pef)

    groundtruth = pd.read_csv(gtf, sep="\t")
    # Evaluate a single prediction
    predictions = pd.read_csv(pef, sep="\t")
    meta_df = None
    if (metaf is not None):
        meta_df = pd.read_csv(metaf, sep="\t")

    # print(meta_df)
    return get_single_result_df(groundtruth, predictions, meta_df, debug=debug)


def get_single_result_df(groundtruth, predictions, meta_df=None, psdsf=None, debug=0):
    out = {}
    if meta_df is None:
        meta_df = pd.DataFrame(groundtruth.append(predictions).groupby(['filename'])['offset'].max().rename('duration'))
        meta_df[meta_df['duration'] < 10] = 10

    if 'filename' in meta_df.columns:
        meta_df = meta_df.set_index('filename')

    def calcs(metric):
        df = pd.DataFrame(metric).T
        df.loc['micro-avg'] = df.sum()
        df['recall'] = df['Ntp']/(df['Ntp']+df['Nfn'])
        df['precision'] = df['Ntp']/(df['Ntp']+df['Nfp'])
        df['f1'] = 2*df['precision']*df['recall']/(df['precision']+df['recall'])
        df.loc['macro-avg'] = df.drop('micro-avg').mean()
        # df['f1']=2*df['precision']*df['recall']/(df['precision']+df['recall'])
        return df
    events_metric = SED.evaluation_measures.event_based_evaluation_df(groundtruth, predictions, t_collar=0.200, percentage_of_length=0.2)
    events_metric_df = calcs(events_metric.class_wise)
    out["collar"] = events_metric_df
    # print('events_metric',events_metric)
    # groundtruth=groundtruth[groundtruth['event_label']=='Blender']
    # predictions=predictions[predictions['event_label']=='Blender']
    segment_metric = SED.evaluation_measures.segment_based_evaluation_df(groundtruth, predictions, meta_df, time_resolution=1.)
    # print(segment_metric.class_wise)
    # print('segment_metric',segment_metric)
    segment_metric_df = calcs(segment_metric.class_wise)

    out["segment"] = segment_metric_df
    # macro_f1_event = events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
    # macro_f1_segment = segment_metric.results_class_wise_average_metrics()['f_measure']['f_measure']

    thresh = np.arange(0.1, 1, .2)  # np.arange(0.1,1,.6)=[0.1,0.7]  ,np.arange(0.1,1,.2)=[0.1, 0.3, 0.5, 0.7, 0.9]
    thresh = [0.1, 0.3, 0.5, 0.8, 0.85, 0.9]
    for t in thresh:
        psds = psds_metric(dtc_threshold=t, gtc_threshold=t, cttc_threshold=.3, ground_truth=groundtruth, metadata=meta_df, predictions=predictions)
        psds_df = calcs(psds)
        out[f'psd d/gtc={t}'] = psds_df

    metadata = {}
    for i, f in meta_df.iterrows():
        metadata[i] = (0, f['duration'])

    gem = computeGem3(groundtruth, metadata, predictions, debug=debug)
    for m in gem:
        out[m] = calcs(pd.DataFrame(gem[m]))
    # print(out)
    return out
