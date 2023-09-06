import unified_ar.metric
import os
import pandas as pd
import numpy as np


def intersection(e1, e2):
    inter = (max(e1[0], e2[0]), min(e1[1], e2[1]))
    if (inter[1] <= inter[0]):
        inter = None
#     print(e1,e2,inter)
    return inter


def dur(e):
    d = e[1]-e[0]
    if (d < 0):
        print('erorr duration is less than zero')
    return d


def eval_my_metric(real, pred, duration=(0, 10), alpha=2, debug=0, calcne=1):
    debug = {'D': 0, 'T': 0, 'M': 0, 'R': 0, 'V': 1}  # V:verbose
    # real=merge_events_if_necessary(real)
    # pred=merge_events_if_necessary(pred)
    # real_tree=_makeIntervalTree(real,'r')
    # pred_tree=_makeIntervalTree(pred,'p')
    duration = (min(duration[0], real[0][0]), max(duration[1], real[-1][1]))
    real = np.append(real, duration[1])  # add a zero duration event in the end for ease comparision the last event
    real[-1] = (duration[1], duration[1])
    pred = np.append(pred, duration[1])  # add a zero duration event in the end for ease comparision the last event
    pred[-1] = (duration[1], duration[1])
    # _ means negative
    rel = {'r+': {}, 'r-': {}, 'p+': {}, 'p-': {}}
    print(real)
    r_0 = (duration[0], real[0][0])
    r_n = (real[-1][1], duration[1])
    metric = {}
    pi = 0
    rcalc = []
    real_ = []
    pred_ = []
    ri_ = -1
    for ri in range(len(real)):
        r = real[ri]
        rp = real[ri-1] if ri > 0 else (duration[0], duration[0])
        r_ = (rp[1], r[0])
        tmpr = {'p+': {}, 'p-': {}}
        tmpr_ = {'p+': {}, 'p-': {}}

        if (dur(r_) > 0):
            real_.append(r_)
            ri_ = len(real_)-1
            rel['r-'][ri_] = tmpr_

        rel['r+'][ri] = tmpr

        cond = pi < len(pred)
        pi_ = -1
        while cond:
            pp = pred[pi-1] if pi > 0 else (duration[0], duration[0])
            p = pred[pi]
            p_ = (pp[1], p[0])

            if (dur(p_) > 0 and (len(pred_) == 0 or pred_[-1] != p_)):
                pred_.append(p_)
                pi_ = len(pred_)-1

            if not (pi in rel['p+']):
                rel['p+'][pi] = {'r+': {}, 'r-': {}}
            if not (pi_ in rel['p-']):
                rel['p-'][pi_] = {'r+': {}, 'r-': {}}
            tmpp = rel['p+'][pi]
            tmpp_ = rel['p-'][pi_]

            rinter = intersection(r, p)
            rinter_ = intersection(r, p_)
            r_inter = intersection(r_, p)
            r_inter_ = intersection(r_, p_)
            if (rinter is not None):
                # tmpr['p+'].append((pi,rinter))
                # tmpp['r+'].append((ri,rinter))
                tmpr['p+'][pi] = rinter
                tmpp['r+'][ri] = rinter
            if (rinter_ is not None):
                # tmpr['p-'].append((pi,rinter_))
                # tmpp_['r+'].append((ri,rinter_))
                tmpr['p-'][pi_] = rinter_
                tmpp_['r+'][ri] = rinter_
            if (r_inter is not None):
                # tmpr_['p+'].append((pi,r_inter))
                # tmpp['r-'].append((ri,r_inter))
                tmpr_['p+'][pi] = r_inter
                tmpp['r-'][ri_] = r_inter
            if (r_inter_ is not None):
                # tmpr_['p-'].append((pi,r_inter_))
                # tmpp_['r-'].append((ri,r_inter_))
                tmpr_['p-'][pi_] = r_inter_
                tmpp_['r-'][ri_] = r_inter_

            if pred[pi][1] < r[1]:
                pi += 1
            else:
                cond = False

        # for k in list(rel.keys()):
        #     if len(rel[k])>0: continue
        #     del rel[k]

    real = np.delete(real, -1, 0)  # real.pop()
    pred = np.delete(pred, -1, 0)  # pred.pop()
#         if(dur(pred_[-1])==0):pred_=np.delete(pred_,-1,0)
#         if(dur(real_[-1])==0):real_=np.delete(real_,-1,0)

    out = {
        'detection':        {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'monotony':         {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'total duration':   {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'relative duration': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    }

    if debug['V']:
        print("real=", real)
        print("pred=", pred)
        print("real_=", real_)
        print("pred_=", pred_)
        # for x in rel:
        [print(f'{x}: {rel[x]}') for x in rel]

    for ri in range(len(real)):
        tpd = int(len(rel['r+'][ri]['p+']) > 0)
        out['detection']['tp'] += tpd
        if debug['D']:
            print(f"D TP+{tpd}      ri={ri}, p+={rel['r+'][ri]['p+']}>0")
        # monotony {
        if (len(rel['r+'][ri]['p+']) == 1):
            for rpi in rel['r+'][ri]['p+']:
                if len(rel['p+'][rpi]['r+']) == 1:
                    out['monotony']['tp'] += 1
                    if debug['M']:
                        print(f"M TP+1     rel[r+][{ri}][p+]={rel['r+'][ri]['p+']}==1 rel[p+][{rpi}][r+]={rel['p+'][rpi]['r+']}==1")
                elif (len(rel['p+'][rpi]['r+']) == 0):
                    print('error it can not be zero')
                elif debug['M']:
                    print(f"M--tp rel[r+][{ri}][p+]={rel['r+'][ri]['p+']}==1 rel[p+][{rpi}][r+]={rel['p+'][rpi]['r+']}>1")
        # }

        for pi in rel['r+'][ri]['p+']:
            tpt = dur(rel['r+'][ri]['p+'][pi])
            tpr = tpt/dur(real[ri])
            out['total duration']['tp'] += tpt
            out['relative duration']['tp'] += tpr
            if debug['T']:
                print(f"T tp+={tpt}             rel[r+][{ri}][p+][{pi}]=dur({rel['r+'][ri]['p+'][pi]})")
            if debug['R']:
                print(f"R tp+={tpr}             rel[r+][{ri}][p+][{pi}]==dur({rel['r+'][ri]['p+'][pi]}) / real[{ri}]=dur({real[ri]})")

        for pi in rel['r+'][ri]['p-']:
            fnt = dur(rel['r+'][ri]['p-'][pi])
            fnr = fnt/dur(real[ri])
            out['total duration']['fn'] += fnt
            out['relative duration']['fn'] += fnr
            if debug['T']:
                print(f"T fn+={fnt}             rel[r+][{ri}][p-][{pi}]=dur({rel['r+'][ri]['p-'][pi]})")
            if debug['R']:
                print(f"R fn+={fnr}             rel[r+][{ri}][p-][{pi}]==dur({rel['r+'][ri]['p-'][pi]}) / real[{ri}]=dur({real[ri]})")

    for ri in range(len(real_)):
        tnd = int(len(rel['r-'][ri]['p-']) > 0)
        out['detection']['tn'] += tnd
        if debug['D']:
            print(f"D TN+{tnd}      ri-={ri}, p-={rel['r-'][ri]['p-']}>0")
        # monotony {

        if (len(rel['r-'][ri]['p-']) == 1):
            for rpi in rel['r-'][ri]['p-']:
                if len(rel['p-'][rpi]['r-']) == 1:
                    out['monotony']['tn'] += 1
                    if debug['M']:
                        print(f"M TN+1     rel[r-][{ri}][p-]={rel['r-'][ri]['p-']}==1 rel[p-][{rpi}][r-]={rel['p-'][rpi]['r-']}==1")
                elif (len(rel['p-'][rpi]['r-']) == 0):
                    print('error it can not be zero')
                elif debug['M']:
                    print(f"M--tn rel[r-][{ri}][p-]={rel['r-'][ri]['p-']}==1 rel[p-][{rpi}][r-]={rel['p-'][rpi]['r-']}>1")
        # }

        for pi in rel['r-'][ri]['p-']:
            tnt = dur(rel['r-'][ri]['p-'][pi])
            tnr = tnt/dur(real_[ri])
            out['total duration']['tn'] += tnt
            out['relative duration']['tn'] += tnr
            if debug['T']:
                print(f"T tn+={tnt}             rel[r-][{ri}][p-][{pi}]=dur({rel['r-'][ri]['p-'][pi]})")
            if debug['R']:
                print(f"R tn+={tnr}             rel[r-][{ri}][p-][{pi}]==dur({rel['r-'][ri]['p-'][pi]}) / real_[{ri}]=dur({real_[ri]})")
        for pi in rel['r-'][ri]['p+']:
            fpt = dur(rel['r-'][ri]['p+'][pi])
            fpr = fpt/dur(real_[ri])
            out['total duration']['fp'] += fpt
            out['relative duration']['fp'] += fpr
            if debug['T']:
                print(f"T fp+={fpt}             rel[r-][{ri}][p+][{pi}]=dur({rel['r-'][ri]['p+'][pi]})")
            if debug['R']:
                print(f"R fp+={fpr}             rel[r-][{ri}][p+][{pi}]==dur({rel['r-'][ri]['p+'][pi]}) / real_[{ri}]=dur({real_[ri]})")

    out['detection']['fp'] = len(real_)-out['detection']['tn']
    if debug['D']:
        print(f"D fp={out['detection']['fp']} #r-={len(real_)} - tn={out['detection']['tn']}")
    out['detection']['fn'] = len(real)-out['detection']['tp']
    if debug['D']:
        print(f"D fn={out['detection']['fn']} #r+={len(real)} - tp={out['detection']['tp']}")

    out['monotony']['fn'] = len(real)-out['monotony']['tp']+len(pred_)-out['monotony']['tn']
    if debug['M']:
        print(f"M fn={out['monotony']['fn']}     #r+={len(real)} - tp={out['monotony']['tp']} + #p-={len(pred_)} - tn={out['monotony']['tn']}")
    out['monotony']['fp'] = len(pred)-out['monotony']['tp']+len(real_)-out['monotony']['tn']
    if debug['M']:
        print(f"M fp={out['monotony']['fp']}     #p+={len(pred)} - tp={out['monotony']['tp']} + #r-={len(real_)} - tn={out['monotony']['tn']}")

    for pi in range(len(pred)):
        fpd = int(len(rel['p+'][pi]['r+']) == 0)
        out['detection']['fp'] += fpd
        if debug['D']:
            print(f"D FP+{fpd}      pi={pi}, r={rel['p+'][pi]['r+']}==0")
#             for ri in rel['p+'][pi]['r-']:
#                 out['total duration']['fp']+=dur(rel['p+'][pi]['r-'][ri])
#                 out['relative duration']['fp']+=dur(rel['p+'][pi]['r-'][ri])/dur(pred[pi])

    for pi in range(len(pred_)):
        fnd = int(len(rel['p-'][pi]['r-']) == 0)
        out['detection']['fn'] += fnd
        if debug['D']:
            print(f"D FN+{fnd}      pi-={pi}, r-={rel['p-'][pi]['r-']}==0")
#             for ri in rel['p-'][pi]['r+']:
#                 out['total duration']['fn']+=dur(rel['p-'][pi]['r+'][ri])
#                 out['relative duration']['fn']+=dur(rel['p-'][pi]['r+'][ri])/dur(pred_[pi])

#         plot_events_with_event_scores(range(len(real)),range(len(pred)),real,pred)
#         plot_events_with_event_scores(range(len(real_)),range(len(pred_)),real_,pred_)
    if debug['V']:
        plot_events(real, pred, real_, pred_)
    return out


def plot_events(real, pred, real_, pred_, label=None):
    from matplotlib.pylab import plt
    import random
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_title(label)
    plt.xlim(0, max(real[-1][1], 10))
    ax.set_xticks(np.arange(0, max(real[-1][1], 10), .1), minor=True)
    maxsize = 20
    for i in range(min(maxsize, len(pred_))):
        d = pred_[i]
        plt.axvspan(d[0], d[1], 0, 0.4, linewidth=1, edgecolor='k', facecolor='m', alpha=.6)

    for i in range(min(maxsize, len(pred))):
        d = pred[i]
        plt.axvspan(d[0], d[1], 0.1, 0.5, linewidth=1, edgecolor='k', facecolor='r', alpha=.6)
#     maxsize=len(real)
    for i in range(min(maxsize, len(real_))):
        gt = real_[i]
        plt.axvspan(gt[0], gt[1], 0.6, 1, linewidth=1, edgecolor='k', facecolor='y', alpha=.6)

    for i in range(min(maxsize, len(real))):
        gt = real[i]
        plt.axvspan(gt[0], gt[1], 0.5, .9, linewidth=1, edgecolor='k', facecolor='g', alpha=.6)
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    # plt.show()


# @interact
# def result_selector(gtf=os.listdir(f'{rootFolder}/metadata/')):
# import SED.my_eval
gtf = 'public.tsv'
rootFolder = '/workspace/sed2020/'
typ = gtf.split('.')[0]
gtf = f'{rootFolder}/metadata/{gtf}'
# meta_dur_df=pd.DataFrame(columns=['filename','duration'])
# meta_dur_df['filename']=groundtruth['filename']
# meta_dur_df['duration']=10
total_dic = {}
for team in sorted(os.listdir(f'{rootFolder}/submissions/')):
    print(f'analysing team {team}')
    for code in sorted(os.listdir(f'{rootFolder}/submissions/{team}')):
        print(f'    {code}')
        base_prediction_path = f'{rootFolder}/submissions/{team}/{code}/{typ}/'
        pef = f'{base_prediction_path}/{code}.output.tsv'
        if not (os.path.isfile(pef)):
            all = [x for x in os.listdir(base_prediction_path) if '.output.tsv' in x]
            if len(all) > 0:
                pef = f'{base_prediction_path}/{all[0]}'
            else:
                print(pef)
                continue
        title = code.replace('_task4', '')
        groundtruth = pd.read_csv(gtf, sep="\t")
        # Evaluate a single prediction
        predictions = pd.read_csv(pef, sep="\t")
        break
    break


# %matplotlib inline

clas = groundtruth.event_label.append(predictions.event_label).unique()
clas
gt = groundtruth
pt = predictions
# m=metric.GEM_NEW
for c in clas:
    gtc = gt.loc[gt.event_label == c]
    ptc = pt.loc[pt.event_label == c]

    for f in gtc.filename.unique():
        g = gtc.loc[gtc.filename == f].apply(lambda l: (l.onset, l.offset), axis=1).values
        p = ptc.loc[ptc.filename == f].apply(lambda l: (l.onset, l.offset), axis=1).values

        print(eval_my_metric(g, p, (0, 10)))
        break
#         print('gtc',
#         print('ptc',ptc.loc[ptc.filename==f]).apply(p=>(p.onset,p.offset))
#         p=pt.loc[pt.event_label==c and pt.filename=g.filename]
#         print(p)
    break
