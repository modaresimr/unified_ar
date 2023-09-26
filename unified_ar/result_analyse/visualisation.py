from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import sklearn.metrics
from matplotlib.pylab import plt
from ..wardmetrics.core_methods import eval_events, eval_segments
from ..wardmetrics.utils import *
from ..wardmetrics.visualisations import *
from IPython.display import display
import unified_ar.metric.MyMetric as MyMetric
import unified_ar.result_analyse.dataset_viewer as dv
import unified_ar.result_analyse.SpiderChart as spiderchart
from unified_ar.metric.CMbasedMetric import CMbasedMetric
from unified_ar.metric.EventBasedMetric import time2int
import unified_ar.general.utils


def filterTime(events, duration=None):
    if (duration is not None):
        events = events.loc[duration[0] <= events.EndTime]
        events = events.loc[duration[1] >= events.StartTime]
    return events


def plot_joint_events(dataset, real_events, pred_events, onlyAct=None, duration=None):
    if duration == None:
        duration = [real_events.iloc[0]['StartTime'], real_events.iloc[0]['StartTime'] + pd.Timedelta('3d')]
    real_events = filterTime(real_events, duration)
    pred_events = filterTime(pred_events, duration)

    # visualize(dataset)
    # real_events, pred_events = remove_gaps(real_events, pred_events)
    print('visualizing real and pred1')
    plotJoinAct(dataset, real_events, pred_events, onlyAct=onlyAct)


def my_result_analyse(dataset, real_events, pred_events, onlyAct=None, duration=None):
    real_events = filterTime(real_events, duration)
    pred_events = filterTime(pred_events, duration)

    # visualize(dataset)
    real_events, pred_events = remove_gaps(real_events, pred_events)
    print('visualizing real and pred')
    plotJoinAct(dataset, real_events, pred_events, onlyAct=onlyAct)
    plotMyMetric2(dataset, real_events, pred_events, onlyAct=onlyAct)
    plotWardMetric(dataset, real_events, pred_events, onlyAct=onlyAct)


def plotWardMetric(dataset, real_events, pred_events, onlyAct=None):
    # Calculate segment results:
    acts = [i for i in dataset.activities_map] if onlyAct == None else [onlyAct]
    revent = {}
    pevent = {}
    for act in acts:
        revent[act] = []
        pevent[act] = []

    for i, e in real_events.iterrows():
        if not (e.Activity in acts):
            continue
        revent[e.Activity].append((time2int(e.StartTime), time2int(e.EndTime)))
    for i, e in pred_events.iterrows():
        if not (e.Activity in acts):
            continue
        pevent[e.Activity].append((time2int(e.StartTime), time2int(e.EndTime)))

    result = {}
    for act in acts:
        ground_truth_test = revent[act]
        detection_test = pevent[act]
        print(dataset.activities_map[act], "==================================")
        if (len(ground_truth_test) == 0 or len(detection_test) == 0):
            print("No Prediction or Ground Truth -> Ward metric can not be ploted")
            continue

        twoset_results, segments_with_scores, segment_counts, normed_segment_counts = eval_segments(ground_truth_test, detection_test)

        fn = segment_counts["D"] + segment_counts["F"] + segment_counts["Us"] + segment_counts["Ue"]
        fp = segment_counts["I"] + segment_counts["M"] + segment_counts["Os"] + segment_counts["Oe"]

        recall = segment_counts['TP'] / (segment_counts['TP'] + fn)
        prec = segment_counts['TP'] / (segment_counts['TP'] + fp)
        f1 = 2 * recall * prec / (recall + prec + 0.000001)
        print({'recall': recall, 'precision': prec, 'f1': f1})
        # Print results:

        print_detailed_segment_results(segment_counts)

        segment_counts['TN'] = 0
        normed_segment_counts['TN'] = 0
        all = np.sum(detailed_segment_results_to_list(segment_counts))
        for item in normed_segment_counts:
            normed_segment_counts[item] = round(segment_counts[item] / all * 100, 2)
        print('Normalized segment_results %')
        print_detailed_segment_results(normed_segment_counts)
        print_twoset_segment_metrics(twoset_results)
        # Access segment results in other formats:
        # print("\nAbsolute values:")
        # print("----------------")
        # print(detailed_segment_results_to_list(segment_counts)) # segment scores as basic python list
        # print(detailed_segment_results_to_string(segment_counts)) # segment scores as string line
        # print(detailed_segment_results_to_string(segment_counts, separator=";", prefix="(", suffix=")\n")) # segment scores as string line

        # print("Normed values:")
        # print("--------------")
        # print(detailed_segment_results_to_list(normed_segment_counts)) # segment scores as basic python list
        # print(detailed_segment_results_to_string(normed_segment_counts)) # segment scores as string line
        # print(detailed_segment_results_to_string(normed_segment_counts, separator=";", prefix="(", suffix=")\n")) # segment scores as string line

        # Access segment metrics in other formats:
        # print("2SET metrics:")
        # print("-------------")
        # print(twoset_segment_metrics_to_list(twoset_results)) # twoset_results as basic python list
        # print(twoset_segment_metrics_to_string(twoset_results)) # twoset_results as string line
        # print(twoset_segment_metrics_to_string(twoset_results, separator=";", prefix="(", suffix=")\n")) # twoset_results as string line

        # Visualisations:
        plot_events_with_segment_scores(segments_with_scores, ground_truth_test, detection_test)
        # plot_segment_counts(segment_counts)
        plot_twoset_metrics(twoset_results, startangle=45)

        # Run event-based evaluation:
        gt_event_scores, det_event_scores, detailed_scores, standard_scores = eval_events(ground_truth_test, detection_test)

        # Print results:
        # print_standard_event_metrics(standard_scores)
        print_detailed_event_metrics(detailed_scores)

        # Access results in other formats:
        print(standard_event_metrics_to_list(standard_scores))  # standard scores as basic python list, order: p, r, p_w, r_w
        print(standard_event_metrics_to_string(standard_scores))  # standard scores as string line, order: p, r, p_w, r_w)
        print(standard_event_metrics_to_string(standard_scores, separator=";", prefix="(",
                                               suffix=")\n"))  # standard scores as string line, order: p, r, p_w, r_w

        print(detailed_event_metrics_to_list(detailed_scores))  # detailed scores as basic python list
        print(detailed_event_metrics_to_string(detailed_scores))  # detailed scores as string line
        print(detailed_event_metrics_to_string(detailed_scores, separator=";", prefix="(", suffix=")\n"))  # standard scores as string line

        # Show results:
        plot_events_with_event_scores(gt_event_scores, det_event_scores, ground_truth_test, detection_test, show=False)
        plot_event_analysis_diagram(detailed_scores)


def remove_gaps(real_events, pred_events, onlyAct=None, max_events=20):
    if (len(real_events) + len(pred_events) < 5):
        return real_events, pred_events
    removdur = pd.to_timedelta('0s')

    pred_events = pred_events[['StartTime', 'EndTime', 'Activity']].copy()
    real_events = real_events[['StartTime', 'EndTime', 'Activity', 'Duration']].copy()
    if not (onlyAct is None):
        real_events = real_events.loc[real_events.Activity == onlyAct]
        pred_events = pred_events.loc[pred_events.Activity == onlyAct]
    # print(str(onlyAct),'===>r:',len(real_events),' p:',len(pred_events))
    real_events = real_events.head(2 * max_events).tail(max_events)
    pred_events = pred_events.loc[pred_events.EndTime < real_events.EndTime.iloc[-1]].loc[pred_events.EndTime >= real_events.StartTime.iloc[0]]
    # print(str(onlyAct),'after===>r:',len(real_events),' p:',len(pred_events))
    allItems = []

    defGap = real_events.Duration.sum() / len(real_events.Duration)
    if len(real_events.Duration) == 0:
        defGap = pd.to_timedelta('2m')
    for i, (k, row) in enumerate(real_events.iterrows()):
        allItems.append({'indx': i, 'type': 'real', 'state': 'start', 'time': row.StartTime})
        allItems.append({'indx': i, 'type': 'real', 'state': 'end', 'time': row.EndTime})
    for i, (k, row) in enumerate(pred_events.iterrows()):
        allItems.append({'indx': i, 'type': 'pred', 'state': 'start', 'time': row.StartTime})
        allItems.append({'indx': i, 'type': 'pred', 'state': 'end', 'time': row.EndTime})

    allItems.sort(key=lambda elem: elem['time'])
    removdur = pd.to_timedelta('0s')
    gap_start = allItems[0]['time']
    gap_end = allItems[0]['time']
    states = {'real': 0, 'pred': 0}
    for i, item in enumerate(allItems):
        isvalid = states['pred'] + states['real'] == 0
        if (item['state'] == 'start'):
            states[item['type']] = 1
            gap_end = item['time']
        else:
            states[item['type']] = 0
            gap_start = item['time']
        dur = max(defGap, pd.to_timedelta('2m'))
        if (isvalid):
            removdur += max(np.timedelta64(), gap_end - gap_start - dur)
        if (item['state'] == 'start'):
            if (item['type'] == 'real'):
                rc = item['indx']
                real_events.iat[rc, 1] -= removdur
                real_events.iat[rc, 0] -= removdur
            if (item['type'] == 'pred'):
                pc = item['indx']
                pred_events.iat[pc, 1] -= removdur
                pred_events.iat[pc, 0] -= removdur

    return real_events, pred_events


def remove_gaps_old(real_events, pred_events, onlyAct=None):
    pc = 0
    removdur = pd.to_timedelta('0s')
    pred_events = pred_events[['StartTime', 'EndTime', 'Activity']]
    if not (onlyAct is None):
        real_events = real_events.loc[real_events.Activity == onlyAct]
        pred_events = pred_events.loc[pred_events.Activity == onlyAct]

    for rc in range(0, len(real_events)):
        lasttime = real_events.iloc[rc]['EndTime']
        real_events.iat[rc, 1] -= removdur
        real_events.iat[rc, 0] -= removdur

        while (pc < len(pred_events) and pred_events.iloc[pc]['StartTime'] - removdur < lasttime):
            pred_events.iat[pc, 1] -= removdur
            pred_events.iat[pc, 0] -= removdur
            pc += 1

        if (pc > 0):
            lasttime = max(lasttime, pred_events.iloc[pc - 1]['EndTime'])

        dur = max(real_events.iloc[rc]['Duration'], pd.to_timedelta('5m'))

        inittime = min(real_events.iloc[rc + 1]['StartTime'], pred_events.iloc[pc]['StartTime'])
        while pc <= len(pred_events) and real_events.iloc[rc + 1]['StartTime'] >= pred_events.iloc[pc]['StartTime']:
            if (rc + 1 >= len(real_events)):
                break
            elif pc >= len(pred_events):
                inittime = real_events.iloc[rc + 1]['StartTime']
            else:
                inittime = min(real_events.iloc[rc + 1]['StartTime'], pred_events.iloc[pc]['StartTime'])
            pred_events.iat[pc, 1] -= removdur
            pred_events.iat[pc, 0] -= removdur
            pc += 1
            removdur += max(np.timedelta64(), inittime - lasttime - dur)

    for i in range(pc, len(pred_events)):
        pred_events.iat[i, 1] -= removdur
        pred_events.iat[i, 0] -= removdur

    return real_events, pred_events


def plotMyMetric2(dataset, real_events, pred_events, onlyAct=None, ax=None, debug=1, calcne=1):
    acts = [i for i in dataset.activities_map] if onlyAct == None else [onlyAct]
    res = MyMetric.eval(real_events, pred_events, acts, debug=debug, calcne=calcne)
    if not (onlyAct is None):
        res2 = res
        res = {}
        res[onlyAct] = res2
    for i in res:
        metrics = res[i]
        # plotJoinAct(dataset,real_events,pred_events,onlyAct=i)
        df = pd.DataFrame(metrics)
        print(dataset.activities_map[i], "========")
        display(df.round(2))

        print('average=', np.average(list(df.loc['f1'])))
        df = df.drop(['tp', 'fp', 'fn'])
        spiderchart.plot(df, [0.25, .5, .75], title=dataset.activities_map[i], ax=ax)


def plotMyMetric(allmetrics, acts, actmap={}):
    acount = len(acts)
    col = min(4, acount)
    row = int(np.ceil((acount) / float(col)))
    import unified_ar.result_analyse.SpiderChart
    # result_analyse.SpiderChart.radar_factory(5, frame='polygon')
    m_fig, m_ax = plt.subplots(row, col, figsize=(col * 3, row * 3), subplot_kw=dict(projection='radar'))
    if type(m_ax) != np.ndarray:
        m_ax = np.array([m_ax])
    m_ax = m_ax.flatten()

    for i, act in enumerate(acts):
        metrics = allmetrics[act]
        if ('avg' in metrics):
            metrics = metrics['avg']
        # plotJoinAct(dataset,real_events,pred_events,onlyAct=i)
        df = pd.DataFrame(metrics)
        name = actmap[act] if act in actmap else act
        unified_ar.result_analyse.SpiderChart.radar_factory(len(metrics.columns), frame='polygon')
        print(name, "========")
        display(df.round(2))

        print('average=', np.average(list(df.loc['f1'])))
        df = df.drop(['tp', 'fp', 'fn'])
        spiderchart.plot(df, [0.25, .5, .75], title=name, ax=m_ax[i])
    m_fig.tight_layout(pad=0, h_pad=-10.0, w_pad=3.0)


def plotJoinMetric(joinmetrics, acts, actmap={}):
    acount = len(acts)
    col = min(4, acount)
    row = int(np.ceil((acount) / float(col)))
    import unified_ar.result_analyse.SpiderChart
    unified_ar.result_analyse.SpiderChart.radar_factory(5, frame='polygon')
    m_fig, m_ax = plt.subplots(row, col, figsize=(col * 3, row * 3), subplot_kw=dict(projection='radar'))
    if type(m_ax) != np.ndarray:
        m_ax = np.array([m_ax])
    m_ax = m_ax.flatten()

    # acts.insert(0, 'avg')
    # actmap['avg'] = 'avg'
    for i, act in enumerate(acts):
        try:
            all = None
            name = actmap[act] if act in actmap else act
            # print(name, "========")
            # actres[act]={(m,e):joinmetrics[m][act]['avg'][e]  for m in joinmetrics for e in joinmetrics[m][act]['avg']}

            # # # # # #{ VISUALISATION
            # if act:
            if act in ['avg', 'avg_weighted']:
                tmp = {(name, m, e): joinmetrics[m][act][e] for m in joinmetrics for e in joinmetrics[m][act]}
            else:
                tmp = {(name, m, e): joinmetrics[m][act]['avg'][e] for m in joinmetrics for e in joinmetrics[m][act]['avg']}
#                 print(actres[k])

            if tmp:
                df = pd.DataFrame(tmp).round(2)
                # if not ('f1' in df.index):
                #         print('f1 is not exist')
                #         continue
                # else:
                for c in df.columns.get_level_values(1).unique():
                    df = df.rename({c: f'{c}, avg={df.loc["f1"][name][c].mean().round(2)}'}, axis=1, level=1)
                keys = ['tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1']

                display(df.loc[[k for k in keys if k in df.index]])

            # # # # # #} VISUALISATION
            for item in joinmetrics:
                metrics = joinmetrics[item][act]
                if ('avg' in metrics):
                    metrics = metrics['avg']
                if len(metrics) < 4:
                    continue

                # plotJoinAct(dataset,real_events,pred_events,onlyAct=i)
                df = pd.DataFrame(metrics)
                # display(df.round(2))
                if not ('f1' in df.index):
                    print('f1 is not exist')
                    continue

                # print('average=', np.average(list(df.loc['f1'])))
                # df=df.drop(['tp','fp','fn','recall','precision'])
                # df=df.loc[['f1']]

                calcf1 = 2 * df.loc['recall'] * df.loc['precision'] / (df.loc['recall'] + df.loc['precision'])
                df.loc['f1'] = calcf1
                df = df.loc[['f1']]
                if all is None:
                    all = df.drop(['f1'])
                all.loc[item] = df.loc['f1']

            if not (all is None):
                # print(f'all={all} title={name}, ax={m_ax[i]}')

                spiderchart.plot(all, [0.25, .5, .75], title=name, ax=m_ax[i])

        except Exception as e:
            print(e)
            import traceback
            import sys
            traceback.print_exception(*sys.exc_info())

    m_fig.tight_layout(pad=0, h_pad=-10.0, w_pad=3.0)


def visualize(dataset):
    dv.displaycontent(dataset)
    dv.view(dataset, 1)
    dv.plotAct(dataset, dataset.activity_events)


# from unified_ar.general.utils import loadState
# dataset,real_events,pred_events=loadState('r1')
# my_result_analyse(dataset,real_events,pred_events)


# +
def plotJoinAct(dataset, real_acts, pred_acts, label=None, onlyAct=None, ax=None):
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    size = 0.45
    acts = dataset.activities if onlyAct is None else [onlyAct]
    if not (onlyAct is None):
        real_acts = real_acts.loc[real_acts.Activity == onlyAct]
        pred_acts = pred_acts.loc[pred_acts.Activity == onlyAct]

    if (len(real_acts) == 0):
        print('not enough data of this type', onlyAct)
        return
    if (len(pred_acts) == 0):
        pred_acts = pred_acts.append({
            'StartTime': real_acts.StartTime.iloc[0],
            'EndTime': real_acts.EndTime.iloc[0],
            'Activity': real_acts.Activity.iloc[0]
        },
            ignore_index=True)
        print('not enough p data of this type', onlyAct)
        print(pred_acts)
        return

#   ft=real_acts.StartTime.iloc[0]
#   dur=ft+pd.to_timedelta('12h')
#   real_acts=real_acts.loc[real_acts.StartTime<dur]
#   pred_acts=pred_acts.loc[pred_acts.StartTime<dur]
# apply(lambda x:dataset.activities_map[x.Activity], axis=1).tolist()
    ract = (real_acts.Activity + (size / 2)).tolist()
    rstart = real_acts.StartTime.tolist()
    rend = real_acts.EndTime.tolist()
    # .apply(lambda x:dataset.activities_map[x.Activity], axis=1).tolist()
    pact = (pred_acts.Activity - (size / 2)).tolist()
    pstart = pred_acts.StartTime.tolist()
    pend = pred_acts.EndTime.tolist()
    if ax == None:
        if (onlyAct):
            fig, ax = plt.subplots(figsize=(10, 0.5))
        else:
            fig, ax = plt.subplots(figsize=(50, len(acts)))
    ax.set_title(label)
    if (len(real_acts) == 0):
        print('no r activity of this type', label)
    else:
        _plotActs(ax, ract, rstart, rend, linewidth=1, edgecolor='k', facecolor='g', size=size, alpha=.6)
    if (len(pred_acts) == 0):
        print('no p activity of this type', label)
    else:
        _plotActs(ax, pact, pstart, pend, linewidth=1, edgecolor='k', facecolor='r', size=size, alpha=.6)


#   data_linewidth_plot(pact, pstart, pend, ax=ax,
#                       colors="red", alpha=1, linewidth=.3)
# plt.hlines(ract, rstart, rend, colors=(0,.5,0,.2), linewidth=1)
# plt.hlines(pact, pstart, pend, colors="red", lw=2)
#   loc = mdates.AutoDateLocator()
#   ax.xaxis.set_major_locator(loc)
#   ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    ax.set_xticks([])
    ax.set_xlim(min(pstart + rstart), max(pend + rend))
    ax.set_yticks([i for i in dataset.activities_map])
    ax.yaxis.grid(True)

    ax.set_yticklabels([l for l in dataset.activities_map_inverse])
    if (onlyAct):
        ax.set_ylim(onlyAct - size, onlyAct + size)
    else:
        ax.set_ylim(0 + size, len(dataset.activities) - size)
    plt.margins(0.1)
    plt.show()


def plotJoinAct2(real_acts, pred_acts, acts, labels, ax=None, duration=None):
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    if not (filterTime == None):
        real_acts = filterTime(real_acts, duration)
        pred_acts = filterTime(pred_acts, duration)

    size = 0.45
    rcond = real_acts.Activity == -1
    pcond = pred_acts.Activity == -1
    for act in acts:
        rcond |= real_acts.Activity == act
        pcond |= pred_acts.Activity == act
    real_acts = real_acts.loc[rcond]
    pred_acts = pred_acts.loc[pcond]

    if (len(real_acts) == 0):
        print('not enough data of this type', acts)
        return
    if (len(pred_acts) == 0):
        # pred_acts=pred_acts.append({'StartTime':real_acts.StartTime.iloc[0],'EndTime':real_acts.EndTime.iloc[0],'Activity':real_acts.Activity.iloc[0]},ignore_index=True)
        print('not enough p data of this type', acts)
        print(pred_acts)
        return

#   ft=real_acts.StartTime.iloc[0]
#   dur=ft+pd.to_timedelta('12h')
#   real_acts=real_acts.loc[real_acts.StartTime<dur]
#   pred_acts=pred_acts.loc[pred_acts.StartTime<dur]
# apply(lambda x:dataset.activities_map[x.Activity], axis=1).tolist()
    ract = (real_acts.Activity + (size / 2)).tolist()
    rstart = real_acts.StartTime.tolist()
    rend = real_acts.EndTime.tolist()
    # .apply(lambda x:dataset.activities_map[x.Activity], axis=1).tolist()
    pact = (pred_acts.Activity - (size / 2)).tolist()
    pstart = pred_acts.StartTime.tolist()
    pend = pred_acts.EndTime.tolist()
    if ax == None:
        fig, ax = plt.subplots(figsize=(10, len(acts) / 5 + .3))
    ax.set_title(labels.__str__())
    if (len(real_acts) == 0):
        print('no r activity of this type', label)
    else:
        _plotActs(ax, ract, rstart, rend, linewidth=1, edgecolor='k', facecolor='g', size=size, alpha=.6)
    if (len(pred_acts) == 0):
        print('no p activity of this type', label)
    else:
        _plotActs(ax, pact, pstart, pend, linewidth=1, edgecolor='k', facecolor='r', size=size, alpha=.6)
#   data_linewidth_plot(pact, pstart, pend, ax=ax,
#                       colors="red", alpha=1, linewidth=.3)
# plt.hlines(ract, rstart, rend, colors=(0,.5,0,.2), linewidth=1)
# plt.hlines(pact, pstart, pend, colors="red", lw=2)
#   loc = mdates.AutoDateLocator()
#   ax.xaxis.set_major_locator(loc)
#   ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    ax.set_xticks([])
    ax.set_xlim(min(pstart + rstart), max(pend + rend))
    ax.set_yticks([i for i in acts])
    ax.yaxis.grid(True)

    ax.set_yticklabels([l for l in labels])
    ax.set_ylim(min(acts) - size, max(acts) + size)
    plt.margins(0.1)


#   plt.show()

# -


def _plotActs(ax, x, s, e, **kwargs):
    size = kwargs.pop("size", 1)
    for i in range(len(x)):
        ax.add_patch(patches.Rectangle((s[i], x[i] - size / 2), e[i] - s[i], size, **kwargs))

        #


def plotJoinAct(dataset, real_acts, pred_acts, label=None, onlyAct=None, ax=None):
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    size = 0.45
    acts = dataset.activities if onlyAct is None else [onlyAct]
    if not (onlyAct is None):
        real_acts = real_acts.loc[real_acts.Activity == onlyAct]
        pred_acts = pred_acts.loc[pred_acts.Activity == onlyAct]

    if (len(real_acts) == 0):
        print('not enough data of this type', onlyAct)
        return
    if (len(pred_acts) == 0):
        pred_acts.loc[0] = {
            'StartTime': real_acts.StartTime.iloc[0],
            'EndTime': real_acts.EndTime.iloc[0],
            'Activity': real_acts.Activity.iloc[0]
        }
        print('not enough p data of this type', onlyAct)
        print(pred_acts)
        return

#   ft=real_acts.StartTime.iloc[0]
#   dur=ft+pd.to_timedelta('12h')
#   real_acts=real_acts.loc[real_acts.StartTime<dur]
#   pred_acts=pred_acts.loc[pred_acts.StartTime<dur]
# apply(lambda x:dataset.activities_map[x.Activity], axis=1).tolist()
    ract = (real_acts.Activity + (size / 2)).tolist()
    rstart = real_acts.StartTime.tolist()
    rend = real_acts.EndTime.tolist()
    # .apply(lambda x:dataset.activities_map[x.Activity], axis=1).tolist()
    pact = (pred_acts.Activity - (size / 2)).tolist()
    pstart = pred_acts.StartTime.tolist()
    pend = pred_acts.EndTime.tolist()
    if ax == None:
        if (onlyAct):
            fig, ax = plt.subplots(figsize=(10, 0.5))
        else:
            fig, ax = plt.subplots(figsize=(10, len(acts) / 5))
    ax.set_title(label)
    if (len(real_acts) == 0):
        print('no r activity of this type', label)
    else:
        _plotActs(ax, ract, rstart, rend, linewidth=1, edgecolor='k', facecolor='g', size=size, alpha=.6)
    if (len(pred_acts) == 0):
        print('no p activity of this type', label)
    else:
        _plotActs(ax, pact, pstart, pend, linewidth=1, edgecolor='k', facecolor='r', size=size, alpha=.6)


#   data_linewidth_plot(pact, pstart, pend, ax=ax,
#                       colors="red", alpha=1, linewidth=.3)
# plt.hlines(ract, rstart, rend, colors=(0,.5,0,.2), linewidth=1)
# plt.hlines(pact, pstart, pend, colors="red", lw=2)
#   loc = mdates.AutoDateLocator()
#   ax.xaxis.set_major_locator(loc)
#   ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    ax.set_xticks([])
    ax.set_xlim(min(pstart + rstart), max(pend + rend))
    ax.set_yticks([i for i in dataset.activities_map])
    ax.yaxis.grid(True)

    ax.set_yticklabels([l for l in dataset.activities_map_inverse])
    if (onlyAct):
        ax.set_ylim(onlyAct - size, onlyAct + size)
    else:
        ax.set_ylim(0 + size, len(dataset.activities) - size)
    plt.margins(0.1)
    # plt.show()


def plotJoinTree(real_acts, pred_acts):
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    size = 0.45

    if (len(real_acts) == 0):
        print('real not enough data of this type')
        print(real_acts)
        return
    if (len(pred_acts) == 0):
        # pred_acts=pred_acts.append({'StartTime':real_acts.StartTime.iloc[0],'EndTime':real_acts.EndTime.iloc[0],'Activity':real_acts.Activity.iloc[0]},ignore_index=True)
        print('pred not enough p data of this type')
        print(pred_acts)
        return

    rstart = [r.begin for r in real_acts]
    rend = [r.end for r in real_acts]
    ract = [1 + (size / 2) for r in real_acts]
    # .apply(lambda x:dataset.activities_map[x.Activity], axis=1).tolist()

    pstart = [r.begin for r in pred_acts]
    pend = [r.end for r in pred_acts]
    pact = [1 - (size / 2) for r in pred_acts]
    ax = plt.gca()

    _plotActs(ax, ract, rstart, rend, linewidth=1, edgecolor='k', facecolor='g', size=size, alpha=.6)

    _plotActs(ax, pact, pstart, pend, linewidth=1, edgecolor='k', facecolor='r', size=size, alpha=.6)

    ax.set_xticks([])
    ax.set_xlim(min(pstart + rstart), max(pend + rend))
    ax.set_yticks([i for i in [1]])
    ax.yaxis.grid(True)

    #   ax.set_yticklabels([l for l in labels])
    ax.set_ylim(1 - size, 1 + size)
    plt.margins(0.1)


#   plt.show()


def fold_avg(dataset, evalres):
    sumcm = evalres[0]['test'].quality
    for i in range(1, len(evalres)):
        cm = evalres[i]['test'].quality
        for x in cm:
            sumcm[x] += cm[x]
    # plot_confusion_matrix(sumcm,evalres[0].Sdata.acts,figsize=[25,12])
    for x in sumcm:
        sumcm[x] /= len(evalres)
    return sumcm


def plot_CM(dataset, evalres):
    sumcm = evalres[0]['test'].event_cm
    for i in range(1, len(evalres)):
        cm = evalres[i]['test'].event_cm
        sumcm += cm
    # plot_confusion_matrix(sumcm,evalres[0].Sdata.acts,figsize=[25,12])
    plot_CM_normal(sumcm, dataset.activities)


def tmp(cm, acts):

    plt.imshow(cm, interpolation='nearest')
    plt.xticks(np.arange(0, len(acts)), acts)
    plt.yticks(np.arange(0, len(acts)), acts)

    # plt.show()


def plot_CM_normal(cm, acts, title=None, ax=None):
    import numpy as np
    import matplotlib.pyplot as plt

    conf_arr = cm

    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(0 if a == 0 else float(j) / float(a))
        norm_conf.append(tmp_arr)

    # fig = plt.figure(figsize=(8, 8))
    if ax == None:
        ax = plt.axes()
    ax.set_aspect(1)
    cax = ax.matshow(np.array(norm_conf), cmap=plt.cm.jet, interpolation='nearest')

    width, height = conf_arr.shape

    # for x in range(width):
    # 	for y in range(height):
    # 		ax.annotate(str(conf_arr[x][y]), xy=(y, x),
    # 					horizontalalignment='center',
    # 					verticalalignment='center')
    # cbar_ax = ax.figure.add_axes([0.85, 0.15, 0.05, 0.7])

    cb = ax.figure.colorbar(cax, ax=ax, shrink=0.5)

    ax.set_xlim(-.5, len(acts) - .5)
    ax.set_ylim(-.5, len(acts) - .5)
    ax.set_title(title)
    # alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), acts, rotation=-90)
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)

    ax.set_xticks(range(width))
    ax.set_xticklabels(acts, rotation=-90)
    plt.yticks(range(height), acts)
    # plt.show()


def plot_per_act(dataset, myevalres):
    # myevalres=sorted(myevalres)
    activities = dataset.activities
    summycm = {}
    sumcm = {}

    print(myevalres)
    for p in myevalres:
        evalres = myevalres[p]
        # for p in ['test','train']:

        summycm[p] = np.zeros((len(activities), len(activities)))
        sumcm[p] = np.zeros((len(activities), len(activities)))
        # from result_analyse.metric import calc_cm_per_s_event
        # summycm[p]=calc_cm_per_s_event(dataset,evalres)
        for i in range(len(evalres)):
            eventcm = evalres[i]['test'].event_cm

            summycm[p] += eventcm
            if hasattr(evalres[i]['test'], 'cm'):
                cm = evalres[i]['test'].cm
            else:
                cm = sklearn.metrics.confusion_matrix(evalres[i]['test'].Sdata.label, evalres[i]['test'].predicted_classes, labels=range(len(activities)))

            sumcm[p] += cm

    import matplotlib.pyplot as plt

    def get_new_fig(fn, figsize=[12, 9]):
        """ Init graphics """
        fig1 = plt.figure(fn, figsize)
        ax1 = fig1.gca()  # Get Current Axis
        ax1.cla()  # clear existing plot
        return fig1, ax1

    bar_count = len(myevalres)
    x = np.arange(len(activities))  # the label locations

    fig, ax = get_new_fig(',', [20, 5])  # plt.subplots()
    rects = []
    width = .7 / (bar_count * 2)
    colors = ['coral', 'cornflowerblue', 'darkgray', 'darkgoldenrod']

    def mylabel(rects, cm):
        for idx, rect in enumerate(rects):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, 'c=%d' % cm[idx], ha='center', va='bottom', rotation=90)

    for i, p in enumerate(sorted(myevalres)):
        e2 = (100 * CMbasedMetric(sumcm[p], average=None)['f1']).round()
        x1 = x - 1.2 * width * (len(myevalres) - i - .5)
        # print(x1 + width/2.,1.05*e2,)
        # +(" total=%e"%np.sum(sumcm[p])
        rect = ax.bar(x1, e2, width, label=p, color=colors[i])
        rects.append(rect)
        # rects.append(ax.text(x1 + width/2., 1.05*e2,np.sum(sumcm[p],0),ha='center', va='bottom', rotation=0))
        mylabel(rect, np.sum(sumcm[p], 1))
        e3 = (100 * CMbasedMetric(summycm[p], average=None)['f1']).round()
        rects.append(ax.bar(x, 30, .1 / (bar_count * 2), label=None))
        x2 = x + (1.2 * width * (i + .5))
        # +(" total=%d"%np.sum(summycm[p])
        rect2 = ax.bar(x2, e3, width, label='TS-' + p, color=colors[i], hatch='//')
        # print('x:%f y:%f'%(x2 + width/2.,1.05*e3))
        # rects.append(ax.text(x2 + width/2., 1.05*e3,np.sum(summycm[p],0),ha='center', va='bottom', rotation=0))
        mylabel(rect2, np.sum(summycm[p], 1))
        rects.append(rect2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    name = 'F1 score'
    ax.set_ylabel(name)
    ax.set_title(dataset.data_dscr)
    ax.set_xticks(x)
    acts = dataset.activities_map_inverse
    a = {x.replace('Not_in_Bed', 'Not in Bed').replace('in_Bed', 'in Bed').replace('_', '\n'): acts[x] for x in acts}
    ax.set_xticklabels(a, rotation=-60)

    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                '{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center',
                va='bottom')

    # for rect in rects:
    # autolabel(rect)

    fig.tight_layout()
    fig.patch.set_facecolor('white')
    plt.show()


def plot_per_act_cm(dataset, cm, event_cm):
    # myevalres=sorted(myevalres)
    import unified_ar.general.utils as utils
    data = utils.Data('tmp')
    data.event_cm = event_cm
    data.cm = cm
    myevalres = {'method':
                 {0: {'test': data}}
                 }
    plot_per_act(dataset, myevalres)


def convert2event(real):
    newreal = []
    mind = datetime(2011, 1, 1)
    for item in real:
        s = mind + pd.to_timedelta(str(item[0]) + 'm')
        e = mind + pd.to_timedelta(str(item[1]) + 'm')
        newreal.append({"StartTime": s, "EndTime": e, "Duration": e - s, "Activity": 1})

    return pd.DataFrame(newreal)


if __name__ == '__main__':
    import unified_ar.result_analyse.resultloader
    import unified_ar.general.utils as utils
    # run_info,dataset,evalres=utils.loadState(result_analyse.resultloader.get_runs()[0][0])
    run_info, dataset, evalres = utils.loadState('ward-a')

    real_events = evalres[0].real_events
    pred_events = evalres[0].pred_events
    # my_result_analyse(dataset,evalres[0].real_events,evalres[0].pred_events,duration=(pd.to_datetime('17-2-2020'),pd.to_datetime('17-2-2021')))
    onlyact = 1

    remove_gaps(real_events, pred_events, onlyact)
    plotJoinAct(dataset, real_events, pred_events, onlyAct=onlyact)
    plotMyMetric(dataset, real_events, pred_events, onlyAct=onlyact)
    plotWardMetric(dataset, real_events, pred_events, onlyAct=onlyact)
