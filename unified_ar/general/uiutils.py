from IPython.display import display
import logging
methods = None
logging.getLogger('matplotlib.font_manager').disabled = True


def runPipelineUI():
    global methods
    from ipywidgets import interact, interactive, fixed, interact_manual
    from unified_ar import constants
    from copy import deepcopy
    if (methods == None):
        methods = deepcopy(constants.methods)

    def res(**args):
        largs = []
        for a in args:
            largs.append('--' + a)
            largs.append(str(args[a]))
        print(largs)
        from unified_ar import main
        main.Main(largs)

    params = {}
    for par in methods.__dict__:
        args = getattr(methods, par)

        if (len(args) <= 1):
            continue
        if (par == 'name'):
            continue
        if par == 'meta_segmentation_sub_tasks':
            continue
        params[par] = {v['method']().shortname() + ' ' + (str(v['params']) if 'params' in v and len(v['params']) > 0 else ''): k for k, v in enumerate(args)}
        if not (par == 'mlstrategy' or par == 'evaluation' or par == 'dataset'):
            params[par]['All -> Find Best'] = -1

    w = interact_manual(res, **params)


def loadGemMetricUI():
    import unified_ar.general.utils as utils
    import unified_ar.result_analyse.resultloader
    import unified_ar.result_analyse.visualisation as vs

    from ipywidgets import interact, interactive, fixed, interact_manual, widgets
    import numpy as np
    import pandas as pd

    @interact
    def result_selector(file=unified_ar.result_analyse.resultloader.get_runs()):
        if (file == None):
            return
        print('Analysing ', file)
        run_info, dataset, evalres = utils.loadState(file)
        stime = dataset.activity_events.iloc[0].StartTime
        # etime=stime+np.timedelta64(1,'D')
        etime = dataset.activity_events.iloc[-1].EndTime

        for i in range(len(evalres)):
            quality = evalres[i]['test'].quality
            print('Evalution quality fold=%d is %s' % (i, quality))
        print(len(dataset.sensor_events))

    #     vs.plot_CM(dataset,evalres)

        @interact
        def viewFold(fold=range(len(evalres))):
            @interact_manual
            def view(start_date=widgets.DatePicker(value=pd.to_datetime(stime)), end_date=widgets.DatePicker(value=pd.to_datetime(etime)), debug=widgets.Checkbox(value=False)):
                duration = (pd.to_datetime(start_date), pd.to_datetime(end_date))
                duration2 = (pd.to_datetime(start_date), pd.to_datetime(start_date) + pd.DateOffset(days=7))
                real_events = vs.filterTime(evalres[fold]['test'].real_events, duration)
                pred_events = vs.filterTime(evalres[fold]['test'].pred_events, duration)
                # vs.plotJoinAct(dataset,real_events,pred_events)
                acts = [p for p in dataset.activities_map]
                labels = [dataset.activities_map[p] for p in acts]
                print(acts)
                print(labels)
                vs.plotJoinAct2(real_events, pred_events, acts, labels, duration=duration2)
                # vs.plot_per_act(dataset,{'test':evalres})

                from matplotlib import pyplot as plt
                plt.rc_context(rc={'figure.max_open_warning': 0})
                import unified_ar.result_analyse.SpiderChart
                unified_ar.result_analyse.SpiderChart.radar_factory(5, frame='polygon')
                acount = len(dataset.activities_map)
                a_fig, a_ax = plt.subplots(acount - 1, 1, figsize=(10, acount * .25),)
    #             a_fig.tight_layout(pad=3.0)
                col = 4
                row = int(np.ceil((acount - 1.0) / float(col)))
                m_fig, m_ax = plt.subplots(row, col, figsize=(col * 3, row * 3), subplot_kw=dict(projection='radar'))
                if type(a_ax) != np.ndarray:
                    print('dddd', a_ax)
                    print(type(a_ax))
                    a_ax = np.array([a_ax])
                else:
                    m_ax = m_ax.flatten()
                for i in range(acount - 1, len(m_ax)):
                    m_ax[i].set_visible(False)

                for i in range(1, len(dataset.activities_map)):
                    #                 real_events2,pred_events2=vs.remove_gaps(real_events,pred_events,i)
                    # real_events2,pred_events2=vs.remove_gaps(real_events,pred_events,i,max_events=10)
                    real_events2, pred_events2 = real_events, pred_events
                    vs.plotJoinAct(dataset, real_events, pred_events, onlyAct=i, ax=a_ax[i - 1])
                    try:
                        #                     vs.plotJoinAct(dataset,real_events2,pred_events2,onlyAct=i,ax=a_ax[i-1])

                        vs.plotMyMetric2(dataset, real_events2, pred_events2, onlyAct=i, ax=m_ax[i - 1], debug=debug, calcne=0)

                    except Exception as e:
                        import sys
                        import traceback
                        print(e, file=sys.stderr)
                        traceback.print_exc()

                    #    print('error in ',i)
                    # vs.plotWardMetric(dataset,real_events,pred_events,onlyAct=i)
    #             vs.plotJoinMyMetric(dataset,real_events2,pred_events2,calcne=0)
                # a_fig.show()
                m_fig.tight_layout(pad=0, h_pad=-20.0, w_pad=3.0)
                # m_fig.show()

    #             @interact
    #             def view2(onlyAct=[(str(i)+" : "+dataset.activities[i],i)for i in dataset.activities_map]):
    #                 print(onlyAct)
    #                 #vs.visualize(dataset)
    #                 # vs.my_result_analyse(evalres[i].real_events,evalres[i].pred_events)
    #                 #vs.plotJoinAct(dataset,real_events,pred_events,onlyAct=onlyAct)
    #                 real_events2,pred_events2=vs.remove_gaps(real_events,pred_events,onlyAct)
    #                 vs.plotJoinAct(dataset,real_events2,pred_events2,onlyAct=onlyAct)
    #                 vs.plotMyMetric(dataset,real_events,pred_events,onlyAct=onlyAct)
    #                 vs.plotWardMetric(dataset,real_events,pred_events,onlyAct=onlyAct)


def loadWardMetricUI():
    import unified_ar.general.utils as utils
    import unified_ar.result_analyse.resultloader
    import unified_ar.result_analyse.visualisation as vs

    from ipywidgets import interact, interactive, fixed, interact_manual, widgets
    import numpy as np
    import pandas as pd

    @interact
    def result_selector(file=result_analyse.resultloader.get_runs()):
        if (file == None):
            return
        print('Analysing ', file)
        run_info, dataset, evalres = utils.loadState(file)
        stime = dataset.activity_events.iloc[0].StartTime
        # etime=stime+np.timedelta64(1,'D')
        etime = dataset.activity_events.iloc[-1].EndTime

        for i in range(len(evalres)):
            quality = evalres[i]['test'].quality
            print('Evalution quality fold=%d is %s' % (i, quality))
        print(len(dataset.sensor_events))

    #     vs.plot_CM(dataset,evalres)

        @interact
        def viewFold(fold=range(len(evalres))):
            @interact_manual
            def view(start_date=widgets.DatePicker(value=pd.to_datetime(stime)), end_date=widgets.DatePicker(value=pd.to_datetime(etime)), debug=widgets.Checkbox(value=False)):
                duration = (pd.to_datetime(start_date), pd.to_datetime(end_date))
                duration2 = (pd.to_datetime(start_date), pd.to_datetime(start_date) + pd.DateOffset(days=7))
                real_events = vs.filterTime(evalres[fold]['test'].real_events, duration)
                pred_events = vs.filterTime(evalres[fold]['test'].pred_events, duration)
                # vs.plotJoinAct(dataset,real_events,pred_events)
                acts = [p for p in dataset.activities_map]
                labels = [dataset.activities_map[p] for p in acts]
                print(acts)
                print(labels)
                vs.plotJoinAct2(real_events, pred_events, acts, labels, duration=duration2)
                # vs.plot_per_act(dataset,{'test':evalres})

                from matplotlib import pyplot as plt
                plt.rc_context(rc={'figure.max_open_warning': 0})
                acount = len(dataset.activities_map)

                for i in range(1, len(dataset.activities_map)):
                    #                 real_events2,pred_events2=vs.remove_gaps(real_events,pred_events,i)
                    # real_events2,pred_events2=vs.remove_gaps(real_events,pred_events,i,max_events=10)
                    real_events2, pred_events2 = real_events, pred_events
                    # vs.plotJoinAct(dataset,real_events,pred_events,onlyAct=i,ax=a_ax[i-1])
                    try:
                        #                     vs.plotJoinAct(dataset,real_events2,pred_events2,onlyAct=i,ax=a_ax[i-1])
                        vs.plotWardMetric(dataset, real_events, pred_events, onlyAct=i)

                    except Exception as e:
                        import sys
                        import traceback
                        print(e, file=sys.stderr)
                        traceback.print_exc()

                    #    print('error in ',i)
                    # vs.plotWardMetric(dataset,real_events,pred_events,onlyAct=i)
    #             vs.plotJoinMyMetric(dataset,real_events2,pred_events2,calcne=0)

                # m_fig.tight_layout(pad=0,h_pad=-20.0, w_pad=3.0)
                # m_fig.show()

    #             @interact
    #             def view2(onlyAct=[(str(i)+" : "+dataset.activities[i],i)for i in dataset.activities_map]):
    #                 print(onlyAct)
    #                 #vs.visualize(dataset)
    #                 # vs.my_result_analyse(evalres[i].real_events,evalres[i].pred_events)
    #                 #vs.plotJoinAct(dataset,real_events,pred_events,onlyAct=onlyAct)
    #                 real_events2,pred_events2=vs.remove_gaps(real_events,pred_events,onlyAct)
    #                 vs.plotJoinAct(dataset,real_events2,pred_events2,onlyAct=onlyAct)
    #                 vs.plotMyMetric(dataset,real_events,pred_events,onlyAct=onlyAct)
    #                 vs.plotWardMetric(dataset,real_events,pred_events,onlyAct=onlyAct)


def loadGemMultiUI():
    from ipywidgets import interact, interactive, fixed, interact_manual, widgets
    import unified_ar.result_analyse.resultloader
    import unified_ar.result_analyse.kfold_analyse as an
    import unified_ar.metric.MyMetric
    import unified_ar.metric.Metrics
    # metrics = {'GEM': lambda: metric.Metrics.GEM(),
    #            'GEM_NEW':lambda: metric.Metrics.GEM_NEW(),
    #            'EventCM': lambda:metric.Metrics.EventCM(),
    #            'Tatbul': lambda:metric.Metrics.Tatbul(),
    #            'Classical': lambda:metric.Metrics.Classical(),
    #            }
    # metrics = metrics.keys()
    metrics = {'GEM', 'GEM_NEW', 'EventCM', 'Tatbul', 'Classical'}
    # import unified_ar.metric.MyMetric as mymetric
    import unified_ar.metric.TatbulMetric as mymetric
    import unified_ar.general.utils as utils
    import unified_ar.result_analyse.visualisation as vs
    from ipywidgets import Button, Layout

    @interact
    def datasets(dataset=['Aruba', 'Home1', 'Home2', 'A4H', 'ward', 'VanKasteren']):
        @interact_manual
        def compare(files=widgets.SelectMultiple(options=unified_ar.result_analyse.resultloader.get_runs_summary(dataset), description='Files', layout=Layout(width='100%', height='180px')), metric=metrics, titles="title1,title2"):
            if len(files) == 0:
                return print('no file is selected')
            utils.reload()
            from IPython.display import display
            import pandas as pd
            import unified_ar.result_analyse.resultloader
            run_info = {}
            dataset = {}
            evalres = {}
            res = {}
            titles = titles.split(',')
            if (len(titles) != len(files)):
                print('Titles are not correct. use files names instead')
                titles = files
            print(files)
            runtables = []
            for i, file in enumerate(files):
                print(i, file)
                # display(result_analyse.resultloader.load_run_table(file))

                t = titles[i]
                run_info[t], dataset[t], evalres[t] = utils.loadState(file)
                if hasattr(evalres[t][0]['test'], 'functions'):
                    display(evalres[t][0]['test'].functions)
    #             print(evalres[t])
#                 for i in evalres[t]:
#                     evalres[t][i]['test'].Sdata=None

                dataset[t].sensor_events = None

                res[t] = an.mergeEvals(dataset[t], evalres[t], metric)

            res = {t: res[t] for t in sorted(res.keys())}


#
#
#             # display(res)
#             acts = dataset[t].activities_map.copy()
#             del acts[0]
#             acts['avg'] = 'avg'
#             actres = {}
#             for k in acts:

#                 # if(k==0):continue
#                 if (k == 0):
#                     continue
#                 key = acts[k]
#                 if key == 'avg':
#                     actres[key] = {(m, e): res[m]['avg'][e] for m in res for e in res[m]['avg']}
#                 else:
#                     actres[key] = {(m, e): res[m][k]['avg'][e] for m in res for e in res[m][k]['avg']}
#                 print(f'act={key}==============================')
# #                 print(actres[k])
#                 if (len(actres[key]) == 0):
#                     print('No Eval')
#                 else:
#                     df2 = pd.DataFrame(actres[key]).round(2)
#                     display(df2)
            vs.plotJoinMetric(res, [k for k in res[t]], dataset[t].activities_map)
            # # vs.plotJoinMetric(res, acts.values())
            # display(res)
            # # vs.plotJoinMetric(res, [k for k in res[t]])
