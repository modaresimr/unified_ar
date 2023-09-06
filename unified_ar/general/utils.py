import multiprocessing
from tqdm.notebook import tqdm

import pandas as pd
import auto_profiler
import os
from os.path import exists
import logging
from intervaltree import intervaltree
from .. import Data
logger = logging.getLogger(__file__)

# Define a Data Object


# Arg Max in a Dic
def argmaxdic(dic):
    mx = {'v': 0, 'i': 0}
    for d in dic:
        tmp = dic[d]
        if (mx['v'] < tmp):
            mx['v'] = tmp
            mx['i'] = d
    return mx['i']


# Defining Interval Tree from Activities.


def makeIntervalTree(acts):
    tree = IntervalTree()

    for act in acts:
        start = act['StartTime'].value
        end = act['EndTime'].value
        if (start == end):
            start = start - 1
        tree[start:end] = act
    return tree


def makeNonOverlapIntervalTree(acts):
    tree = makeIntervalTree(acts)
    tree.split_overlaps()
    tree.merge_equals(data_reducer=lambda x, y: y)
    return tree


# Find overlap between 2 event in Minutes


def findOverlap(a, b):
    return ((min(a['EndTime'], b['EndTime']) - max(a['StartTime'], b['StartTime'])).value) / pd.to_timedelta('60s').value


# Buffer data type for stacking stream
class Buffer:

    def __init__(self, input, minsize, maxsize):
        self.data = input
        self.times = input.time.values
        self.datavalues = input.values
        self.minsize = minsize
        self.maxsize = maxsize
        self.start_index = 0

    def removeTop(self, idx):
        self.start_index = idx

    def getEventsInRange(self, starttime, endtime):
        sindex = self.searchTime(starttime, -1)
        eindex = self.searchTime(endtime, +1)
        if (sindex is None):
            return None
        if (eindex is None):
            return None
        return self.data.iloc[sindex:eindex + 1]

    def searchTime(self, time, operator=0):
        times = self.times
        n = len(times)
        L = self.start_index
        R = n

        if operator == 1:
            while L < R:
                m = int((L + R) / 2)

                if times[m] <= time:
                    L = m + 1
                else:
                    R = m
            return L - 1 if L > self.start_index else None
        else:
            while L < R:
                m = int((L + R) / 2)

                if times[m] < time:
                    L = m + 1
                else:
                    R = m
            return L if L < n else None


def instantiate(method):
    m = method['method']()
    m.applyParams(method['params'])


def saveState(vars, file, name='data'):
    import compress_pickle

    if not (os.path.exists(f'save_data/{file}/')):
        os.makedirs(f'save_data/{file}/')
    pklfile = f'save_data/{file}/{name}.pkl'
    # with open(file+name+'.pkl', 'wb') as f:
    # pickle.dump(vars, f)
    compress_pickle.dump(vars, pklfile + '.lz4')


def loadState(file, name='data', raiseException=True):
    import compress_pickle
    pklfile = f'save_data/{file}/{name}.pkl'
    try:
        if (os.path.exists(pklfile)):
            # with open(pklfile, 'rb') as f:
            res = compress_pickle.load(pklfile)
            # f.close()
            saveState(res, file, name)
            os.remove(pklfile)
            return res
        # if(name=='data'):
        # from unified_ar.metric.CMbasedMetric import CMbasedMetric
        # from unified_ar.metric.event_confusion_matrix import event_confusion_matrix
        #     [run_info,datasetdscr,evalres]=compress_pickle.load(pklfile+'.lz4')
        #     for i in evalres:
        #         data=evalres[i]['test']
        #         Sdata=data.Sdata
        #         import unified_ar.combiner.SimpleCombiner
        #         com=combiner.SimpleCombiner.EmptyCombiner2()
        #         evalres[i]['test'].Sdata.pred_events =com.combine(Sdata.s_event_list,Sdata.set_window,data.predicted)
        #         evalres[i]['test'].event_cm     =event_confusion_matrix(Sdata.a_events,Sdata.pred_events,datasetdscr.activities)
        #         evalres[i]['test'].quality      =CMbasedMetric(data.event_cm,'macro',None)
        #     return [run_info,datasetdscr,evalres]
        return compress_pickle.load(pklfile + '.lz4')
    except:
        if (raiseException):
            raise
        return None


def saveFunctions(func, file):
    file = 'save_data/' + file + '/'
    if not (os.path.exists(file)):
        os.makedirs(file)
    for k in func.__dict__:
        obj = func.__dict__[k]

        if isinstance(obj, MyTask):
            tmpfunc = obj.func
            obj.func = ''
            obj.save(file + '_' + k + '_' + type(obj).__module__ + '_')
            obj.func = tmpfunc


def loadall(file):
    import pickle
    data = loadState(file)
    file = 'save_data/' + file + '/'
    func = Data('Saved Functions')
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(file) if isfile(join(file, f))]
    for f in onlyfiles:
        x = f.split('_')
        if ('data.pkl' in f):
            continue

        if ('.pkl' in f):
            with open(file + f, 'rb') as fl:
                func.__dict__[x[1]] = pickle.load(fl)
        elif ('pyact.h5' in f):
            from classifier.PyActLearnClassifier import PAL_NN
            classifier = PAL_NN()
            classifier.load(file + f)
            func.__dict__[x[1]] = classifier
        elif ('.h5' in f):
            from classifier.KerasClassifier import KerasClassifier
            classifier = KerasClassifier()
            classifier.load(file + f)
            func.__dict__[x[1]] = classifier
        else:
            logger.error('unsupported' + f)

    return [data, func]


def configurelogger(file, dir, logparam=''):
    from datetime import datetime
    # Default parameters
    log_filename = os.path.basename(file).split('.')[0] + \
        '-%s-%s.log' % (datetime.now().strftime('%H-%M-%S'), logparam)
    # Setup output directory
    output_dir = (dir if dir else 'logs')+datetime.now().strftime('/%Y-%m-%d/')

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    if os.path.exists(output_dir):
        # Found output_dir, check if it is a directory
        if not os.path.isdir(output_dir):
            exit('Output directory %s is found, but not a directory. Abort.' % output_dir)
    else:
        # Create directory
        os.makedirs(output_dir)

    log_filename = os.path.join(output_dir, log_filename)
    # Setup Logging as early as possible
    import sys
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s] %(filename)-10s %(funcName)-10s %(levelname)-8s %(message)s',
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)],
                        datefmt='%H:%M:%S')
    logging.getLogger().setLevel(logging.DEBUG)


def logProfile(p):
    title = 'Time   [Hits * PerHit] Function name [Called from] [Function Location]\n' +\
        '-----------------------------------------------------------------------\n'
    logger.debug("TimeProfiling\n%s%s" % (title, auto_profiler.Tree(p.root, threshold=1)))


if __name__ == '__main__':
    loadState('200515_10-31-57-Home1')


def convertAsghari():
    import pandas as pd
    pred = pd.read_csv('save_data/asghari/b1/output1.csv', header=0, names=["StartTime", "EndTime", "Activity"])
    st = pd.to_datetime(pred['StartTime'], format='%Y-%m-%d %H:%M:%S')
    et = pd.to_datetime(pred['EndTime'], format='%Y-%m-%d %H:%M:%S')
    pred['StartTime'] = st

    pred['EndTime'] = et
    pred['Activity'] = pred.Activity.apply(lambda x: dataset.activities_map_inverse[x])
    evalres[0].pred_events = pred
    ######
    run_info, dataset, evalres = utils.loadState('200211_12-39-09-Home1')
    ######
    evalres[0].pred_events = pred
    stime = evalres[0].pred_events.iloc[0].StartTime
    etime = evalres[0].pred_events.iloc[-1].EndTime
    rstime = evalres[0].real_events.iloc[0].StartTime
    retime = evalres[0].real_events.iloc[-1].EndTime

    stime, etime, rstime, retime

    #######
    evalres[0].real_events = evalres[0].real_events.loc[evalres[0].real_events.EndTime >= stime].loc[evalres[0].real_events.StartTime <= etime]
    #######
    evalres[0] = evalres[4]
    #######

    evalres[0].Sdata = None
    evalres[0].predicted = None
    evalres[0].shortrunname = "Asghari_b1"
    evalres[0].predicted_classes = None
    evalres[0].event_cm = None
    evalres[0].quality = {'accuracy': 0, 'precision': .45, 'recall': 0.61, 'f1': 0.52}
    evalres[0].pred_events = pred

    #######
    utils.saveState([run_info, dataset, {0: evalres[0]}], 'asghari-Home1')


def convertSED(name, dataset, pe):
    import unified_ar.datatool.seddata
    ######
    from datetime import datetime
    run_date = datetime.now().strftime('%y%m%d_%H-%M-%S')
    run_info = {'dataset': 'SED2020', 'run_date': run_date, 'dataset_path': '', 'strategy': 'EIN', 'evalution': '-'}
    ######
    pred = datatool.seddata.SED(pe, name, dataset)
    pred.load()

    ########
    evalres = {0: {'test': Data('SED')}}

    evalres[0]['test'].real_events = dataset.activity_events
    evalres[0]['test'].Sdata = None
    evalres[0]['test'].predicted = None
    evalres[0]['test'].shortrunname = name
    evalres[0]['test'].predicted_classes = None
    evalres[0]['test'].event_cm = None
    evalres[0]['test'].quality = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    evalres[0]['test'].pred_events = pred.activity_events

    #######
    saveState([run_info, dataset, {0: evalres[0]}], name)
    return pred


def convert2SED(filename):
    import numpy as np
    [run_info, dataset, evalres] = loadState(filename)
    real = []
    pred = []

    for i in evalres:
        pred_events = fastcombine(evalres[i]['test'].pred_events)
        real_events = evalres[i]['test'].real_events
        pred_events['StartTime'] = (pred_events['StartTime'].astype(np.int64) / 1000000000).astype(np.int64)
        pred_events['EndTime'] = (pred_events['EndTime'].astype(np.int64) / 1000000000).astype(np.int64)
        real_events['StartTime'] = (real_events['StartTime'].astype(np.int64) / 1000000000).astype(np.int64)
        real_events['EndTime'] = (real_events['EndTime'].astype(np.int64) / 1000000000).astype(np.int64)

        for k, p in pred_events.iterrows():
            pred.append({"filename": i, "onset": p['StartTime'], "offset": p['EndTime'], "event_label": dataset.activities[p['Activity']]})
        for k, r in real_events.iterrows():
            real.append({"filename": i, "onset": r['StartTime'], "offset": r['EndTime'], "event_label": dataset.activities[r['Activity']]})
    df_real = pd.DataFrame(data=real, columns=["filename", "onset", "offset", "event_label"])
    df_pred = pd.DataFrame(data=pred, columns=["filename", "onset", "offset", "event_label"])
    df_real = df_real.sort_values(by=['onset', 'offset']).drop_duplicates(subset=["filename", "onset", "offset", "event_label"], keep='last')
    df_pred = df_pred.sort_values(by=['onset', 'offset']).drop_duplicates(subset=["filename", "onset", "offset", "event_label"], keep='last')

    #######
    folder = f"{run_info['dataset']}-{run_info['strategy']}-{run_info['evalution']}"
    dir = f'/workspace/AR-MME-eval/saved/{folder}'
    if not (os.path.exists(dir)):
        os.makedirs(dir)
    gte_file = f'{dir}/real.tsv'
    if (os.path.exists(gte_file)):

        old_gte = pd.read_csv(gte_file, sep='\t', comment='#')
        # from IPython.display import display
        # display(old_gte['event_label'])
        # display(old_gte['event_label'] == df_real['event_label'])
        if (sum(old_gte['event_label'] == df_real['event_label']) != len(df_real['event_label'])):
            print('error---- GTE is different-cancel converting to sed format')
            return

    df_real.to_csv(path_or_buf=f'{dir}/real.tsv', sep='\t', index=False, header=True)
    try:
        pes_file = f'{dir}/{evalres[0]["test"].functions["segmentor"]}-{run_info["run_date"]}.tsv'
    except:
        pes_file = f'{dir}/HHMM-{run_info["run_date"]}.tsv'

    with open(pes_file, 'w') as f:
        f.write(f'# {run_info!r}\n')
        short = {e: evalres[e]['test'].shortrunname for e in evalres}
        # funcs = {e: {
        #     f: evalres[e]['test'].functions[f] for f in evalres[e]['test'].functions if f not in ['classifier_metric', 'event_metric']
        #     } for e in evalres}
        # print(funcs)
        # f.write(f'# {short}\n')

        try:
            for func in evalres[0]['test'].functions:
                if func in ['classifier_metric', 'event_metric']:
                    continue
                f.write(f"# {func}: {evalres[0]['test'].functions[func]}\n")
        except:
            pass

        # f.write(f'# {funcs}\n')
        df_pred.to_csv(f, sep='\t', index=False, header=True)


# convert2SED('210909_12-53-32-Home1')


def fastcombine(predicted):
    from intervaltree.intervaltree import IntervalTree
    events = []
    ptree = {}
    epsilon = pd.to_timedelta('1s')

    for i, p in predicted.iterrows():
        start = p['StartTime']
        end = p['EndTime']
        # pclass = np.argmax(predicted[i])
        pclass = p['Activity']

        if not (pclass in ptree):
            ptree[pclass] = IntervalTree()
        ptree[pclass][start:end + epsilon] = {'Activity': pclass, 'StartTime': start, 'EndTime': end}
        # if(i>0 and pclass>0 and predicted[i-1]==predicted[i] and False):
        #     #fix gap
        #     start   = times[i-1][1]
        #     end     = times[i][0]
        #     if(end>start):
        #     #pclass = np.argmax(predicted[i])
        #         ptree[pclass][start:end] = {
        #             'Activity': pclass, 'StartTime': start, 'EndTime': end
        #         }

    tree = IntervalTree()

    def datamerger(x, y):
        start = min(x['StartTime'], y['StartTime'])
        end = max(x['EndTime'], y['EndTime'])
        return {'Activity': x['Activity'], 'StartTime': start, 'EndTime': end}

    for a in ptree:
        ptree[a].merge_overlaps(data_reducer=datamerger)
        tree |= ptree[a]

    tree.split_overlaps()

    def data_reducer(x, y):
        if (x['EndTime'] > y['EndTime']):
            return y
        return x

    tree.merge_equals(data_reducer=data_reducer)
    for inv in tree:
        if (inv.end - inv.begin > epsilon):
            events.append({'Activity': inv.data['Activity'], 'StartTime': inv.begin, 'EndTime': inv.end})

    events = pd.DataFrame(events)
    events = events.sort_values(['StartTime'])
    events = events.reset_index()
    events = events.drop(['index'], axis=1)
    return events


def reload():
    import importlib
    import sys
    from os.path import dirname, basename, isfile
    import glob
    for module in list(sys.modules.values()):
        if '.conda' in f'{module}':
            continue
        # if '_' in f'{module}':
        #     continue
        if 'unified_ar' not in f'{module}':
            continue

        # print(module)
        importlib.reload(module)


def parallelRunner(parallel, runner, items):
    pbar = tqdm(total=len(items))
    if parallel:
        import os
        cpus = len(os.sched_getaffinity(0))
        pool = multiprocessing.Pool(cpus, maxtasksperchild=4)
        result = pool.imap(runner, items)
        try:
            for _ in items:
                res = result.next()
                pbar.update(1)
                yield res
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            pool.close()
            raise KeyboardInterrupt
    else:
        for item in items:
            res = runner(item)
            pbar.update(1)
            yield res
