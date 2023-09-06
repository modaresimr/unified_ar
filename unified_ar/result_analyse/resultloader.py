import unified_ar.combiner.SimpleCombiner
from unified_ar.metric.CMbasedMetric import CMbasedMetric
from tqdm.auto import tqdm
import sklearn.metrics
import pandas as pd
import os
import unified_ar.general.utils as utils
from ipywidgets import interact, interactive, fixed, interact_manual
import logging
logger = logging.getLogger(__file__)


def getRunTable(run_info, dataset, evalres):
    allitems = []
    # combin=combiner.SimpleCombiner.EmptyCombiner()
    for i in range(len(evalres)):
        for mode in ['train', 'test']:
            if not mode in evalres[i]:
                continue
            evaldata = evalres[i][mode]
            # Sdata=evaldata.Sdata
        # pred_events=combin.combine(Sdata.s_event_list,Sdata.set_window,evaldata.predicted)

            quality = evaldata.quality
            # print('Evalution quality fold=%d is %s' % (i, quality))
            item = {
                'split': mode,
                'runname': evaldata.shortrunname,
                'fold': i,
                'accuracy': quality['accuracy'],
                'precision': quality['precision'],
                'recall': quality['recall'],
                'f1': quality['f1'],
            }
            for f in run_info:
                item[f] = run_info[f]
            if hasattr(evaldata, 'functions'):
                for f in evaldata.functions:
                    item[f] = evaldata.functions[f][0]
                    item[f+'_params'] = evaldata.functions[f][1]

            item['type'] = 'event'
            allitems.append(item)
            item = item.copy()
            if hasattr(evaldata.Sdata, 'label'):
                cm = sklearn.metrics.confusion_matrix(evaldata.Sdata.label, evaldata.predicted_classes, labels=range(len(dataset.activities)))
                quality = CMbasedMetric(cm, 'macro')
                # print('classical quality:%s'%(quality))
                item['accuracy'] = quality['accuracy']
                item['precision'] = quality['precision']
                item['recall'] = quality['recall']
                item['f1'] = quality['f1']

            else:
                import numpy as np
                item['accuracy'] = np.nan
                item['precision'] = np.nan
                item['recall'] = np.nan
                item['f1'] = np.nan
            item['type'] = 'classic'
            allitems.append(item)
    # df = pd.DataFrame(allitems, columns=['fold', 'type', 'accuracy', 'precision', 'recall', 'f1', 'runname'])
    df = pd.DataFrame(allitems)
    return df


def load_run_table(file):
    try:
        runtable = utils.loadState(file, 'runtable1', raiseException=False)
        if (runtable is None):
            res = utils.loadState(file)
            if (len(res) != 3):
                # raise Error
                logger.warn('File %s can not import!' % file)
                return
            [run_info, datasetdscr, evalres] = res
            runtable = getRunTable(run_info, datasetdscr, evalres)
            utils.saveState(runtable, file, 'runtable1')
        return runtable
    except Exception as e:
        logger.warn(f'File {file} can not import! {e}')
        return


def getRunInfo(run_info, datasetdscr, evalres):
    # run_info={'dataset':datasetdscr.shortname(),'run_date':run_date,'dataset_path':datasetdscr.data_path, 'strategy':strategy.shortname(),'evalution':evaluation.shortname()}
    compressdata = {'run_info': run_info, 'folds': {k: {'quality': evalres[k]['test'].quality, 'runname': evalres[k]['test'].shortrunname} for k in evalres}}
    # compressdata={'run_info':run_info, 'folds':{0:{'quality':{'f1':0},'runname':run_info}}}
    return compressdata


def load_run_info(file):
    runinfo = utils.loadState(file, 'info', raiseException=False)
    if (runinfo is None):
        try:
            res = utils.loadState(file)
            if (len(res) != 3):
                # raise Error
                logger.warn('File %s can not import!' % file)
                return
            [run_info, datasetdscr, evalres] = res
            runinfo = getRunInfo(run_info, datasetdscr, evalres)
            utils.saveState([runinfo], file, 'info')
            return runinfo
        except:
            logger.warn('File %s can not import' % file)
            raise
            return
    return runinfo[0]


def single_runtable_loader(item):
    import os
    # if not ('A4H' in item):continue
    if not os.path.exists(f'save_data/{item}/data.pkl.lz4'):
        return
    try:
        return load_run_table(item)

    except Exception as e:
        print('File %s can not import error ' % item)
        import sys
        import traceback
        print(e, file=sys.stderr)
        traceback.print_exc()
        return


def get_all_runs_table():
    listdir = os.listdir('save_data/')
    listdir.sort(key=lambda f: os.path.getmtime('save_data/'+f), reverse=True)
    result = []

    import unified_ar.general.utils as utils
    result = [r for r in utils.parallelRunner(True, single_runtable_loader, listdir) if not (r is None)]
    return pd.concat(result)


def get_runs():
    list = os.listdir('save_data/')
    list.sort(key=lambda f: os.path.getmtime('save_data/'+f), reverse=True)
    result = []
    for item in list:

        try:
            # res=utils.loadState(item)
            # if(len(res)!=3):
            #     raise Error
            # [run_info,datasetdscr,evalres]=res
            # disp_name='dataset:%s date:%s %s'%(run_info['dataset'],run_info['run_date'], evalres[0].shortrunname)
            disp_name = item
            result.append((disp_name, item))
        except:
            logger.warn('File %s can not import' % item)

    return result

# +


def get_runs_summary(dataset=''):
    list = os.listdir('save_data/')
    list.sort(key=lambda f: os.path.getmtime('save_data/'+f), reverse=True)
    result = []

    def loader(file):
        if dataset not in file:
            return
        if not os.path.exists(f'save_data/{file}/data.pkl.lz4'):
            return
        res = load_run_info(file)
        if res is None:
            return

        f1 = res['folds'][0]['quality']['f1']
        disp_name = file+":f1="+str(f1)+"=="+res['folds'][0]['runname']+"===="+str(res['folds'])
        return (disp_name, file)

    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
#     results = Parallel(n_jobs=num_cores)(delayed(loader)(file) for file in list)
    results = [loader(file) for file in list]
    for item in results:
        if (item is None):
            continue
        result.append(item)
    # result.sort(key=lambda x: (x[0].split('_')[2],x[0].split('=')[1]))
    # result.sort(key=lambda x: (x[0].split('_')[0],x[0].split('=')[1]),reverse=True)
    result.sort(key=lambda x: (x[0]), reverse=True)
    return result


def get_runs_summary3(dataset=''):
    list = os.listdir('save_data/')
    list.sort(key=lambda f: os.path.getmtime('save_data/'+f), reverse=True)
    result = []

    def loader(file):
        if dataset not in file:
            return

        try:
            res = utils.load_run_info(file)
            f1 = res['folds'][0]['test']['quality']['f1']
            disp_name = file+":f1="+str(f1)+"=="+res['folds'][0]['test']['shortrunname']+"===="+str(res['folds'])
            return (disp_name, file)
        except:
            try:
                res = utils.loadState(file)
                if (len(res) != 3):
                    raise Error
                [run_info, datasetdscr, evalres] = res
                disp_name = 'dataset:%s date:%s %s' % (run_info['dataset'], run_info['run_date'], evalres[0]['test'].shortrunname)
                return (disp_name, file)
            except Exception as e:
                logger.warn('File %s can not import' % file)
                import sys
                import traceback
                print(e, file=sys.stderr)
                traceback.print_exc()

#            raise

    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
#     results = Parallel(n_jobs=num_cores)(delayed(loader)(file) for file in list)
    results = [loader(file) for file in list]
    for item in results:
        if (item is None):
            continue
        result.append(item)

    return result


def get_runs_summary2(dataset=''):
    list = os.listdir('save_data/')
    list.sort(key=lambda f: os.path.getmtime('save_data/'+f), reverse=True)
    result = []
    for item in list:
        if dataset not in item:
            continue
        if not os.path.exist(f'save_data/{item}/data.pkl.lz4'):
            continue
        try:
            res = utils.loadState(item, 'info')
            # if(len(res)!=3):
            #     raise Error
            # [run_info,datasetdscr,evalres]=res
            # disp_name='dataset:%s date:%s %s'%(run_info['dataset'],run_info['run_date'], evalres[0].shortrunname)
            f1 = res['folds'][0]['quality']['f1']
            disp_name = item+":f1="+str(f1)+"=="+res['folds'][0]['shortrunname']+"===="+str(res['folds'])
            result.append((disp_name, item))
        except:
            logger.warn('File %s can not import' % item)
#            raise
    return result


# -

def display_result(file):
    [run_info, datasetdscr, evalres] = utils.loadState(file)
    for i in range(len(evalres)):
        evaldata = evalres[i]['test']
        quality = evaldata.quality
        logger.debug('Evalution quality fold=%d is f1=%.2f acc=%.2f precision=%.2f recall=%.2f' %
                     (i, quality.f1, quality.accuracy, quality.precision, quality.recall))


if __name__ == '__main__':
    get_runs_summary('ward')
