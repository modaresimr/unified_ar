from joblib.parallel import Parallel, delayed, parallel_backend

from general.utils import Data, MyTask
from optimizer.OptLearn import OptLearn

import logging
logger = logging.getLogger(__file__)


def method_param_selector(callback,uniquekey):
    import itertools
    from constants import methods
    s = [methods.preprocessing, methods.segmentation,
         methods.activity_fetcher, methods.feature_extraction, methods.classifier]
    permut = list(itertools.product(*s))

    allpool = []
    for item in permut:
        func = Data('Functions')
        func.uniquekey=uniquekey
        func.preprocessor = createFunction(item[0])
        func.segmentor = createFunction(item[1])
        func.activityFetcher = createFunction(item[2])
        func.featureExtractor = createFunction(item[3])
        func.classifier = createFunction(item[4])
        
        func.combiner = createFunction(methods.combiner[0])
        func.classifier_metric = createFunction(methods.classifier_metric[0])
        func.event_metric = createFunction(methods.classifier_metric[0])

        func.shortrunname = ''
        for k in func.__dict__:
            obj = func.__dict__[k]
            if isinstance(obj, MyTask):
                obj.func = func
                func.shortrunname += obj.shortname()+'_'

        optl = OptLearn(func, callback)
        allpool.append(optl)
        # break

    success, fail = run(allpool, True)

    bestJobscore = success[0].result['optq']['q']
    bestJob=success[0]
    for job in success:
        if(bestJobscore > job.result['optq']['q']):
            bestJobscore = job.result['optq']['q']
            bestJob = job

    return bestJob


def runOptLearn(optl, test=0):
    # try:
    optl.run()
    optl.success=True   
    # except Exception as e:
    #     if test:
    #         raise e
    #     import sys
    #     import traceback
    #     print(e, file=sys.stderr)
    #     traceback.print_exc()
    #     optl.success=False
    return optl


def run(allpool, test):
    # return [runOptLearn(allpool[0])]
    if test:
        result = [runOptLearn(p, test) for p in allpool]
    else:
        with parallel_backend('threading'):
            result = (Parallel()(delayed(runOptLearn)(p) for p in allpool))

    success = []
    fail = []
    for i in range(len(result)):
        if(result[i].success):
            success.append(result[i])
        else:
            logger.error("!!!FAILED: %d %s", i, result[i].shortname)
            fail.append(result[i])
    # for pool in allpool:
    #     runOptLearn(pool)
    #     break;
    logger.debug('finish')
    return success, fail

def createFunction(function):
    res=function['method']()
    res.findopt=function['findopt'] if 'findopt' in function else False
    res.defparams=function['params'] if 'params' in function else []
    return res

