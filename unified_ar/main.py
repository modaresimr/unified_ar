import argparse
from datetime import datetime
import logging

import general.utils as utils

import auto_profiler
import importlib
import constants

import general.utils


@auto_profiler.Profiler(depth=8, on_disable=general.utils.logProfile)
def run(args):
    importlib.reload(constants)
    from constants import methods

    logger = logging.getLogger(__file__)
    logger.debug(f'args={args}')

    if (args.dataset < 0):
        logger.error('Invalid dataset argument')
        return

    if (args.mlstrategy < 0):
        logger.error('Invalid mlstrategy argument')
        return

    if (args.evaluation < 0):
        logger.error('Invalid evaluation argument')
        return

    # logger.info(f'dataset={datasetdscr}')
    strategy = methods.mlstrategy[args.mlstrategy]['method']()
    evaluation = methods.evaluation[args.evaluation]['method']()

    if (args.feature_extraction >= 0):
        methods.feature_extraction = [methods.feature_extraction[args.feature_extraction]]
    if (args.segmentation >= 0):
        methods.segmentation = [methods.segmentation[args.segmentation]]
    if args.seg_params:
        for allkp in args.seg_params:
            for kp in allkp.split(','):
                k, p = kp.split('=')
                for entry in methods.segmentation[0]['params']:
                    if k in entry:
                        entry[k] = float(p)
                    if 'var' in entry and entry['var'] == k:
                        entry['init'] = float(p)
    print(args.seg_params)
    # print('segme', methods.segmentation)
    # return
    if (args.classifier >= 0):
        methods.classifier = [methods.classifier[args.classifier]]

    datasetdscr = methods.dataset[args.dataset]['method']().load()
    run_date = datetime.now().strftime('%y%m%d_%H-%M-%S')
    filename = f'{run_date}-{datasetdscr.shortname()}-s={args.segmentation}'
    methods.run_names = {'out': filename}

    evalres = evaluation.evaluate(datasetdscr, strategy)

    run_info = {
        'dataset': datasetdscr.shortname(),
        'run_date': run_date,
        'dataset_path': datasetdscr.data_path,
        'strategy': strategy.shortname(),
        'evalution': evaluation.shortname()
    }
    compressdata = {'run_info': run_info, 'folds': {k: {'quality': evalres[k]['test'].quality, 'runname': evalres[k]['test'].shortrunname} for k in evalres}}

    utils.saveState([compressdata], filename, 'info')
    utils.saveState([run_info, datasetdscr, evalres], filename)
    # utils.convert2SED(filename)
    for i in range(len(evalres)):
        logger.debug(f'Evalution quality fold={i} is {evalres[i]["test"].quality}')

    logger.debug(f'run finished args={args}')


def Main(argv):
    strargs = str(argv)

    auto_profiler.Profiler.GlobalDisable = True
    parser = argparse.ArgumentParser(description='Run on datasets.')
    parser.add_argument('--dataset', '-d', help=' to original datasets', type=int, default=0)
    parser.add_argument('--output', '-o', help='Output folder', default='logs')
    parser.add_argument('--mlstrategy', '-st', help='Strategy', type=int, default=0)
    parser.add_argument('--segmentation', '-s', help='segmentation', type=int, default=-1)
    parser.add_argument('--seg_params', '-sp', action='append', help='segmentation parameter')

    parser.add_argument('--feature_extraction', '-f', type=int, default=0)
    parser.add_argument('--classifier', type=int, default=0)
    parser.add_argument('--evaluation', help='evaluation', type=int, default=0)
    parser.add_argument('--comment', '-c', help='comment', default='')
    #parser.add_argument('--h5py', help='HDF5 dataset folder')
    args = parser.parse_args(argv)

    utils.configurelogger(__file__, args.output, strargs)
    import numpy
    import os
    os.system("taskset -a -pc 0-1000 %d >>/dev/null" % os.getpid())
    run(args)
    # print(args.seg_param)


if __name__ == '__main__':
    import sys
    strargs = str(sys.argv[1:])
    Main(sys.argv[1:])
# https://stackoverflow.com/questions/39063676/how-to-boost-a-keras-based-neural-network-using-adaboost
