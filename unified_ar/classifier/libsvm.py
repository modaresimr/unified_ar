from .classifier_abstract import *
from libsvm.svmutil import *
import numpy as np
import logging
logger = logging.getLogger(__file__)


class LibSVM(Classifier):

    def createmodel(self, inputsize, outputsize, update_model=False):
        if update_model:
            try:
                return self.model
            except:
                pass
        self.outputsize = outputsize
        pass

    def _train(self, x, y):
        prob = svm_problem(y, x)
        param = svm_parameter('-t 2')
        self.model = svm_train(prob, param)

    def _predict(self, testset):
        res = np.zeros((len(testset), self.outputsize))
        cls = self.predict_classes(testset)
        for i in range(0, len(testset)):
            res[i] = cls[i]
        return res

    def _predict_classes(self, testset):
        tmp = np.zeros(len(testset))
        p_labs, p_acc, p_vals = svm_predict(tmp, testset, self.model)
        return p_labs

    def save(self, file):
        logger.debug('saving %s', file)
        svm_save_model(file+'.libsvm', self.model)

    def loadmodel(self, file):
        logger.debug('loading %s', file)
        self.model = svm_load_model(file+'.libsvm')
