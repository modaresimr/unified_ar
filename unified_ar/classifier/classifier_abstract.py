from general.utils import MyTask
import numpy as np

class Classifier(MyTask):
    def createmodel(self, inputsize, outputsize, update_model=False):
        tmpsample=np.array([np.zeros(inputsize)])
        newshape=self._reshape(tmpsample).shape[1:]
        return self._createmodel(newshape, outputsize,update_model=update_model)

    def train(self, trainset, trainlabel):
        return self._train(self._reshape(trainset), trainlabel)

    def evaluate(self, testset, testlabel):
        return self._evaluate(self._reshape(testset), testlabel)

    def predict(self, testset):
        return self._predict(self._reshape(testset))

    def predict_classes(self, testset):
        return self._predict_classes(self._reshape(testset))

    def setWeight(self,weight):
        self.weight=weight
        
    def _reshape(self, data):
        if(len(data.shape) == 2):
            return data

        if(len(data.shape) == 1):
            raise np.reshape(data, (data.shape[0], 1))

        return np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]))
    
    def _createmodel(self, inputsize, outputsize, update_model=False):
        raise NotImplementedError

    def _train(self, trainset, trainlabel):
        raise NotImplementedError

    def _evaluate(self, testset, testlabel):
        raise NotImplementedError

    def _predict(self, testset):
        raise NotImplementedError

    def _predict_classes(self, testset):
        raise NotImplementedError
