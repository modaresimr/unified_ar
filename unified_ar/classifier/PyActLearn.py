import tensorflow as tf
import numpy as np
import unified_ar.classifier.pal_nn.lstm as lstm

from .classifier_abstract import Classifier
from pyActLearn.learning.nn.mlp import MLP
from pyActLearn.learning.nn.sda import SDA


class PyActLearnClassifier(Classifier):
    def _createmodel(self, inputsize, outputsize):
        raise NotImplementedError

    def _train(self, trainset, trainlabel):
        raise NotImplementedError

    def _evaluate(self, testset, testlabel):
        raise NotImplementedError

    def _predict(self, testset):
        raise NotImplementedError

    def _predict_classes(self, testset):
        raise NotImplementedError

    def save(self, desc):
        raise NotImplementedError

    def load(self, desc):
        raise NotImplementedError


class PAL_NN(Classifier):

    def getmodel(self, inputsize, outputsize):
        raise NotImplementedError

    def _createmodel(self, inputsize, outputsize, update_model=False):
        if update_model:
            try:
                return self.model
            except:
                pass
        self.outputsize = outputsize
        self.model = self.getmodel(inputsize, outputsize)
        return self.model

    def _train(self, trainset, trainlabel):
        self.model.fit(trainset, trainlabel, iter_num=5000, batch_size=100,
                       criterion='monitor_based', summaries_dir='logs')

    def _evaluate(self, testset, testlabel):
        raise NotImplementedError

    def _predict(self, testset):
        return self.model.predict_proba(testset)

    def _predict_classes(self, testset):
        return self.model.predict(testset)

    def save(self, file):
        tf.saved_model.save(self.model.sess, file+'pyact.h5')

    def load(self, file):
        self.model = MLP(0, 0, [1000])
        self.model.sess = tf.saved_model.load(file+"pyact.h5")


class PAL_MLP(PAL_NN):

    def getmodel(self, inputsize, outputsize):
        return MLP(inputsize, outputsize, [1000])


class PAL_LSTM(PAL_NN):

    def getmodel(self, inputsize, outputsize):
        return lstm.LSTM(inputsize, outputsize, num_hidden=1000,
                         num_units=300, num_skip=100)

    def train(self, trainset, trainlabel):
        train_y = np.zeros((trainlabel.shape[0], self.outputsize))
        for i in range(trainset.shape[0]):
            train_y[i, trainlabel[i]] = 1
        super().train(trainset, train_y)


class PAL_LSTM_Simple(PAL_LSTM):

    def getmodel(self, inputsize, outputsize):
        return lstm.SimpleLSTM(inputsize, outputsize, numhidden=1000,
                               num_units=300, num_skip=100)


class PAL_LSTM_Legacy(PAL_LSTM):

    def getmodel(self, inputsize, outputsize):
        return lstm.LSTM_Legacy(
            inputsize, outputsize, num_units=300, num_steps=100)


class PAL_SDA(PAL_NN):
    """Stacked Auto-encoder
    """

    def getmodel(self, inputsize, outputsize):
        return SDA(inputsize, outputsize, [300, 300, 300])
