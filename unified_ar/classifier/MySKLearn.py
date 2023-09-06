from classifier.classifier_abstract import Classifier
import sklearn.ensemble
import sklearn.neighbors
import sklearn.svm
import sklearn.tree
import sklearn.multiclass
import numpy as np


class sklearnClassifier(Classifier):

    def _createmodel(self, inputsize, outputsize, update_model=False):
        if update_model:
            try:
                return self.model
            except:
                pass
        self.outputsize = outputsize
        self.inputsize = inputsize
        self.model = self.getmodel(inputsize, outputsize)
        return self.model

    def getmodel(self, inputsize, outputsize):
        raise NotImplementedError

    def _train(self, trainset, trainlabel):
        return self.model.fit(trainset, trainlabel)

    def _evaluate(self, testset, testlabel):
        self.model.evaluate(testset, testlabel)

    def _predict(self, testset):
        try:
            return self.model.predict_proba(testset)
        except AttributeError:
            res = np.zeros((len(testset), self.outputsize))
            cls = self.predict_classes(testset)
            for i in range(0, len(testset)):
                res[i] = cls[i]
            return res

    def _predict_classes(self, testset):
        return self.model.predict(testset)


class UAR_KNN(sklearnClassifier):

    def getmodel(self, inputsize, outputsize):
        return sklearn.neighbors.KNeighborsClassifier(n_neighbors=self.k)


class UAR_RandomForest(sklearnClassifier):

    def getmodel(self, inputsize, outputsize):
        return sklearn.ensemble.RandomForestClassifier(
            class_weight='balanced',
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=self.max_depth,
            max_features=int(max(inputsize)*self.max_features_rate)
        )


class UAR_SVM(sklearnClassifier):

    def getmodel(self, inputsize, outputsize):
        # return sklearn.svm.SVC(kernel=self.kernel,gamma=0.001, C=100., probability=True)
        return sklearn.svm.SVC(kernel=self.kernel, gamma=self.gamma, C=self.C, decision_function_shape='ovr', probability=True)


class UAR_SVM2(sklearnClassifier):

    def getmodel(self, inputsize, outputsize):
        # return sklearn.svm.SVC(kernel=self.kernel,gamma=0.001, C=100., probability=True)
        model1 = sklearn.svm.SVC(kernel=self.kernel, gamma=self.gamma, C=self.C, decision_function_shape='ovr', probability=True)
        return sklearn.multiclass.OneVsOneClassifier(model1)


class UAR_DecisionTree(sklearnClassifier):

    def getmodel(self, inputsize, outputsize):
        return sklearn.tree.DecisionTreeClassifier(class_weight='balanced')


class CustomSklearn(sklearnClassifier):

    def __init__(self, skmodel):
        self.skmodel = skmodel

    def getmodel(self, inputsize, outputsize):
        return self.skmodel
