import logging
import numpy as np
from tqdm.keras import TqdmCallback
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from .classifier_abstract import Classifier
import unified_ar as ar
import tensorflow as tf
from tensorflow.keras import backend as K
# import tensorflow_addons as tfa

# tf.config.set_visible_devices([], 'GPU')

logger = logging.getLogger(__file__)

from .keras_utils import F1Score, categorical_focal_loss


class KerasClassifier(Classifier):

    def get_loss_functions(self):
        # return categorical_focal_loss()
        return 'categorical_crossentropy'

    def get_metrics(self, num_classes):

        # a=tfa.metrics.F1Score(num_classes=outputsize,average='micro')
        # a.average ='macro'
        METRICS = [
            #   tf.keras.metrics.TruePositives(name='tp'),
            #   tf.keras.metrics.FalsePositives(name='fp'),
            #   tf.keras.metrics.TrueNegatives(name='tn'),
            #   tf.keras.metrics.FalseNegatives(name='fn'),
            # CategoricalTruePositives(name='tp',num_classes=outputsize,batch_size=500),
            # KerasClassifier.tf_f1_score,
            # a,
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            # tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            # tf.keras.metrics.Precision(name='precision'),
            # tf.keras.metrics.Recall(name='recall'),
            # tf.keras.metrics.AUC(name='auc'),
        ]

        # loss=tfa.losses.sparsemax_loss
        # loss=tfa.losses.sigmoid_focal_crossentropy
        # loss = 'sparse_categorical_crossentropy'

        # f1_score_metric = F1Score(num_classes, average='weighted')
        # return ['accuracy', f1_score_metric]
        return ['accuracy']

    def _createmodel(self, inputsize, outputsize, update_model=False):
        self.outputsize = outputsize
        self.tqdmcallback = TqdmCallback(verbose=1)
        if update_model and not hasattr(self, 'model'):
            from unified_ar.constants import methods
            meta_path = methods.run_names.get('meta_base', '')
            if meta_path:
                logger.debug(f'loading meta train model {meta_path}')
                metrics = self.get_metrics(outputsize)
                metrics.append(self.get_loss_functions())
                custom_objects = {m.__name__: m for m in metrics if callable(m)}

                self.model = tf.keras.models.load_model(f'save_data/{meta_path}/keras', custom_objects=custom_objects)

            return self.model

        

        model = self.getmodel(inputsize, outputsize)
        if not methods.run_names.get('meta_base', ''):
            model.summary()

        # model.compile(optimizer='adam', loss=loss, metrics=METRICS)
        model.compile(optimizer='adam', loss=self.get_loss_functions(), metrics=self.get_metrics(outputsize))
        self.model = model
        
        return model

    def getmodel(self, inputsize, outputsize):
        raise NotImplementedError

    def _train(self, trainset, trainlabel):
        from unified_ar.constants import methods
        if methods.run_names.get('meta_base', ''):
            return 
        classes = np.unique(trainlabel)
        try:
            cw = compute_class_weight("balanced", classes=classes, y=trainlabel)
        except:
            cw = np.ones(self.outputsize)
        if hasattr(self, 'weight') and not (self.weight is None):
            cw *= self.weight
        cw = {c: cw[i] for i, c in enumerate(classes)}

        for i in range(self.outputsize):
            if not (i in cw):
                cw[i] = 0
        

        trainlabel = tf.keras.utils.to_categorical(trainlabel, num_classes=self.outputsize)

        # mc = tf.keras.callbacks.ModelCheckpoint(path, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        # tf.keras.backend.set_value(self.model.optimizer.lr, .01)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=0, patience=20, restore_best_weights=True)

        save_folder = ar.general.utils.get_save_folder()

        logpath = f'{save_folder}/{methods.run_names["out"]}.csv'
        from pathlib import Path
        Path(logpath).parent.mkdir(parents=True, exist_ok=True)
        csv_logger = tf.keras.callbacks.CSVLogger(logpath, append=True, separator=',')

        filepath = f"{save_folder}/weights-best"
        checkpoint = ModelCheckpoint(
            filepath, save_weights_only=True,
            # monitor='val_f1_score',
            monitor='val_accuracy',
            verbose=0,
            save_best_only=True,
            mode='max')
        tensorboard_cb = tf.keras.callbacks.TensorBoard(save_folder)
        callbacks = [self.tqdmcallback, es, csv_logger, checkpoint, tensorboard_cb]
        # callbacks = [self.tqdmcallback, es, csv_logger, checkpoint]

        self.model.fit(
            trainset,
            trainlabel,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            shuffle=True,
            # class_weight=cw,
            callbacks=callbacks,
            verbose=0)
        self.trained = True

        path = f'save_data/{methods.run_names["out"]}/keras'

        logger.debug(f'save model to {path}')
        tf.keras.models.save_model(model=self.model, filepath=path)
        # self.model = tf.keras.models.load_model(path)

    def _evaluate(self, testset, testlabel):
        # if(self.trained):
        testlabel = tf.keras.utils.to_categorical(testlabel)
        self.model.evaluate(testset, testlabel, callbacks=[self.tqdmcallback], verbose=0)

    # else:
    #     print("model not trained")

    def _predict(self, testset):
        # if(self.trained):
        return self.model.predict(testset, callbacks=[self.tqdmcallback], verbose=0)

    # else:
    #     return self.model.predict(testset)*0

    def _predict_classes(self, testset):
        # if(self.trained):
        return np.argmax(self._predict(testset), axis=-1)
        # return self.model.predict_classes(testset)

    # else:
    #     return self.model.predict_classes(testset)*0

    def save(self, file):
        logger.debug('saving model to %s', file)
        self.model.save(file + '.keras')

    def load(self, file):
        logger.debug('loading model from %s', file)
        if not ('.keras' in file):
            file = file + '.keras'
        self.model = tf.keras.models.load_model(file)


class SequenceNN(KerasClassifier):

    def _reshape(self, data):
        print("shape", data.shape)
        if (len(data.shape) == 2):
            return np.reshape(data, (data.shape[0], data.shape[1], 1))
        return data


class LSTMTest(SequenceNN):

    def getmodel(self, inputsize, outputsize):

        return tf.keras.models.Sequential(
            [
                # tf.keras.layers.Dense(128, input_shape=inputsize),
                # tf.keras.layers.Embedding(input_dim=inputsize,output_dim=100),
                tf.keras.layers.LSTM(128, activation=tf.nn.relu, input_shape=inputsize),
                # tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(outputsize, activation=tf.nn.softmax)
            ],
            name=self.shortname())


class LSTMAE(SequenceNN):

    def getmodel(self, inputsize, outputsize):

        return tf.keras.models.Sequential(
            [
                tf.keras.layers.LSTM(inputsize, activation=tf.nn.relu, input_shape=inputsize),
                tf.keras.layers.LSTM(inputsize // 2, activation=tf.nn.relu),
                tf.keras.layers.LSTM(inputsize, activation=tf.nn.relu),
                # tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(outputsize, activation=tf.nn.softmax)
            ],
            name=self.shortname())


class SimpleKeras(KerasClassifier):

    def getmodel(self, inputsize, outputsize):
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, input_shape=inputsize),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(outputsize, activation=tf.nn.softmax)
        ],
            name=self.shortname())


class NormalKeras(KerasClassifier):

    def getmodel(self, inputsize, outputsize):
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, input_shape=inputsize, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.layers.LeakyReLU()),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.layers.LeakyReLU()),
            tf.keras.layers.Dense(32),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.layers.LeakyReLU()),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.layers.LeakyReLU()),
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.layers.LeakyReLU()),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(outputsize, activation=tf.nn.softmax)
        ], name=self.shortname())


class WangMLP(KerasClassifier):

    def getmodel(self, inputsize, outputsize):
        from pyActLearn.learning.nn.mlp import MLP
        return MLP(inputsize, outputsize, [1000])


class CategoricalTruePositives(tf.keras.metrics.Metric):
    import tensorflow.keras.backend as K

    def __init__(self, num_classes, batch_size, name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)

        self.batch_size = batch_size
        self.num_classes = num_classes

        self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = K.argmax(y_true, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.flatten(y_true)

        true_poss = K.sum(K.cast((K.equal(y_true, y_pred)), dtype=tf.float32))

        self.cat_true_positives.assign_add(true_poss)

    def result(self):

        return self.cat_true_positives
