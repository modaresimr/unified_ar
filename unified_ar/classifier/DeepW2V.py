import logging

import numpy as np
import tensorflow as tf
import unified_ar as ar
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv1D,
                                     Dense, Dropout, Embedding,
                                     GlobalAveragePooling1D, Input, Masking)
from tensorflow.keras.models import Model, Sequential
from tqdm.keras import TqdmCallback

from .classifier_abstract import Classifier
from .Keras import SequenceNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, BatchNormalization,
                                            Activation, GlobalAveragePooling1D, Dense, Masking)


class FCN(SequenceNN):

    def getmodel(self, inputsize, outputsize):
        nb_classes = outputsize
        s1, s2 = inputsize

        model = Sequential(name="FCN")
        model.add(Masking(mask_value=0.0, input_shape=(s1, s2)))
        model.add(Conv1D(filters=128, kernel_size=8, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(activation='relu'))
        model.add(Conv1D(filters=256, kernel_size=5, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(128, kernel_size=3, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(nb_classes, activation='softmax'))

        return model

class FCNEmbedded(SequenceNN):

    def getmodel(self, inputsize, outputsize):

        nb_classes = outputsize
        n_timesteps = inputsize[1]

        model = Sequential(name="FCN_Embedded")

        model.add(Embedding(input_dim=self.vocab_size + 1, output_dim=64, 
                    input_length=n_timesteps, mask_zero=True))
        model.add(Conv1D(filters=128, kernel_size=8, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(activation='relu'))
        model.add(Conv1D(filters=256, kernel_size=5, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(128, kernel_size=3, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        return model


class LiciottiBiLSTM(SequenceNN):

    def getmodel(self, inputsize, outputsize):
        from tensorflow.keras.layers import (LSTM, Activation,
                                             BatchNormalization, Bidirectional,
                                             Conv1D, Dense, Dropout, Embedding,
                                             GlobalAveragePooling1D, Input,
                                             Lambda)
        from tensorflow.keras.models import Model
        nb_timesteps = inputsize[1]
        nb_classes = outputsize
        emb_dim = self.emb_dim
        #vocab = list(self.classifier_dataset_encoder.eventDict.keys())
        output_dim = self.nb_units

        # build the model

        # create embedding

        # embedding_layer = Embedding(input_dim=self.vocab_size + 1, output_dim=emb_dim, input_length=nb_timesteps, mask_zero=True)

        # # classifier
        # feature_1 = Input(shape=((nb_timesteps,)))

        # embedding = embedding_layer(feature_1)
        
        # lstm_1 = Bidirectional(LSTM(output_dim))(embedding)

        # output_layer = Dense(nb_classes, activation='softmax')(lstm_1)

        # model = Model(inputs=feature_1, outputs=output_layer, name="liciotti_Bi_LSTM")

                
        model = tf.keras.models.Sequential(name="liciotti_Bi_LSTM")
        # Dummy layer to specify input shape
        model.add(Lambda(lambda x: x, input_shape=(nb_timesteps,)))

        # Embedding layer
        model.add(Embedding(input_dim=self.vocab_size + 1, 
                            output_dim=emb_dim, 
                            input_length=nb_timesteps, 
                            mask_zero=True))

        # Bidirectional LSTM
        model.add(Bidirectional(LSTM(output_dim)))

        # Dense output layer
        model.add(Dense(nb_classes, activation='softmax'))
        return model
