import logging
import numpy as np
from tqdm.keras import TqdmCallback
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from .classifier_abstract import Classifier
import unified_ar as ar
import tensorflow as tf

from .Keras import SequenceNN


class FCN(SequenceNN):

    def getmodel(self, inputsize, outputsize):

        import tensorflow as tf
        from tensorflow.keras.layers import Input, Masking, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense
        from tensorflow.keras.models import Model
        # from tensorflow.keras.activations import *

        nb_classes = outputsize
        print(f'inputsize {inputsize}')
        s1 = inputsize[0]
        s2 = inputsize[1]

        input_layer = Input(shape=((s1, s2)))

        mask = Masking(mask_value=0.0)(input_layer)

        conv1 = Conv1D(filters=128, kernel_size=8, padding='same')(mask)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(activation='relu')(conv1)

        conv2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)

        conv3 = Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)

        gap_layer = GlobalAveragePooling1D()(conv3)

        output_layer = Dense(nb_classes, activation='softmax')(gap_layer)

        model = Model(inputs=input_layer, outputs=output_layer, name="FCN")

        return model


class FCNEmbedded(SequenceNN):

    def getmodel(self, inputsize, outputsize):

        import tensorflow as tf
        from tensorflow.keras.layers import Embedding, Input, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense, Dropout
        from tensorflow.keras.models import Model
        # from tensorflow.keras.activations import *
        nb_classes = outputsize

        n_timesteps = inputsize[1]

        input_layer = Input(shape=((n_timesteps,)))
        # if  self.func.featureExtractor.hasattr('tokenizer'):
        #     weights=tokenizer.
        embedding = Embedding(input_dim=self.vocab_size + 1, output_dim=64, input_length=n_timesteps, mask_zero=True)(input_layer)

        conv1 = Conv1D(filters=128, kernel_size=8, padding='same')(embedding)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(activation='relu')(conv1)

        conv2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)

        conv3 = Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)

        gap_layer = GlobalAveragePooling1D()(conv3)

        x = Dropout(0.5)(gap_layer)

        output_layer = Dense(nb_classes, activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output_layer, name="FCN_Embedded")

        return model


class LiciottiBiLSTM(SequenceNN):

    def getmodel(self, inputsize, outputsize):
        from tensorflow.keras.layers import Embedding, Input, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense, Dropout, Bidirectional, LSTM
        from tensorflow.keras.models import Model
        nb_timesteps = inputsize[1]
        nb_classes = outputsize
        emb_dim = self.emb_dim
        #vocab = list(self.classifier_dataset_encoder.eventDict.keys())
        output_dim = self.nb_units

        # build the model

        # create embedding

        embedding_layer = Embedding(input_dim=self.vocab_size + 1, output_dim=emb_dim, input_length=nb_timesteps, mask_zero=True)

        # classifier
        feature_1 = Input(shape=((nb_timesteps,)))

        embedding = embedding_layer(feature_1)

        lstm_1 = Bidirectional(LSTM(output_dim))(embedding)

        output_layer = Dense(nb_classes, activation='softmax')(lstm_1)

        model = Model(inputs=feature_1, outputs=output_layer, name="liciotti_Bi_LSTM")
        return model
