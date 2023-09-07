from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Flatten, LSTM, MaxPooling1D, TimeDistributed
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import regularizers
from .Keras import KerasClassifier, SequenceNN


class CNN_LSTM(SequenceNN):
    def getmodel(self, inputsize, outputsize):
        model = Sequential(name=self.shortname())
        print(inputsize)
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=inputsize[1]))
        model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu')))
        model.add(TimeDistributed(Conv1D(filters=16, kernel_size=3, activation='relu')))
        model.add(TimeDistributed(Dropout(0.4)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(units=200, return_sequences=True))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(LSTM(units=100))
        model.add(Dropout(0.4))
        # model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        # model.add(Dense(100, activation='relu'))
        # model.add(Dense(100, activation='relu'))
        model.add(Dense(80, activation='relu'))
        model.add(Dense(outputsize, activation='softmax'))
        return model

    # def _reshape(self, data):
    #     if (len(data.shape) == 2):
    #         return np.reshape(data, (data.shape[0], data.shape[1], 1))
    #     return data
