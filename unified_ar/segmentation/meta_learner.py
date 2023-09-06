

def createKerasModel(X,columns):
            from tensorflow import keras
            from tensorflow.keras import layers
            inputs = keras.Input(shape=(X.shape[1],))
            layer1 = layers.Dense(100, activation='relu')(inputs)
            layer2 = layers.Dense(200, activation='relu')(layer1)
            layer3 = layers.Dense(100, activation='relu')(layer2)
            classifier = layers.Dense(1, activation='softmax',name='method')(layer3)
            regressions = [layers.Dense(1, activation='linear',name=x)(layer3) for x in columns.drop('method')]

            mdl = keras.Model(inputs=inputs, outputs=[classifier, *regressions])

            mdl.compile(loss=['categorical_crossentropy',*(['mse']*len(regressions))], optimizer='adam', metrics=['accuracy'])

def createSVRModel(X,columns):
    from sklearn.svm import SVR
    from sklearn.multioutput import MultiOutputRegressor
    mdl=MultiOutputRegressor(SVR())
