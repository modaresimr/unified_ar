from feature_extraction.feature_abstract import FeatureExtraction
import tensorflow as tf
import logging
logger = logging.getLogger(__file__)
class DeepLearningFeatureExtraction(FeatureExtraction):
    def precompute(self,datasetdscr,windows):
        super().precompute(datasetdscr,windows)
        self.simpleFE=SimpleFeatureExtraction()
        wins=np.array([self.simpleFE.featureExtract(w,None) for w in windows])
        win=wins[0]
        dimention=self.params['size']
        input_size=len(win)
        input=tf.keras.layers.Input(shape=(input_size,));

        # input_img=tf.keras.layers.Input(shape=(input_size,))
        # autoencoder=tf.keras.models.Sequential();
        # autoencoder.add(input_img)
        # autoencoder.add(tf.keras.layers.Dense(dimention,activation='relu'))
        # autoencoder.add(tf.keras.layers.Dense(input_size,activation='sigmoid'))
        
        # encoder=tf.keras.models.Model(input_img,autoencoder.layers[0](input_img))

        model = tf.keras.models.Sequential([
            input,
            tf.keras.layers.Dense(dimention, activation=tf.nn.relu),
            tf.keras.layers.Dense(dimention/2, activation=tf.nn.relu),
            tf.keras.layers.Dense(dimention, activation=tf.nn.sigmoid),
            tf.keras.layers.Dense(input_size, activation=tf.nn.sigmoid)
        ],name="deep-autoencoder")
        
        self.encoder=tf.keras.models.Sequential([
                                                 input,
                                                 model.layers[0],
                                                 model.layers[1]
                                                ],name=self.shortname()+"-deep-encoder")
        
        # model=autoencoder;
        # self.encoder=encoder;
        # self.decoder=model.layers[3](model.layers[2])
        logger.debug('shape %d inputsize %d',wins.shape, input_size)
        #model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        logger.debug('encoder========================')
        self.encoder.summary()
        logger.debug('autoencoder========================')
        
        
        model.compile(optimizer='adamax', loss=tf.keras.losses.mean_absolute_error)
        
        model.fit(wins,wins,epochs=5, batch_size=100, shuffle=True)
        model.summary()
        self.model=model
        
    def featureExtract(self,window):
        f=self.simpleFE.featureExtract(window)
        f=f.reshape(1,len(f))
        return self.encoder.predict(f)[0]
        
