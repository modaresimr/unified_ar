import tensorflow as tf
from tensorflow.keras import backend as K

import tensorflow as tf
from tensorflow.keras import backend as K

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, average='macro', name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.true_positives = self.add_weight(name='tp', initializer='zeros', shape=(num_classes,))
        self.false_positives = self.add_weight(name='fp', initializer='zeros', shape=(num_classes,))
        self.false_negatives = self.add_weight(name='fn', initializer='zeros', shape=(num_classes,))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, tf.int32)

        for i in range(self.num_classes):
            y_true_i = tf.cast(tf.equal(y_true, i), tf.float32)
            y_pred_i = tf.cast(tf.equal(y_pred, i), tf.float32)

            self.true_positives[i].assign_add(tf.reduce_sum(y_true_i * y_pred_i))
            self.false_positives[i].assign_add(tf.reduce_sum((1 - y_true_i) * y_pred_i))
            self.false_negatives[i].assign_add(tf.reduce_sum(y_true_i * (1 - y_pred_i)))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

        if self.average == 'macro':
            return tf.reduce_mean(f1)
        elif self.average == 'micro':
            return tf.reduce_sum(f1) / self.num_classes
        elif self.average == 'weighted':
            total_true = tf.reduce_sum(self.true_positives + self.false_negatives)
            return tf.reduce_sum(f1 * (self.true_positives + self.false_negatives)) / total_true

    def reset_states(self):
        self.true_positives.assign(tf.zeros(self.num_classes))
        self.false_positives.assign(tf.zeros(self.num_classes))
        self.false_negatives.assign(tf.zeros(self.num_classes))




def categorical_focal_loss(gamma=2., alpha=.25):
    def focal_loss(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * tf.keras.backend.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini-batch
        return tf.keras.backend.mean(loss, axis=1)

    return focal_loss
