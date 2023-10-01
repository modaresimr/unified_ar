import tensorflow as tf
from tensorflow.keras import backend as K

import tensorflow as tf
from tensorflow.keras import backend as K


def F1Score(num_classes, average='macro'):
    def f1_score(y_true, y_pred):
        y_pred = tf.round(y_pred)
        tp = [tf.reduce_sum(y_true[:, c] * y_pred[:, c]) for c in range(num_classes)]
        fp = [tf.reduce_sum((1 - y_true[:, c]) * y_pred[:, c]) for c in range(num_classes)]
        fn = [tf.reduce_sum(y_true[:, c] * (1 - y_pred[:, c])) for c in range(num_classes)]

        precision = [tp[c] / (tp[c] + fp[c] + K.epsilon()) for c in range(num_classes)]
        recall = [tp[c] / (tp[c] + fn[c] + K.epsilon()) for c in range(num_classes)]
        f1 = [2 * precision[c] * recall[c] / (precision[c] + recall[c] + K.epsilon()) for c in range(num_classes)]

        if average == 'macro':
            f1_score = tf.reduce_mean(f1)
        elif average == 'micro':
            f1_score = tf.reduce_sum(f1) / num_classes
        elif average == 'weighted':
            total_true = tf.reduce_sum(y_true)
            f1_score = tf.reduce_sum(f1 * tf.reduce_sum(y_true, axis=0)) / total_true
        else:
            raise ValueError("Unknown average type. Must be one of ['macro', 'micro', 'weighted']")

        return f1_score

    return f1_score


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




    @staticmethod
    def tf_f1_score(y_true, y_pred):
        """Computes 3 different f1 scores, micro macro
        weighted.
        micro: f1 score accross the classes, as 1
        macro: mean of f1 scores per class
        weighted: weighted average of f1 scores per class,
                weighted from the support of each class


        Args:
            y_true (Tensor): labels, with shape (batch, num_classes)
            y_pred (Tensor): model's predictions, same shape as y_true

        Returns:
d            tuple(Tensor): (micro, macro, weighted)
                        tuple of the computed f1 scores
        """

        f1s = [0, 0, 0]

        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)

        for i, axis in enumerate([None, 0]):
            TP = tf.math.count_nonzero(y_pred * y_true, axis=axis)
            FP = tf.math.count_nonzero(y_pred * (y_true - 1), axis=axis)
            FN = tf.math.count_nonzero((y_pred - 1) * y_true, axis=axis)

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)

            f1s[i] = tf.reduce_mean(f1)

        weights = tf.reduce_sum(y_true, axis=0)
        weights /= tf.reduce_sum(weights)

        f1s[2] = tf.reduce_sum(f1 * weights)

        micro, macro, weighted = f1s
        return macro

    # @staticmethod
    # def f1(y_true, y_pred):
    #     K=tf.keras.backend
    #     y_true=K.cast(y_true, 'float')
    #     y_pred = K.round(y_pred)
    #     tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    #     tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    #     fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    #     fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    #     p = tp / (tp + fp + K.epsilon())
    #     r = tp / (tp + fn + K.epsilon())

    #     f1 = 2*p*r / (p+r+K.epsilon())
    #     f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    #     return K.mean(f1)
