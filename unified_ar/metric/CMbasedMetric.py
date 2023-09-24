
import numpy as np
from unified_ar.general.utils import Data

epsilon = np.finfo(float).eps


def CMbasedMetric(cm, average=None, weight=None):

    TP, FP, FN, TN,support = get_tp_fp_fn_tn_support(cm)

    accuracy = TP.sum()/cm.sum()
    precision = TP/(TP+FP+epsilon)
    recall = TP/(TP+FN+epsilon)
    f1 = 2*recall*precision/(recall+precision+epsilon)

    result = {}

    result['accuracy'] = round(accuracy, 4)
    if (average is None):
        result['precision'] = precision
        result['recall'] = recall
        result['f1'] = f1
        return result

    s = TP+FN
    # weight = np.ones(len(s)) if weight is None else np.array(weight)
    # weight[0] = 0
    # for i in range(len(s)):
    #     if (s[i] == 0):
    #         weight[i] = 0

    if average == 'weighted':
        if weight is None:
            weight=support
        result['precision'] = round(np.average(precision, weights=weight), 4)
        result['recall'] = round(np.average(recall, weights=weight), 4)
        result['f1'] = round(np.average(f1, weights=weight), 4)
    elif average == 'macro':
        result['precision'] = round(np.mean(precision), 4)
        result['recall'] = round(np.mean(recall), 4)
        result['f1'] = round(np.mean(f1), 4)
    elif average == 'micro':
        total_tp = TP.sum()
        total_fp = FP.sum()
        total_fn = FN.sum()
        micro_precision = total_tp / (total_tp + total_fp + epsilon)
        micro_recall = total_tp / (total_tp + total_fn + epsilon)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + epsilon)
        result['precision'] = round(micro_precision, 4)
        result['recall'] = round(micro_recall, 4)
        result['f1'] = round(micro_f1, 4)

    return result


def get_tp_fp_fn_tn(cm):
    TP, FP, FN, TN,_ =get_tp_fp_fn_tn_support(cm)
    return TP, FP, FN, TN


def get_tp_fp_fn_tn_support(cm):
    cm = np.array(cm)
    np.seterr(divide='ignore', invalid='ignore')
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1).T - TP
    support = np.sum(cm, axis=1)
    num_classes = len(cm)
    TN = []
    for i in range(num_classes):
        temp = np.delete(cm, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(temp.sum())
    return TP, FP, FN, TN, support