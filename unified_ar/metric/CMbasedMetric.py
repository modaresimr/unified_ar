
import numpy as np
from unified_ar.general.utils import Data

epsilon = np.finfo(float).eps


def CMbasedMetric(cm, average=None, weight=None):

    TP, FP, FN, TN = get_tp_fp_fn_tn(cm)

    accuracy = TP.sum()/cm.sum()
    precision = TP/(TP+FP+epsilon)
    recall = TP/(TP+FN+epsilon)
    f1 = 2*recall*precision/(recall+precision+epsilon)

    result = {}

    result['accuracy'] = round(accuracy, 2)
    if (average is None):
        result['precision'] = precision
        result['recall'] = recall
        result['f1'] = f1
        return result

    s = TP+FN
    weight = np.ones(len(s)) if weight is None else np.array(weight)
    weight[0] = 0
    for i in range(len(s)):
        if (s[i] == 0):
            weight[i] = 0

    result['precision'] = round(np.average(precision, weights=weight), 2)
    result['recall'] = round(np.average(recall, weights=weight), 2)

    result['f1'] = round(np.average(f1, weights=weight), 2)
    # result['f1'] = round(2*result['precision']*result['recall'] / \
    #         (result['precision']+result['recall']+epsilon),2)  # np.average(f1[validres])
    return result


def get_tp_fp_fn_tn(cm):
    cm = np.array(cm)
    np.seterr(divide='ignore', invalid='ignore')
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1).T - TP
    num_classes = len(cm)
    TN = []
    for i in range(num_classes):
        temp = np.delete(cm, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(temp.sum())
    return TP, FP, FN, TN
