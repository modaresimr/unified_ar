import numpy as np
from unified_ar.general.utils import Data
import unified_ar.general.utils as utils

import unified_ar.result_analyse.visualisation as vs
import pandas as pd


def create(real, pred, filename):

    evalres = [{}]
    evalres[0]['test'] = Data('test res')
    evalres[0]['test'].real_events = vs.convert2event(real)
    evalres[0]['test'].pred_events = vs.convert2event(pred)
    evalres[0]['test'].quality = {}

    dataset = Data('MyDataset')
    dataset.activities = ['None', 'Act']
    dataset.activity_events = evalres[0]['test'].real_events
    dataset.activities_map_inverse = {k: v for v, k in enumerate(dataset.activities)}
    dataset.activities_map = {v: k for v, k in enumerate(dataset.activities)}
    dataset.sensor_events = pd.DataFrame()
    runinfo = filename

    utils.saveState([runinfo, dataset, evalres], filename)


gt = np.array([(65, 141), (157, 187), (260, 304), (324, 326), (380, 393), (455, 470), (475, 485), (505, 555),	(666, 807), (814, 888), (903, 929)])/3

a = np.array([(66, 73), (78, 126), (135, 147), (175, 186), (225, 236), (274, 318), (349, 354), (366, 372), (423, 436), (453, 460), (467, 473),
             (487, 493), (501, 506), (515, 525), (531, 542), (545, 563), (576, 580), (607, 611), (641, 646), (665, 673), (678, 898), (907, 933)])/3
b = np.array([(63, 136), (166, 188), (257, 310), (451, 473), (519, 546), (663, 916)])/3

create(gt, a, 'ward-a')
create(gt, b, 'ward-b')
