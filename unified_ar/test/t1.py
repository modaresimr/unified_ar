
import general.utils as utils
import datatool.testdata as testdata
import unified_ar.result_analyse.visualisation as vs
from unified_ar.metric import *
result = '200203_20-10-20-KaryoAdlNormal'
run_info, dataset, evalres = utils.loadState(result)

print(CMbasedMetric(evalres[0].event_cm, ''))


# vs.my_result_analyse(dataset,real_events,pred_events)

# utils.saveState(result,'a/b/c')
