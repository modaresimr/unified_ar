
from general import utils
import numpy as np
from activity_fetcher.CookActivityFetcher import CookActivityFetcher
def calc_cm_per_s_event(dataset,evalres):
    activities=dataset.activities  
    summycm=np.zeros((len(activities),len(activities)))
    
    for i in range(len(evalres)):
        # print(evalres[i]['test'].__dict__)
        sdata=evalres[i]['test'].Sdata
        cook=CookActivityFetcher()
        cook.precompute(sdata)
        c=0
        for k in range(0,len(sdata.s_event_list)):
            real=cook.getActivity2(sdata.s_event_list,[k])
            while (c<len(sdata.set_window) and  k>=max(sdata.set_window[c])):c=c+1
            if(c>=len(sdata.set_window)):
                break
            pred=sdata.label[c]
            summycm[real][pred]=summycm[real][pred]+1
        # evalres[i]['test'].Sdata.s_event_list[evalres[i]['test'].Sdata.set_window[400][-1]][1]    
        
    return summycm


if __name__ == '__main__':
    [run_info,datasetdscr,evalres]=utils.loadState('200515_13-22-21-Home2')
    calc_cm_per_s_event(datasetdscr,evalres)