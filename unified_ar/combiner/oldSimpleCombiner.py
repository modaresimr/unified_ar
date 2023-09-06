def convertAndMergeToEvent_old(set_window,predictedclasses):
    events=[]
    for i in range(0,len(set_window)):
        start=set_window[i]['start'];
        end=set_window[i]['end'];
        pclass=predictedclasses[i]
        if(len(events)>0):
            last=events[len(events)-1]
            if(last['Activity']==pclass):
                last['EndTime']=end
                continue
        events.append({'Activity':pclass, 'StartTime':start, 'EndTime':end});
    return pd.DataFrame(events)