from general.utils import Data
from preprocessing.preprocessing_abstract import Preprocessing


class SimplePreprocessing(Preprocessing):

    # remove invalid changes in stream
    def process(self, datasetdscr, dataset):
        removekeys = []
        for sid, info in datasetdscr.sensor_desc.iterrows():
            
            if (info.Nominal | info.OnChange):
                continue
            if not info.Cumulative:
                continue
            

            xs  = dataset.s_events.loc[dataset.s_events.SID == sid]

            # min = xs.value.min()
            # max = xs.value.max()
            last = xs.iloc[0].copy()
            for key, event in xs.iterrows():
                newval=event['value']-last['value']
                last=event.copy()
                event['value']=newval

                # if event['value']<last['value'] or event['value']-last['value']> invalid_changes:
            # acceptable_change_rate=.2
            # #print(min, max, max-min)
            # invalid_changes = (max-min)*acceptable_change_rate
            # for key, event in xs.iterrows():
            #     if event['value']<last['value'] or event['value']-last['value']> invalid_changes:
            #         removekeys.append(key)
            #     else:
            #         last = event    
            #     # invalid_changes = event['value']*acceptable_change_rate
            #     # if abs(last['value']-) > invalid_changes:
            #     #     #print (event)
            #     #     removekeys.append(key)
            #     #     continue
            #     # last = event
            # # print(removekeys)
        d = Data(dataset.name)
        d.s_events = dataset.s_events.drop(removekeys)
        d.a_events = dataset.a_events
        d.s_event_list = d.s_events.values
        d.acts = dataset.acts
        d.act_map = dataset.act_map
        return d
