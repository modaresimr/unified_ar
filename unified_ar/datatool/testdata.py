from general.utils import Data
import pandas as pd
def load():

    gt=[(65,141), (157,187), (260,304), (324,326), (380,393), (455,470), (475,485), (505,555),	(666,807), (814,888),(903,929)]
    a=[(66,73),(78,126),(135,147),(175,186),(225,236),(274,318),(349,354),(366,372),(423,436),(453,460),(467,473),(487,493),(501,506),(515,525),(531,542),(545,563),(576,580),(607,611),(641,646),(665,673),(678,898),(907,933)]
    b=[(63,136),(166,188),(257,310),(451,473),(519,546),(663,916)]


    dataset=Data('test')
    dataset.activities=['null','Act']
    dataset.activities_map={0:'null',1:'Act'}
    dataset.activities_map_inverse={'null':0,'Act':1}
    dataset.activity_events=pd.DataFrame(columns=["StartTime", "EndTime", "Activity", 'Duration'])

    init=pd.to_datetime('1/1/2020')
    for k in gt:
        actevent = {"StartTime": init+pd.to_timedelta(str(k[0])+'s'),	"EndTime": init+pd.to_timedelta(str(k[1])+'s'),
                            "Activity": 1, 'Duration': pd.to_timedelta(str(k[1]-k[0])+'s')}

        dataset.activity_events = dataset.activity_events.append(actevent, ignore_index=True)

    aevents=pd.DataFrame(columns=["StartTime", "EndTime", "Activity", 'Duration'])
    for k in a:
        actevent = {"StartTime": init+pd.to_timedelta(str(k[0])+'s'),	"EndTime": init+pd.to_timedelta(str(k[1])+'s'),
                            "Activity": 1, 'Duration': pd.to_timedelta(str(k[1]-k[0])+'s')}
        aevents=aevents.append(actevent, ignore_index=True)

    bevents=pd.DataFrame(columns=["StartTime", "EndTime", "Activity", 'Duration'])
    for k in b:
        actevent = {"StartTime": init+pd.to_timedelta(str(k[0])+'s'),	"EndTime": init+pd.to_timedelta(str(k[1])+'s'),
                            "Activity": 1, 'Duration': pd.to_timedelta(str(k[1]-k[0])+'s')}
        bevents = bevents.append(actevent, ignore_index=True)

    return dataset,aevents,bevents


