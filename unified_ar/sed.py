import numpy as np
import pandas as pd
import os
#@interact
#def result_selector(gtf=os.listdir(f'{rootFolder}/metadata/')):
import SED.my_eval
gtf='public.tsv'
rootFolder='/workspace/sed2020/'
typ=gtf.split('.')[0]
gtf=f'{rootFolder}/metadata/{gtf}'
# meta_dur_df=pd.DataFrame(columns=['filename','duration'])
# meta_dur_df['filename']=groundtruth['filename']
# meta_dur_df['duration']=10
total_dic={}
for team in sorted(os.listdir(f'{rootFolder}/submissions/')):
    print(f'analysing team {team}')
    for code in sorted(os.listdir(f'{rootFolder}/submissions/{team}')):
        print(f'    {code}')
        base_prediction_path=f'{rootFolder}/submissions/{team}/{code}/{typ}/'
        pef = f'{base_prediction_path}/{code}.output.tsv'
        if not(os.path.isfile(pef)):
            all=[x for x in os.listdir(base_prediction_path) if '.output.tsv' in x]
            if len(all)>0:
                pef=f'{base_prediction_path}/{all[0]}'
            else:
                print(pef)
                continue
        title=code.replace('_task4','')
        res1=SED.my_eval.get_single_result(gtf,pef)
        total_dic[title]={c:res1[c].loc['macro-avg']['f1'] for c in res1.keys() if c!='gem'}
        if('gem' in res1):
            for k in res1['gem'].keys():
                total_dic[title][f'gem-{k}']=res1['gem'].loc['avg'][k]
#        break
#    break

total=pd.DataFrame(total_dic).T
total['y']=total.index

import general.utils
general.utils.saveState(total,'total')
