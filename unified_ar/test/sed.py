
import numpy as np
import pandas as pd
import os
import SED.my_eval
gtf='public.tsv'
rootFolder='/workspace/sed2020/'
typ=gtf.split('.')[0]
gtf=f'{rootFolder}/metadata/{gtf}'
total_dic={}


res1=SED.my_eval.get_single_result('/workspace/sed2020//metadata/public.tsv','/workspace/sed2020//submissions/CTK_NU/CTK_NU_task4_SED_1/public//CTK_NU_task4_SED_1.output.tsv')
res2=SED.my_eval.get_single_result('/workspace/sed2020//metadata/public.tsv','/workspace/sed2020//submissions/CTK_NU/CTK_NU_task4_SED_3/public//CTK_NU_task4_SED_3.output.tsv','/workspace/sed2020//submissions/CTK_NU/CTK_NU_task4_SED_1/public//CTK_NU_task4_SED_3.output_PSDS')


total=pd.DataFrame({'res1':{c:res1[c].loc['macro-avg']['f1'] for c in res1.keys()},
                   'res2':{c:res2[c].loc['macro-avg']['f1'] for c in res2.keys()} }).T

print(total)