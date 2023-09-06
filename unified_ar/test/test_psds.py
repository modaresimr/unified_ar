# print('ali')
# import pandas as pd
# import SED.my_eval
# gtf='/workspace/sed2020/metadata/public.tsv'
# pef='/workspace/sed2020/submissions/CTK_NU/CTK_NU_task4_SED_1/public/CTK_NU_task4_SED_1.output.tsv'
# res=SED.my_eval.get_single_result(gtf,pef,debug=0)
# out={}
# out['segment']=res['segment'][['Ntp','Nfp','Nfn']].loc['Blender']
# out['total duration']=res['total duration'][['Ntp','Nfp','Nfn']].loc['Blender']
# out['diff']=out['segment']-out['total duration']
# pd.DataFrame(out).T



print('ali')
import pandas as pd
import SED.my_eval
path="/tmp/test/"

gtf='/workspace/sed2020/metadata/public.tsv'
pef='/workspace/sed2020/submissions/CTK_NU/CTK_NU_task4_SED_1/public/CTK_NU_task4_SED_1.output.tsv'

gt = pd.read_csv(gtf, sep="\t")
    # Evaluate a single prediction
pt = pd.read_csv(pef, sep="\t")

clas=gt.event_label.append(pt.event_label).unique()
all=None
result={}
for f in gt.filename.unique():

        gtc=gt.loc[gt.filename==f]
        ptc=pt.loc[pt.filename==f]
 
        res=SED.my_eval.get_single_result_df(gtc,ptc,debug=0)
        # for m in res:
        out={}
        if('Blender' not in res['segment'].index):continue
        out['segment']=res['segment'][['Ntp','Nfp','Nfn']].loc['Blender']
        out['total duration']=res['total duration'][['Ntp','Nfp','Nfn']].loc['Blender']
        out['diff']=out['segment']-out['total duration']
        out=pd.DataFrame(out).T
        if(all is None):
            all=out
        else:
            all+=out
        # if (out.loc['diff']['Nfp'])>.1:
        print(f'=======================file={f}')
        print(out)
#             res=SED.my_eval.get_single_result(f'{path}/g.tsv',f'{path}/p.tsv',f'{path}/meta.tsv',debug=1)
#             break
#         break

print(all)

