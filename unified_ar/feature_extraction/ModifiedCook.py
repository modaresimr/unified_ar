from feature_extraction.feature_abstract import FeatureExtraction
import pandas as pd
import numpy as np
class Cook1(FeatureExtraction):
	def getShape(self):
		scount=sum(1 for x in self.datasetdscr.sensor_id_map);
		return (scount+3,)
		
	sec_in_day=(60*60*24)
	def prepareData(self, s_event_list):
		self.sid_events_idx={}
		self.sid_pre_nex={}
		for sname in self.datasetdscr.sensor_id_map_inverse:
			sid_events=s_event_list.loc[s_event_list['SID']==sname].index
			self.sid_events_idx[sname]=sid_events
			# self.sid_pre_nex[sname]={x:
			# 		(indx.iloc[i-1] if i>0 else None,#pre
			# 		indx.iloc[i+1] if i<len(indx)-1 else None)#next
			# 		for  i,x in enumerate(indx)}

    
	def featureExtract2(self,s_event_list,idx):
		
		window=s_event_list
		scount=len(self.datasetdscr.sensor_id_map)
		f=np.zeros(scount + 3)
		dic={id:np.zeros(0) for id in self.datasetdscr.sensor_id_map_inverse}
		for i in idx:
			# sname=window[i,0]
			# # tim=window[i,1]
			# value=window[i,2]
			dic[sname]=np.append(dic[sname],idx)
			
			# sid=self.datasetdscr.sensor_id_map_inverse[sname]
			# f[sid]=value   #f[sensor_id_map_inverse[x.SID]]=1

		for sname in dic:
			scount = dic[sname].count()
			sid    = self.datasetdscr.sensor_id_map_inverse[sname]
			sinfo  = self.datasetdscr.sensor_desc.loc[sid]
			
			if sinfo.Cumulative:
				sdate=window[idx[0],1]
				edate=window[idx[-1],1]
				pres=self.sid_events_idx[sname][self.sid_events_idx<idx[0]]
				afters=self.sid_events_idx[sname][self.sid_events_idx>idx[-1]]
				if len(pres):
					pre=pres[0]
					pdate=s_event_list[pre,1]
					pvalue=s_event_list[pre,2]
					# svalue=

				self.sid_events.loc[self.sid_events.index<idx[0]][-1]
				self.sid_events.loc[self.sid_events.index>idx[-1]][0]
				pre,_=self.sid_pre_nex[sname][dic[sname][0]]
				_,nex=self.sid_pre_nex[sname][dic[sname][-1]]
				
				if pre:
					

					
				window[i,2]
			if f[sid] >1:
				f[sid] = dic[id].max() - dic[id].min()
			

		

		stime=window[idx[0] ,1]#startdatetime
		etime=window[idx[-1],1]#enddatetime
		ts=(stime-pd.to_datetime(stime.date())).total_seconds()
		f[scount+0]=ts/self.sec_in_day
		f[scount+1]=(etime-stime).total_seconds()/self.sec_in_day
		f[scount+2]=len(idx)
		return f		


	def normalize(self, windows ,istrain):
        return windows