from segmentation.segmentation_abstract import Segmentation



class Probabilistic_old(Segmentation):
    def precompute(self,datasetdscr,s_events,a_events,acts):
        ws= a_events.groupby('Activity')['Duration'].mean(numeric_only=False)
        
       
        L=10
        w={}
        w[0]=min(ws)
        w[L]=ws.mean()
        for l in range(1,L):
            w[l]=(w[L]-w[0])* l/L+w[0]
        buffer=Buffer(s_events,0,0)  
        
        a_s={sid:{a:0 for a in [-1]+acts} for sid,desc in sensor_desc.iterrows()}
        
        w_a={a:{i:0 for i in range(-1,L+1)} for a in acts}
        for i,a in a_events.iterrows():
            ##print(i)
            six=buffer.searchTime(a.StartTime,-1)
            eix=buffer.searchTime(a.EndTime,+1)
            #window=buffer.data.SID.iloc[six:eix+1];
            for si in range(six,eix+1):
                s=buffer.data.SID.iat[si]
                a_s[s][a.Activity]+=1
                a_s[s][-1]+=1
                
            for l in range(L+1):
                w_a[a.Activity][l]+=1-min(1,abs(w[l]-a.Duration)/w[l])
            w_a[a.Activity][-1]+=1
                
        Pw_a={a:{i:0 for i in range(L+1)} for a in acts}
        Pa_s={sid:{a:0 for a in acts} for sid,desc in sensor_desc.iterrows()}
        for a in acts:
            for l in range(L+1):
                Pw_a[a][l]=0 if w_a[a][-1]==0 else w_a[a][l]/w_a[a][-1]
            for sid in a_s:
                Pa_s[sid][a]=0 if a_s[sid][-1]==0 else a_s[sid][a]/a_s[sid][-1]
            
        Pw_s={sid:{i:0 for i in range(L+1)} for sid,desc in sensor_desc.iterrows()}
        w_s={sid:{i:0 for i in range(L+1)} for sid,desc in sensor_desc.iterrows()}
        for sid,desc in sensor_desc.iterrows():
            for l in range(L+1):
                a=argmaxdic(Pa_s[sid])
                Pw_s[sid][l]=Pa_s[sid][a]*Pw_a[a][l]
            w_s[sid]=argmaxdic(Pw_s[sid])
        self.w_s=w_s
        self.w=w
        self.Pw_a=Pw_a
        self.Pw_s=Pw_s
        self.Pa_s=Pa_s
        self.w_a=w_a
        self.a_s=a_s

        
    def reset(self):
        self.lastindex=-1   
    
    def segment(self,w_history,buffer):   
        
        sindex=self.lastindex+1
        self.lastindex=sindex
        if(sindex >=len(buffer.data)):
            return None
        sensor=buffer.data.iloc[sindex]
        stime=buffer.times[sindex]
        
        sindex=buffer.searchTime(stime,-1)
        
        size=self.w[self.w_s[sensor.SID]]
        etime=stime+size
        eindex=buffer.searchTime(etime,+1)
        #etime=buffer.times[eindex]
        
        window=buffer.data.iloc[sindex:eindex+1];
        #buffer.removeTop(sindex)
        window.iat[0,1].value
        return {'window':window,'start':stime, 'end':etime}
            
    
