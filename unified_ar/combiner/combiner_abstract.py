from general.utils import MyTask

class Combiner(MyTask):
    def combine(self, s_event_list,set_window, act_data):
        times=[]

        for i in range(0, len(set_window)):
            idx     = set_window[i]
            start   = s_event_list[idx[0],1]
            end     = s_event_list[idx[-1],1]
            times.append({'end':end,'begin':start})
            
        return  self.combine2(times,act_data)

    def combine2(self,times,act_data):
        pass