from activity_fetcher.activity_fetcher_abstract import AbstractActivityFetcher
class CookActivityFetcher(AbstractActivityFetcher):
    def getActivity(self,dataset,window):
        end=window.iat[window.shape[0]-1,1].value
        acts=(self.a_events_tree[end])
        if(len(acts)==0):
            acts=(self.a_events_tree[end-1:end])
            if(len(acts)==0):
                return 0
            #print('new Fetcher-------')
        return next(iter(acts)).data.Activity  

    def getActivity2(self,dataset,idx):
        window=dataset
        end=window[idx[-1],1].value
        acts=(self.a_events_tree[end-1:end])
        if(len(acts)==0):
            return 0
        return acts.pop().data.Activity
        # return next(iter(acts)).data.Activity