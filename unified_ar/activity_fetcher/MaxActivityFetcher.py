from .activity_fetcher_abstract import AbstractActivityFetcher
import numpy as np


class MaxActivityFetcher(AbstractActivityFetcher):
    def getActivity(self, dataset, window):
        start = window.iat[0, 1].value  # .first.time
        end = window.iat[window.shape[0]-1, 1].value
        acts = (dataset.a_events_tree[start:end])

        if (len(acts) == 0):
            return 0
        pacts = []
        nacts = []
        for x in acts:
            pacts.append((min(x.end, end)-max(x.begin, start))/(end-start))
            nacts.append(x.data)
        # np.argmax(pacts)
        return nacts[np.argmax(pacts)].Activity

    def getActivity2(self, dataset, idx):
        window = dataset
        start = window[idx[0], 1].value
        end = window[idx[-1], 1].value
        racts = (self.a_events_tree[start:end])
        if (len(racts) == 0):
            return 0
        pacts = np.zeros(len(self.acts))
        for x in racts:
            pacts[x.data.Activity] += (min(x.end, end)-max(x.begin, start))/(end-start)
        # np.argmax(pacts)
        return np.argmax(pacts)
        # return next(iter(acts)).data.Activity

    def getActivity2old(self, dataset, idx):
        window = dataset
        start = window[idx[0], 1].value
        end = window[idx[-1], 1].value
        acts = (self.a_events_tree[start:end])
        if (len(acts) == 0):
            return 0
        pacts = np.zeros(len(acts))
        nacts = np.zeros(len(acts))
        i = 0
        for x in acts:
            pacts[i] = (min(x.end, end)-max(x.begin, start))/(end-start)
            nacts[i] = x.data.Activity
            i += 1
        # np.argmax(pacts)
        return int(nacts[np.argmax(pacts)])
        # return next(iter(acts)).data.Activity
