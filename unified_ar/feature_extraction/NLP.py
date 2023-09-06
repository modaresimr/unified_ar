from .feature_abstract import FeatureExtraction
import numpy as np


class SensorWord(FeatureExtraction):

    def precompute(self, datasetdscr, windows):
        self.max_win = 50  # max([len(w) for w in windows])
        # self.max_win = max([len(w) for w in windows])
        from tensorflow.keras.preprocessing.text import Tokenizer
        self.tokenizer = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n', num_words=1000)  # , oov_token='other')
        items = datasetdscr.sensor_events['SID'].astype(str)+datasetdscr.sensor_events['value'].astype(str).values
        self.tokenizer.fit_on_texts(items)
        # print(f'precompute finished dictsize={len(self.tokenizer.word_index)}')
        self.shape = (self.max_win,)

        super().precompute(datasetdscr, windows)

    def prepareData(self, s_event_list):

        pass

    def featureExtract2(self, s_event_list, idx):
        window = s_event_list
        f = np.zeros(self.shape)

        # for i in range(len(idx)-1, max(-1, len(idx)-self.max_win-1), -1):
        sindx = max(0, len(idx)-self.max_win)
        eindx = len(idx)
        t = max(0, self.max_win-eindx-sindx)
        for i in range(sindx, eindx):
            sname = window[idx[i], 0]
            # svalue = int(window[idx[i], 2])
            svalue = window[idx[i], 2]

            # wordidx = self.tokenizer.texts_to_sequences([f'{sname}{svalue}'])

            # print(f'{sname}-{int(svalue)}===>{wordidx}')
            # return
            # f[wordidx][t] = 1
            # f[t] = wordidx[0][0]
            # f[t] = self.tokenizer.word_index.get(f'{sname}{svalue}'.lower(), 1)
            f[t] = self.tokenizer.word_index.get(f'{sname}{svalue}'.lower(), 0)
            if f[t] > 1000:
                f[t] = 0
            # if f[t] > 0:
            t += 1

            # f[self.datasetdscr.sensor_id_map_inverse[window[idx[i], 0]]] = 1  #f[sensor_id_map_inverse[x.SID]]=1
        return f

    def normalize(self, windows, istrain):
        return super().normalize(windows, istrain)


class SensorWordNormal(FeatureExtraction):

    def precompute(self, datasetdscr, windows):
        wordidx = 0
        self.max_win = max([len(w) for w in windows])
        self.wordmap = {}
        for s in datasetdscr.sensor_desc_map:
            items = datasetdscr.sensor_desc_map[s]
            for r in items:
                wordidx += 1
                self.wordmap[f'{s}-{r}'] = wordidx
        wordidx += 1
        self.wordmap[f'other'] = wordidx
        display(f'wordmap={self.wordmap}')
        self.shape = (self.max_win,)

        super().precompute(datasetdscr, windows)

    def prepareData(self, s_event_list):

        pass

    def featureExtract2(self, s_event_list, idx):
        window = s_event_list
        f = np.zeros(self.shape)
        for t, i in enumerate(range(len(idx)-1, max(-1, len(idx)-self.max_win-1), -1)):
            sname = window[idx[i], 0]
            svalue = int(window[idx[i], 2])

            wordidx = self.wordmap.get(f'{sname}-{svalue}', self.wordmap['other'])

            # print(f'{sname}-{int(svalue)}===>{wordidx}')
            # return
            # f[wordidx][t] = 1
            f[t] = wordidx
            # f[self.datasetdscr.sensor_id_map_inverse[window[idx[i], 0]]] = 1  #f[sensor_id_map_inverse[x.SID]]=1
        return f

    def normalize(self, windows, istrain):
        return super().normalize(windows, istrain)
