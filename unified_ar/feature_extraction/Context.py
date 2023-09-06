from .feature_abstract import FeatureExtraction
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


class Diff(FeatureExtraction):

    def makeFeatureMapper(self, ds):
        from sklearn.preprocessing import MinMaxScaler

        nominal = [ds.sensor_id_map_inverse[x] for x in ds.sensor_desc[ds.sensor_desc['Nominal'] == 1].index]

        numtmp = {ds.sensor_id_map_inverse[x]: ds.sensor_desc.loc[x]['ItemRange']
                  for x in ds.sensor_desc[ds.sensor_desc['Nominal'] == 0 & (ds.sensor_desc['Cumulative'] >= 0)].index}
        numtmp = {x: numtmp[x] for x in sorted(numtmp.keys())}
        numeric_data = pd.DataFrame(numtmp)
        minmaxscaler = MinMaxScaler()
        # minmaxscaler.fit([numeric_data.loc['min'].values,numeric_data.loc['max'].values])

        nominal_categories = {ds.sensor_id_map_inverse[k]: ds.sensor_desc_map[k].keys() for k in ds.sensor_desc_map}
        nominal_categories = [list(nominal_categories[k]) for k in sorted(nominal_categories.keys())]

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
        )

        categorical_transformer = OneHotEncoder(categories=nominal_categories, handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", minmaxscaler, [k for k in numtmp]),
                ("cat", categorical_transformer, nominal),
            ]
        )
        z = np.zeros((2, len(ds.sensor_id_map)))
        for i in numeric_data.columns:
            z[0][i] = numeric_data.loc['min'][i]
            z[1][i] = numeric_data.loc['max'][i]

        preprocessor.fit(z)
        shape = preprocessor.transform(z).shape[1]
        return preprocessor, shape

    def precompute(self, datasetdscr, windows):

        self.currentState = np.zeros((len(datasetdscr.sensor_id_map)))

        self.transformer, flen = self.makeFeatureMapper(datasetdscr)
        self.shape = (2*flen,)

        self.currentId = 0

        super().precompute(datasetdscr, windows)

    def featureExtract2(self, s_event_list, idx):
        window = s_event_list
        if idx[0] < self.currentId:
            raise Error('error idx is not sequentioal')

        while self.currentId < idx[0]:
            self.currentState[self.datasetdscr.sensor_id_map_inverse[window[self.currentId, 0]]] = window[self.currentId, 2]
            self.currentId += 1

        newState = self.currentState.copy()
        nid = self.currentId
        while nid < idx[1]:
            newState[self.datasetdscr.sensor_id_map_inverse[window[nid, 0]]] = window[nid, 2]
            nid += 1

        return np.concatenate([self.transformer.transform([self.currentState]), self.transformer.transform([newState])], axis=None)
