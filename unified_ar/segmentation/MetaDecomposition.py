import unified_ar.general.Cache as Cache
import unified_ar.general.utils as utils
from .segmentation_abstract import Segmentation
import pandas as pd
import logging
logger = logging.getLogger(__file__)


class SWMeta(Segmentation):
    def applyParams(self, params):
        if not super().applyParams(params):
            return False
        self.meta_size = pd.to_timedelta(params['meta_size'])
        self.meta_step_period = self.meta_size * params['meta_overlap_rate']
        self.meta_mode = params['meta_mode']
        return True

    def customSplit(self, sensor_events, activity_events, start, end):
        Train0 = utils.Data(f'train_random_days {start}-{end}')
        Train0.s_events = sensor_events.loc[(sensor_events.time >= start) & (sensor_events.time < end)]
        act_filter = [] if activity_events is None else (activity_events.StartTime >= start) & (activity_events.StartTime < end)
        Train0.a_events = activity_events.loc[act_filter]
        Train0.s_event_list = Train0.s_events.values
        return Train0

    def createMetaDataset(self, datasetdscr, s_events, a_events, acts):
        Cache.GlobalDisable = True
        meta_features = []
        meta_targets = []
        from unified_ar.constants import methods
        def_segments = methods.segmentation
        self.segmentor_dic = {p['method']().shortname(): p for p in methods.meta_segmentation_sub_tasks}
        import unified_ar.ml_strategy.Simple
        import logging

        fast_strategy = unified_ar.ml_strategy.Simple.NormalStrategy()

        selectedSeg = methods.meta_segmentation_sub_tasks[0].copy()
        selectedSeg['findopt'] = False
        methods.segmentation = [selectedSeg]
        logger.debug('train using all the data with the first segment method')
        logger.debug(f'method={selectedSeg["method"]().shortname()}')
        seg_params = [f"{p['var']}:{p['init']}" for p in selectedSeg["params"]]
        logger.debug(f'params={seg_params}')
        fast_strategy.ui_debug = {'seg': 1}
        Train0 = utils.Data(f'train_all_days')
        Train0.s_events = s_events
        Train0.s_events_list = s_events.values
        Train0.a_events = a_events
        def_run_names = methods.run_names.copy()

        methods.run_names['out'] = f"{def_run_names['out']}/base"
        result = fast_strategy.train(datasetdscr, Train0, acts, update_model=False)
        methods.run_names['meta_base'] = methods.run_names['out']

        logger.debug(f'result of full data training {result}')
        logger.debug(fast_strategy.get_info().functions)

        logger.debug(f'now finding the best seg method for each meta segmets size={self.meta_size} step={self.meta_step_period}')
        # logging.getLogger().setLevel(logging.INFO)
        logging.getLogger().handlers[1].setLevel(logging.INFO)
        fast_strategy.ui_debug = {'seg': 0}
        methods.segmentation = methods.meta_segmentation_sub_tasks

        starts = s_events.time.dt.floor(self.meta_step_period).unique()
        ends = starts + self.meta_size

        for s, e in zip(starts, ends):
            methods.run_names['out'] = f"{def_run_names['out']}/{s}-{e}"
            data2 = self.customSplit(s_events, a_events, s, e)
            logger.debug(f's={s} : {e}============= #sevent={len(data2.s_events)} #aevents={len(data2.a_events)}')
            if len(data2.s_events) == 0 or len(data2.a_events) == 0:
                continue

            result = fast_strategy.train(datasetdscr, data2, acts, update_model=True)
            if result is None:
                continue
            logger.debug(fast_strategy.get_info().functions)
            # result=fast_strategy.pipeline(fast_strategy.functions,self.traindata,train=True)
            fea = {'start_time': s, 'end_time': e}
            # aggr = data2.s_events.groupby('SID').count()
            # fea = {k: aggr.loc[k]['value'] if k in aggr.index else 0 for k in datasetdscr.sensor_id_map_inverse}
            # fea['time'] = s
            meta_features.append(fea)
            seg = result.functions['segmentor']
            meta_targets.append({'method': seg[0], **result.quality, **{p: seg[1][p] for p in seg[1]}})
            # self.strategy=fast_strategy

        methods.segmentation = def_segments
        methods.run_names = def_run_names
        # logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger().handlers[1].setLevel(logging.DEBUG)
        meta_dataset = {'meta_features': pd.DataFrame(meta_features), 'meta_targets': pd.DataFrame(meta_targets)}

        utils.saveState(meta_dataset, f"meta_dataset/{methods.run_names['out']}_{methods.run_names['fold']}")
        return meta_dataset

    def create_train_model(self, meta_dataset):
        feat_df = meta_dataset['meta_features']
        target_df = meta_dataset['meta_targets']

        ntarget = target_df.drop(['accuracy', 'precision', 'recall', 'f1'], axis=1)

        self.targetTransformer = MyTargetTransformer(self.meta_mode)
        self.targetTransformer.fit(ntarget)
        self.featTransformer = self.create_feat_transformer(feat_df)
        self.featTransformer.fit(feat_df)
        logger.debug(f"metainfo=============== \nfeat={feat_df}\n target={target_df}")
        X = self.featTransformer.transform(feat_df)
        y = self.targetTransformer.transform(ntarget)
        if self.meta_mode == 'keras':
            from tensorflow import keras
            from tensorflow.keras import layers
            inputs = keras.Input(shape=(X.shape[1],))
            layer1 = layers.Dense(16, activation='relu')(inputs)
            layer2 = layers.Dense(8, activation='relu')(layer1)
            layer3 = layers.Dense(16, activation='relu')(layer2)
            classifier = layers.Dense(1, activation='softmax', name='method')(layer3)
            regressions = [layers.Dense(1, activation='linear', name=x)(layer3) for x in ntarget.columns.drop('method')]

            mdl = keras.Model(inputs=inputs, outputs=[classifier, *regressions])

            mdl.compile(loss=['categorical_crossentropy', *(['mse'] * len(regressions))], optimizer='adam', metrics=['accuracy'])
        else:
            from sklearn.svm import SVR
            from sklearn.multioutput import MultiOutputRegressor
            mdl = MultiOutputRegressor(SVR())
        mdl.fit(X, y)
        return mdl

    def precompute(self, datasetdscr, s_events, a_events, acts):
        self.datasetdscr = datasetdscr

        if 1:
            meta_dataset = self.createMetaDataset(datasetdscr, s_events, a_events, acts)
        else:
            meta_dataset = utils.loadState(
                "meta_dataset 220602_23-52-11-A4H-Namespace(classifier=0, comment='0', dataset=3, evaluation=0, feature_extraction=0, mlstrategy=0, output='logs', segmentation=0) 0")

        self.ml = self.create_train_model(meta_dataset)

    def create_feat_transformer(self, feat):
        import numpy as np
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from feature_extraction.features import TimeSpliter, periodic_spline_transformer

        numeric_features = feat.columns.drop(['time'])
        numeric_transformer = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ('date', make_pipeline(TimeSpliter(), ColumnTransformer(
                    transformers=[
                        ("weekday", periodic_spline_transformer(7, n_splines=3), ["time_dayofweek"]),
                        ("month", periodic_spline_transformer(12, n_splines=6), ["time_month"]),
                    ]
                )), ['time']),
            ]
        )

        return preprocessor
        # y=target.loc[target['method']=='FixedSlidingWindow'][['size','shift']]
        # X=feat.loc[target['method']=='FixedSlidingWindow']

    def reset(self):
        self.lastindex = -1
        # self.ml=None
        self.meta_predict = None
        self.last_meta_index = None
        # self.targetTransformer=None
        # self.featTransformer=None
        self.last_segmentor = None

    def prepare_meta_analysis(self, buffer):
        s_events = buffer.data
        starts = s_events.time.dt.floor(self.meta_step_period).unique()
        ends = starts + self.meta_size

        meta_features = []
        ranges = []
        for s, e in zip(starts, ends):
            data2 = self.customSplit(s_events, None, s, e)
            if len(data2.s_events) == 0:
                continue
            ranges.append([s, e])
            aggr = data2.s_events.groupby('SID').count()
            fea = {k: aggr.loc[k]['value'] if k in aggr.index else 0 for k in self.datasetdscr.sensor_id_map_inverse}
            fea['time'] = s
            meta_features.append(fea)

        feat_df = pd.DataFrame(meta_features)
        X = self.featTransformer.transform(feat_df)
        y2 = self.ml.predict(X)
        y = self.targetTransformer.inverse_transform(y2)
        ranges_df = pd.DataFrame(ranges, columns=['start', 'end'])
        self.meta_predict = pd.concat([ranges_df, y], axis=1)
        print(self.meta_predict)

    def segment2(self, w_history, buffer):
        params = self.params
        if self.meta_predict is None:
            self.prepare_meta_analysis(buffer)

        if len(w_history) == 0:
            self.last_meta_index = 0
            lastStart = buffer.data.iloc[0]['time']
        else:
            #   print(w_history)
            lastStart = buffer.times[w_history[len(w_history) - 1][0]]

        while 1:
            prd = self.meta_predict.iloc[self.last_meta_index]
            if self.last_segmentor is None:

                # from segmentation.FixedSlidingWindow import FixedSlidingWindow
                seg_info = self.segmentor_dic[prd['method']]
                self.last_segmentor = seg_info['method']()

                prd2 = prd.drop(['start', 'end']).to_dict()
                print(prd2)
                if not self.last_segmentor.applyParams(prd2):
                    # self.last_segmentor.applyParams(seg_info[]
                    if prd['method'] == 'FixedSlidingWindow':
                        prd2['size'] = 30
                        prd2['shift'] = 15
                    else:
                        prd2['size'] = 10
                        prd2['shift'] = 10
                    print(f'predicted segment is not valid.. choose default{prd2}')
                    self.last_segmentor.applyParams(prd2)
            if prd['start'] <= lastStart and prd['end'] > lastStart:
                break

            self.last_meta_index += 1
            self.last_segmentor = None

        return self.last_segmentor.segment2(w_history, buffer), None


class MyTargetTransformer:
    def __init__(self, mode):
        self.mode = mode
    is_fit = False

    def fit(self, y):
        import pandas as pd
        from sklearn import preprocessing

        self.is_fit = True
        self.trans = {'method': preprocessing.LabelEncoder(), 'other': preprocessing.StandardScaler()}

        self.trans['method'].fit(y['method'])
        self.trans['other'].fit(y.drop(['method'], axis=1))
        self.columns = y.columns

    def transform(self, y):
        if not self.is_fit:
            raise Error('not fit')

        y1 = self.trans['method'].transform(y['method'])

        y2 = self.trans['other'].transform(y.drop(['method'], axis=1))
        if self.mode == 'keras':
            res = [y1.reshape(-1, 1)]
            for i, c in enumerate(self.columns.drop(['method'])):
                res.append(y2[:, i].reshape(-1, 1))
            return res
        else:
            return y2
    #         return [self.trans[c].transform(y[[c]]) for c in self.trans]

    def inverse_transform(self, y):
        import numpy as np
        if not self.is_fit:
            raise Error('not fit')
        if self.mode == 'keras':
            y2 = np.hstack(y)
            y2[:, 1:] = self.trans['other'].inverse_transform(y2[:, 1:])
            df = pd.DataFrame(y2)
            df.columns = self.columns
            df['method'] = self.trans['method'].inverse_transform(y2[:, 0].astype(int) * 0)
            # df['method']='s'
            return df
        else:
            df = pd.DataFrame(self.trans['other'].inverse_transform(y))
            df.columns = self.columns.drop(['method'])
            return df
