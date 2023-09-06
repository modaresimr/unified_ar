import pandas as pd




def periodic_spline_transformer(period, n_splines=None, degree=3):
    from sklearn.preprocessing import SplineTransformer
    import numpy as np
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )
    


from sklearn.base import TransformerMixin
class TimeSpliter(TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dfs = []
        self.column_names = []
        for column in X:
            dt = X[column].dt
            # Assign custom column names
            newcolumnnames = [column+'_'+col for col in ['year','month','day', 'dayofweek','hour','minute','second']]
            df_dt = pd.concat([dt.year,dt.month,dt.day, dt.dayofweek,dt.hour,dt.minute,dt.second], axis=1)
            # Append DF to list to assemble list of DFs
            df_dt.columns=newcolumnnames
            dfs.append(df_dt)
            # Append single DF's column names to blank list
            self.column_names.append(newcolumnnames)
        # Horizontally concatenate list of DFs
        dfs_dt = pd.concat(dfs, axis=1)
        return dfs_dt

    def get_feature_names(self):
        # Flatten list of column names
        self.column_names = [c for sublist in self.column_names for c in sublist]
        return self.column_names
