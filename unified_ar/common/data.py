import pandas as pd
from collections.abc import Sequence
import numpy as np
import io
class Data:

    def __init__(self, name):
        self.name = name

    def __str__(self):
        out=[]
        for d, v in self.__dict__.items():
            if d.startswith("_"):continue
            if isinstance(v, pd.DataFrame):
                out.append(f'{d}:[{v.columns}]')
            elif isinstance(v, np.ndarray):
                out.append(f'{d}: {type(v).__name__}({v.shape})')
            elif isinstance(v, Sequence):
                out.append(f'{d}: {type(v).__name__}({len(v)})')
            else:
                out.append(f'{d}: {v}')
        
        return f'<{self.name} ' + ' '.join(out)+">"

    # def __repr__(self):
    #     return '<' + self.name + '> ' + repr(self.__dict__)
    
    def __reduce__(self):
        """Return state information for pickling"""
        state = self.__dict__.copy()
        hdf_dict = {}
        for key, value in state.items():
            if isinstance(value, pd.DataFrame):
                hdf_buffer = io.BytesIO()
                value.to_pickle(hdf_buffer)
                hdf_dict[key] = hdf_buffer.getvalue()
                
        for key in hdf_dict:
            del state[key]
        state['hdf_dict'] = hdf_dict
        # print("==================getstate",state)
        return (self.__class__, (self.name,), state)

    def __setstate__(self, state):
        """Restore state from the unpickling operation"""
        # print(state)
        hdf_dict = state.pop('hdf_dict', {})
        # print("==================setstate",hdf_dict)
        for key, hdf_serialized in hdf_dict.items():
            hdf_buffer = io.BytesIO(hdf_serialized)
            setattr(self, key, pd.read_pickle(hdf_buffer))
        self.__dict__.update(state)