import pandas as pd
import io
class Data:

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return '<' + self.name + '> ' + str(self.__dict__)
    
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