import pandas as pd
from datetime import *

from memory_profiler import profile


@profile
def load_dataframe(path, **kwargs):
    print(f'Reading file {path} started at {datetime.now()}')
    df = pd.read_csv(path, **kwargs)
    if 'chunksize' in kwargs.keys():
        df = pd.concat(df, ignore_index=True)
    print(f'Reading file {path} finished at {datetime.now()}')
    return df
