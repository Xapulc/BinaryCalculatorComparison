import os
import pandas as pd

from pathlib import Path


def save_file(data, file_name, dir_list=None):
    if dir_list is None:
        dir_path = "."
    else:
        dir_path = os.path.join(*dir_list)
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(dir_path, file_name)
    data.to_csv(file_path, index=True)


def load_file(file_name, dir_list=None, index_col_list=None):
    if dir_list is None:
        dir_path = "."
    else:
        dir_path = os.path.join(*dir_list)

    file_path = os.path.join(dir_path, file_name)

    if Path(file_path).exists():
        return pd.read_csv(file_path, index_col=index_col_list)
    else:
        return None
