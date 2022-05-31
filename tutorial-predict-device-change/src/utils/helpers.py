from pandas.tseries.offsets import MonthEnd
import pandas as pd
from box import Box
from typing import Text


def get_config(config_file_path: Text):
    with open(config_file_path) as f:
        config = Box.from_yaml(f)
    return config


def custom_ts_split(months, train_period=0):
    for k, month in enumerate(months):

        start_train = pd.to_datetime(months.min())
        end_train = pd.to_datetime(start_train) + MonthEnd(train_period - 1 + k)

        test_period = pd.to_datetime(end_train + MonthEnd(1))

        if test_period <= pd.to_datetime(months.max()):

            yield start_train, end_train, test_period

        else:
            print(test_period)
            print(months.max())