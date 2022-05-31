import pandas as pd
import argparse
from typing import Text

from src.utils.logging import get_logger
from src.utils.helpers import get_config


logger = get_logger("DATA_LOAD")


class DataLoader:

    def __init__(self, config_path: Text):

        config = get_config(config_path)
        logger.setLevel(config.Base.log_level)
        self.train_y_path = config.Data.target_raw
        self.train_x_path = config.Data.user_features_raw
        self.test_x_path = config.Data.scoring_user_features_raw

    @staticmethod
    def _load_x(path) -> pd.DataFrame:
        x_df = pd.read_feather(path)
        x_df['month'] = pd.to_datetime(x_df['month'])
        return x_df

    def load_train_x(self) -> pd.DataFrame:
        logger.info("Load raw data for train..")
        return self._load_x(path=self.train_x_path)

    def load_test_x(self) -> pd.DataFrame:
        logger.info("Load raw data for prediction..")
        return self._load_x(path=self.test_x_path)

    def load_train_y(self) -> pd.DataFrame:
        logger.info("Load raw target for train..")
        y_df = pd.read_feather(self.train_y_path)
        y_df['month'] = pd.to_datetime(y_df['month'])
        return y_df

    def load_train_xy(self) -> pd.DataFrame:
        train_y_df = self.load_train_y()
        train_x_df = self.load_train_x()

        train_xy_df = pd.merge(
            left=train_x_df,
            right=train_y_df,
            how='left',
            on=['user_id', 'month']
        )

        logger.info("Train_XY loaded")

        return train_xy_df




