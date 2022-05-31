import pandas as pd
import argparse
from typing import Text

from src.data_load import DataLoader
from src.utils.logging import get_logger
from src.utils.helpers import get_config


logger = get_logger("FEATURES")


class FeatureCollector:

    def __init__(self, config_path: Text):

        config = get_config(config_path)
        logger.setLevel(config.Base.log_level)
        self.train_x_path = config.Features.features_path
        self.test_x_path = config.Features.scoring_features_path

    @staticmethod
    def drop_nulls(df) -> pd.DataFrame:
        df = df.dropna()
        return df

    @staticmethod
    def save_featurized_data(df, path) -> None:
        df.to_feather(path)

    def create_features(self, df, path) -> None:
        df = self.drop_nulls(df)

        # ...
        # add other features here
        # ...

        self.save_featurized_data(df, path)

    def create_features_train(self, df):
        logger.info("Create train features...")
        self.create_features(df, path=self.train_x_path)
        logger.info("Done")

    def create_features_test(self, df):
        logger.info("Create test features...")
        self.create_features(df, path=self.test_x_path)
        logger.info("Done")


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_loader = DataLoader(config_path=args.config)
    train_xy = data_loader.load_train_xy()

    featurizer = FeatureCollector(config_path=args.config)
    featurizer.create_features_train(train_xy)




