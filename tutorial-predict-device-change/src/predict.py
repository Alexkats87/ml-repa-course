import pandas as pd
import argparse
from typing import Text

from src.data_load import DataLoader
from src.featurize import FeatureCollector
from src.train import Model

from src.utils.logging import get_logger


logger = get_logger("PREDICT")


def predict(config_path):

    data_loader = DataLoader(config_path=config_path)
    test_x = data_loader.load_test_x()

    featurizer = FeatureCollector(config_path=config_path)
    featurizer.create_features_test(test_x)

    model = Model(config_path=config_path)
    model.predict()


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    predict(config_path=args.config)
