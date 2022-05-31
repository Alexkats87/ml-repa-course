import pandas as pd
import joblib
import json
from typing import Text
import argparse

from catboost import CatBoostClassifier

from src.utils.helpers import custom_ts_split, get_config
from src.utils.metrics import precision_at_k_score, recall_at_k_score, lift_score
from src.utils.logging import get_logger


logger = get_logger("MODEL_TRAIN")


class Model:
    def __init__(self, config_path: Text):

        config = get_config(config_path)
        logger.setLevel(config.Base.log_level)

        self.model_params = config.Train.model_params
        self.model_path = config.Train.model_path
        self.eval_metrics_path = config.Train.train_metrics

        self.train_data_path = config.Features.features_path
        self.test_data_path = config.Features.scoring_features_path
        self.predicted_target_path = config.Features.predicted_target_path

        self.drop_col = ['user_id', 'month']
        self.target_col = 'target'
        self.clf = None

        self.random_seed = config.Base.random_state
        self.top_k_coef = config.Train.top_K_coef

    @staticmethod
    def load_data(path):
        return pd.read_feather(path)

    @staticmethod
    def save_data(df, path):
        df.to_feather(path)

    def predict(self):

        logger.info('Load features for prediction..')
        test_x_df = self.load_data(self.test_data_path)

        logger.info('Load and apply model..')
        clf = joblib.load(self.model_path)
        probas_scoring = clf.predict_proba(test_x_df.drop(columns=self.drop_col, axis=1))

        test_x_df['target_proba'] = probas_scoring[:, 1]
        test_y_df = test_x_df[['user_id', 'month', 'target_proba']].copy()

        logger.info('Save prediction prediction result ..')
        self.save_data(test_y_df, path=self.predicted_target_path)
        logger.info('Done')

    def evaluate_model(self):

        logger.info("Start training model..")

        metrics = ['lift', 'precision_at_k', 'recall_at_k']
        metrics_df = pd.DataFrame(columns=['test_period'] + metrics)

        train_df = self.load_data(self.train_data_path)
        top_k = int(train_df.shape[0] * self.top_k_coef)
        months = train_df.month.sort_values().unique()

        clf = CatBoostClassifier(**self.model_params, random_seed=self.random_seed)
        k = 1

        for start_train, end_train, test_period in custom_ts_split(months, train_period=1):
            logger.info(f'Fold {k}:')
            logger.info(f'Train: {start_train} - {end_train}')
            logger.info(f'Test: {test_period} \n')

            # Get train / test data for the split
            x_train = (train_df[(train_df.month >= start_train) & (train_df.month <= end_train)]
                       .drop(columns=self.drop_col + [self.target_col], axis=1))

            x_test = (train_df[(train_df.month == test_period)]
                      .drop(columns=self.drop_col + [self.target_col], axis=1))

            y_train = train_df.loc[(train_df.month >= start_train) & (train_df.month <= end_train), self.target_col]
            y_test = train_df.loc[(train_df.month == test_period), self.target_col]

            logger.info(f'Train shapes: X - {x_train.shape}, y - {y_train.shape}')
            logger.info(f'Test shapes: X - {x_test.shape}, y - {y_test.shape}')

            # Fit estimator
            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)
            probas = clf.predict_proba(x_test)
            logger.info(f'Max probas: {probas[:, 1].max()}')

            print(y_test.shape, y_pred.shape, probas[:, 1].shape)

            lift = lift_score(y_test, y_pred, probas[:, 1], top_k)
            precision_at_k = precision_at_k_score(y_test, y_pred, probas[:, 1], top_k)
            recall_at_k = recall_at_k_score(y_test, y_pred, probas[:, 1], top_k)

            metrics_df = metrics_df.append(
                dict(zip(metrics_df.columns, [test_period, lift, precision_at_k, recall_at_k])),
                ignore_index=True
            )

            k += 1

            logger.info(f'Precision at {top_k}: {precision_at_k}')
            logger.info(f'Recall at {top_k}: {recall_at_k}')
            # plot_lift_curve(y_test[:top_k], y_pred[:top_k], step=0.1)
            logger.info('\n')

        joblib.dump(clf, self.model_path)
        self.clf = clf
        logger.info("Model trained and saved")

        metrics_aggs = metrics_df[metrics].agg(['max', 'min', 'std', 'mean'])
        metrics = {
            f'{metric}_{agg}': metrics_aggs.loc[agg, metric]
            for metric in metrics_aggs.columns
            for agg in metrics_aggs.index
        }

        with open(self.eval_metrics_path, 'w') as metrics_f:
            json.dump(obj=metrics, fp=metrics_f, indent=4)

        logger.info("Metrics saved")
        logger.info(metrics_aggs)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    model = Model(config_path=args.config)
    model.evaluate_model()


