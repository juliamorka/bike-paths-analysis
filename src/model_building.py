import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split


class ModelBuilder(ABC):
    """
    Abstract base class for building and evaluating machine learning models.

    Attributes:
        data (pd.DataFrame or None): The dataset used for modeling.
        dataset_name (str or None): Name of the dataset.
        predictors (list of str): Features used as predictors.
        target_feature (str): Target feature to predict.
        train_set (pd.DataFrame or None): Training dataset.
        test_set (pd.DataFrame or None): Testing dataset.
        validation_set (pd.DataFrame or None): Validation dataset.
        model (sklearn estimator or None): Machine learning model instance.
        model_build (sklearn estimator or None): Built machine learning model.
        hyperparams (dict): Hyperparameters for the model.
        use_validation_set (bool): Flag indicating whether a validation set is used.
        predictions (list): Predicted values on the test set.
        metrics_values (pd.DataFrame or None): DataFrame to store evaluation metrics.
        save_dir (str): Directory path to save model artifacts.
        log_mlflow (bool): Flag to enable MLflow logging.
        scaler (object or None): Scaler object for data normalization.
        timestamp (str): Current timestamp for folder naming and logging.
    """

    @abstractmethod
    def __init__(
        self,
        predictors,
        target_feature,
        save_dir="",
        log_mlflow=False,
        experiment_name="",
    ):
        """
        Initialize a ModelBuilder object.

        Args:
            predictors (list of str): Features used as predictors.
            target_feature (str): Target feature to predict.
            save_dir (str, optional): Directory path to save model artifacts. Defaults to "".
            log_mlflow (bool, optional): Flag to enable MLflow logging. Defaults to False.
            experiment_name (str, optional): Name of the MLflow experiment. Defaults to "".
        """
        self.data = None
        self.dataset_name = None
        self.predictors = predictors
        self.target_feature = target_feature
        self.train_set = None
        self.test_set = None
        self.validation_set = None
        self.model = None
        self.model_build = None
        self.hyperparams = {}
        self.use_validation_set = False
        self.predictions = []
        self.metrics_values = None
        self.save_dir = save_dir
        self.log_mlflow = log_mlflow
        self.scaler = None
        self.timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.log_mlflow:
            mlflow.set_experiment(f"{experiment_name}_{self.timestamp}")

    def set_model(self, model):
        """
        Set the machine learning model to be used.

        Args:
            model (sklearn estimator): Machine learning model instance.
        """
        self.model = model

    def set_data(self, data, dataset_name: str):
        """
        Set the dataset and its name.

        Args:
            data (pd.DataFrame): Dataset to use for modeling.
            dataset_name (str): Name of the dataset.
        """
        self.data = data.copy()
        self.dataset_name = dataset_name

    def split_data(self, test_size=0.2, validation_set=False):
        """
        Split the dataset into training, testing, and optionally validation sets.

        Args:
            test_size (float, optional): Fraction of the dataset to be used as the test set.
                Defaults to 0.2.
            validation_set (bool, optional): Whether to use a validation set. Defaults to False.
        """
        if test_size == 1:
            self.test_set = self.data
        else:
            self.train_set, self.test_set = train_test_split(
                self.data, test_size=test_size, random_state=42
            )
            if validation_set:
                self.train_set, self.validation_set = train_test_split(
                    self.train_set,
                    test_size=test_size / (1 - test_size),
                    random_state=42,
                )
                self.use_validation_set = True

    def build_model(self):
        """
        Build the machine learning model on training set using class configuration.
        """
        self.model_build = self.model(**self.hyperparams)
        self.model_build.fit(
            self.train_set[self.predictors],
            self.train_set[self.target_feature],
        )

    def predict(self):
        """
        Generate predictions on the test set using the already built model
        and save them to .parquet.
        """
        self.predictions = self.model_build.predict(self.test_set[self.predictors])
        pd.DataFrame(self.predictions).to_parquet(
            (
                os.path.join(
                    self.save_dir, f"predicted_values_{self.dataset_name}.parquet"
                )
            )
        )

    def calculate_metrics_values(self):
        """
        Calculate evaluation metrics based on the predictions made on the test set.
        """
        actual = self.test_set[self.target_feature]
        predicted = self.predictions
        metrics_values = {
            "dataset": self.dataset_name,
            "mse": mean_squared_error(actual, predicted),
            "r2": r2_score(actual, predicted),
            "mae": mean_absolute_error(actual, predicted),
            "mape": mean_absolute_percentage_error(actual, predicted),
        }
        if self.metrics_values is None:
            self.metrics_values = pd.DataFrame([metrics_values])
        else:
            self.metrics_values = pd.concat(
                [self.metrics_values, pd.DataFrame([metrics_values])], ignore_index=True
            )

    def save_model_info(self):
        """
        Save the built model and evaluation metrics to files.
        """
        with open(os.path.join(self.save_dir, "model.pkl"), "wb") as file:
            pickle.dump(self.model_build, file)
        self.metrics_values.to_csv(os.path.join(self.save_dir, "metrics.csv"))


class LinearRegressionModelBuilder(ModelBuilder):
    """
    Class for building and evaluating Linear Regression models.

    Attributes:
        model (sklearn estimator): Linear Regression model instance.
        forced_positive_predictions (bool): Flag indicating if predictions
            are forced to be positive.
    """

    def __init__(self, **kwargs):
        """
        Initialize a LinearRegressionModelBuilder object.

        Args:
            **kwargs: Additional arguments to be passed to the parent class constructor.
        """
        super().__init__(**kwargs)
        self.model = LinearRegression
        self.forced_positive_predictions = False

    def forward_feature_selection(
        self, num_features=6, scoring="neg_mean_squared_error"
    ):
        """
        Perform forward feature selection to select the best features for modeling.

        Args:
            num_features (int, optional): Number of features to select. Defaults to 6.
            scoring (str, optional): Scoring metric used for feature selection.
                Defaults to "neg_mean_squared_error".
        """
        predictors_data = self.train_set[self.predictors].astype(float)
        target_data = self.train_set[self.target_feature].astype(float)

        remaining_features = self.predictors
        best_features = []

        for round_ in range(num_features):
            best_score = np.inf
            best_candidate = ""
            best_model = None
            for candidate in remaining_features:
                features = best_features + [candidate]
                model = self.model(**self.hyperparams)
                mse_scores = -cross_val_score(
                    model, predictors_data[features], target_data, scoring=scoring
                )
                score = np.mean(mse_scores)
                if score < best_score:
                    best_score = score
                    best_candidate = candidate
                    best_model = model
            best_features.append(best_candidate)
            remaining_features = list(
                set(remaining_features).difference(
                    set(
                        predictors_data.columns[
                            np.abs(predictors_data.corr()[best_candidate]) > 0.7
                        ]
                    )
                )
            )
            if self.log_mlflow:
                with mlflow.start_run():
                    mlflow.log_param("predictors", best_features)
                    mlflow.log_metrics({"mse": best_score})
                    mlflow.sklearn.log_model(
                        sk_model=best_model,
                        artifact_path="sklearn-model",
                        registered_model_name=f"sk-learn-linear-reg-model-ffs",
                    )
            if not remaining_features:
                print("FFS finished at round", round_)
                self.predictors = best_features
                return
        self.predictors = best_features

    def force_positive_predictions(self):
        """
        Force predictions to be non-negative - swap negative values to 0.
        """
        self.predictions = np.maximum(self.predictions, 0)
        self.forced_positive_predictions = True

    def plot_actuals_vs_predicted(self):
        """
        Create actual versus predicted values plot and save it to file.
        """
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.plot(
            range(0, len(self.predictions)),
            self.test_set[self.target_feature],
            label="Actual",
        )
        ax.plot(range(0, len(self.predictions)), self.predictions, label="Predicted")
        ax.set_xlabel("Sample number")
        ax.set_ylabel(self.target_feature)
        ax.set_title(f"Actual vs predicted plot of {self.target_feature}")
        plt.legend()
        fig.savefig(
            os.path.join(
                self.save_dir,
                f"actual_vs_pred_{self.target_feature}_{self.dataset_name}_"
                f"{'only_pos' if self.forced_positive_predictions else ''}_"
                f"{self.timestamp}.png",
            )
        )
        plt.close(fig)

    def plot_predictors_correlation(self):
        """
        Create correlation heatmap plot of the predictors and save it to file.
        """
        to_plot = self.data[self.predictors]
        features_hash = hash(tuple(self.predictors))
        num_feat = len(self.predictors)
        num_feat_large = num_feat > 15
        fig, ax = plt.subplots(1, 1, figsize=(30, 25) if num_feat_large else (15, 10))
        sns.heatmap(to_plot.corr(), annot=not num_feat_large, vmin=-1, vmax=1, ax=ax)
        ax.set_title(
            "Predictors correlation heatmap", fontsize=30 if num_feat_large else 20
        )
        fig.savefig(
            os.path.join(
                self.save_dir,
                f"corr_heatmap_{self.dataset_name}_num_features_{num_feat}"
                f"_features_hash_{features_hash}_{self.timestamp}.png",
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
