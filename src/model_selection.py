import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from src.transformations import get_predictors

def split_data(data, to_predict, validation_set=False):
    """
    Splits the data into training, validation, and test sets.

    Parameters:
        data (DataFrame): The input data.
        to_predict (str): The name of the target column.
        validation_set (bool): Whether to create a validation set. Default is False.

    Returns:
        tuple: a tuple containing X_train, y_train, X_test, y_test, X_holdout, y_holdout
    """
    X, y = data[get_predictors(data, to_predict)], data[to_predict]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if not validation_set:
        return X_temp, y_temp, X_test, y_test
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    return X_train, y_train, X_test, y_test, X_validation, y_validation


def get_ffs_features(
    X, y, num_features=6, log_mlflow=False, scoring="neg_mean_squared_error"
):
    """
    Performs forward feature selection to select the best features for the model.

    Parameters:
        X (DataFrame): Input features (training set).
        y (Series): Target variable (training set).
        num_features (int): Number of features to select.
        log_mlflow (bool): Whether to log results to MLflow.
        scoring (str): Scoring method for cross-validation.

    Returns:
        list: selected feature names.
    """
    X = X.astype(float)
    y = y.astype(float)

    remaining_features = list(X.columns)
    best_features = []

    for _ in range(num_features):
        scores_with_candidates = []
        for candidate in remaining_features:
            features = best_features + [candidate]
            model = LinearRegression()
            mse_scores = -cross_val_score(model, X[features], y, scoring=scoring)
            score = np.mean(mse_scores)
            scores_with_candidates.append((score, candidate))
            if log_mlflow:
                with mlflow.start_run():
                    mlflow.log_param("predictors", features)
                    mlflow.log_metrics({"mse": score})
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="sklearn-model",
                        registered_model_name="sk-learn-linear-reg-model-3",
                    )

        scores_with_candidates = sorted(scores_with_candidates)
        _, best_candidate = scores_with_candidates[0]
        print(best_candidate)
        remaining_features.remove(best_candidate)
        best_features.append(best_candidate)

    return best_features


def get_metrics_summary(y_actual: pd.Series, y_pred: pd.Series) -> tuple:
    """
    Calculates various evaluation metrics for regression.

    Parameters:
        y_actual (pd.Series): Actual target values.
        y_pred (pd.Series): Predicted target values.

    Returns:
        tuple:
            - mean_squared_error (float): Mean squared error.
            - r2_score (float): R-squared score.
            - mean_absolute_error (float): Mean absolute error.
            - mean_absolute_percentage_error (float): Mean absolute percentage error.
    """
    return (
        mean_squared_error(y_actual, y_pred),
        r2_score(y_actual, y_pred),
        mean_absolute_error(y_actual, y_pred),
        mean_absolute_percentage_error(y_actual, y_pred),
    )
