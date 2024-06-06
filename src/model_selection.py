from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import mlflow
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
)


def split_data(data, predictors, to_predict):
    X, y = data[predictors], data[to_predict]
    X_temp, X_holdout, y_temp, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    return X_train, y_train, X_test, y_test, X_holdout, y_holdout


def get_ffs_features(
    X, y, num_features=6, log_mlflow=False, scoring="neg_mean_squared_error"
):
    X = X.astype(float)
    y = y.astype(float)

    remaining_features = list(X.columns)
    best_features = []

    for _ in range(num_features):
        scores_with_candidates = []
        for candidate in remaining_features:
            features = best_features + [candidate]
            model = LinearRegression()
            mse_scores = -cross_val_score(model, X, y, scoring=scoring)
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
        best_score, best_candidate = scores_with_candidates[0]

        remaining_features.remove(best_candidate)
        best_features.append(best_candidate)

    return best_features


def get_metrics_summary(y_actual, y_pred):
    return (
        mean_squared_error(y_actual, y_pred),
        r2_score(y_actual, y_pred),
        mean_absolute_error(y_actual, y_pred),
        mean_absolute_percentage_error(y_actual, y_pred),
    )
