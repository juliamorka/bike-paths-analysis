import numpy as np
import pandas as pd


def get_predictors(data: pd.DataFrame, to_predict: str) -> list:
    """
    Get a list of predictor columns from the given DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        to_predict (str): The column to be predicted.

    Returns:
        list: A list of column names containing numerical predictors
              excluding the column to be predicted.
    """
    return [
        col
        for col in data.select_dtypes(include=[np.number]).columns
        if to_predict not in col
    ]


def get_logged_predictors(predictors: pd.Series) -> pd.Series:
    """
    Apply logarithmic transformation to the given predictors.

    Parameters:
        predictors (pd.Series): series containing numerical predictors.

    Returns:
        pd.Series: Transformed series.
    """
    return predictors.apply(lambda value: np.log(value + 1))


def get_powered_predictors(predictors: pd.Series, power: float) -> pd.Series:
    """
    Apply power transformation to the given predictors.

    Parameters:
        predictors (pd.Series): Series containing numerical predictors.
        power (float): The exponent to raise the values by.

    Returns:
        pd.Series: Transformed series.
    """
    return predictors.apply(lambda value: value**power)


def get_inverse_predictors(predictors: pd.Series, epsilon: float = 1e-10) -> pd.Series:
    """
    Apply inverse transformation to the given predictors.

    Parameters:
        predictors (pd.Series): Series containing numerical predictors.
        epsilon (float, optional): Small value added to avoid division by zero.

    Returns:
        pd.Series: Transformed series.
    """
    return predictors.apply(lambda value: 1 / (value + epsilon))


def get_transformed_predictors(
    data: pd.DataFrame, predictor_cols: list
) -> pd.DataFrame:
    """
    Generate transformed predictors based on the given columns.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        predictor_cols (list): List of column names containing numerical predictors.

    Returns:
        pd.DataFrame: Transformed DataFrame with additional columns representing
                      logarithmic, powered, square root, and inverse transformations
                      of the predictor columns.
    """
    transformed_data = data.copy()
    for col in predictor_cols:
        transformed_data[col + "_log"] = get_logged_predictors(data[col])
        transformed_data[col + "_sqr"] = get_powered_predictors(data[col], 2)
        transformed_data[col + "_sqrt"] = get_powered_predictors(data[col], 0.5)
        transformed_data[col + "_inv"] = get_inverse_predictors(data[col])
    return transformed_data
