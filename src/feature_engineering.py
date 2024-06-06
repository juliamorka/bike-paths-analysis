import numpy as np


def get_logged_variables(data, to_predict):
    numerical_cols = [
        col
        for col in data.select_dtypes(include=[np.number]).columns
        if to_predict not in col
    ]
    return data[numerical_cols].apply(lambda x: np.log(x + 1))
