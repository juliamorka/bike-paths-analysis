import json

import numpy as np


def get_sum_wrapper(func):
    """
    Wrapper function that takes another function as input
    and returns the sum of its output.

    Args:
        func (callable): Function to be wrapped.

    Returns:
        callable: Wrapped function that computes the sum of the output of `func`.
    """

    def inner(*args, **kwargs):
        to_sum = func(*args, **kwargs)
        return np.sum(to_sum)

    return inner


def get_json_configs(paths_list):
    """
    Read multiple JSON configuration files and return their contents
    as a list of dictionaries.

    Args:
        paths_list (list of str): List of file paths to JSON
            configuration files.

    Returns:
        list of dict: List containing dictionaries loaded
            from each JSON file.
    """
    configs = []
    for path in paths_list:
        with open(path, "r") as json_config:
            configs.append(json.load(json_config))
    return configs
