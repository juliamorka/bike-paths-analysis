import json

import numpy as np


def get_sum_wrapper(func):
    def inner(*args, **kwargs):
        to_sum = func(*args, **kwargs)
        return np.sum(to_sum)

    return inner


def get_json_configs(paths_list):
    configs = []
    for path in paths_list:
        with open(path) as json_config:
            configs.append(json.load(json_config))
    return configs
