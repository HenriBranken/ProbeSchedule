import numpy as np


class NoProperWMATrend(Exception):
    pass


def get_prized_index(n_bumps_list):
    print(n_bumps_list)
    some_list = np.where(np.asarray(n_bumps_list) == 1)[0]
    if len(some_list) > 0:
        return some_list[-1]
    else:
        raise NoProperWMATrend("No index at which n_extrema = 1.")


num_bumps = []
prized_index = get_prized_index(num_bumps)
