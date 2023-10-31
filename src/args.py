# This code is borrowed from the HINNPerf (Hierarchical Interaction Neural 
# Network for Performance Prediction of Configurable Systems) implementation.
# Available here: https://drive.google.com/drive/folders/1qxYzd5Om0HE1rK0syYQsTPhTQEBjghLh

from itertools import product as prod

"""
Function to convert dictionary of lists to list of dictionaries of all combinations of listed variables. 
Example:
    list_of_param_dicts({'a': [1, 2], 'b': [3, 4]}) ---> [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
"""
def list_of_param_dicts(param_dict):
    """
    Arguments:
        param_dict   -(dict) dictionary of parameters
    """
    vals = list(prod(*[v for k, v in param_dict.items()]))
    keys = list(prod(*[[k]*len(v) for k, v in param_dict.items()]))
    return [dict([(k, v) for k, v in zip(key, val)]) for key, val in zip(keys, vals)]