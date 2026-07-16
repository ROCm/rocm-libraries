import gentune_utils
import random


def create_random_instance(params):
    success = False
    while not success:
        success = True
        replace_strs = []
        replace_vals = [[]]
        for param in params:
            replace_strs, replace_vals = gentune_utils.add_param_dim(
                replace_strs, replace_vals, param
            )
            # select one randomly
            n_vals = len(replace_vals)
            if n_vals >= 1:
                select = random.randint(0, n_vals - 1)
                replace_vals = [replace_vals[select]]
            else:
                success = False
                break
    return replace_strs, replace_vals
