import gentune_utils
import random


def create_small_variation(old_replace_vals, params, p_change):
    success = False
    max_attempts = 10
    attempts = 0
    while not success and attempts < max_attempts:
        success = True
        replace_strs = []
        replace_vals = [[]]
        param_idx = 0
        for param in params:
            replace_strs, replace_vals = gentune_utils.add_param_dim(
                replace_strs, replace_vals, param
            )
            n_vals = len(replace_vals)
            if n_vals >= 1:
                do_ran_selection = True
                for val in replace_vals:
                    if val[-1] == old_replace_vals[param_idx]:  # is original in list?
                        ran = random.random()
                        if ran > p_change:  # choose original
                            replace_vals = [val]
                            do_ran_selection = False
                if do_ran_selection:
                    select = random.randint(0, n_vals - 1)
                    replace_vals = [replace_vals[select]]
            else:
                attempts += 1
                success = False
                break
            param_idx += 1
    return replace_vals, success
