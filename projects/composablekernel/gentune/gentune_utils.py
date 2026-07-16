# guessing the size of the parameter space can be rather difficult. To
def add_param_dim(replace_strs, instances, param, max_expansion_size=10000):
    new_instances = []
    replace_strs.append(param["Names"])
    for instance in instances:
        for val in param["possible_vals"]:
            new_instance = instance.copy()
            val_c = val.copy()
            if param["subst_active"]:
                for v in range(len(val_c)):
                    for rep_idx in range(len(replace_strs) - 1):
                        for k in range(len(replace_strs[rep_idx])):
                            val_c[v] = val_c[v].replace(
                                replace_strs[rep_idx][k].strip(),
                                new_instance[rep_idx][k].strip(),
                            )
            new_instance.append(val_c)
            if "constraint" in param.keys():
                constr_txt = param["constraint"]
                # swap all replace strings into constraint str and eval it
                for rep_idx in range(len(replace_strs)):
                    for k in range(len(replace_strs[rep_idx])):
                        constr_txt = constr_txt.replace(
                            replace_strs[rep_idx][k].strip(),
                            new_instance[rep_idx][k].strip(),
                        )
                if not eval(constr_txt):
                    continue
            new_instances.append(new_instance)
            if len(new_instances) >= max_expansion_size:
                raise Exception("reached maximum expansion size")
    return replace_strs, new_instances


PARAM_SPACE_TO_LARGE_TO_COMPUTE = -1


def get_param_space_size(params, prev_rv=[], prev_rs=[[]], max_expansion_size=10000):
    replace_strs = prev_rv
    instances = prev_rs
    for param in params:
        try:
            replace_strs, instances = add_param_dim(
                replace_strs, instances, param, max_expansion_size
            )
        except Exception:
            return PARAM_SPACE_TO_LARGE_TO_COMPUTE  # parameter space is to large to explicitly compute
    return len(instances)


def create_all_combos(params):
    replace_strs = []
    replace_vals = [[]]
    for param in params:
        replace_strs, replace_vals = add_param_dim(replace_strs, replace_vals, param)
    return replace_strs, replace_vals


def apply_replace_values_to_str(replace_strs, replace_vals, line):
    if len(replace_strs) != len(replace_vals):
        raise Exception("Length of replace strings must equal length of replace vals")
    for j in range(len(replace_strs)):
        for k in range(len(replace_strs[j])):
            line = line.replace(replace_strs[j][k].strip(), replace_vals[j][k].strip())
    return line
