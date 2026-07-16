import re
import copy


def extract_brackets(str):
    num_open_brackets = 1
    str_idx = 0
    while num_open_brackets != 0:
        if str_idx == len(str):
            raise Exception("expected ) while parsing " + str)
        if str[str_idx] == "(":
            num_open_brackets += 1
        elif str[str_idx] == ")":
            num_open_brackets -= 1

        str_idx += 1
    return str[: str_idx - 1], str[str_idx + 1 :]


def create_config_from_expr(expr, base_gentune_dir, lval=[]):
    if isinstance(expr, inc_interpret_node):  # recursive expansion
        return expr.extract_config(base_gentune_dir, copy.deepcopy(lval))
    if isinstance(expr, list):
        return expr  # has already been expanded, just return
    return parse_input_from_file_internal(expr, base_gentune_dir, copy.deepcopy(lval))


# by definition, lval has already been generated, rval will be
# yeah, i know operator could be a class and all, but we only have two and probably wont get more so that seems excessive
def apply_config_add(lval, rval, base_gentune_dir, base_lval):
    # append to list list
    return copy.deepcopy(lval) + create_config_from_expr(
        rval, base_gentune_dir, base_lval
    )


def apply_config_mul(lval, rval, base_gentune_dir):
    # cartesian product
    return create_config_from_expr(rval, base_gentune_dir, lval)


class inc_interpret_node:
    known_operators = ["*", "+"]  # in order of precedence

    def __init__(self, string):
        # process input string until it is empty
        self.expressions = []
        self.operators = []
        while True:
            # parse expressions
            string = string.lstrip(" \t\n")
            if string == "":  # cannot be empty or end after an operator
                raise Exception("expected an expression while parsing input string")
            if string[0] == '"':
                # find the next "
                index = string[1:].find('"') + 1
                if index == 0:
                    raise Exception('Error: expected " while parsing ' + string)
                self.expressions.append(string[1:index])
                string = string[index + 1 :]
            elif string[0] == "(":
                next_node_str, string = extract_brackets(string[1:])
                self.expressions.append(inc_interpret_node(next_node_str))
            else:
                raise Exception(
                    'Ill-formed include str: expeced start of expression (" or ( ). Occured while parsing + '
                    + string
                )

            # now, we expect either the end of the statement or an operator to guide us to the next expression
            string = string.lstrip(" \t\n")
            if string == "":
                break
            if string[0] not in inc_interpret_node.known_operators:
                raise Exception(
                    "Ill-formed include str: expected an operator. Occured while parsing "
                    + string
                )
            # extract the operator and start over again
            self.operators.append(string[0])
            string = string[1:]

    def extract_config(self, base_gentune_dir, base_lval):
        # only one element. Just generate it and be done with it
        if len(self.operators) == 0:
            return create_config_from_expr(
                self.expressions[0], base_gentune_dir, copy.deepcopy(base_lval)
            )
        # in order of precedence
        for operator in inc_interpret_node.known_operators:
            op_idx = 0
            while op_idx < len(self.operators):
                if self.operators[op_idx] == operator:
                    # check if lval has been expanded

                    if not isinstance(self.expressions[op_idx], list):
                        self.expressions[op_idx] = create_config_from_expr(
                            self.expressions[op_idx],
                            base_gentune_dir,
                            copy.deepcopy(base_lval),
                        )
                    if operator == "*":
                        # no need for passing base_lval as it has already been applied to lval
                        self.expressions[op_idx] = apply_config_mul(
                            copy.deepcopy(self.expressions[op_idx]),
                            self.expressions[op_idx + 1],
                            base_gentune_dir,
                        )
                    elif operator == "+":
                        # concatenate two solutions
                        self.expressions[op_idx] = apply_config_add(
                            copy.deepcopy(self.expressions[op_idx]),
                            self.expressions[op_idx + 1],
                            base_gentune_dir,
                            copy.deepcopy(base_lval),
                        )
                    else:
                        raise Exception("Encountered unknown operator " + operator)

                    # now, delete operator and rval config
                    del self.expressions[op_idx + 1]
                    del self.operators[op_idx]
                else:
                    op_idx += 1
        # everything is in the 0th element now

        return self.expressions[0]


def process_param_exclusion(param, excluded_symbol_list):
    if "constraint" in param.keys():
        for excluded_symbol in excluded_symbol_list:
            if excluded_symbol in param["constraint"]:
                raise Exception(
                    "Error while parsing constraints: constraints may only refer to parameters defined ABOVE given parameter and gen parameters may not refer to tune parameters( parameter "
                    + excluded_symbol
                    + " encountered while processing constraint of parameter "
                    + param["Names"][0]
                    + ")"
                )
    for name in param["Names"]:
        excluded_symbol_list.append(name)
    return excluded_symbol_list


def parse_input_from_file_internal(path, base_gentune_dir, configs=[]):
    with open(base_gentune_dir + path, "r") as file:
        lines = file.readlines()
        str = "".join(lines)

    # convention: ALL_CAPS for keys taken directly from config, lowercase for generated ones
    known_keys = [
        "NAME",
        "TEST_INSTANCE_TEMPLATE_PATH",
        "TEST_INSTANCE_DIR",
        "PARAMS",
        "BENCH_ARGS",
        "VERIFY_ARGS",
        "OUT_FILENAME",
        "BENCH_CMD",
        "COMPILE_CMD",
        "PROFILER_DATATYPE_ID",
        "PROFILER_LAYOUT_ID",
        "OUTPUT_HPP_TEMPLATE_PATH",
        "OUTPUT_HPP_TEMPLATE_CODE_LINE",
    ]

    # check for #INCLUDE, #INCLUDE_STRUCTURE and #CACHE_BEST_RESULTS

    if "#INCLUDE" in str:
        include_split_str = str.split("#INCLUDE")
        orig_configs = configs
        new_confs = []
        for i in range(1, len(include_split_str)):
            new_confs += parse_input_from_file_internal(
                include_split_str[i].split("\n")[0].strip(),
                base_gentune_dir,
                copy.deepcopy(orig_configs),
            )
        configs = new_confs

    if "#IMPORT_STRUCTURE" in str:
        include_struct_split_str = str.split("#IMPORT_STRUCTURE")
        if len(include_struct_split_str) != 2:
            raise Exception("Only one include structure may be defined per file")
        include_struct_split_str = include_struct_split_str[1].split(
            "#END_IMPORT_STRUCTURE"
        )
        if len(include_struct_split_str) != 2:
            raise Exception(
                "if #IMPORT_STRUCTURE is defined, you need to define #END_IMPORT_STRUCTURE exactly once"
            )
        include_struct_str = include_struct_split_str[0]
        base_interpreter_node = inc_interpret_node(include_struct_str)
        configs = base_interpreter_node.extract_config(
            base_gentune_dir, copy.deepcopy(configs)
        )

    if len(configs) == 0:
        # append an empty config if there is none yet
        configs.append({})

    cache_best_results = False
    if "#CACHE_BEST_RESULTS" in str:
        cache_best_results = True
    # remove comments
    str = re.sub(r"#.*$", "", str, flags=re.MULTILINE)
    str = str.strip()
    # build a key-params dict from which we will build our json
    str = str.split(":")

    # iterate over all configurations that we have
    for config in configs:
        segments = [part.split("\n") for i, part in enumerate(str)]
        cleaned_segment = ""
        curr_key_name = ""

        # splitting by : can be a bit dirty if a namespace is in the template. To fix this, we split only if the last element is a known key
        for i in range(len(segments)):
            for j in range(len(segments[i]) - 1):
                if segments[i][j] != "" and cleaned_segment != "" and j != 0:
                    cleaned_segment = cleaned_segment + "\n" + segments[i][j]
                else:
                    cleaned_segment = cleaned_segment + segments[i][j]
            if segments[i][-1] not in known_keys:
                end_char = ":"  # dont add : on last element
                if i == len(segments) - 1:
                    end_char = ""
                if (
                    segments[i][-1] != ""
                    and cleaned_segment != ""
                    and len(segments[i]) != 1
                ):
                    cleaned_segment = (
                        cleaned_segment + "\n" + segments[i][-1] + end_char
                    )
                else:
                    cleaned_segment = cleaned_segment + segments[i][-1] + end_char
            else:
                # we now know we've reached a true end. Thus add to dict (except for first entry, which may be arbitrary text and is ignored)
                if curr_key_name != "":
                    config[curr_key_name] = cleaned_segment  # add to dict
                cleaned_segment = ""  # reset
                curr_key_name = segments[i][-1]
        config[curr_key_name] = cleaned_segment
        # check for consistency and process keys (where neccessary)

        if "PARAMS" in config.keys():
            if "gen_params" not in config.keys():
                config["gen_params"] = {}
            if "tune_params" not in config.keys():
                config["tune_params"] = {}

            # for legibility, parameter definitions may span multiple lines.
            # create a new parameter once keyword GEN or TUNE appears
            parameter_strings = []
            parameter_split_lines = config["PARAMS"].split("\n")
            for param_line in parameter_split_lines:
                if " GEN" not in param_line and " TUNE" not in param_line:
                    if len(parameter_strings) == 0:
                        raise Exception(
                            "First line of any parameter string must begin with GEN or TUNE keyword"
                        )
                    parameter_strings[-1] += " " + param_line
                else:
                    parameter_strings.append(param_line)
            for parameter_str in parameter_strings:
                parameter = {}
                # extract constraint (if it exists)
                if "CONSTRAINT" in parameter_str:
                    split_str = parameter_str.split("CONSTRAINT")
                    if len(split_str) != 2:
                        raise Exception(
                            "Error processing parameters: more than one constraint specified. Encountered while processing "
                            + parameter_str
                        )
                    parameter["constraint"] = split_str[1]
                    parameter_str = split_str[0]
                is_tune = False
                is_gen = False
                if " GEN" in parameter_str:
                    is_gen = True
                    parameter_str = parameter_str.split(" GEN")
                if " TUNE" in parameter_str:
                    is_tune = True
                    parameter_str = parameter_str.split(" TUNE")
                # to avoid accidental substitution, user must explicitly turn on substitution mode
                if parameter_str[1].startswith(" REPLACE"):
                    parameter["subst_active"] = True
                    parameter_str[1] = parameter_str[1][8:]
                else:
                    parameter["subst_active"] = False
                # check for multiple occurrances of GEN or TUNE
                if len(parameter_str) != 2:
                    raise Exception(
                        "GEN or TUNE keywords may occur only once in parameter"
                    )
                # check if user accidentially put key in new line
                if parameter_str[0].strip() == "":
                    raise Exception(
                        "Need name of parameter. Note that keywords GEN or TUNE may not be in new line, must be in same line as Name(s) of Parameter(s)"
                    )

                # check if is GEN or TUNE
                if (not is_tune and not is_gen) or (is_tune and is_gen):
                    raise Exception(
                        "Error processing parameters: parameter must be either GEN or TUNE Encountered while processing "
                        + parameter_str
                        + " (Note that keyword GEN or TUNE must be in same line as name of parameter)"
                    )

                parameter_key = parameter_str[0].strip()
                parameter["Names"] = [n.strip() for n in parameter_str[0].split(";")]
                # check if any previous parameter has the name of this parameter in the possible values
                for name in parameter["Names"]:
                    for prev_parameter in list(config["tune_params"].values()) + list(
                        config["gen_params"].values()
                    ):
                        for prev_name in prev_parameter["Names"]:
                            if prev_name in name and (
                                name != prev_name
                            ):  # Note: overwriting parameters is allowed
                                print("Note:")
                                print(prev_name)
                                print(name)
                                raise Exception(
                                    "Parameter name may not conatain name of previous parameter. Pleas make sure no other parameter name is a substring of another"
                                )

                possible_vals_str = parameter_str[1].split(";")
                possible_vals = []
                constexpr_conditions = []
                preprocessor_commands = []
                for value in possible_vals_str:
                    # process PREPROCESSOR and CONDITION
                    if "PREPROCESSOR " in value:
                        value = value.split("PREPROCESSOR ")
                        # checks
                        if not is_gen:
                            raise Exception(
                                "Only gen params may have PREPROCESSOR commands"
                            )
                        if len(value) != 2:
                            raise Exception(
                                "Keyword PREPROCESSOR may occur only once per possible value"
                            )
                        if "CONDITION " in value[1]:
                            raise Exception(
                                "Keyword CONDITION must occur before keyword PREPROCESSOR"
                            )

                        preprocessor_commands.append(value[1])
                        value = value[0]
                    else:
                        preprocessor_commands.append(
                            ""
                        )  # enter blank so that list remains aligned

                    if "CONDITION " in value:
                        value = value.split("CONDITION ")
                        # checks
                        if not is_gen:
                            raise Exception(
                                "Only gen params may have CONDITION commands"
                            )
                        if len(value) != 2:
                            raise Exception(
                                "Keyword CONDITION may occur only once per possible value"
                            )

                        constexpr_conditions.append(value[1])
                        value = value[0]
                    else:
                        constexpr_conditions.append(
                            ""
                        )  # enter blank so that list remains aligned

                    # now, process the actual values
                    values = [value] * len(parameter["Names"])

                    # strip values check if any is empty
                    for v in values:
                        v = v.strip()
                        if v == "":
                            print(
                                "WARNING: empty value encountered while processing values. Do you have a trailing ;?"
                            )
                    possible_vals.append(values)
                parameter["possible_vals"] = possible_vals
                parameter["preprocessor_commands"] = preprocessor_commands
                parameter["constexpr_conditions"] = constexpr_conditions

                if is_tune:
                    config["tune_params"][parameter_key] = parameter
                if is_gen:
                    config["gen_params"][parameter_key] = parameter

            # check that constranits refer only to parameters mentioned before and gen parameters' constraints do not refer to tune params
            # we do this to enable reduction of computational complexity in generation
            tune_params = list(config["tune_params"].values())
            gen_params = list(config["gen_params"].values())

            excluded_symbol_list = []
            for param_idx in range(len(tune_params)):
                excluded_symbol_list = process_param_exclusion(
                    tune_params[len(tune_params) - param_idx - 1], excluded_symbol_list
                )
            for param in reversed(gen_params):
                excluded_symbol_list = process_param_exclusion(
                    param, excluded_symbol_list
                )

        if "BENCH_ARGS" in config.keys():
            # TEST_EXEC - for now we only support a call to ckProfiler in verification mode
            # note that we currently dont have a way to ensure that verification mode is turned on

            # split values and remove potential
            args_tmp = config["BENCH_ARGS"].split("\n")
            cleaned_args = []
            for arg in args_tmp:
                if arg != "":
                    cleaned_args.append(arg)
            if len(cleaned_args) == 0:
                raise Exception(
                    "need to specify at least one sample set of arguments to BENCH_ARGS (delimited by newline)"
                )
            config["BENCH_ARGS"] = cleaned_args

        if "VERIFY_ARGS" in config.keys():
            # TEST_EXEC - for now we only support a call to ckProfiler in verification mode
            # note that we currently dont have a way to ensure that verification mode is turned on

            # split values and remove potential
            args_tmp = config["VERIFY_ARGS"].split("\n")
            cleaned_args = []
            for arg in args_tmp:
                if arg != "":
                    cleaned_args.append(arg)
            if len(cleaned_args) == 0:
                raise Exception(
                    "need to specify at least one sample set of arguments to VERIFY_ARGS (delimited by newline)"
                )
            config["VERIFY_ARGS"] = cleaned_args

        # write out most optimal solutions at set positions to enable sharing of optimal
        if cache_best_results:
            cache_point_str = path.replace("/", "_").replace(".", "-")
            if "cache_points" in config.keys():
                config["cache_points"].append(cache_point_str)
            else:
                config["cache_points"] = [cache_point_str]
    return configs


# basically call parse_input_from_file and perform checks
# need this as files may be incomplete segments
def parse_input_from_file(path, base_gentune_dir, output_dir=""):
    configs = parse_input_from_file_internal(path, base_gentune_dir)

    for config in configs:
        # TEMPLATE - a template for exe in which the parameter
        if "TEST_INSTANCE_TEMPLATE_PATH" not in config.keys():
            raise Exception(
                "TEST_INSTANCE_TEMPLATE_PATH must be specified (path to file to be used as a template, based off examples)"
            )

        # PARAMS - here we will build two seperate dicts TUNE_PARAMS and GEN_PARAMS
        if "PARAMS" not in config.keys():
            raise Exception(
                "need to specify at least one parameter (Example: PARAMS:ALayout GEN Row, Col)"
            )

        if "BENCH_ARGS" not in config.keys() or "VERIFY_ARGS" not in config.keys():
            raise Exception(
                "need to specify arguments for benchmark and verification. "
            )
        if output_dir != "":
            if (
                "OUTPUT_HPP_TEMPLATE_PATH" not in config.keys()
                or "OUTPUT_HPP_TEMPLATE_CODE_LINE" not in config.keys()
            ):
                raise Exception(
                    "Output can only be specified if OUTPUT_HPP_TEMPLATE_PATH and OUTPUT_HPP_TEMPLATE_CODE_LINE are specified in gentune file"
                )

            with open(config["OUTPUT_HPP_TEMPLATE_PATH"].strip(), "r") as file:
                lines = file.readlines()
                config["hpp_out_template"] = "".join(lines)
    # pprint.pprint(configs)
    return configs
