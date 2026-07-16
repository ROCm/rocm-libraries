import gentune_utils


def create_file(gen_file_configs, out_filename):
    text = generate_instance_file_text(gen_file_configs)
    with open(
        out_filename,
        "w+",
    ) as file:
        file.write(text)


def generate_instance_file_text(gen_file_configs):
    template_txt = gen_file_configs[0]["hpp_out_template"].split("#ADD_INSTANCES_HERE")
    if len(template_txt) != 2:
        raise Exception(
            "Error while processing hpp output: Expected exactly one occurance of #ADD_INSTANCES_HERE in hpp out template"
        )

    text = template_txt[0]

    for gen_file_config in gen_file_configs:
        preprocessor_command = ""
        for command in gen_file_config["preprocessor_commands"]:
            if command != "":
                if preprocessor_command == "":
                    preprocessor_command = "#if defined(" + command + ")"
                else:
                    preprocessor_command += "&& defined(" + command + ")"
        if preprocessor_command != "":
            text += "\n" + preprocessor_command + "\n"
        constexpr_condition = ""
        for condition in gen_file_config["constexpr_conditions"]:
            if constexpr_condition != "" and condition != "":
                constexpr_condition += " && "
            constexpr_condition += condition

        if constexpr_condition != "":
            text += "       if constexpr(" + constexpr_condition + ")\n    {\n"

        n_instances = len(gen_file_config["replace_vals"])

        if n_instances > 0:
            text += "           add_device_operation_instances(op_ptrs, std::tuple<\n            "

            for i in range(n_instances):
                if len(gen_file_config["replace_vals"][i]) == 0:
                    continue  # early out if we don't have ideal instances yet
                line = gentune_utils.apply_replace_values_to_str(
                    gen_file_config["replace_strs"],
                    gen_file_config["replace_vals"][i],
                    gen_file_config["hpp_out_code_line"],
                )

                text += line
                if i != n_instances - 1:
                    text += ",\n            "
            text += "\n         >{});"

        if constexpr_condition != "":
            text += "\n    }\n"

        if preprocessor_command != "":
            text += "\n#endif\n"
    text += template_txt[1]

    return text
