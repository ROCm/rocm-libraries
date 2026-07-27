import argparse
import interpreter
import v0_optimizer
import os
import gentune_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--in_filename", help="path to gentune file relative to base_gentune_dir"
    )
    parser.add_argument(
        "-g",
        "--base_gentune_dir",
        help="path to folder where gentune files live",
        required=False,
        default="gentune_files/",
    )
    parser.add_argument(
        "-b",
        "--base_code_dir",
        help="path to main folder of code",
        required=False,
        default="../",
    )
    parser.add_argument(
        "-t",
        "--num_threads",
        help="number of threads to build with",
        required=False,
        default=16,
    )
    parser.add_argument(
        "-p",
        "--print_every_n_sec",
        help="print the output after x seconds",
        default=60,
        required=False,
    )
    parser.add_argument(
        "-v",
        "--verbose-level",
        help="level of verbosity. 1=verification fails only. 2: + build fails + don't delete failed instances 3: + unsupported problems",
        default=0,
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="generate output hpp file to path. Requires hpp_template and hpp_template_code to be in gentune file",
        default="",
        required=False,
    )

    args = parser.parse_args()
    configs = interpreter.parse_input_from_file(
        args.in_filename, args.base_gentune_dir, args.output
    )

    gen_file_config = {}
    gen_file_config["base_code_dir"] = args.base_code_dir
    gen_file_config["verbose_level"] = args.verbose_level

    os.makedirs(configs[0]["TEST_INSTANCE_DIR"].replace("'", "").strip(), exist_ok=True)

    # print some stats about the parameter space
    print("Overall there are " + str(len(configs)) + " configurations")

    conf_idx = 0
    for config in configs:
        print(
            "\n--------------------------\nStats for config number "
            + str(conf_idx)
            + " with gen parameter combination:"
        )
        print(str(list(config["gen_params"].values())) + "\n")

        replace_strs, gen_param_combos = gentune_utils.create_all_combos(
            config["gen_params"].values()
        )
        gen_param_combo_idx = 0
        for gen_param_combo in gen_param_combos:
            print(
                "\nStats for gen param combo number "
                + str(gen_param_combo_idx)
                + " of config number with gen params"
            )
            print(str(gen_param_combo))
            naive_param_space_size = 1
            for tune_param in config["tune_params"].values():
                naive_param_space_size *= len(tune_param["possible_vals"])
            print(
                "Naive parameter space size (without constraints): "
                + str(naive_param_space_size)
            )

            param_space_size = gentune_utils.get_param_space_size(
                config["tune_params"].values(), replace_strs, [gen_param_combo]
            )
            if param_space_size == gentune_utils.PARAM_SPACE_TO_LARGE_TO_COMPUTE:
                print("Parameter space with constraints is too large to compute")
            else:
                print("Parameter space size with constraints: " + str(param_space_size))
            config["parameter_space_size"] = param_space_size
            gen_param_combo_idx += 1
        conf_idx += 1

    print("begin optimizing")
    v0_optimizer.optimize_configs(
        configs,
        int(args.num_threads),
        gen_file_config,
        int(args.print_every_n_sec),
        args.output,
    )


main()
