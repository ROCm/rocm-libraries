# kabraham: a basic optimizer good enough at what it does to serve as a proof-of concept for the gentune project but not much more than that

import gentune_utils
import benchmarker
import time
import queue
import threading
import multithreaded_compiler
import v0_configuration_generator
import instance_writer
import os

# global values for tracking
n_build_failures = 0
n_bench_failures = 0
n_success = 0

n_improvments_mc = 0
n_improvments_rv = 0
n_improvments_cache = 0
n_perf_extracted = 0
# def write_best configs()


def extract_constexpr_preprocessor_conditions(
    config, all_combos_replace_strs, replace_combo
):
    constexpr_conditions = []
    preprocessor_commands = []
    if "gen_params" in config.keys():
        for param_idx in range(len(all_combos_replace_strs)):
            parameter = {}
            for param in config["gen_params"].values():
                if param["Names"] == all_combos_replace_strs[param_idx]:
                    parameter = param
            for val_idx in range(len(parameter["possible_vals"])):
                if parameter["possible_vals"][val_idx] == replace_combo[param_idx]:
                    constexpr_conditions.append(
                        parameter["constexpr_conditions"][val_idx]
                    )
                    preprocessor_commands.append(
                        parameter["preprocessor_commands"][val_idx]
                    )
    return constexpr_conditions, preprocessor_commands


def optimize_configs(configs, nthreads, gen_file_config, print_every_n_sec, output_dir):
    best_configs = {}
    cached_results = {}

    global n_improvments_cache, n_improvments_mc, n_improvments_rv
    global n_build_failures, n_bench_failures, n_success, n_perf_extracted

    n_improvments_cache = 0
    n_improvments_mc = 0
    n_improvments_rv = 0
    n_build_failures = 0
    n_bench_failures = 0
    n_success = 0
    n_perf_extracted = 0
    time_start = time.time()

    dispatch_queue = queue.Queue()
    work_queue = queue.Queue(maxsize=1)
    benchmark_queue = queue.Queue(maxsize=2)

    # start worker threads
    compile_threads = []
    instance_gen_threads = []
    for threadIdx in range(nthreads):
        t = threading.Thread(
            target=multithreaded_compiler.compiler_thread,
            args=(threadIdx, work_queue, benchmark_queue),
        )
        t.start()
        compile_threads.append(t)

    best_configs = {}
    bench_configs = {}
    gen_file_configs = {}

    # loop over all configurations
    for config in configs:
        # loop over all combinations of GEN parameters
        all_combos_replace_strs = []
        all_combos_replace_vals = [[]]

        if "gen_params" in config.keys():
            all_combos_replace_strs, all_combos_replace_vals = (
                gentune_utils.create_all_combos(list(config["gen_params"].values()))
            )

        for replace_combo in all_combos_replace_vals:
            # extract preprocessor and constexpr_condition values
            constexpr_conditions, preprocessor_commands = (
                extract_constexpr_preprocessor_conditions(
                    config, all_combos_replace_strs, replace_combo
                )
            )
            if "hpp_out_template" in config.keys():
                gen_file_config["hpp_out_template"] = config["hpp_out_template"]
                gen_file_config["hpp_out_code_line"] = config[
                    "OUTPUT_HPP_TEMPLATE_CODE_LINE"
                ]

            gen_file_config["constexpr_conditions"] = constexpr_conditions
            gen_file_config["preprocessor_commands"] = preprocessor_commands

            # apply GEN params directly to template
            gen_file_config["gen_replace_strs"] = all_combos_replace_strs
            gen_file_config["gen_replace_vals"] = replace_combo

            gen_file_config["uid"] = str(replace_combo)
            gen_file_config["parameter_space_size"] = config["parameter_space_size"]
            best_configs[gen_file_config["uid"]] = {}
            if "cache_points" in config.keys():
                gen_file_config["cache_points"] = config["cache_points"]
            else:
                gen_file_config["cache_points"] = ""

            bench_config = {}

            # replace IDs for datatype and layout in test args
            bench_args = config["BENCH_ARGS"]
            verify_args = config["VERIFY_ARGS"]

            bench_config["bench_args"] = bench_args
            bench_config["verify_args"] = verify_args

            bench_config["compile_command"] = config["COMPILE_CMD"]
            bench_config["test_instance_dir"] = config["TEST_INSTANCE_DIR"]
            bench_config["test_instance_template_path"] = config[
                "TEST_INSTANCE_TEMPLATE_PATH"
            ]

            # replace values of gen params occuring in tune params
            tune_params = list(config["tune_params"].values()).copy()
            for tune_param in tune_params:
                if "constraint" in tune_param.keys():
                    tune_param["constraint"] = (
                        gentune_utils.apply_replace_values_to_str(
                            all_combos_replace_strs,
                            replace_combo,
                            tune_param["constraint"],
                        )
                    )
            my_gen_file_config = gen_file_config.copy()
            my_bench_config = bench_config.copy()
            t = threading.Thread(
                target=v0_configuration_generator.generate_candidates_specific_config,
                args=(
                    my_gen_file_config,
                    my_bench_config,
                    list(config["tune_params"].values()),
                    work_queue,
                    dispatch_queue,
                    best_configs,
                    cached_results,
                ),
            )
            instance_gen_threads.append(t)
            t.start()
            bench_configs[gen_file_config["uid"]] = my_bench_config
            gen_file_configs[gen_file_config["uid"]] = my_gen_file_config

    benchmark_thread = threading.Thread(
        target=benchmarker_thread,
        args=(
            benchmark_queue,
            best_configs,
            bench_configs,
            gen_file_configs,
            cached_results,
        ),
    )
    benchmark_thread.start()

    time_start = time.time()
    time_last_print = time_start
    file_idx = 0
    shutdown = False
    while True:
        if not shutdown:
            dispatch_queue.put(file_idx)
        file_idx += 1
        if time.time() - time_last_print > print_every_n_sec or shutdown:
            time_last_print = time.time()
            print("Total time: " + str(time_last_print - time_start))
            print("Total files tested: " + str(file_idx))
            print("Internat stats: ")
            print("compile queue entries:" + str(work_queue.qsize()))
            print("benchmark queue entries: " + str(benchmark_queue.qsize()))
            print_stats(best_configs)

            # write output, if specified
            if output_dir != "":
                for gen_config in gen_file_configs.values():
                    gen_config["replace_vals"] = best_configs[gen_config["uid"]][
                        "best_params"
                    ]
                instance_writer.create_file(list(gen_file_configs.values()), output_dir)
            if shutdown:
                print("Shutting down")
                break
        min_one_thread_alive = False
        for instance_gen_thread in instance_gen_threads:
            if instance_gen_thread.is_alive():
                min_one_thread_alive = True
        if not min_one_thread_alive:
            print("No more work in queue. Initiating shutdown sequence")
            for compile_thread in compile_threads:
                work_queue.put([None, None, None, None, None])
            for compile_thread in compile_threads:
                compile_thread.join()
            benchmark_queue.put([None, None, None, None, None])
            benchmark_thread.join()

            shutdown = True
            continue
        dispatch_queue.join()


def benchmarker_thread(
    benchmark_queue, best_configs, bench_configs, gen_file_configs, cached_results
):
    global n_improvments_mc, n_improvments_rv, n_improvments_cache, n_perf_extracted
    global n_build_failures, n_bench_failures, n_success
    n_verify_failures = 0
    verify_time = 0
    n_valid_benchmarked = 0

    is_first = True
    instance_file_name = ""
    while True:
        print("Hello bench alive")

        # delete last file
        if not is_first:
            try:
                os.remove(instance_file_name)
                os.remove(instance_file_name + ".o")
            except OSError:
                pass
        is_first = False
        instance_file_name, build_result, params, source, uid = benchmark_queue.get()
        if instance_file_name is None:
            break
        bconf = best_configs[uid]
        bench_config = bench_configs[uid]
        gen_file_config = gen_file_configs[uid]

        print("N valid benchmarked %d build fails %d ver fails %d time verifying %f" % (n_valid_benchmarked, n_build_failures, n_verify_failures, verify_time))

        if build_result == 0:
            print("Going to do benches for params ", params)
            bench_results = benchmarker.benchmark_example(
                instance_file_name,
                bench_config["bench_args"],
                (int(gen_file_config["verbose_level"]) >= 1),
                (int(gen_file_config["verbose_level"]) >= 1),
            )
            n_valid_benchmarked += 1
        else:
            n_build_failures += 1
            continue

        n_success += 1

        for i in range(len(bench_results)):
            if bench_results[i] > 0:
                n_perf_extracted += 1
                if (
                    bench_results[i] < bconf["best_score_so_far"][i]
                    or bconf["best_score_so_far"][i] == 0
                ):
                    # now, run again in verify mode to be sure it actually is a valid configuration
                    ver_t1 = time.time()
                    bench_results_verify = benchmarker.benchmark_example(
                        instance_file_name,
                        [bench_config["verify_args"][i]],
                        (int(gen_file_config["verbose_level"]) >= 1),
                    )
                    verify_time += time.time() - ver_t1
                    if bench_results_verify[0] < 0:
                        n_verify_failures += 1
                        n_bench_failures += 1
                        continue

                    # verbose output - only if new best solution for config found
                    bconf["best_score_so_far"][i] = bench_results[i]
                    bconf["best_params"][i] = params[
                        len(gen_file_config["gen_replace_strs"]) :
                    ]
                    print("-------------------------------------------------------")
                    print(
                        "found new better solution for "
                        + gen_file_config["uid"]
                        + " "
                        + bench_config["bench_args"][i]
                    )
                    print("score is " + str(bconf["best_score_so_far"][i]))
                    print("params are: ")
                    print(bconf["best_params"][i])
                    print("came from " + source)
                    print("")
                    if source == " random variation":
                        n_improvments_rv += 1
                    if source == " monte carlo":
                        n_improvments_mc += 1
                    if "cache from" in source:
                        n_improvments_cache += 1
                    # store in all cache levels
                    # create a unique id for gen_params and test config
                    cache_id = gen_file_config["uid"] + bench_config["bench_args"][i]
                    for cache_level in gen_file_config["cache_points"]:
                        # create dict if it dosent exist yet
                        if cache_level not in cached_results.keys():
                            cached_results[cache_level] = {}

                        cached_results[cache_level][
                            cache_id
                        ] = {}  # overwrite if it already exists
                        # enter a key for every param we have
                        dict_write_param_idx = 0
                        for rep_str in gen_file_config["replace_strs"]:
                            cached_results[cache_level][cache_id][str(rep_str)] = bconf[
                                "best_params"
                            ][i][dict_write_param_idx]
                            dict_write_param_idx += 1

                    print("\n===== CURRENT BEST RESULTS =====")
                    for qqi, qqscore in enumerate(bconf["best_score_so_far"]):
                        if qqscore and qqscore > 0:
                            print(f"{bench_config["bench_args"][qqi]} | score={qqscore:.3f} | params={bconf["best_params"][qqi]}")
                    print("================================\n")


def print_stats(best_configs):
    global \
        n_build_failures, \
        n_success, \
        n_bench_failures, \
        n_improvments_mc, \
        n_improvments_rv, \
        n_improvments_cache, \
        n_perf_extracted
    print("Report card since last print:")
    print(str(n_bench_failures) + " Benchmark failures")
    print(
        str(n_success)
        + " successful benchmarks with "
        + str(n_perf_extracted)
        + " performance figures gathered"
    )
    print("")
    print(
        "Overall, "
        + str(n_improvments_cache + n_improvments_mc + n_improvments_rv)
        + " improvements were made"
    )
    print(str(n_improvments_mc) + " came from monte carlo generation")
    print(
        str(n_improvments_cache)
        + " came from cached configurations of other setups (eg. bf16->f16)"
    )
    print(
        str(n_improvments_rv)
        + " came from random variation of current known best solutions"
    )

    print("Individual performances: \n")
    for known_config in best_configs.keys():
        print(
            "perf for "
            + known_config
            + " is "
            + str(best_configs[known_config]["best_score_so_far"])
        )
        print("parameters for config:")
        if "best_params" in best_configs[known_config].keys():
            for param_set in best_configs[known_config]["best_params"]:
                if param_set != "":
                    print(param_set)

    n_build_failures = 0
    n_success = 0
    n_bench_failures = 0
    n_improvments_cache = 0
    n_improvments_mc = 0
    n_improvments_rv = 0
    n_perf_extracted = 0
