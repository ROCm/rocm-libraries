import gentune_utils
import candidate_generation_monte_carlo
import candidate_generation_random_variation
import random

# global definitions
p_monte_carlo = 0.05
p_use_random_var = 0.99
p_random_change = 0.05


# brief: this function generates configurations for the tuner threads to work on
# for this, it uses multiple strategies, like monte carlo, random variation or cache
# each configuration runs in it's own thread
# function is conrolled from main thread through dispatch_queue and puts it's configurations in work_queue,
# from wich the worker threads pull work
def generate_candidates_specific_config(
    gen_file_config,
    bench_config,
    tune_params,
    work_queue,
    dispatch_queue,
    best_configs,
    cached_results,
):
    num_res = len(bench_config["bench_args"])
    global p_monte_carlo, p_random_change, p_use_random_var

    bconf = best_configs[gen_file_config["uid"]]
    bconf["best_score_so_far"] = []
    bconf["best_params"] = []
    bconf["tested_so_far"] = {}
    for i in range(num_res):
        bconf["best_score_so_far"].append(0)
        bconf["best_params"].append("")

    newreplace_strs, new_replace_vals = (
        candidate_generation_monte_carlo.create_random_instance(tune_params)
    )

    test_instance_config = {}
    gen_file_config["replace_strs"] = newreplace_strs
    test_instance_config["replace_strs"] = (
        gen_file_config["gen_replace_strs"] + newreplace_strs
    )
    test_instance_config["test_instance_template_path"] = bench_config[
        "test_instance_template_path"
    ]
    test_instance_config["test_instance_dir"] = bench_config["test_instance_dir"]
    test_instance_config["compile_command"] = bench_config["compile_command"].replace(
        "\n", ""
    )
    test_instance_config["print_build_failures"] = (
        int(gen_file_config["verbose_level"]) >= 2
    )

    # note that, depending on the gentune file configuration, there is the possibility that there
    # are no valid parameter combinations. Thus we implement an abort if the attempts are to many
    num_attempts = 0
    max_attempts_per_round = 1000

    brute_force_mode = False
    brute_force_combos = []
    brute_force_idx = 0
    while True:
        # first round: block until we're told to run
        if num_attempts == 0:
            file_idx = dispatch_queue.get()
        # increment counters and check for abort
        if num_attempts > max_attempts_per_round:
            print(
                "WARNING: could only generate valid parameter set for "
                + gen_file_config["uid"]
                + " despite "
                + str(num_attempts)
                + " attempts"
            )
            print(
                "Either CONSTRAINTS do not allow valid combianations or the search space has been exhausted"
            )
            print("Aborting/Continuing to next parameter set")
            dispatch_queue.task_done()
            num_attempts = 0
            continue
        num_attempts += 1

        ran = random.random()
        # generate random instance
        source = ""  # inform user of source of new optimal solution
        # if we've covered more than 50% of all configs so far, switch to brute force mode
        if (
            not brute_force_mode
            and not gen_file_config["parameter_space_size"] == -1
            and len(bconf["tested_so_far"].keys())
            >= 0.5 * gen_file_config["parameter_space_size"]
        ):
            brute_force_mode = True
            print(
                "More than 50 percent of parameter space for "
                + str(gen_file_config["uid"])
                + "has been exhausted. Switching to brute force mode"
            )
            newreplace_strs, brute_force_combos = gentune_utils.create_all_combos(
                tune_params
            )
            print("Total brute force combinations: " + str(len(brute_force_combos)))
        if brute_force_mode:
            source = "Brute Force"
            if (
                len(bconf["tested_so_far"].keys())
                == gen_file_config["parameter_space_size"]
            ):
                print(
                    "Parameter space for "
                    + str(
                        gen_file_config["uid"]
                        + " has been exhaused. Exiting (note this applies only to the thread generating candidates for this combo. There may or may not be other threads generating candidates)"
                    )
                )
                dispatch_queue.task_done()
                break
            new_replace_vals = [brute_force_combos[brute_force_idx]]
            brute_force_idx += 1
            ran = 2

        if ran < p_monte_carlo:
            source = " monte carlo"
            newreplace_strs, new_replace_vals = (
                candidate_generation_monte_carlo.create_random_instance(tune_params)
            )
        elif ran < p_use_random_var:  # generate random variation of known instances
            source = " random variation"
            ran = random.randint(0, num_res - 1)
            if bconf["best_score_so_far"][ran] != 0:
                new_replace_vals, success = (
                    candidate_generation_random_variation.create_small_variation(
                        bconf["best_params"][ran], tune_params, p_random_change
                    )
                )
                if not success:
                    continue
            else:
                continue
        elif ran < 1.0:
            # use stored solution from one of the cache points
            if gen_file_config["cache_points"]:
                # pick a random lookup dict along the hiarchy
                len_cache_points = len(gen_file_config["cache_points"])
                ran = random.randint(0, len_cache_points - 1)
                lookup_key = gen_file_config["cache_points"][ran]
                source = "cache from " + lookup_key
                if lookup_key in cached_results.keys():
                    # pick a random entry
                    lookup_dict = cached_results[lookup_key]
                    n_applicable_cache_entries = len(lookup_dict.keys())
                    ran = random.randint(0, n_applicable_cache_entries - 1)
                    lookup_dict = list(lookup_dict.values())[ran]

                    # kabraham: the purpose of the caching is to share results between similar but different configs
                    # these configs may have slightly different parameters.
                    # thus, we have to check if the parameters exist, are allowed by our constraints,
                    # and generate them by random if they don't
                    newreplace_strs = []
                    new_replace_vals = [[]]
                    matching_success = True
                    for param in tune_params:
                        # create all combinations
                        newreplace_strs, new_replace_vals = gentune_utils.add_param_dim(
                            newreplace_strs, new_replace_vals, param
                        )

                        # it's possible we've navigated into a dead end
                        if len(new_replace_vals) == 0:
                            matching_success = False
                            break
                        # check if this replace str is in cache
                        take_from_dict = False
                        if str(newreplace_strs[-1]) in lookup_dict.keys():
                            # check if the value from dict is in our possible values (note that we may have other constraints)
                            for combo in new_replace_vals:
                                # print("tacking from dict")
                                if lookup_dict[str(newreplace_strs[-1])] == combo[-1]:
                                    take_from_dict = True
                                    new_replace_vals = [combo]
                                    break

                        if not take_from_dict:
                            # print("tacking at random")
                            # if it wasn't possible to take from dict, pick one at random
                            ran = random.randint(0, len(new_replace_vals) - 1)
                            new_replace_vals = [new_replace_vals[ran]]
                    if not matching_success:
                        continue
                else:
                    continue
            else:
                continue

        # check if this exact config has been tested so far
        if str(new_replace_vals) in bconf["tested_so_far"].keys():
            continue
        else:
            bconf["tested_so_far"][str(new_replace_vals)] = 1

        # if we've arrived at this point, we've successfully created a parameter set.
        # we can append this to the work queue, increment indexes and continue
        work_queue.put(
            [
                (gen_file_config["gen_replace_vals"] + new_replace_vals[0]),
                file_idx,
                source,
                gen_file_config["uid"],
                test_instance_config,
            ]
        )
        num_attempts = 0
        dispatch_queue.task_done()
