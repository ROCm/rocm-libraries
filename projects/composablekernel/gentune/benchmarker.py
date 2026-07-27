import instance_writer
import subprocess
import re

ERROR_BENCHMARK_FAILURE = -100
ERROR_COMPILE_FAILURE = -3
ERROR_EXECUTION_FAILED = -2
ERROR_UNSUPPORTED = -1
SUCCESS = 0
NO_PERF_GATHERED = -4


def benchmark_example(
    instance_name, bench_args, print_failures=False, print_nosupport=False
):
    performances = []
    for benchIdx in range(len(bench_args)):
        fail_mode = SUCCESS
        res = subprocess.getstatusoutput(
            "./" + instance_name + ".o " + bench_args[benchIdx]
        )
        # check for errors
        if "Error" in res[1]:
            fail_mode = ERROR_BENCHMARK_FAILURE
            if print_failures:
                print("benchmark failure on instance " + instance_name)
                print("Args: " + bench_args)
                print(res[1])

        if res[0] != SUCCESS:
            fail_mode = ERROR_EXECUTION_FAILED
            performances.append(ERROR_EXECUTION_FAILED)

            if print_failures:
                print("bench fail")
                print(instance_name)
                print(res)
        else:
            txt = res[1].split("\n")
            extracted_perf = [
                x.strip()
                for x in txt
                if x.strip().startswith("Perf:")
                or x.strip().endswith("does not support this problem")
            ]
            if len(extracted_perf) == 0:
                performances.append(ERROR_UNSUPPORTED)
            elif len(extracted_perf) != 1:
                raise Exception("examples must contain exaclty one instance")

            for line in extracted_perf:
                if fail_mode < 0:
                    performances.append(ERROR_EXECUTION_FAILED)
                elif line.endswith("does not support this problem"):
                    performances.append(ERROR_UNSUPPORTED)
                    if print_nosupport:
                        print(line)
                else:
                    performances.append(float(re.findall("\\d+\\.\\d+", line)[0]))
    return performances


# note: for benchmarking multiple instances in a single file, not currently in use
def benchmark_configuration(gen_file_config, bench_config):
    instance_writer.create_file([gen_file_config], bench_config["out_filename"].strip())
    res = subprocess.getstatusoutput(
        "cd "
        + gen_file_config["base_code_dir"].strip()
        + " && "
        + bench_config["compile_command"]
    )
    num_entries = len(gen_file_config["replace_vals"])

    # if there are errors, divide-and conquer to find cause of error
    fail_mode = SUCCESS
    performances = []
    if res[0] != 0:
        fail_mode = ERROR_COMPILE_FAILURE
    else:
        for benchIdx in range(len(bench_config["bench_args"])):
            performance = []
            best_perf = NO_PERF_GATHERED
            res = subprocess.getstatusoutput(
                gen_file_config["base_code_dir"].strip()
                + bench_config["bench_cmd"].strip()
                + " "
                + bench_config["bench_args"][benchIdx]
            )
            if res[0] != 0:
                fail_mode = ERROR_EXECUTION_FAILED
                print("bench fail")
                print(
                    gen_file_config["base_code_dir"].strip()
                    + bench_config["bench_cmd"].strip()
                    + " "
                    + bench_config["bench_args"][benchIdx]
                )
                print(res)
            else:
                txt = res[1].split("\n")
                extracted_perf = [
                    x.strip()
                    for x in txt
                    if x.strip().startswith("Perf:")
                    or x.strip().endswith("does not support this problem")
                ]

                for line in extracted_perf:
                    if line.endswith("does not support this problem"):
                        performance.append(ERROR_UNSUPPORTED)
                    else:
                        performance.append(float(re.findall("\\d+\\.\\d+", line)[0]))
                        if performance[-1] > 0 and (
                            best_perf == NO_PERF_GATHERED or performance[-1] < best_perf
                        ):
                            best_perf = performance[-1]
            performances.append(performance)
    if fail_mode != SUCCESS:
        if num_entries == 1:
            return [[fail_mode] * len(bench_config["bench_args"])]
        else:
            lower_config = gen_file_config.copy()
            upper_config = gen_file_config.copy()

            lower_config["replace_vals"] = gen_file_config["replace_vals"][
                : int(num_entries / 2)
            ]
            upper_config["replace_vals"] = gen_file_config["replace_vals"][
                int(num_entries / 2) :
            ]

            lower_perf = benchmark_configuration(lower_config, bench_config)
            upper_perf = benchmark_configuration(upper_config, bench_config)
            for i in range(len(lower_perf)):
                lower_perf[i].append(upper_perf[i])
            return lower_perf
    return performances
