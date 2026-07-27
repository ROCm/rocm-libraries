import gentune_utils
import subprocess
import benchmarker


def write_test_instance(test_instance_config, replace_vals, file_name):
    test_instance_str = ""
    with open(test_instance_config["test_instance_template_path"].strip(), "r") as file:
        lines = file.readlines()
        test_instance_str = "".join(lines)

    final_file_text = gentune_utils.apply_replace_values_to_str(
        test_instance_config["replace_strs"], replace_vals, test_instance_str
    )

    with open(file_name, "w+") as file:
        file.write(final_file_text)


# create and benchmark a new instance
def compile_test_instance(test_instance_config, replace_vals, threadId, fileIdx):
    file_name = (
        test_instance_config["test_instance_dir"].strip()
        + "test_instance_n"
        + str(fileIdx)
    )

    write_test_instance(test_instance_config, replace_vals, file_name)

    # attempt to compile and return
    my_file_compile_cmd = gentune_utils.apply_replace_values_to_str(
        [["INPUT_FILENAME"], ["OUTPUT_NAME"]],
        [["test_instance_n" + str(fileIdx)], ["test_instance_n" + str(fileIdx) + ".o"]],
        test_instance_config["compile_command"],
    )
    res = subprocess.getstatusoutput(
        "cd " + test_instance_config["test_instance_dir"] + " && " + my_file_compile_cmd
    )

    bench_result = 0
    if res[0] != 0:
        bench_result = benchmarker.ERROR_COMPILE_FAILURE
        if test_instance_config["print_build_failures"]:
            print("Build failure in thread " + str(threadId))
            print(res[1])
    return file_name, bench_result
