import os
import argparse
import subprocess
import time

from utils import parse_input_log, parse_hipblaslt_output, export_csv, dynamic_iters, convert_command

def run_baseline(input_file, args, tuning_info):
    # Delete Tuning File Environment Variable to avoid tuning
    hipblaslt_tuning_file = ""
    if "HIPBLASLT_TUNING_FILE" in os.environ:
        hipblaslt_tuning_file = os.environ["HIPBLASLT_TUNING_FILE"]
        del os.environ["HIPBLASLT_TUNING_FILE"]

    with open(input_file, 'r') as f, open(args.output_path + '/baseline_reproduce_commands.log', 'w+') as f_out:
        for line in f:
            command = convert_command(line, args, tuning_info, "baseline")
            f_out.write(command + '\n')
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            parse_hipblaslt_output(result.stdout, line, tuning_info, "baseline")

    # Restore HIPBLASLT_TUNING_FILE
    if hipblaslt_tuning_file != "":
        os.environ["HIPBLASLT_TUNING_FILE"] = hipblaslt_tuning_file

def run_tuning(input_file, args, tuning_info):
    # set default tuning file if it is not in the environment
    default_tuning_file = False
    if "HIPBLASLT_TUNING_FILE" not in os.environ:
        default_tuning_file = True
        os.environ["HIPBLASLT_TUNING_FILE"] = args.output_path + "/tuning.txt"

    with open(input_file, 'r') as f, open(args.output_path + '/tuning_reproduce_commands.log', 'w+') as f_out:
        for line in f:
            command = convert_command(line, args, tuning_info, "tuning")
            f_out.write(command + '\n')
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            parse_hipblaslt_output(result.stdout, line, tuning_info, "tuning")

    # Remove HIPBLASLT_TUNING_FILE
    if default_tuning_file:
        del os.environ["HIPBLASLT_TUNING_FILE"]

def gpu_setup(device):
    os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
    subprocess.run(["rocm-smi", "--setperfdeterminism", "1900", "-d", str(device)], check=True)

def gpu_reset(device):
    del os.environ["HIP_FORCE_DEV_KERNARG"]
    subprocess.run(["rocm-smi", "-r", "-d", str(device)], check=True)

def main():
    parser = argparse.ArgumentParser(description="Execute Gemm Tuning")
    parser.add_argument("--input_file", type=str, help="Path to the list of gemm ops")
    parser.add_argument("--output_path", type=str, default='./tuning_result', help="Path to output file")
    parser.add_argument("--swizzleA", action='store_true', help="Whether to enable swizzleA tuning")
    parser.add_argument("--swizzleB", action='store_true', help="Whether to enable swizzleB tuning")
    parser.add_argument("--requested_solution", type=int, default=128, help="Searching space for gemm tuning")
    parser.add_argument("--cold_iters", type=int, default=-1, help="warm-up iteration to measure kernel performance")
    parser.add_argument("--iters", type=int, default=-1, help="iteration to measure kernel performance")
    parser.add_argument("--gpu_id", type=int, default=0, help="select gpu devices")
    parser.add_argument("--stablize_gpu", action='store_true', help="whether to stablelize GPU frequency")
    args = parser.parse_args()

    # set_up gpu status
    os.environ["HIPBLASLT_LOG_MASK"] = "32"
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.stablize_gpu:
        gpu_setup(args.gpu_id)

    tuning_info = dict()
    os.makedirs(args.output_path, exist_ok=True)

    start_time = time.time()
    unique_log_name = parse_input_log(args, tuning_info)
    end_time = time.time()
    print("parsing time elapsed = {}".format(end_time - start_time))

    start_time = time.time()
    run_baseline(unique_log_name, args, tuning_info)
    end_time = time.time()
    print("baseline time elapsed = {}".format(end_time - start_time))

    start_time = time.time()
    run_tuning(unique_log_name, args, tuning_info)
    end_time = time.time()
    print("tuning time elapsed = {}".format(end_time - start_time))

    # reset gpu status
    if args.stablize_gpu:
        gpu_reset(args.gpu_id)

    export_csv(unique_log_name, tuning_info, args)


if __name__ == "__main__":
    main()









