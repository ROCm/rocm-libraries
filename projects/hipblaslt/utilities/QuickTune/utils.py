# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import os
import re
import csv

def parse_hipblaslt_output(output, line, tuning_info, mode):
    outputs = output.split('\n')
    print(output)

    # Check if NO solution was found
    if "NO solution found!" in output:
        print(f"WARNING: No solution found for {mode} mode")
        latency = -1
        solution_idx = "NO_SOLUTION"
    else:
        try:
            if mode == 'baseline':
                latency = float(outputs[-5].split(',')[-1])
            else:
                latency = float(outputs[-5].split(',')[-2])
            solution_idx = outputs[-4].split(':')[-1]
        except (ValueError, IndexError) as e:
            print(f"ERROR: Failed to parse output for {mode} mode: {e}")
            latency = -1
            solution_idx = "PARSE_ERROR"

    if mode == 'baseline':
        tuning_info[line].update({"baseline_latency(us)": latency})
        tuning_info[line].update({"baseline_solution_idx": solution_idx})
        # Set status if baseline failed
        if latency == -1:
            tuning_info[line].update({"status": "BASELINE_FAILED"})
    elif mode == 'tuning':
        tuning_info[line].update({"tuned_latency(us)": latency})
        tuning_info[line].update({"tuned_solution_idx": solution_idx})
        baseline_latency = tuning_info[line].get('baseline_latency(us)', -1)
        if baseline_latency > 0 and latency > 0:
            ratio = baseline_latency / latency * 100
            ratio = round(ratio, 2)
            tuning_info[line].update({"baseline/tuned": f"{ratio}%"})
            tuning_info[line].update({"status": "SUCCESS"})
        else:
            tuning_info[line].update({"baseline/tuned": "N/A"})
            # Set status based on what failed
            if baseline_latency == -1 and latency == -1:
                tuning_info[line].update({"status": "BOTH_FAILED"})
            elif latency == -1:
                tuning_info[line].update({"status": "TUNING_FAILED"})
            # If only baseline failed, status is already "BASELINE_FAILED" from baseline mode

def export_csv(input_file, tuning_info, args):
    fieldnames = ['m', 'n', 'k', 'lda', 'ldb', 'ldc', 'ldd', 'a_type', 'b_type', 'c_type', 'd_type', 'count', 'baseline_latency(us)', 'baseline_solution_idx', 'tuned_latency(us)', 'tuned_solution_idx', 'baseline/tuned', 'status']
    tuning_info_list = []

    success_count = 0
    total_count = 0

    with open(input_file, 'r') as f:
        for line in f:
            info = tuning_info[line]
            status = info.get('status', 'UNKNOWN')

            total_count += 1
            if status == 'SUCCESS':
                success_count += 1

            tuning_info_list.append(info)

    with open(args.output_path + '/tuning_result.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tuning_info_list)

    return success_count, total_count

def dynamic_iters(input_cmd, cold_iters, iters, tuning_info):
    if cold_iters == -1 or iters == -1:
        mnk = (float(tuning_info[input_cmd]['m']) * float(tuning_info[input_cmd]['n']) * float(tuning_info[input_cmd]['k'])) / (1024*1024)
        if mnk <= 4000:
            cold_iters, iters = 200, 100
        elif mnk <= 80000:
            cold_iters, iters = 50, 20
        else:
            cold_iters, iters = 10, 2
    return cold_iters, iters

def convert_command(input_cmd, args, tuning_info, mode):
    requested_solution = args.requested_solution
    iters = args.iters
    cold_iters = args.cold_iters

    output_cmd = input_cmd.strip()
    cold_iters, iters = dynamic_iters(input_cmd, cold_iters, iters, tuning_info)

    output_cmd = re.sub(f'--algo_method\s+\S+', '', output_cmd)
    output_cmd = re.sub(f'--solution_index\s+\S+', '', output_cmd)
    output_cmd = re.sub(f'--requested_solution\s+\S+', '', output_cmd)
    output_cmd = re.sub(f'--cold_iters\s+\S+', '', output_cmd)
    output_cmd = re.sub(f'--iters\s+\S+', '', output_cmd)
    output_cmd = re.sub(f'--rotating\s+\S+', '', output_cmd)
    output_cmd = re.sub(f'--aux_type\s+\S+', '', output_cmd)
    output_cmd = re.sub(f'\s{2,}', '', output_cmd)

    output_cmd = output_cmd + f" --algo_method heuristic"
    output_cmd = output_cmd + f" --cold_iters {cold_iters}"
    output_cmd = output_cmd + f" --iters {iters}"
    output_cmd = output_cmd + f" --rotating 512 --flush --print_kernel_info"
    if mode == "baseline":
        output_cmd = output_cmd + f" --requested_solution 1"
    else:
        output_cmd = output_cmd + f" --requested_solution {requested_solution}"
        output_cmd = output_cmd + f" --skip_slow_solution_ratio 0.8"
        if args.swizzleA == True:
            output_cmd = re.sub("--transA\s+\S+", "--transA T", output_cmd)
            output_cmd = output_cmd + f" --swizzleA"

    return output_cmd
