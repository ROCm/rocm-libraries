# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import re
import argparse
import time
import os

def parse_input_log(args, tuning_info):
    
    start_time = time.time()

    # save unique lines in hipblaslt log
    os.makedirs(args.output_path, exist_ok=True)
    unique_log_name = args.output_path + '/unique_' + args.input_file.split('/')[-1]
    f_out = open(unique_log_name, 'w')
    
    with open(args.input_file, 'r') as f:
        for line in f:
            if 'hipblaslt-bench' not in line:
                continue

            if line in tuning_info:
                tuning_info[line]['count'] += 1
                continue
            else:
                tuning_info[line] = {}
                tuning_info[line]['count'] = 1
                f_out.write(line)
            matches = re.findall(r"-(m|n|k)\s+(\d+)", line)
            matches.extend(re.findall(r"--(lda|ldb|ldc|ldd)\s+(\d+)", line))
            matches.extend(re.findall(r"--(a_type|b_type|c_type|d_type)\s+(\S+)", line))
            tuning_info[line].update({key: value for key, value in matches})
    
    end_time = time.time()
    print("parsing time elapsed = {} seconds".format(end_time - start_time))
    print("After removing duplicate, the hipblaslt log is saved at\n", unique_log_name, "\n")

    return unique_log_name

def main():
    parser = argparse.ArgumentParser(description="Execute Gemm Tuning")
    parser.add_argument("--input_file", type=str, help="Path to the list of gemm ops")
    parser.add_argument("--output_path", type=str, default='.', help="Path to output file")
    args = parser.parse_args()
    
    tuning_info = dict()
    unique_log_name = parse_input_log(args, tuning_info)

if __name__ == "__main__":
    main()
