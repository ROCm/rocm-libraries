# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import os
import sys
import subprocess
import argparse
import time


def run_gtest_shards(test_binary, num_shards, gtest_filter=None):
    start_time = time.time()
    processes = []
    binary_dir = os.path.dirname(os.path.abspath(test_binary))
    print(f"Test binary directory: {binary_dir}")
    for shard_index in range(num_shards):
        env = os.environ.copy()
        env['GTEST_TOTAL_SHARDS'] = str(num_shards)
        env['GTEST_SHARD_INDEX'] = str(shard_index)
        cmd = [os.path.abspath(test_binary)]
        if gtest_filter:
            cmd.append(f'--gtest_filter={gtest_filter}')
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=binary_dir
        )
        processes.append(proc)
    exit_codes = [p.wait() for p in processes]
    total_time = time.time() - start_time
    print(f"Total test time: {total_time * 1000:.0f} ms")
    if any(code != 0 for code in exit_codes):
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run a Google Test binary with sharding.')
    parser.add_argument('test_binary', help='Path to the Google Test binary')
    parser.add_argument('--num_shards', type=int, required=True, help='Number of shards (processes) to launch')
    parser.add_argument('--gtest_filter', type=str, default=None, help='Google Test filter pattern')
    args = parser.parse_args()
    run_gtest_shards(args.test_binary, args.num_shards, args.gtest_filter)

if __name__ == '__main__':
    main()