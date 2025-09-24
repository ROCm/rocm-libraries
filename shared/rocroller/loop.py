#!/usr/bin/env python3

import os
import subprocess
import shutil
import glob
import argparse
import itertools
import json
from dataclasses import dataclass


@dataclass
class TestCombination:
    instr_width: int
    write: bool  # False for read
    stride: int
    iters: int
    
    @property
    def mode_name(self) -> str:
        return "write" if self.write else "read"
    
    def to_env_vars(self) -> dict:
        """Convert this combination to environment variables"""
        return {
            "INSTR_WIDTH": str(self.instr_width),
            "WRITE": "1" if self.write else "0",
            "BYTE_STRIDE": str(self.stride),
            "ITERS": str(self.iters)
        }
    
    def get_output_dir_name(self) -> str:
        return f"ds_{self.mode_name}_b{self.instr_width * 32}_stride_{self.stride}_iters_{self.iters}"
    
    def get_working_dir_name(self) -> str:
        return f"{self.get_output_dir_name()}_rocprof"


def main():
    parser = argparse.ArgumentParser(
        description="rocprofv3 profiling of ds instructions with various options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    env_group = parser.add_argument_group('Options', 
                                          'Environment variables that are passed as options for the test')

    env_group.add_argument(
        "instr_widths",
        type=int,
        nargs="*",
        default=[1, 2, 4],
        help="INSTR_WIDTH values in dwords",
    )

    env_group.add_argument(
        "-s",
        "--strides",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="BYTE_STRIDE values",
    )

    env_group.add_argument(
        "-w", "--write", action="store_true", help="Test ds_write_*"
    )

    env_group.add_argument(
        "-r", "--read", action="store_true", help="Test ds_read_*"
    )

    env_group.add_argument(
        "-i",
        "--iters",
        type=int,
        nargs="+",
        default=range(4, 129),
        help="ITERS values (number of iterations)",
    )

    # Sometimes no work is done on CU thus needs to be re-ran (for small workgroup counts)
    # If the output is too short (i.e. only csv headers), repeat the run
    parser.add_argument(
        "--char-limit",
        type=int,
        default=77,
        help="Minimum character limit for output validation (default: 77)",
    )

    parser.add_argument(
        "--rocprof-path",
        type=str,
        default="/opt/rocm/bin/rocprofv3",
        help="Path to rocprofv3 executable (default: /opt/rocm/bin/rocprofv3)",
    )

    parser.add_argument(
        "--exe-cmd",
        type=str,
        default="./test/rocroller-tests --gtest_filter=*GPU_LoopLDSKernel*",
        help="Executable command to profile (default: rocroller-tests with GPU_KernelTest filter)",
    )

    args = parser.parse_args()

    # Install ATT decoder if not present
    if not os.path.isfile("/opt/rocm-7.1.0/lib/librocprof-trace-decoder.so"):
        wget_cmd = [
            "wget",
            "https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.2/rocprof-trace-decoder-ubuntu-22.04-0.1.2-Linux.deb",
        ]
        subprocess.run(wget_cmd, check=True)
        dpkg_cmd = [
            "sudo",
            "dpkg",
            "-i",
            "rocprof-trace-decoder-ubuntu-22.04-0.1.2-Linux.deb",
        ]
        subprocess.run(dpkg_cmd, check=True)

    modes_to_test = []
    if args.write and args.read:
        modes_to_test = [False, True]  # False for read, True for write
    elif args.write:
        modes_to_test = [True]
    elif args.read:
        modes_to_test = [False]
    else: # default both
        modes_to_test = [False, True]

    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    test_combinations = [TestCombination(*combo) for combo in itertools.product(
        args.instr_widths,  
        modes_to_test,      
        args.strides,
        args.iters  # Added ITERS to the product
    )]
    
    print(f"Instruction widths: {args.instr_widths}")
    print(f"Modes: {['write' if mode else 'read' for mode in modes_to_test]}")
    print(f"Strides: {args.strides}")
    print(f"Iterations: {args.iters}")
    print(f"Total test combinations: {len(test_combinations)}")

    for i, test in enumerate(test_combinations, 1):
        print(f"\n[{i}/{len(test_combinations)}] Testing combination: {test}")
        
        os.environ.update(test.to_env_vars())
        
        rocprof_working_dir = os.path.join(output_dir, test.get_working_dir_name())

        # Loop in case of rocprofiler issues
        for attempt in range(1, 20):
            print(f"Attempt {attempt} for combination: {test}")
                  
            if os.path.exists(rocprof_working_dir):
                shutil.rmtree(rocprof_working_dir)

            rocprof_cmd = [
                args.rocprof_path,
                "--att",
                "-d",
                rocprof_working_dir + "/",
                "--att-perfcounter-ctrl=1",
                "--att-perfcounters=SQ_LDS_BANK_CONFLICT,SQ_LDS_IDX_ACTIVE,SQ_INST_LEVEL_LDS,SQ_ACCUM_PREV_HIRES",
                "--att-target-cu=1",
                "--att-shader-engine-mask=0x1",
                "--",
            ] + args.exe_cmd.split()
            
            print("Running command:", " ".join(rocprof_cmd))

            try:
                subprocess.run(rocprof_cmd, check=True)

                # Adjust output csv file if multiple dispatches (e.g. due to hipmemcpy)
                csv_files = glob.glob(
                    f"{rocprof_working_dir}/stats_ui_output_agent_*_dispatch_1.csv"
                )
                if not csv_files:
                    print("No CSV output file found, retrying...")
                    continue

                if len(csv_files) > 1:
                    raise Exception("Multiple CSV files found", csv_files)

                with open(csv_files[0], "r") as f:
                    output = f.read()

                output_len = len(output)

                target_dir = os.path.join(output_dir, test.get_output_dir_name())

                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                shutil.move(rocprof_working_dir, target_dir)
                
                env_vars_file = os.path.join(target_dir, "env_vars.json")
                with open(env_vars_file, "w") as f:
                    json.dump(test.to_env_vars(), f, indent=2)

                print(f"length: {output_len}")
                print(f"Results for {test} saved in {target_dir}")

                if output_len > args.char_limit:
                    break

            except Exception as e:
                print(f"Unexpected error: {e}")
                return
            
        else: # no break
            print(f"Failed to get valid output after multiple attempts for {test}")
            return

if __name__ == "__main__":
    main()
