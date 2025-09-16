#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
import glob
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run rocprofv3 profiling ds instructions with different strides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "instr_widths",
        type=int,
        nargs="*",
        default=[1, 2, 4],
        help="Instruction width values in dwords",
    )

    parser.add_argument(
        "-s",
        "--strides",
        type=int,
        nargs="+",
        default=[2**i for i in range(0, 4)],
        help="List of byte stride values to test",
    )

    parser.add_argument(
        "-w", "--write", action="store_true", help="Test write mode (sets WRITE=1)"
    )

    parser.add_argument(
        "-r", "--read", action="store_true", help="Test read mode (sets WRITE=0)"
    )

    parser.add_argument(
        "--char-limit",
        type=int,
        default=77,
        help="Minimum character limit for output validation (default: 76)",
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

    # Determine which modes to test
    modes_to_test = []
    if args.write and args.read:
        modes_to_test = [("read", False), ("write", True)]
    elif args.write:
        modes_to_test = [("write", True)]
    elif args.read:
        modes_to_test = [("read", False)]
    else:
        # Default: test both modes
        modes_to_test = [("read", False), ("write", True)]

    # Set static environment variables
    os.environ["ROCROLLER_BUILD_DIR"] = "./"
    os.environ["ROCROLLER_SAVE_ASSEMBLY"] = "1"

    # Due to how rocprofv3 works, sometimes data is not recorded and thus needs to be re-ran
    # If the output is too short (i.e. only csv headers), repeat the run
    CHAR_LIMIT = args.char_limit

    output_dir = "output"

    if os.path.exists(output_dir):
        print(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    print(f"Testing instruction widths: {args.instr_widths}")
    print(f"Testing modes: {[mode[0] for mode in modes_to_test]}")
    print(f"Testing strides: {args.strides}")
    print(
        f"Output will be saved to: {output_dir}/ds_<mode>_b<instr_width_bits>_stride_<stride>/"
    )

    # Loop through instruction widths
    for instr_width in args.instr_widths:
        print(f"\n=== Testing INSTR_WIDTH={instr_width} ===")

        # Set INSTR_WIDTH environment variable
        os.environ["INSTR_WIDTH"] = str(instr_width)

        # Loop through modes
        for mode_name, write_flag in modes_to_test:
            print(
                f"\n--- Testing {mode_name.upper()} mode for INSTR_WIDTH={instr_width} ---"
            )

            # Set WRITE environment variable for this mode
            os.environ["WRITE"] = "1" if write_flag else "0"

            # Loop through user-specified strides
            for stride in args.strides:
                # Set stride environment variable
                os.environ["BYTE_STRIDE"] = str(stride)

                rocprof_dir = f"{output_dir}/rocprof_{instr_width}_{mode_name}_{stride}"

                # Loop in case of rocprofiler issues
                while True:
                    print(
                        f"Trying INSTR_WIDTH={instr_width} BYTE_STRIDE={stride} WRITE={os.environ['WRITE']} ({mode_name} mode)"
                    )

                    # Remove rocprof directory if it exists
                    if os.path.exists(rocprof_dir):
                        shutil.rmtree(rocprof_dir)

                    # Build the rocprof command
                    rocprof_cmd = [
                        args.rocprof_path,
                        "--att",
                        "-d",
                        rocprof_dir + "/",
                        "--att-perfcounter-ctrl=8",
                        "--att-perfcounters=SQ_LDS_BANK_CONFLICT,SQ_LDS_IDX_ACTIVE,SQ_LDS_MEM_VIOLATIONS,SQ_INST_LEVEL_LDS",
                        "--att-target-cu=1",
                        "--att-shader-engine-mask=0x1",
                        "--",
                    ] + args.exe_cmd.split()

                    try:
                        # Run rocprof command
                        subprocess.run(rocprof_cmd, check=True)

                        # Read the output file
                        csv_files = glob.glob(
                            f"{rocprof_dir}/stats_ui_output_agent_*_dispatch_1.csv"
                        )
                        if not csv_files:
                            print("No CSV output file found, retrying...")
                            continue

                        with open(csv_files[0], "r") as f:
                            output = f.read()

                        output_len = len(output)

                        # Move rocprof directory to organized output directory
                        stats_dir_name = (
                            f"ds_{mode_name}_b{instr_width * 32}_stride_{stride}"
                        )
                        target_dir = os.path.join(output_dir, stats_dir_name)

                        if os.path.exists(target_dir):
                            shutil.rmtree(target_dir)
                        shutil.move(rocprof_dir, target_dir)

                        print(f"length: {output_len}")
                        print(f"Results saved to: {target_dir}")

                        # Check if output is long enough
                        if output_len > CHAR_LIMIT:
                            break

                    except Exception as e:
                        print(f"Unexpected error: {e}")
                        return


if __name__ == "__main__":
    main()
