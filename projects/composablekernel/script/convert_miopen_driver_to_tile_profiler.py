# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Convert miopen driver command to ck Profiler
# Example (single command mode):
#   python3 ../script/convert_miopen_driver_to_tile_profiler.py
#   /opt/rocm/bin/MIOpenDriver conv -n 32 -c 64 -H 28 -W 28 -k 64 -y 3 -x 3
#   -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 32 -F 1 -t 1
#
# Example (batch mode):
#   python3 ../script/convert_miopen_driver_to_tile_profiler.py
#   --input-file commands.txt --output-file results.txt

import argparse
import copy
import subprocess
import shlex
import sys
from io import StringIO


def init_const_args(args, profiler_path=None):
    args.ck_profiler_cmd = profiler_path if profiler_path else "../build/bin/ckProfiler"
    # use decimal values
    args.init_method = 2
    # don't print tensor values
    args.log_value = 0


def filter_to_best_config(output):
    """Filter output to only show the best configuration section."""
    if not output:
        return output

    lines = output.split('\n')
    result_lines = []
    in_best_config = False
    no_instance_found = False

    valid_prefixes = ('Best configuration parameters:', 'name:', 'avg_time:', 'tflops:', 'GB/s:')

    if len(lines) > 1:
        for line in lines:
            if 'Best configuration parameters:' in line:
                in_best_config = True

            if in_best_config:
                stripped = line.strip()
                # Detect sentinel value (FLT_MAX) that indicates no applicable instance
                if stripped.startswith('avg_time:') and '3.40282e+38' in stripped:
                    no_instance_found = True
                if stripped.startswith('Error:') or stripped.startswith('max err:'):
                    continue
                if stripped == '' or any(stripped.startswith(p) for p in valid_prefixes):
                    result_lines.append(line)
                elif stripped and not any(stripped.startswith(p) for p in valid_prefixes):
                    continue

    while result_lines and not result_lines[-1].strip():
        result_lines.pop()

    if no_instance_found:
        return "No applicable instance found"

    return '\n'.join(result_lines) if result_lines else output


def run_ck_profiler_cmd(cmd, capture_output=False):
    cmd_concatenated_str = " ".join(cmd)

    if capture_output:
        output = StringIO()
        output.write("ckProfiler command:\n")
        output.write(cmd_concatenated_str + "\n")
        stderr_text = ""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output.write(result.stdout)
            if result.stderr:
                stderr_text = result.stderr
        except Exception as e:
            stderr_text = f"Error running command: {e}\n"
        return (output.getvalue(), stderr_text)
    else:
        print("ckProfiler command:")
        print(cmd_concatenated_str)
        subprocess.run(cmd)
        return None


def parse_layouts(args):
    if args.in_layout == "NCW" or args.in_layout == "NCHW" or args.in_layout == "NCDHW":
        if args.ck_profier_op == "grouped_conv_bwd_weight_tile":
            args.layout = 4
        elif (
            args.ck_profier_op == "grouped_conv_fwd_tile"
            or args.ck_profier_op == "grouped_conv_bwd_data_tile"
        ):
            args.layout = 3
        else:
            print("Not supported layout for this op")
            exit(1)
    elif (
        args.in_layout == "NWC" or args.in_layout == "NHWC" or args.in_layout == "NDHWC"
    ):
        if args.ck_profier_op == "grouped_conv_bwd_weight_tile":
            args.layout = 2
        elif (
            args.ck_profier_op == "grouped_conv_bwd_data_tile"
            or args.ck_profier_op == "grouped_conv_fwd_tile"
        ):
            args.layout = 1
    else:
        print("Not supported layout for this op")
        exit(1)


def parse_data_type(args):
    if args.data_type == "fp32":
        if (
            args.ck_profier_op == "grouped_conv_bwd_weight_tile"
            or args.ck_profier_op == "grouped_conv_bwd_data_tile"
            or args.ck_profier_op == "grouped_conv_fwd_tile"
        ):
            args.data_type = 0
    if args.data_type == "fp16":
        if (
            args.ck_profier_op == "grouped_conv_bwd_weight_tile"
            or args.ck_profier_op == "grouped_conv_bwd_data_tile"
            or args.ck_profier_op == "grouped_conv_fwd_tile"
        ):
            args.data_type = 1
    if args.data_type == "int8":
        if args.ck_profier_op == "grouped_conv_bwd_weight_tile":
            args.data_type = 4
        if args.ck_profier_op == "grouped_conv_bwd_data_tile":
            print("Not supported data type for grouped_conv_bwd_data_tile")
            exit(1)
        if args.ck_profier_op == "grouped_conv_fwd_tile":
            args.data_type = 3
    if args.data_type == "bfp16":
        if args.ck_profier_op == "grouped_conv_bwd_weight_tile":
            args.data_type = 5
        if (
            args.ck_profier_op == "grouped_conv_bwd_data_tile"
            or args.ck_profier_op == "grouped_conv_fwd_tile"
        ):
            args.data_type = 2


def add_conv_params_to_cmd(args, cmd):
    if args.spatial_dim == 1:
        cmd += [str(args.fil_w), str(args.in_w)]
        cmd += [str(args.conv_stride_w), str(args.dilation_w)]
        cmd += [str(args.pad_w), str(args.pad_w)]
    elif args.spatial_dim == 2:
        cmd += [str(args.fil_h), str(args.fil_w)]
        cmd += [str(args.in_h), str(args.in_w)]
        cmd += [str(args.conv_stride_h), str(args.conv_stride_w)]
        cmd += [str(args.dilation_h), str(args.dilation_w)]
        cmd += [str(args.pad_h), str(args.pad_w)]
        cmd += [str(args.pad_h), str(args.pad_w)]
    elif args.spatial_dim == 3:
        cmd += [str(args.fil_d), str(args.fil_h), str(args.fil_w)]
        cmd += [str(args.in_d), str(args.in_h), str(args.in_w)]
        cmd += [str(args.conv_stride_d), str(args.conv_stride_h)]
        cmd += [str(args.conv_stride_w)]
        cmd += [str(args.dilation_d), str(args.dilation_h), str(args.dilation_w)]
        cmd += [str(args.pad_d), str(args.pad_h), str(args.pad_w)]
        cmd += [str(args.pad_d), str(args.pad_h), str(args.pad_w)]
    else:
        print("Not supported spatial dim (supported: 1, 2, 3)")
        exit(1)


def build_grouped_conv_fwd_cmd(args):
    args.ck_profier_op = "grouped_conv_fwd_tile"
    parse_data_type(args)
    parse_layouts(args)
    # use int32 by default
    args.index_type = 0

    cmd = [str(args.ck_profiler_cmd), str(args.ck_profier_op)]
    cmd += [str(args.data_type), str(args.layout), str(args.index_type)]
    cmd += [str(args.verify), str(args.init_method)]
    cmd += [str(args.log_value), str(args.time)]
    cmd += [str(args.spatial_dim), str(args.group_count)]
    cmd += [str(args.batchsize), str(args.out_channels)]
    cmd += [str(args.in_channels)]
    add_conv_params_to_cmd(args, cmd)

    # Add optional named arguments
    if args.instance != -1:
        cmd += ["--instance", str(args.instance)]
    if args.list_instances:
        cmd += ["--list-instances"]

    return cmd


def run_ck_grouped_conv_fwd(args, capture_output=False):
    cmd = build_grouped_conv_fwd_cmd(args)
    return run_ck_profiler_cmd(cmd, capture_output)


def build_grouped_conv_bwd_data_cmd(args):
    args.ck_profier_op = "grouped_conv_bwd_data_tile"
    parse_data_type(args)
    parse_layouts(args)
    # Only test split-K = 1.
    args.split_k_value = 1

    cmd = [str(args.ck_profiler_cmd), str(args.ck_profier_op)]
    cmd += [str(args.data_type), str(args.layout)]
    cmd += [str(args.verify), str(args.init_method)]
    cmd += [str(args.log_value), str(args.time)]
    cmd += [str(args.spatial_dim), str(args.group_count)]
    cmd += [str(args.batchsize), str(args.out_channels)]
    cmd += [str(args.in_channels)]
    add_conv_params_to_cmd(args, cmd)

    cmd += [str(args.split_k_value)]

    # Add optional named arguments
    if args.instance != -1:
        cmd += ["--instance", str(args.instance)]
    if args.list_instances:
        cmd += ["--list-instances"]

    return cmd


def run_ck_grouped_conv_bwd_data(args, capture_output=False):
    cmd = build_grouped_conv_bwd_data_cmd(args)
    return run_ck_profiler_cmd(cmd, capture_output)


def run_ck_grouped_conv_bwd_weight(args, capture_output=False):
    args.ck_profier_op = "grouped_conv_bwd_weight_tile"
    parse_data_type(args)
    parse_layouts(args)
    # Test all split K value from the list {1, 2, 4, 8, 32, 64, 128}
    args.split_k_value = "all"

    cmd = [str(args.ck_profiler_cmd), str(args.ck_profier_op)]
    cmd += [str(args.data_type), str(args.layout)]
    cmd += [str(args.verify), str(args.init_method)]
    cmd += [str(args.log_value), str(args.time)]
    cmd += [str(args.spatial_dim), str(args.group_count)]
    cmd += [str(args.batchsize), str(args.out_channels)]
    cmd += [str(args.in_channels)]
    add_conv_params_to_cmd(args, cmd)

    cmd += [str(args.split_k_value)]

    # Add optional named arguments
    if args.instance != -1:
        cmd += ["--instance", str(args.instance)]
    if args.list_instances:
        cmd += ["--list-instances"]

    return run_ck_profiler_cmd(cmd, capture_output)


# Get name of miopen driver, remove it from unknown
def process_miopen_driver_name(args, unknown):
    if "convint8" in unknown:
        args.data_type = "int8"
        unknown.remove("convint8")
    elif "convbfp16" in unknown:
        args.data_type = "bfp16"
        unknown.remove("convbfp16")
    elif "convfp16" in unknown:
        args.data_type = "fp16"
        unknown.remove("convfp16")
    elif "conv" in unknown:
        args.data_type = "fp32"
        unknown.remove("conv")
    else:
        print("Not supported driver (supported: conv, convfp16, convint8, convbfp16).")
        exit(1)


def run_ck_profiler(args, capture_output=False):
    # MIOpen get number of channel per all groups, CK profiler get number of
    # channel per group
    args.in_channels = int(args.in_channels / args.group_count)
    args.out_channels = int(args.out_channels / args.group_count)

    outputs = []
    stderr_lines = []

    if args.forw == 0 or args.forw == 1 or args.forw == 3 or args.forw == 5:
        result = run_ck_grouped_conv_fwd(args, capture_output)
        if capture_output and result:
            outputs.append(result[0])
            if result[1]:
                stderr_lines.append(result[1])
    if args.forw == 0 or args.forw == 2 or args.forw == 3 or args.forw == 6:
        result = run_ck_grouped_conv_bwd_data(args, capture_output)
        if capture_output and result:
            outputs.append(result[0])
            if result[1]:
                stderr_lines.append(result[1])
    if args.forw == 0 or args.forw == 4 or args.forw == 5 or args.forw == 6:
        result = run_ck_grouped_conv_bwd_weight(args, capture_output)
        if capture_output and result:
            outputs.append(result[0])
            if result[1]:
                stderr_lines.append(result[1])

    if capture_output:
        return ("\n".join(outputs), "\n".join(stderr_lines))
    return None


def convert_to_profiler_cases(command_line, parser, gpu_verify=False):
    """Convert one MIOpen driver command line into ckProfiler argument strings.

    Returns a list of ``(section, args)`` tuples where ``section`` is ``"fwd"``
    or ``"bwd_data"`` and ``args`` is the ckProfiler argument string excluding
    the executable and the subcommand name (i.e. the same column format used by
    the direct-conv ``cases`` file). The ``-F`` flag selects which directions
    are emitted; ``bwd_weight`` is intentionally omitted as the direct-conv
    bench has no such section.

    Kernel timing is forced on (``time=1``) since the bench measures TFLOPS.
    """
    argv = shlex.split(command_line)
    args, unknown = parser.parse_known_args(argv)
    init_const_args(args)
    process_miopen_driver_name(args, unknown)

    if gpu_verify:
        args.verify = 2
    # The bench reads TFLOPS from the profiler, which requires kernel timing.
    args.time = 1

    # MIOpen channel counts are per-all-groups; ckProfiler expects per-group.
    args.in_channels = int(args.in_channels / args.group_count)
    args.out_channels = int(args.out_channels / args.group_count)

    cases = []
    if args.forw in (0, 1, 3, 5):
        cmd = build_grouped_conv_fwd_cmd(copy.copy(args))
        cases.append(("fwd", " ".join(cmd[2:])))
    if args.forw in (0, 2, 3, 6):
        cmd = build_grouped_conv_bwd_data_cmd(copy.copy(args))
        cases.append(("bwd_data", " ".join(cmd[2:])))
    return cases


def process_single_command(command_line, parser, capture_output=False, profiler_path=None, verbose=False, gpu_verify=False):
    """Process a single MIOpen driver command line."""
    try:
        argv = shlex.split(command_line)
    except ValueError as e:
        error_msg = f"Error parsing command line: {e}\n"
        if capture_output:
            return error_msg
        print(error_msg)
        return None

    args, unknown = parser.parse_known_args(argv)
    init_const_args(args, profiler_path)
    process_miopen_driver_name(args, unknown)

    if gpu_verify:
        args.verify = 2

    if not capture_output:
        print("Ignored args:")
        print(unknown)

    result = run_ck_profiler(args, capture_output)

    if capture_output and result:
        output_str, stderr_str = result
        if verbose:
            return output_str + (f"\nSTDERR:\n{stderr_str}" if stderr_str else "")
        filtered = filter_to_best_config(output_str)
        if stderr_str:
            filtered += f"\nSTDERR:\n{stderr_str}"
        return filtered

    return result


def process_batch_file(input_file, output_file, parser, profiler_path=None, verbose=False, start_line=0, gpu_verify=False):
    """Process a batch file of MIOpen driver commands."""
    try:
        try:
            with open(input_file, 'r', encoding='utf-8') as f_in:
                lines = f_in.readlines()
        except UnicodeDecodeError:
            with open(input_file, 'r', encoding='utf-16') as f_in:
                lines = f_in.readlines()
    except IOError as e:
        print(f"Error reading input file '{input_file}': {e}")
        sys.exit(1)

    total_lines = len(lines)

    try:
        f_out = open(output_file, 'w')
    except IOError as e:
        print(f"Error opening output file '{output_file}': {e}")
        sys.exit(1)

    try:
        if start_line > 1:
            print(f"Continuing from command {start_line} (skipping first {start_line - 1} commands)")
            lines = lines[start_line - 1:]
            total_lines -= (start_line - 1)

        for i, line in enumerate(lines, 0):
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            print(f"Processing command {i}/{total_lines}: {line[:80]}...")

            f_out.write(f"{'='*80}\n")
            f_out.write(f"Input command: {line}\n")
            f_out.write(f"{'='*80}\n")

            output = process_single_command(
                line, parser,
                capture_output=True,
                profiler_path=profiler_path,
                verbose=verbose,
                gpu_verify=gpu_verify,
            )
            if output:
                f_out.write(output)
                f_out.write("\n")
            f_out.write("\n")

            f_out.flush()

        print(f"\nResults written to '{output_file}'")
    finally:
        f_out.close()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="converter",
        description="Convert miopen driver command to ck tile Profiler"
        "\nExample (single command): python3 "
        "../script/convert_miopen_driver_to_tile_profiler.py "
        "/opt/rocm/bin/MIOpenDriver conv -n 32 -c 64 -H 28 -W 28 "
        "-k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g "
        "32 -F 1 -t 1"
        "\nExample (batch mode): python3 "
        "../script/convert_miopen_driver_to_tile_profiler.py "
        "--input-file commands.txt --output-file results.txt",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=False,
        default=None,
        help="Input file containing MIOpen driver commands (one per line). "
        "Enables batch mode.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        default=None,
        help="Output file to store profiler results (required with --input-file).",
    )
    parser.add_argument(
        "--profiler-path",
        type=str,
        required=False,
        default=None,
        help="Path to ckProfiler executable (default: ../build/bin/ckProfiler).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show full profiler output. Default shows only best configuration.",
    )
    parser.add_argument(
        "--continue-from",
        type=int,
        required=False,
        default=0,
        help="Line number to continue from in batch mode (default: 0).",
    )
    parser.add_argument(
        "--gpu-verify",
        action="store_true",
        default=False,
        help="Use GPU verification (passes verify=2 to ckProfiler, overriding the -V flag).",
    )
    parser.add_argument(
        "-in_layout",
        "-I",
        "--in_layout",
        "--I",
        default="NCHW",
        type=str,
        required=False,
        help="Input Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
    )
    parser.add_argument(
        "-forw",
        "-F",
        "--forw",
        "--F",
        default=0,
        type=int,
        required=False,
        help="Flag enables fwd, bwd, wrw convolutions"
        "\n0 fwd+bwd+wrw (default)"
        "\n1 fwd only"
        "\n2 bwd only"
        "\n4 wrw only"
        "\n3 fwd+bwd"
        "\n5 fwd+wrw"
        "\n6 bwd+wrw",
    )
    parser.add_argument(
        "-spatial_dim",
        "-_",
        "--spatial_dim",
        "--_",
        default=2,
        type=int,
        required=False,
        help="convolution spatial dimension (Default-2)",
    )
    parser.add_argument(
        "-batchsize",
        "-n",
        "--batchsize",
        "--n",
        default=100,
        type=int,
        required=False,
        help="Mini-batch size (Default=100)",
    )
    parser.add_argument(
        "-in_channels",
        "-c",
        "--in_channels",
        "--c",
        default=3,
        type=int,
        required=False,
        help="Number of Input Channels (Default=3)",
    )
    parser.add_argument(
        "-in_d",
        "-!",
        "--in_d",
        "--!",
        default=32,
        type=int,
        required=False,
        help="Input Depth (Default=32)",
    )
    parser.add_argument(
        "-in_h",
        "-H",
        "--in_h",
        "--H",
        default=32,
        type=int,
        required=False,
        help="Input Height (Default=32)",
    )
    parser.add_argument(
        "-in_w",
        "-W",
        "--in_w",
        "--W",
        default=32,
        type=int,
        required=False,
        help="Input Width (Default=32)",
    )
    parser.add_argument(
        "-out_channels",
        "-k",
        "--out_channels",
        "--k",
        default=32,
        type=int,
        required=False,
        help="Number of Output Channels (Default=32)",
    )
    parser.add_argument(
        "-fil_d",
        "-@",
        "--fil_d",
        "--@",
        default=3,
        type=int,
        required=False,
        help="Filter Depth (Default=3)",
    )
    parser.add_argument(
        "-fil_h",
        "-y",
        "--fil_h",
        "--y",
        default=3,
        type=int,
        required=False,
        help="Filter Height (Default=3)",
    )
    parser.add_argument(
        "-fil_w",
        "-x",
        "--fil_w",
        "--x",
        default=3,
        type=int,
        required=False,
        help="Filter Width (Default=3)",
    )
    parser.add_argument(
        "-conv_stride_d",
        "-#",
        "--conv_stride_d",
        "--#",
        default=1,
        type=int,
        required=False,
        help="Convolution Stride for Depth (Default=1)",
    )
    parser.add_argument(
        "-conv_stride_h",
        "-u",
        "--conv_stride_h",
        "--u",
        default=1,
        type=int,
        required=False,
        help="Convolution Stride for Height (Default=1)",
    )
    parser.add_argument(
        "-conv_stride_w",
        "-v",
        "--conv_stride_w",
        "--v",
        default=1,
        type=int,
        required=False,
        help="Convolution Stride for Width (Default=1)",
    )
    parser.add_argument(
        "-pad_d",
        "-$",
        "--pad_d",
        "--$",
        default=1,
        type=int,
        required=False,
        help="Zero Padding for Depth (Default=0)",
    )
    parser.add_argument(
        "-pad_h",
        "-p",
        "--pad_h",
        "--p",
        default=1,
        type=int,
        required=False,
        help="Zero Padding for Height (Default=0)",
    )
    parser.add_argument(
        "-pad_w",
        "-q",
        "--pad_w",
        "--q",
        default=1,
        type=int,
        required=False,
        help="Zero Padding for Width (Default=0)",
    )
    parser.add_argument(
        "-verify",
        "-V",
        "--verify",
        "--V",
        default=1,
        type=int,
        required=False,
        help="Verify Each Layer (Default=1)",
    )
    parser.add_argument(
        "-time",
        "-t",
        "--time",
        "--t",
        default=0,
        type=int,
        required=False,
        help="Time Each Layer (Default=0)",
    )
    parser.add_argument(
        "-dilation_d",
        "-^",
        "--dilation_d",
        "--^",
        default=1,
        type=int,
        required=False,
        help="Dilation of Filter Depth (Default=1)",
    )
    parser.add_argument(
        "-dilation_h",
        "-l",
        "--dilation_h",
        "--l",
        default=1,
        type=int,
        required=False,
        help="Dilation of Filter Height (Default=1)",
    )
    parser.add_argument(
        "-dilation_w",
        "-j",
        "--dilation_w",
        "--j",
        default=1,
        type=int,
        required=False,
        help="Dilation of Filter Width (Default=1)",
    )
    parser.add_argument(
        "-group_count",
        "-g",
        "--group_count",
        "--g",
        type=int,
        default=1,
        required=False,
        help="Number of Groups (Default=1)",
    )
    parser.add_argument(
        "-instance",
        "--instance",
        type=int,
        default=-1,
        required=False,
        help="Instance index (Default=-1)",
    )
    parser.add_argument(
        "-list-instances",
        "--list-instances",
        action="store_true",
        default=False,
        required=False,
        help="List valid instances without running",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()

    preliminary_args, _ = parser.parse_known_args()

    if preliminary_args.input_file is not None:
        # Batch mode
        if preliminary_args.output_file is None:
            print("Error: --output-file is required when using --input-file")
            sys.exit(1)

        print(f"Batch mode: Reading commands from '{preliminary_args.input_file}'")
        profiler_path = preliminary_args.profiler_path
        if profiler_path:
            print(f"Using profiler: '{profiler_path}'")
        if not preliminary_args.verbose:
            print("Output mode: best configuration only (use --verbose for full output)")
        if preliminary_args.gpu_verify:
            print("GPU verification enabled (verify=2)")
        process_batch_file(
            preliminary_args.input_file,
            preliminary_args.output_file,
            parser,
            profiler_path,
            preliminary_args.verbose,
            preliminary_args.continue_from,
            preliminary_args.gpu_verify,
        )
    else:
        # Single command mode
        args, unknown = parser.parse_known_args()
        profiler_path = args.profiler_path
        if profiler_path:
            print(f"Using profiler: '{profiler_path}'")
        init_const_args(args, profiler_path)
        process_miopen_driver_name(args, unknown)
        if args.gpu_verify:
            args.verify = 2
        print("Ignored args:")
        print(unknown)
        run_ck_profiler(args)
