#!/usr/bin/env python3

################################################################################
#
# MIT License
#
# Copyright 2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

import argparse
import origami
import math


def parseArguments():
    parser = argparse.ArgumentParser(description="Test StreamK Grid Selection.")
    parser.add_argument("-m", type=int, default=8192, help="Problem M dimension")
    parser.add_argument("-n", type=int, default=8192, help="Problem N dimension")
    parser.add_argument("-k", type=int, default=8192, help="Problem K dimension")
    parser.add_argument(
        "--trans_a", type=bool, default=True, help="Whether to transpose A"
    )
    parser.add_argument(
        "--trans_b", type=bool, default=False, help="Whether to transpose B"
    )
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--type_a", type=str, default="f16", help="Size of each element in A in bits"
    )
    parser.add_argument(
        "--type_b", type=str, default="f16", help="Size of each element in B in bits"
    )
    parser.add_argument(
        "--type_acc",
        type=str,
        default="f32",
        help="Size of each element in partial tile in bits",
    )
    parser.add_argument(
        "--type_d",
        type=str,
        default="f16",
        help="Size of each element in the output in bits",
    )
    parser.add_argument(
        "--type_compute", type=str, default=None, help="Instruction input type"
    )
    parser.add_argument(
        "--workspace_size",
        type=int,
        default=0,
        help="Amount of workspace available in bytes",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--print", action="store_true", help="Print hardware info")
    parser.add_argument(
        "--wgm", type=int, default=6, help="Wave granularity multiplier"
    )
    # New macro-tile size arguments:
    parser.add_argument("--mt_m", type=int, default=32, help="Macro-tile dimension M")
    parser.add_argument("--mt_n", type=int, default=32, help="Macro-tile dimension N")
    parser.add_argument("--mt_k", type=int, default=256, help="Macro-tile dimension K")
    parser.add_argument(
        "--mi_m", type=int, default=16, help="Machine Instruction dimension M"
    )
    parser.add_argument(
        "--mi_n", type=int, default=16, help="Machine Instruction dimension N"
    )
    parser.add_argument(
        "--mi_k", type=int, default=16, help="Machine Instruction dimension K"
    )
    parser.add_argument("--occupancy", type=int, default=1, help="Occupancy of kernel")

    parser.add_argument(
        "--dynamic_grid_version",
        type=int,
        default=5,
        help="Version of Dynamic Grid Selection to use",
    )

    args = parser.parse_args()

    if args.type_compute is None:
        if origami.datatype_to_bits(
            origami.string_to_datatype(args.type_a)
        ) > origami.datatype_to_bits(origami.string_to_datatype(args.type_b)):
            args.type_compute = args.type_a
        else:
            args.type_compute = args.type_b

    return args


def main():
    args = parseArguments()

    hardware = origami.get_hardware_for_device(args.device)

    if args.print:
        hardware.print()

    # Create problem description
    problem = origami.problem_t()
    problem.size = origami.dim3_t(args.m, args.n, args.k)
    problem.batch = args.batch
    problem.a_transpose = (
        origami.transpose_t.T if args.trans_a else origami.transpose_t.N
    )
    problem.b_transpose = (
        origami.transpose_t.T if args.trans_b else origami.transpose_t.N
    )
    problem.a_dtype = origami.string_to_datatype(args.type_a)
    problem.b_dtype = origami.string_to_datatype(args.type_b)
    problem.d_dtype = origami.string_to_datatype(args.type_d)
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = origami.string_to_datatype(args.type_compute)
    problem.a_mx_block_size = 0
    problem.b_mx_block_size = 0

    # Create config
    config = origami.config_t()
    config.mt = origami.dim3_t(args.mt_m, args.mt_n, args.mt_k)
    config.mi = origami.dim3_t(args.mi_m, args.mi_n, args.mi_k)
    config.occupancy = args.occupancy
    config.workgroup_mapping = args.wgm

    # Select reduction strategy
    grid_algorithm = origami.grid_selection_t.analytical  # default to analytical
    if args.dynamic_grid_version == 0:
        grid_algorithm = origami.grid_selection_t.number_of_cus
    elif args.dynamic_grid_version == 1:
        grid_algorithm = origami.grid_selection_t.min_resources
    elif args.dynamic_grid_version == 2:
        grid_algorithm = origami.grid_selection_t.energy_aware
    elif args.dynamic_grid_version == 3:
        grid_algorithm = origami.grid_selection_t.reduction_cost_aware
    elif args.dynamic_grid_version == 4:
        grid_algorithm = origami.grid_selection_t.data_parallel
    elif args.dynamic_grid_version == 5:
        grid_algorithm = origami.grid_selection_t.analytical
    elif args.dynamic_grid_version == 6:
        grid_algorithm = origami.grid_selection_t.k_split_aware

    reduction = origami.select_reduction(problem, hardware, config, grid_algorithm)

    winner_grid = origami.select_grid_size(
        problem, hardware, config, grid_algorithm, hardware.N_CU
    )

    print(f"Best reduction algo : {reduction}")
    print(f"Best grid : {winner_grid}")

    return 0


if __name__ == "__main__":
    exit(main())
