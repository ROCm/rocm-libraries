# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
        "--type_acc", type=str, default="f32", help="Size of each element in partial tile in bits"
    )
    parser.add_argument(
        "--type_d", type=str, default="f16", help="Size of each element in the output in bits"
    )
    parser.add_argument("--type_compute", type=str, default=None, help="Instruction input type")
    parser.add_argument("--workspace_size", type=int, default=0, help="Amount of workspace available in bytes")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--print", action="store_true", help="Print hardware info")
    parser.add_argument(
        "--wgm", type=int, default=6, help="Wave granularity multiplier"
    )
    # New macro-tile size arguments:
    parser.add_argument("--mt_m", type=int, default=32, help="Macro-tile dimension M")
    parser.add_argument("--mt_n", type=int, default=32, help="Macro-tile dimension N")
    parser.add_argument("--mt_k", type=int, default=256, help="Macro-tile dimension K")
    parser.add_argument("--mi_m", type=int, default=16, help="Machine Instruction dimension M")
    parser.add_argument("--mi_n", type=int, default=16, help="Machine Instruction dimension N")
    parser.add_argument("--mi_k", type=int, default=16, help="Machine Instruction dimension K")
    parser.add_argument("--occupancy", type=int, default=1, help="Occupancy of kernel")

    parser.add_argument("--dynamic_grid_version", type=int, default=5, help="Version of Dynamic Grid Selection to use")

    args = parser.parse_args()

    if args.type_compute is None:
        if origami.datatype_to_bits(origami.string_to_datatype(args.type_a)) > origami.datatype_to_bits(origami.string_to_datatype(args.type_b)):
            args.type_compute = args.type_a
        else:
            args.type_compute = args.type_b

    return args


def main():
    args = parseArguments()

    hardware = origami.get_hardware_for_device(args.device)

    if args.print:
        hardware.print()

    reduction = origami.select_reduction(
        args.m,
        args.n,
        args.k,
        args.batch,
        args.mt_m,
        args.mt_n,
        args.mt_k,
        hardware,
        args.dynamic_grid_version
    )

    winner_grid = origami.select_grid(
        args.m,
        args.n,
        args.k,
        args.batch,
        args.trans_a,
        args.trans_b,
        origami.datatype_to_bits(origami.string_to_datatype(args.type_a)),
        origami.datatype_to_bits(origami.string_to_datatype(args.type_b)),
        origami.datatype_to_bits(origami.string_to_datatype(args.type_d)),
        origami.string_to_datatype(args.type_compute),
        args.workspace_size,
        args.mt_m,
        args.mt_n,
        args.mt_k,
        args.mi_m,  # MI_M
        args.mi_n,  # MI_N
        args.mi_k,  # MI_K
        args.wgm,
        origami.datatype_to_bits(origami.string_to_datatype(args.type_acc)) // 8,
        args.occupancy,
        hardware,
        args.dynamic_grid_version,
        reduction
    )

    print(f"Best reduction algo : {reduction}")
    print(f"Best grid : {winner_grid}")

    return 0


if __name__ == "__main__":
    exit(main())
