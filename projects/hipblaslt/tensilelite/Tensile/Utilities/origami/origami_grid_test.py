#!/usr/bin/env python3

import argparse
import origami
import math


def parseArguments():
    parser = argparse.ArgumentParser(description="Test Origami.")
    parser.add_argument("-m", type=int, default=8192, help="Problem M dimension")
    parser.add_argument("-n", type=int, default=8192, help="Problem N dimension")
    parser.add_argument("-k", type=int, default=8192, help="Problem K dimension")
    parser.add_argument(
        "--transA", type=bool, default=True, help="Whether to transpose A"
    )
    parser.add_argument(
        "--transB", type=bool, default=False, help="Whether to transpose B"
    )
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--element_size", type=int, default=2, help="Size of each element in bytes"
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

    return parser.parse_args()


def main():
    args = parseArguments()

    hardware = origami.getHardwareForDevice(args.device)

    if args.print:
        hardware.print()

    winner_grid = origami.select_best_grid_size(
        args.m,
        args.n,
        args.k,
        args.batch,
        args.transA,
        args.transB,
        hardware,
        args.mt_m,
        args.mt_n,
        args.mt_k,
        16,  # MI_M
        16,  # MI_N
        16,  # MI_K
        args.element_size * 8,
        args.element_size * 8,
        args.element_size * 8,
        0,  # mx_block_size
        0.8,  # H_L2,
        False,  # debug
        6,  # WGM
        8,  # biggest_allowable_split
    )

    print(f"Best grid : {winner_grid}")

    return 0


if __name__ == "__main__":
    exit(main())
