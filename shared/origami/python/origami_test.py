#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import origami
import csv
import os

# Usage:
# individual size: $ python3 origami_test.py -m 2048 -n 2048 -k 2048 --trans_a T --trans_b N --debug --print
# read sizes from a file: $ python3 origami_test.py -m 1 -n 1 -k 1 -b 1 --trans_a T --trans_b N --type_a bf16 --type_b bf16 --type_d  bf16 --sizes True --path ./sizes.csv --arch "gfx950" --debug --print

# legend for datatypes (from hardware.hpp):
#   Float/SGEMM == "f32")
#   ComplexFloat/CGEMM== "c32")
#   ComplexDouble/ZGEMM == "c64"
#   Double/DGEMM == "f64"
#   Half/HHS/HSS== "f16"
#   Int32 == "i32" --> Is it i8II?
#   BFloat16/BBS/BSS == "bf16"
#   Int8/I8I8I == "i8" --> Is it i8II?
#   XFloat32/TF32 == "xf32"
#   Float8/ all F8 GEMMs == "f8"
#   BFloat8 == "bf8"
#   Float6 == "f6"
#   BFloat6 == "bf6"
#   Float4 == "f4"

MatInst = {'gfx950': {}, 'gfx942': {}}
MatInst['gfx950']["f32"] = [
    (16,16,4,1),
    (32,32,2,1)
    ]
# MatInst['gfx950']["c32"] = []
# MatInst['gfx950']["c64"] = []
MatInst['gfx950']["f64"] = [
    (16,16,4,1)
    ]
MatInst['gfx950']["f16"]  = [
    # (4,4,4,16), #gfx942
    #[16,16,4,4] # never use 16x16x4x4
    #[16,16,16,1] #gfx942
    #[32,32,4,2] # never use 32x32x4x2
    #[32,32,8,1] #gfx942
    (16,16,32,1), #gfx950
    (32,32,16,1) #gfx950
    ]
# MatInst['gfx950']["i32"] =[]
MatInst['gfx950']["bf16"]  = MatInst['gfx950']["f16"]
# MatInst['gfx950']["i8"] =[
#   (32,32,16,1),
#   (16,16,32,1),
#   (4,4,4,16)
#   ]
MatInst['gfx950']["xf32"] = [
    # (4,4,4,16), #gfx942
    #[16,16,4,4] # never use 16x16x4x4
    #[16,16,16,1] #gfx942
    #[32,32,4,2] # never use 32x32x4x2
    #[32,32,8,1] #gfx942
    (16,16,32,1), #gfx950
    (32,32,16,1) #gfx950
    ]
MatInst['gfx950']["f8"] = [
    (4,4,4,16),    #gfx950, gfx942
    (16,16,128,1), #gfx950
    (32,32,64,1)   #gfx950
    ]
MatInst['gfx950']["bf8"] = MatInst['gfx950']["f8"]

def parseArguments():
    parser = argparse.ArgumentParser(description="""Get hypothetical Origami MTxDU selection for a size""")
    parser.add_argument("-m", type=int, default=8192)
    parser.add_argument("-n", type=int, default=8192)
    parser.add_argument("-b", type=int, default=1)
    parser.add_argument("-k", type=int, default=8192)
    parser.add_argument("--trans_a", type=bool, default=True)
    parser.add_argument("--trans_b", type=bool, default=False)
    parser.add_argument("--device", type=int, default=0) # to get hardware specs
    parser.add_argument("--type_a", type=str, default="f16")
    parser.add_argument("--type_b", type=str, default="f16")
    parser.add_argument("--type_d", type=str, default="f16")
    parser.add_argument("--scale_block_size", type=int, default=0)
    parser.add_argument("--wgm", type=int, default=6)
    parser.add_argument("--sizes", type=bool, default=False) # to load the sizes from a csv file. -m/-n/-b/-k will be ignored if True.
    parser.add_argument("--path", type=str, default="./sizes.csv") # path to the csv file. Fails if sizes is True, and path or file does not exist.
    parser.add_argument("--arch", type=str, default="gfx950")
    parser.add_argument("--print", action="store_true")

    return parser.parse_args()

def createTileList(arch, gemmType):

    LIST_OF_WAVEs_TO_INCLUDE = [[4, 1], [2, 2], [1, 4], [1, 2], [2, 1], [1, 1]]
    MIN_MT0 = MIN_MT1 = 16
    MAX_MT0 = MAX_MT1 = 512

    # generate all MTs for each datatype:
    bm_max = 0
    tile_list = set()
    for MI in MatInst[arch][gemmType]:
        for bm in range(bm_max + 1):
            MIBlockM = 2 ** bm

            for wave in LIST_OF_WAVEs_TO_INCLUDE:
                waveTileM = 0
                waveTileN = 0

                while True:
                    waveTileM+=1
                    waveTileN=0
                    MatrixInstM = MI[0] * MIBlockM
                    MT0 = MatrixInstM * waveTileM * wave[0]
                    if MT0 < MIN_MT0:
                        continue
                    if MT0 > MAX_MT0:
                        break

                    while True:
                        waveTileN+=1
                        MatrixInstN = MI[1] / MIBlockM * MI[3]
                        MT1 = int(MatrixInstN * waveTileN * wave[1])

                        if MT1 < MIN_MT1:
                            continue
                        if MT1 > MAX_MT1:
                            break

                        # LDS size check for LSU
                        LSU = max(1, 4//wave[0]//wave[1])
                        if LSU > 1 and MT0*MT1*4*LSU > 256*256:
                            continue

                        if MT0*MT1 > 256*256:
                            continue
                        for DU in [16, 32, 64, 128, 256, 512, 1024]:
                            tile_list.add((MT0, MT1, DU, MI[0], MI[1], MI[2], 1, 6, 0, 0))

    return [tile for tile in tile_list]

def main():
    args = parseArguments()

    hardware = origami.get_hardware_for_device(args.device)

    tile_list = createTileList(args.arch, args.type_a)

    print(" Number of unique MTxDU: ", len(tile_list))

    if args.sizes: # sizes from a file
      try:
        with open(args.path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            print(f"M,N,Batch,K,MT0,MT1,DU,MI0,MI1,MI2,MI3,latency")
            for row in csv_reader:
                M = int(row[0])
                N = int(row[1])
                B = int(row[2])
                K = int(row[3])
                ret = origami.select_best_macro_tile_size(
                    M,
                    N,
                    K,
                    B,
                    args.trans_a,
                    args.trans_b,
                    hardware,
                    tile_list,
                    origami.datatype_to_bits(origami.string_to_datatype(args.type_a)),
                    origami.datatype_to_bits(origami.string_to_datatype(args.type_b)),
                    origami.datatype_to_bits(origami.string_to_datatype(args.type_d)),
                    origami.string_to_datatype(args.type_a),
                    args.scale_block_size,
                    0.8,
                    args.print,
                    args.wgm,
                )
                #MxNxBxK, MT0xMT1xDU, MI0xMI1xMI2xMI3, latency/cycles
                print(f"{M},{N},{B},{K},{ret[0][1]},{ret[0][2]},{ret[0][3]},{ret[0][4]},{ret[0][5]},{ret[0][6]},{ret[0][7]},{ret[0][0]:0.3f}")
      except FileNotFoundError:
         raise FileNotFoundError(f"Error: The size file: '{args.path}' does not exist.")
    else: # one size from the command line.
        ret = origami.select_best_macro_tile_size(
            args.m,
            args.n,
            args.k,
            args.b,
            args.trans_a,
            args.trans_b,
            hardware,
            tile_list,
            origami.datatype_to_bits(origami.string_to_datatype(args.type_a)),
            origami.datatype_to_bits(origami.string_to_datatype(args.type_b)),
            origami.datatype_to_bits(origami.string_to_datatype(args.type_d)),
            origami.string_to_datatype(args.type_a),
            args.scale_block_size,
            0.8,
            args.print,
            args.wgm,
        )
        print(f"The best MTxDU for [{args.m}, {args.n}, {args.b}, {args.k}] is: {ret[0]}") # Match this with the top condition
        # add an option to list top 10~15
        print(" full list of MTs: \n", ret)

    if args.print:
        hardware.print()
        hardware.print_debug_info()
        with open("MTxDU.log",'w') as file:
            for tile in tile_list:
                file.write(f'{tile}\n')

    return 0

if __name__ == "__main__":
    exit(main())
