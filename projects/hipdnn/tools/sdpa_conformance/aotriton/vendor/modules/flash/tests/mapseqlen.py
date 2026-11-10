#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

REGULAR_SEQLEN = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
PRIME_SEQLEN_Q = [11, 17, 37,  67, 157, 257,  523, 1033, 2063, 4919]
PRIME_SEQLEN_K = [13, 31, 41,  71, 223, 337,  571, 1063, 2081, 5237]

POT_HEADDIMS = [16, 32, 64, 128, 256, 512]
NPOT_HEADDIMS = [48, 80, 96, 160, 192, 224]
REG_HEADDIMS = sorted(POT_HEADDIMS + NPOT_HEADDIMS)
M8_HEADDIMS = [8, 24, 40, 56, 72, 88, 96, 120, 152, 184, 216, 248, 408]

def reg_map_hdim(hdim):
    for target in REG_HEADDIMS:
        if target >= hdim:
            return target

REG_TO_M8 = { reg_map_hdim(m8) : m8 for m8 in M8_HEADDIMS }

def hrr_map_hdim(hdim):
    reg_hdim = reg_map_hdim(hdim)
    return REG_TO_M8[reg_hdim]

reg_map_sq = lambda x : 2 ** (x-1).bit_length()
reg_map_sk = lambda x : 2 ** (x-1).bit_length()
irr_map_sq = lambda x : PRIME_SEQLEN_Q[reg_map_sq(x).bit_length() - 5]
irr_map_sk = lambda x : PRIME_SEQLEN_K[reg_map_sk(x).bit_length() - 5]

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('pass_files', type=Path, help='Output .cfg file from pytest2entry.py', nargs='+')
    args = p.parse_args()
    return args

def process(fn, regfn, hrrfn, irrfn):
    with open(fn) as f, open(regfn, 'w') as reg, open(hrrfn, 'w') as hrr, open(irrfn, 'w') as irr:
        for line in f:
            j = json.loads(line)
            seqlen_q = j['seqlen_q']
            seqlen_k = j['seqlen_k']
            orig_hdim = j['d_head']
            j['seqlen_q'] = reg_map_sq(seqlen_q)
            j['seqlen_k'] = reg_map_sk(seqlen_k)

            d_head = reg_map_hdim(j['d_head'])
            j['d_head'] = d_head
            print(json.dumps(j), file=reg)

            j['d_head'] = hrr_map_hdim(d_head)
            print(json.dumps(j), file=hrr)

            j['seqlen_q'] = irr_map_sq(seqlen_q)
            j['seqlen_k'] = irr_map_sk(seqlen_k)
            print(json.dumps(j), file=irr)

def main():
    args = parse()
    for fn in args.pass_files:
        reg = fn.with_stem(fn.stem+'.reg')
        hrr = fn.with_stem(fn.stem+'.hrr')
        irr = fn.with_stem(fn.stem+'.irr')
        process(fn, reg, hrr, irr)

if __name__ == '__main__':
    main()
