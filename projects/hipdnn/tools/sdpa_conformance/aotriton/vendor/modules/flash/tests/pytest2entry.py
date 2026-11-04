#!/usr/bin/env python3

import argparse
import sys
import re
import json
from abc import abstractmethod
import subprocess

SEP = re.compile(r'[ :\[\]]')
# Example: ['FAILED', 'test/test_backward.py', '', 'test_irregulars', 'Split-BiasOn-True-l1-dtype1-0.0-CausalOff-41-257-hdim32-5-3', '']
INDICES = (1, 3, 4)
NATIVE_ARCH = subprocess.check_output(['rocm_agent_enumerator'], shell=True).decode('utf8', errors='ignore').strip().split()[0]

DTYPE = {
    'dtype0' : 'float16',
    'dtype1' : 'bfloat16',
    'dtype2' : 'float32',
}

def validate_cfg(out_str):
    if not out_str.endswith('.cfg'):
        raise argparse.ArgumentTypeError(f"Invalid --out file name: '{out_str}'. Must ends with .cfg")
    return out_str

class Translator(object):
    SEP = re.compile(r'-')
    INDICES = None

    def __call__(self, utparam):
        parts = re.split(self.SEP, utparam)
        if self.INDICES is not None:
            parts = [ parts[i] for i in self.INDICES ]
        return self.parts2cfg(parts)

    @abstractmethod
    def parts2cfg(self, parts):
        pass

class Irregulars(Translator):
    def parts2cfg(self, parts):
        _bwd, bias, storage, sm_scale_str, dtype, dropout_p, causal, seqlen_k, seqlen_q, hdim, nheads, batch = parts
        return {
            "causal_type" : 0 if causal == 'CausalOff' else 1,
            "d_head" : int(hdim.removeprefix('hdim')),
            "dropout_p" : float(dropout_p),
            "dtype" : DTYPE[dtype],
            "bias_type" : 1 if bias == 'BiasOn' else 0,
            "seqlen_q" : int(seqlen_q),
            "seqlen_k" : int(seqlen_k),
            "nheads" : int(nheads),     # unread by tune_flash.py, but no harm as well
            "batch" : int(batch),       # unread by tune_flash.py, but no harm as well
            "sm_scale" : sm_scale_str.lower(),
            "storage_flip": storage != 'False',
        }

class Regulars(Translator):
    def parts2cfg(self, parts):
        bias = 'BiasOff'
        _bwd, storage, sm_scale_str, dtype, dropout_p, causal, seqlen_k, seqlen_q, hdim, nheads, batch = parts
        return {
            "causal_type" : 0 if causal == 'CausalOff' else 1,
            "d_head" : int(hdim.removeprefix('hdim')),
            "dropout_p" : float(dropout_p),
            "dtype" : DTYPE[dtype],
            "bias_type" : 1 if bias == 'BiasOn' else 0,
            "seqlen_q" : int(seqlen_q),
            "seqlen_k" : int(seqlen_k),
            "nheads" : int(nheads),     # unread by tune_flash.py, but no harm as well
            "batch" : int(batch),       # unread by tune_flash.py, but no harm as well
            "sm_scale" : sm_scale_str.lower(),
            "storage_flip": storage != 'False',
        }

class RegularBias(Translator):
    def parts2cfg(self, parts):
        bias = 'BiasOn'
        causal = 'CausalOff'
        _bwd, storage, sm_scale_str, dtype, dropout_p, seqlen_k, seqlen_q, hdim, nheads, batch = parts
        return {
            "causal_type" : 0 if causal == 'CausalOff' else 1,
            "d_head" : int(hdim.removeprefix('hdim')),
            "dropout_p" : float(dropout_p),
            "dtype" : DTYPE[dtype],
            "bias_type" : 1 if bias == 'BiasOn' else 0,
            "seqlen_q" : int(seqlen_q),
            "seqlen_k" : int(seqlen_k),
            "nheads" : int(nheads),     # unread by tune_flash.py, but no harm as well
            "batch" : int(batch),       # unread by tune_flash.py, but no harm as well
            "sm_scale" : sm_scale_str.lower(),
            "storage_flip": storage != 'False',
        }

class Gqa(Translator):
    def parts2cfg(self, parts):
        bias = 'BiasOff'
        _bwd, storage, sm_scale_str, dtype, dropout_p, causal, seqlen_k, seqlen_q, hdim, _, batch = parts
        return {
            "causal_type" : 0 if causal == 'CausalOff' else 1,
            "d_head" : int(hdim.removeprefix('hdim')),
            "dropout_p" : float(dropout_p),
            "dtype" : DTYPE[dtype],
            "bias_type" : 1 if bias == 'BiasOn' else 0,
            "seqlen_q" : int(seqlen_q),
            "seqlen_k" : int(seqlen_k),
            "nheads" : (16, 8),
            "batch" : int(batch),       # unread by tune_flash.py, but no harm as well
            "sm_scale" : sm_scale_str.lower(),
            "storage_flip": storage != 'False',
        }

UT2TR = {
    'test_irregulars' : Irregulars(),
    'test_regular_bwd' : Regulars(),
    'test_op_bwd_with_matrix_bias' : RegularBias(),
    'test_gqa' : Gqa(),
}

EPILOG = r'''Script to parse pytest output and generate .cfg file for tune_flash.py.
Example Usage:
    COLUMNS=400 BWD_IMPL=triton_split FOR_RELEASE=2 PYTHONPATH=build-TEST/install_dir/lib/ pytest -rfEsx test/test_backward.py -v 1>ut_X.out 2>ut_X.err
    python test/pytest2entry.py ut_X.out --out pass_X.cfg
    vim pass_X.cfg  # run :sort u
    PYTHONPATH=build-TUNE/install_dir/lib/ python test/tune_flash.py --json_file ~/pass_X.json --entry_from_json pass_X.cfg --use_multigpu -1

Note:
    * COLUMNS=400, -rfE and -v are ALL CRUCIAL to make pytest2entry function correctly.
    * Using -rfEsx instead of -rfE allows users have more information about
      skips/xfails controlled by env vars like SMALL_VRAM/PROBE_UNSUPPORTED.
    * Use -n ${NGPUS} to run UT in parallel. Add --max-worker-restart to
      increase the restarting time (default is NGPU*4 which is pretty limited for full coverage)
'''

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG)
    p.add_argument('pytest_outs', help='PLAIN Standard output of pytest. Do NOT use --color=yes', nargs='+')
    p.add_argument('--out', help='Output .cfg file for tune_flash.py --entry_from_json', type=validate_cfg, default=None)
    p.add_argument('--arch', help='arch field in .cfg. Guess from current GPU by default since this tool should be run on the testing machine.', default=NATIVE_ARCH)
    args = p.parse_args()
    # print(args)
    return args

def parse_pytest_out(args, f, out):
    for line in f:
        if not line.startswith('FAILED'):
            continue
        if 'OutOfMemoryError' in line:
            continue
        parts = re.split(SEP, line)
        utfn, utname, utparam = [ parts[i] for i in INDICES ]
        tr2cfg = {"arch": args.arch}
        if utname not in UT2TR:
            print(f'No handler. Skip {utname}')
            continue
        tr2cfg.update(UT2TR[utname](utparam))
        print(json.dumps(tr2cfg), file=out)

def main():
    args = parse()
    def loop(out):
        for fn in args.pytest_outs:
            with open(fn) as f:
                parse_pytest_out(args, f, out)
    if args.out is None:
        loop(sys.stdout)
    else:
        with open(args.out, 'w') as out:
            loop(out)

if __name__ == '__main__':
    main()
