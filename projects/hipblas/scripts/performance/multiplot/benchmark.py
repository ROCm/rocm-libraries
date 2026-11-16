#!/usr/bin/env python3

import argparse
import re
import subprocess
import os
from pathlib import Path
import yaml
import sys

script_dir = os.path.dirname( __file__ )
getspec_module_dir = os.path.join(script_dir, '..', 'blas')
sys.path.append(getspec_module_dir)
import getspecs

def L1_theo_max(GBps):
    L1_theo_max_dict = {}

    L1_theo_max_dict['saxpy'] = GBps * ( 1 /  6 )
    L1_theo_max_dict['daxpy'] = GBps * ( 1 / 12 )
    L1_theo_max_dict['caxpy'] = GBps * ( 1 /  3 )
    L1_theo_max_dict['zaxpy'] = GBps * ( 1 /  6 )

    L1_theo_max_dict['sdot'] = GBps * ( 1 / 4 )
    L1_theo_max_dict['ddot'] = GBps * ( 1 / 8 )
    L1_theo_max_dict['cdot'] = GBps * ( 1 / 2 )
    L1_theo_max_dict['zdot'] = GBps * ( 1 / 4 )

    L1_theo_max_dict['sscal'] = GBps * ( 1 /  8 )
    L1_theo_max_dict['dscal'] = GBps * ( 1 / 16 )
    L1_theo_max_dict['cscal'] = GBps * ( 3 /  8 )
    L1_theo_max_dict['zscal'] = GBps * ( 3 / 16 )

    L1_theo_max_dict['snrm2'] = GBps * ( 1 / 2 )
    L1_theo_max_dict['dnrm2'] = GBps * ( 1 / 4 )
    L1_theo_max_dict['cnrm2'] = GBps * ( 1     )
    L1_theo_max_dict['znrm2'] = GBps * ( 1 / 2 )

    L1_theo_max_dict['siamax'] = GBps * ( 1 /  4 )
    L1_theo_max_dict['diamax'] = GBps * ( 1 /  8 )
    L1_theo_max_dict['ciamax'] = GBps * ( 1 /  8 )
    L1_theo_max_dict['ziamax'] = GBps * ( 1 / 16 )

    return L1_theo_max_dict

def L2_theo_max(GBps):
    L2_theo_max_dict = {}

    # matrix vector multiplication
    L2_theo_max_dict['sgemv'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['dgemv'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['cgemv'] = GBps * ( 1     )
    L2_theo_max_dict['zgemv'] = GBps * ( 1 / 2 )

#   L2_theo_max_dict['sgbmv'] = GBps * ( 1 / 2 )
#   L2_theo_max_dict['dgbmv'] = GBps * ( 1 / 4 )
#   L2_theo_max_dict['cgbmv'] = GBps * ( 1     )
#   L2_theo_max_dict['zgbmv'] = GBps * ( 1 / 2 )

    L2_theo_max_dict['strmv'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['dtrmv'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['ctrmv'] = GBps * ( 1     )
    L2_theo_max_dict['ztrmv'] = GBps * ( 1 / 2 )

#   L2_theo_max_dict['stbmv'] = GBps * ( 1 / 2 )
#   L2_theo_max_dict['dtbmv'] = GBps * ( 1 / 4 )
#   L2_theo_max_dict['ctbmv'] = GBps * ( 1     )
#   L2_theo_max_dict['ztbmv'] = GBps * ( 1 / 2 )

    L2_theo_max_dict['stpmv'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['dtpmv'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['ctpmv'] = GBps * ( 1     )
    L2_theo_max_dict['ztpmv'] = GBps * ( 1 / 2 )

    # Symmetric matrix vector multiplication
    L2_theo_max_dict['ssymv'] = GBps * ( 1     )
    L2_theo_max_dict['dsymv'] = GBps * ( 1 / 2 )

#   L2_theo_max_dict['ssbmv'] = GBps * ( 1     )
#   L2_theo_max_dict['dsbmv'] = GBps * ( 1 / 2 )

    L2_theo_max_dict['sspmv'] = GBps * ( 1     )
    L2_theo_max_dict['dspmv'] = GBps * ( 1 / 2 )

    # Hermitian matrix vector multiplication
    L2_theo_max_dict['chemv'] = GBps * ( 2     )
    L2_theo_max_dict['zhemv'] = GBps * ( 1     )

    L2_theo_max_dict['chpmv'] = GBps * ( 2     )
    L2_theo_max_dict['zhpmv'] = GBps * ( 1     )

#   L2_theo_max_dict['chbmv'] = GBps * ( 2     )
#   L2_theo_max_dict['zhbmv'] = GBps * ( 1     )

    # rank 1 updates
    L2_theo_max_dict['sger'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['dger'] = GBps * ( 1 / 8 )

    L2_theo_max_dict['ssyr'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['dsyr'] = GBps * ( 1 / 8 )

    L2_theo_max_dict['sspr'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['dspr'] = GBps * ( 1 / 8 )

    L2_theo_max_dict['ssyr2'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['dsyr2'] = GBps * ( 1 / 4 )

    L2_theo_max_dict['sspr2'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['dspr2'] = GBps * ( 1 / 4 )

    # Hermitian rank 1 updates
    L2_theo_max_dict['cher'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['zher'] = GBps * ( 1 / 4 )

    L2_theo_max_dict['chpr'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['zhpr'] = GBps * ( 1 / 4 )

    L2_theo_max_dict['cher2'] = GBps * ( 1     )
    L2_theo_max_dict['zher2'] = GBps * ( 1 / 2 )

    L2_theo_max_dict['chpr2'] = GBps * ( 1     )
    L2_theo_max_dict['zhpr2'] = GBps * ( 1 / 2 )

    # triangle solves
    L2_theo_max_dict['strsv'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['dtrsv'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['ctrsv'] = GBps * ( 1     )
    L2_theo_max_dict['ztrsv'] = GBps * ( 1 / 2 )

#   L2_theo_max_dict['stbsv'] = GBps * ( 1 / 2 )
#   L2_theo_max_dict['dtbsv'] = GBps * ( 1 / 4 )
#   L2_theo_max_dict['ctbsv'] = GBps * ( 1     )
#   L2_theo_max_dict['ztbsv'] = GBps * ( 1 / 2 )

    L2_theo_max_dict['stpsv'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['dtpsv'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['ctpsv'] = GBps * ( 1     )
    L2_theo_max_dict['ztpsv'] = GBps * ( 1 / 2 )

    return L2_theo_max_dict

GBPS_BY_ARCH = {
    'gfx906': 1000,
    'gfx908': 1100,
    'gfx90a': 1600,
    'gfx942': 5300,
    'gfx950': 8000,
}

def detect_cuda():
    # Prefer ROCm when a real gfx agent is present, otherwise fall back to cuda.
    if getspecs.getgfx(0, False) != 'N/A':
        return False
    success, cout = getspecs._subprocess_helper(["nvidia-smi", "-L"])
    return success and bool(cout.strip())

def write_machine_spec_yaml(machine_spec_filename, device_number=0, gbps=None):
    machine_spec_dict = {}
    cuda = detect_cuda()
    machine_spec_dict['arch'] = getspecs.getgfx(device_number, cuda)
    if machine_spec_dict['arch'] == 'N/A':
        print("Unable to detect GPU architecture for device {}".format(device_number))
        print("quitting ", sys.argv[0])
        quit()

    arch = machine_spec_dict['arch']
    GBps = gbps if gbps is not None else GBPS_BY_ARCH.get(arch)
    if GBps is not None:
        machine_spec_dict['GBps'] = GBps
        machine_spec_dict['theo_max'] = True
        machine_spec_dict.update(L1_theo_max(GBps))
        machine_spec_dict.update(L2_theo_max(GBps))
    else:
        print("Warning: unknown GBps for {}, theoretical max disabled".format(arch))
        print("         pass --gbps <value> to enable theoretical max")
        machine_spec_dict['theo_max'] = False

    with open(machine_spec_filename, 'w') as outfile:
        yaml.dump(machine_spec_dict, outfile)


def bench_yaml_for_run(input_yaml_file, flush_bytes):
    """Input YAML for hipblas-bench, with flush_memory_size injected into every test when requested."""
    if not flush_bytes:
        return input_yaml_file
    text = Path(input_yaml_file).read_text()
    injected, count = re.subn(r'^(\s*)(- name:.*)$',
                              lambda m: '{0}{1}\n{0}  flush_memory_size: {2}'.format(m.group(1), m.group(2), flush_bytes),
                              text, flags=re.MULTILINE)
    if count == 0:
        raise SystemExit("no '- name:' test entries found in {}".format(input_yaml_file))
    # Written next to the original so relative 'include:' paths still resolve.
    flushed = os.path.splitext(input_yaml_file)[0] + '.flush.yaml'
    Path(flushed).write_text(injected)
    return flushed


def run_command(bench_command, input_yaml_file, output_csv_file):

    output = subprocess.check_output([bench_command,
                                      '--log_function_name',
                                      '--log_datatype',
                                      '--yaml', input_yaml_file])
    output = output.decode('utf-8')

    with open(output_csv_file, 'w') as f:
        f.write(output)

if __name__ =='__main__':

    parser = argparse.ArgumentParser(
            description='run hipblas-bench for multiple yaml files',
            epilog='Example usage: python3 benchmark.py ' +
                    '-l blas1 -t gfx906  -f scal -f axpy ' +
                    '-b ../../../build_tensile/release/clients/staging/hipblas-bench ')

    parser.add_argument('-l', help='level.',          dest='level',          default="blas1")
    parser.add_argument('-t', help='tag.',            dest='tag',            default="gfx906")
    parser.add_argument('-f', help='function names.', dest='function_names', required=True, action='append')
    parser.add_argument('-b', help='bench command.',  dest='bench_command',  default="../../../build/release/clients/staging/hipblas-bench")
    parser.add_argument('-d', help='GPU device number.', dest='device_number', type=int, default=0)
    parser.add_argument('--gbps', help='memory bandwidth in GB/s for theoretical max; overrides built-in per-arch value.', dest='gbps', type=float, default=None)
    parser.add_argument('--sizes', help='sweep size set: default ({level}/*.yaml) or large ({level}/large/*.yaml).', dest='sizes', choices=['default', 'large'], default='default')
    parser.add_argument('--flush-mb', help='pass flush_memory_size (MiB) to hipblas-bench so inputs are evicted from cache between calls.', dest='flush_mb', type=float, default=None)

    args = parser.parse_args()

    cuda = detect_cuda()
    arch = getspecs.getgfx(args.device_number, cuda)
    if arch == 'N/A':
        print("Unable to detect GPU architecture for device {}".format(args.device_number))
        quit()
    if arch not in args.tag:
        print("Warning: tag does not contain architecture, it is recommended that tag contain the architecture")
        print("         tag, architecture = ", args.tag, ", ", arch)

    output_dir = os.path.join(args.level,args.tag)

    if os.path.exists(output_dir):
        print("Warning: directory exists: ", output_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    machine_spec_filename = os.path.join(args.level, args.tag, "machine_spec.yaml")

    write_machine_spec_yaml(machine_spec_filename, args.device_number, args.gbps)

    input_dir = args.level if args.sizes == 'default' else os.path.join(args.level, 'large')
    flush_bytes = int(args.flush_mb * 1024 * 1024) if args.flush_mb else 0

    for function_name in args.function_names:

        input_filename = os.path.join(input_dir, function_name+".yaml")
        if not os.path.isfile(input_filename):
            print("Warning: skipping {}: no {}".format(function_name, input_filename))
            continue
        input_filename = bench_yaml_for_run(input_filename, flush_bytes)
        output_filename = os.path.join(args.level, args.tag, function_name+".csv")

        try:
            run_command(args.bench_command, input_filename, output_filename)
        finally:
            if input_filename.endswith('.flush.yaml'):
                os.remove(input_filename)
