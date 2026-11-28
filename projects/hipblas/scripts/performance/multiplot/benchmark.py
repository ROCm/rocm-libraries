#!/usr/bin/env python3

import argparse
import re
import statistics
import subprocess
import os
from pathlib import Path
import yaml
import sys

script_dir = os.path.dirname( __file__ )
getspec_module_dir = os.path.join(script_dir, '..', 'blas')
sys.path.append(getspec_module_dir)
sys.path.append(script_dir)
import getspecs
import perf_model

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

def clock_summary(values):
    if not values:
        return None
    return {'min': min(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'max': max(values), 'count': len(values)}


def bench_clocks(output_csv_files):
    """
    Clocks hipblas-bench reported with each result when run with HIPBLAS_BENCH_FREQ: the lowest
    per-XCD engine clock and the memory clock in MHz, as (sclk list, mclk list) over all results.
    """
    sclk, mclk = [], []
    for filename in output_csv_files:
        if not os.path.isfile(filename):
            continue
        lines = Path(filename).read_text().splitlines()
        for header, values in zip(lines, lines[1:]):
            if 'hipblas-Gflops' not in header:
                continue
            row = {k.strip(): v.strip() for k, v in zip(header.split(','), re.split(r',\s*(?![^()]*\))', values))}
            result_sclk, result_mclk = perf_model.result_clocks(row)
            if result_sclk:
                sclk.append(result_sclk)
            if result_mclk:
                mclk.append(result_mclk)
    return sclk, mclk


def load_machine_file(filename):
    """Optional user YAML with gbps, peak_gflops, sclk_peak_mhz, mclk_max_mhz, l2_mb, partitions."""
    if not filename:
        return {}
    data = yaml.safe_load(Path(filename).read_text()) or {}
    known = {'gbps', 'peak_gflops', 'sclk_peak_mhz', 'mclk_max_mhz', 'l2_mb', 'partitions'}
    unknown = set(data) - known
    if unknown:
        raise SystemExit("unknown keys in {}: {}".format(filename, sorted(unknown)))
    peaks = data.get('peak_gflops') or {}
    for key in peaks:
        if key not in perf_model.PEAK_KEYS:
            raise SystemExit("unknown peak_gflops key '{}' in {}".format(key, filename))
    return data


def resolve_settings(args, machine_file, sclk_stats=None, mclk_stats=None):
    """Merge CLI options over the machine file and derive the effective ceilings."""
    def pick(cli_value, key):
        return cli_value if cli_value is not None else machine_file.get(key)

    settings = {
        'gbps': pick(args.gbps, 'gbps'),
        'peak_gflops': dict(machine_file.get('peak_gflops') or {}),
        'sclk_peak_mhz': pick(args.sclk_peak_mhz, 'sclk_peak_mhz'),
        'mclk_max_mhz': pick(args.mclk_max_mhz, 'mclk_max_mhz'),
        'l2_mb': pick(args.l2_mb, 'l2_mb'),
        'partitions': pick(args.partitions, 'partitions') or 1,
    }
    settings['peak_gflops'].update(perf_model.parse_peak_gflops(args.peak_gflops))

    clock_scale = args.clock_scale
    mclk_scale = args.mclk_scale
    # Clocks reported with the results scale each result's ceiling (per_result_*); the medians give
    # the scale for the run as a whole. An explicit --clock-scale / --mclk-scale overrides both.
    settings['per_result_sclk'] = clock_scale is None and bool(sclk_stats and settings['sclk_peak_mhz'])
    settings['per_result_mclk'] = mclk_scale is None and bool(mclk_stats and settings['mclk_max_mhz'])
    if settings['per_result_sclk']:
        clock_scale = sclk_stats['median'] / float(settings['sclk_peak_mhz'])
    if settings['per_result_mclk']:
        mclk_scale = mclk_stats['median'] / float(settings['mclk_max_mhz'])
    settings['clock_scale'] = clock_scale if clock_scale is not None else 1.0
    settings['mclk_scale'] = mclk_scale if mclk_scale is not None else 1.0
    return settings


def write_machine_spec_yaml(machine_spec_filename, device_number=0, gbps=None, settings=None, telemetry=None):
    machine_spec_dict = {}
    cuda = detect_cuda()
    machine_spec_dict['arch'] = getspecs.getgfx(device_number, cuda)
    if machine_spec_dict['arch'] == 'N/A':
        print("Unable to detect GPU architecture for device {}".format(device_number))
        print("quitting ", sys.argv[0])
        quit()

    settings = dict(settings or {})
    if gbps is not None:
        settings['gbps'] = gbps
    partitions = settings.get('partitions') or 1

    arch = machine_spec_dict['arch']
    GBps = settings.get('gbps')
    if GBps is None:
        GBps = GBPS_BY_ARCH.get(arch)
    peaks = settings.get('peak_gflops') or {}

    machine_spec_dict['clock_scale'] = settings.get('clock_scale', 1.0)
    machine_spec_dict['mclk_scale'] = settings.get('mclk_scale', 1.0)
    machine_spec_dict['partitions'] = partitions
    for key in ('per_result_sclk', 'per_result_mclk'):
        if settings.get(key):
            machine_spec_dict[key] = True
    for key in ('sclk_peak_mhz', 'mclk_max_mhz', 'l2_mb'):
        if settings.get(key) is not None:
            machine_spec_dict[key] = settings[key]
    if telemetry:
        machine_spec_dict.update(telemetry)

    if peaks:
        machine_spec_dict['peak_gflops'] = dict(peaks)
        machine_spec_dict['peak_gflops_effective'] = {
            k: v * machine_spec_dict['clock_scale'] / partitions for k, v in peaks.items()}

    if GBps is not None:
        GBps_effective = GBps * machine_spec_dict['mclk_scale'] / partitions
        machine_spec_dict['GBps'] = GBps
        machine_spec_dict['GBps_effective'] = GBps_effective
        machine_spec_dict.update(L1_theo_max(GBps_effective))
        machine_spec_dict.update(L2_theo_max(GBps_effective))
    else:
        print("Warning: unknown GBps for {}, memory-bound theoretical max disabled".format(arch))
        print("         pass --gbps <value> to enable it")

    machine_spec_dict['theo_max'] = GBps is not None or bool(peaks)

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


def run_command(bench_command, input_yaml_file, output_csv_file, env=None):

    output = subprocess.check_output([bench_command,
                                      '--log_function_name',
                                      '--log_datatype',
                                      '--yaml', input_yaml_file], env=env)
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
    parser.add_argument('--peak-gflops', help='compute peaks in GFLOP/s, e.g. f32=100000,f64=2000 (keys: f16, bf16, f32, f64).', dest='peak_gflops', default=None)
    parser.add_argument('--machine-file', help='YAML with gbps, peak_gflops, sclk_peak_mhz, mclk_max_mhz, l2_mb, partitions; command line options override it.', dest='machine_file', default=None)
    parser.add_argument('--clock-scale', help='scale for compute peaks, sclk_during_run / sclk_peak.', dest='clock_scale', type=float, default=None)
    parser.add_argument('--mclk-scale', help='scale for memory bandwidth, mclk_during_run / mclk_max.', dest='mclk_scale', type=float, default=None)
    parser.add_argument('--auto-clock-scale', help='have hipblas-bench report the clocks of each result (HIPBLAS_BENCH_FREQ) and scale each ceiling by them (needs --sclk-peak-mhz / --mclk-max-mhz).', dest='auto_clock_scale', action='store_true')
    parser.add_argument('--sclk-peak-mhz', help='reference engine clock for --auto-clock-scale.', dest='sclk_peak_mhz', type=float, default=None)
    parser.add_argument('--mclk-max-mhz', help='reference memory clock for --auto-clock-scale.', dest='mclk_max_mhz', type=float, default=None)
    parser.add_argument('--partitions', help='number of compute partitions the peaks are split across.', dest='partitions', type=int, default=None)
    parser.add_argument('--l2-mb', help='L2 cache size in MiB, used to flag cache-resident results in plots.', dest='l2_mb', type=float, default=None)
    parser.add_argument('--sizes', help='sweep size set: default ({level}/*.yaml) or large ({level}/large/*.yaml).', dest='sizes', choices=['default', 'large'], default='default')
    parser.add_argument('--flush-mb', help='pass flush_memory_size (MiB) to hipblas-bench so inputs are evicted from cache between calls.', dest='flush_mb', type=float, default=None)

    args = parser.parse_args()

    machine_file = load_machine_file(args.machine_file)

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

    input_dir = args.level if args.sizes == 'default' else os.path.join(args.level, 'large')
    flush_bytes = int(args.flush_mb * 1024 * 1024) if args.flush_mb else 0

    bench_env = None
    if args.auto_clock_scale:
        bench_env = dict(os.environ, HIPBLAS_BENCH_FREQ='1')
        bench_env.pop('HIPBLAS_BENCH_FREQ_ALL', None)

    output_filenames = []
    for function_name in args.function_names:

        input_filename = os.path.join(input_dir, function_name+".yaml")
        if not os.path.isfile(input_filename):
            print("Warning: skipping {}: no {}".format(function_name, input_filename))
            continue
        input_filename = bench_yaml_for_run(input_filename, flush_bytes)
        output_filename = os.path.join(args.level, args.tag, function_name+".csv")

        try:
            run_command(args.bench_command, input_filename, output_filename, bench_env)
        finally:
            if input_filename.endswith('.flush.yaml'):
                os.remove(input_filename)
        output_filenames.append(output_filename)

    telemetry = {}
    sclk_stats = mclk_stats = None
    if args.auto_clock_scale:
        sclk, mclk = bench_clocks(output_filenames)
        sclk_stats = clock_summary(sclk)
        mclk_stats = clock_summary(mclk)
        # over results; each result's own clocks scale its ceiling in plot.py
        telemetry['sclk_mhz'] = sclk_stats
        telemetry['mclk_mhz'] = mclk_stats
        telemetry['perf_level'] = getspecs.getperflevel(args.device_number, cuda)
        if not sclk_stats and not mclk_stats:
            print("Warning: hipblas-bench reported no clocks; --auto-clock-scale needs a hipblas-bench with"
                  " HIPBLAS_BENCH_FREQ support, built with amd-smi. Clock scales left at 1.0")
        elif (sclk_stats and not (args.sclk_peak_mhz or machine_file.get('sclk_peak_mhz'))) or \
             (mclk_stats and not (args.mclk_max_mhz or machine_file.get('mclk_max_mhz'))):
            print("Warning: --auto-clock-scale needs --sclk-peak-mhz / --mclk-max-mhz to turn clocks into scales")
    if flush_bytes:
        telemetry['flush_memory_size'] = flush_bytes

    settings = resolve_settings(args, machine_file, sclk_stats, mclk_stats)
    write_machine_spec_yaml(machine_spec_filename, args.device_number, settings=settings, telemetry=telemetry)
