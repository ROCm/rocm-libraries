#!/usr/bin/env python3

import argparse
import re
import statistics
import subprocess
import os
import threading
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

# Clock samples taken while the GPU is idle (host setup, gaps between calls) do not describe the
# clocks the kernels ran at; with activity available, only samples at or above this are used.
BUSY_ACTIVITY_PCT = 50


def get_gfx_activity(device_number, cuda):
    """GFX engine activity in percent from amd-smi, or None when it cannot be read."""
    if cuda:
        return None
    cmd = [getspecs.get_smi_exec(cuda), "metric", "-g", str(device_number), "-u", "--csv"]
    success, cout = getspecs._subprocess_helper(cmd)
    if not success:
        return None
    for col in ('gfx_activity', 'GFX_ACTIVITY', 'gfx_busy_inst'):
        value = ClockSampler._mhz(getspecs.get_csv_val(cout, col, device_number))
        if value is not None:
            return value
    return None


class ClockSampler:
    """Samples sclk, mclk and GFX activity with amd-smi in a background thread while benchmarks run."""

    def __init__(self, device_number, cuda, interval):
        self.device_number = device_number
        self.cuda = cuda
        self.interval = interval
        self.samples = []   # (sclk MHz or None, mclk MHz or None, activity % or None)
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def _mhz(value):
        match = re.search(r'[-+]?\d+(\.\d+)?', str(value))
        return float(match.group(0)) if match else None

    def _run(self):
        while not self._stop.is_set():
            sclk = self._mhz(getspecs.getsclk(self.device_number, self.cuda))
            mclk = self._mhz(getspecs.getmclk(self.device_number, self.cuda))
            activity = get_gfx_activity(self.device_number, self.cuda)
            if sclk or mclk:
                self.samples.append((sclk, mclk, activity))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
        return False

    @staticmethod
    def busy_samples(samples):
        """
        Samples taken while the GPU was busy, and whether activity was available to decide.
        Without activity readings every sample is kept.
        """
        if not any(a is not None for _, _, a in samples):
            return list(samples), False
        return [smp for smp in samples if smp[2] is not None and smp[2] >= BUSY_ACTIVITY_PCT], True

    @staticmethod
    def summary(samples):
        if not samples:
            return None
        return {'min': min(samples), 'median': statistics.median(samples),
                'mean': statistics.mean(samples), 'max': max(samples), 'count': len(samples)}


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
    if clock_scale is None and sclk_stats and settings['sclk_peak_mhz']:
        clock_scale = sclk_stats['median'] / float(settings['sclk_peak_mhz'])
    if mclk_scale is None and mclk_stats and settings['mclk_max_mhz']:
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
    parser.add_argument('--peak-gflops', help='compute peaks in GFLOP/s, e.g. f32=100000,f64=2000 (keys: f16, bf16, f32, f64).', dest='peak_gflops', default=None)
    parser.add_argument('--machine-file', help='YAML with gbps, peak_gflops, sclk_peak_mhz, mclk_max_mhz, l2_mb, partitions; command line options override it.', dest='machine_file', default=None)
    parser.add_argument('--clock-scale', help='scale for compute peaks, sclk_during_run / sclk_peak.', dest='clock_scale', type=float, default=None)
    parser.add_argument('--mclk-scale', help='scale for memory bandwidth, mclk_during_run / mclk_max.', dest='mclk_scale', type=float, default=None)
    parser.add_argument('--auto-clock-scale', help='sample sclk/mclk with amd-smi during the run and derive the scales from the median (needs --sclk-peak-mhz / --mclk-max-mhz).', dest='auto_clock_scale', action='store_true')
    parser.add_argument('--sclk-peak-mhz', help='reference engine clock for --auto-clock-scale.', dest='sclk_peak_mhz', type=float, default=None)
    parser.add_argument('--mclk-max-mhz', help='reference memory clock for --auto-clock-scale.', dest='mclk_max_mhz', type=float, default=None)
    parser.add_argument('--telemetry-interval', help='seconds between amd-smi samples.', dest='telemetry_interval', type=float, default=0.5)
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

    sampler = ClockSampler(args.device_number, cuda, args.telemetry_interval) if args.auto_clock_scale else None
    if sampler:
        sampler.__enter__()
    try:
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
    finally:
        if sampler:
            sampler.__exit__(None, None, None)

    telemetry = {}
    sclk_stats = mclk_stats = None
    if sampler:
        busy, have_activity = ClockSampler.busy_samples(sampler.samples)
        sclk_stats = ClockSampler.summary([smp[0] for smp in busy if smp[0]])
        mclk_stats = ClockSampler.summary([smp[1] for smp in busy if smp[1]])
        telemetry['sclk_mhz'] = sclk_stats
        telemetry['mclk_mhz'] = mclk_stats
        telemetry['telemetry_samples'] = len(sampler.samples)
        telemetry['telemetry_busy_samples'] = len(busy) if have_activity else None
        telemetry['perf_level'] = getspecs.getperflevel(args.device_number, cuda)
        if sampler.samples and not have_activity:
            print("Warning: amd-smi reported no GFX activity; clock statistics include idle samples")
        if not sclk_stats and not mclk_stats:
            print("Warning: --auto-clock-scale collected no busy amd-smi samples; clock scales left at 1.0")
        elif (sclk_stats and not (args.sclk_peak_mhz or machine_file.get('sclk_peak_mhz'))) or \
             (mclk_stats and not (args.mclk_max_mhz or machine_file.get('mclk_max_mhz'))):
            print("Warning: --auto-clock-scale needs --sclk-peak-mhz / --mclk-max-mhz to turn samples into scales")
    if flush_bytes:
        telemetry['flush_memory_size'] = flush_bytes

    settings = resolve_settings(args, machine_file, sclk_stats, mclk_stats)
    write_machine_spec_yaml(machine_spec_filename, args.device_number, settings=settings, telemetry=telemetry)
