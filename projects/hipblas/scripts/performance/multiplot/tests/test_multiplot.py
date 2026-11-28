# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ########################################################################
"""
Host-only tests for the multiplot scripts (no GPU needed):
    python3 -m pytest scripts/performance/multiplot/tests
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

import pytest
import yaml

MULTIPLOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MULTIPLOT_DIR)

import benchmark  # noqa: E402
import perf_model  # noqa: E402

PREFIX_TO_PREC = {'s': 'f32_r', 'd': 'f64_r', 'c': 'f32_c', 'z': 'f64_c'}

# Large enough that O(n) vector terms are negligible next to the matrix terms.
LARGE = {'N': 200000, 'M': 200000, 'K': 64, 'KL': 64, 'KU': 64, 'transA': 'N'}
LARGE_VECTOR = {'N': 10**9}


def _legacy_factors():
    factors = dict(benchmark.L1_theo_max(1.0))
    factors.update(benchmark.L2_theo_max(1.0))
    return factors


@pytest.mark.parametrize('label,factor', sorted(_legacy_factors().items()))
def test_model_matches_legacy_factors(label, factor):
    prec = PREFIX_TO_PREC[label[0]]
    func = label[1:]
    params = LARGE_VECTOR if func in ('axpy', 'dot', 'scal', 'nrm2', 'iamax') else LARGE
    assert perf_model.intensity(func, prec, params) == pytest.approx(factor, rel=0.01)


def test_fp64_compute_peak_binds_on_low_fp64_parts():
    gbps, peaks = 10000, {'f32': 100000, 'f64': 2000}
    assert perf_model.ceiling_gflops('gemv', 'f64_r', LARGE, gbps, peaks) == 2000
    assert perf_model.ceiling_gflops('hemv', 'f64_c', LARGE, gbps, peaks) == 2000
    assert perf_model.ceiling_gflops('axpy', 'f32_r', LARGE_VECTOR, gbps, peaks) == pytest.approx(gbps / 6, rel=1e-6)
    assert perf_model.ceiling_gflops('axpy', 'f64_r', LARGE_VECTOR, gbps, peaks) == pytest.approx(gbps / 12, rel=1e-6)


def test_ceiling_terms_are_optional():
    assert perf_model.ceiling_gflops('gemm', 'f32_r', LARGE, None, {'f32': 1000}) == 1000
    assert perf_model.ceiling_gflops('axpy', 'f32_r', LARGE_VECTOR, 600, None) == pytest.approx(100, rel=1e-6)
    assert perf_model.ceiling_gflops('axpy', 'f32_r', LARGE_VECTOR, None, None) is None
    assert perf_model.ceiling_gflops('not_a_function', 'f32_r', LARGE_VECTOR, 600, None) is None


def test_banded_intensity_depends_on_bandwidth():
    narrow = perf_model.intensity('gbmv', 'f32_r', dict(LARGE, KL=1, KU=1))
    wide = perf_model.intensity('gbmv', 'f32_r', dict(LARGE, KL=4096, KU=4096))
    # wide band approaches gemv (1/2); a tridiagonal band is dominated by vector traffic
    assert wide == pytest.approx(0.5, rel=0.01)
    assert narrow < wide
    assert perf_model.intensity('tbmv', 'f64_r', {'M': 200000, 'K': 4096}) == pytest.approx(0.25, rel=0.01)


def test_parse_peak_gflops():
    assert perf_model.parse_peak_gflops('f32=100000, f64=2000') == {'f32': 100000.0, 'f64': 2000.0}
    with pytest.raises(ValueError):
        perf_model.parse_peak_gflops('fp32=1')


def _args(**overrides):
    defaults = dict(gbps=None, peak_gflops=None, sclk_peak_mhz=None, mclk_max_mhz=None,
                    l2_mb=None, partitions=None, clock_scale=None, mclk_scale=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_resolve_settings_cli_overrides_machine_file():
    machine_file = {'gbps': 1000, 'peak_gflops': {'f32': 10, 'f64': 5}, 'l2_mb': 8}
    settings = benchmark.resolve_settings(_args(gbps=2000, peak_gflops='f64=7'), machine_file)
    assert settings['gbps'] == 2000
    assert settings['peak_gflops'] == {'f32': 10, 'f64': 7.0}
    assert settings['l2_mb'] == 8
    assert settings['clock_scale'] == 1.0 and settings['mclk_scale'] == 1.0
    assert settings['partitions'] == 1


def test_resolve_settings_auto_clock_scale():
    sclk = {'median': 1500.0}
    mclk = {'median': 900.0}
    settings = benchmark.resolve_settings(_args(sclk_peak_mhz=2000, mclk_max_mhz=1000), {}, sclk, mclk)
    assert settings['clock_scale'] == pytest.approx(0.75)
    assert settings['mclk_scale'] == pytest.approx(0.9)
    assert settings['per_result_sclk'] and settings['per_result_mclk']
    # an explicit scale wins over the reported clocks
    settings = benchmark.resolve_settings(_args(sclk_peak_mhz=2000, mclk_max_mhz=1000, clock_scale=0.5),
                                          {}, sclk, mclk)
    assert settings['clock_scale'] == 0.5
    assert settings['per_result_mclk'] and not settings['per_result_sclk']
    # clocks without a reference clock leave the scale at 1
    settings = benchmark.resolve_settings(_args(), {}, sclk, mclk)
    assert settings['clock_scale'] == 1.0 and settings['mclk_scale'] == 1.0
    assert not settings['per_result_sclk'] and not settings['per_result_mclk']


def test_write_machine_spec_effective_values(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, 'detect_cuda', lambda: False)
    monkeypatch.setattr(benchmark.getspecs, 'getgfx', lambda dev, cuda: 'gfx_test')
    settings = {'gbps': 20000, 'peak_gflops': {'f64': 4000}, 'clock_scale': 0.9,
                'mclk_scale': 0.5, 'partitions': 2, 'l2_mb': 100, 'per_result_sclk': True,
                'per_result_mclk': False}
    spec_file = tmp_path / 'machine_spec.yaml'
    benchmark.write_machine_spec_yaml(str(spec_file), settings=settings, telemetry={'perf_level': 'HIGH'})
    spec = yaml.safe_load(spec_file.read_text())
    assert spec['arch'] == 'gfx_test'
    assert spec['GBps'] == 20000
    assert spec['GBps_effective'] == pytest.approx(5000)
    assert spec['peak_gflops_effective']['f64'] == pytest.approx(1800)
    assert spec['saxpy'] == pytest.approx(5000 / 6)
    assert spec['theo_max'] is True
    assert spec['perf_level'] == 'HIGH'
    assert spec['per_result_sclk'] is True and 'per_result_mclk' not in spec


def test_write_machine_spec_unknown_arch_without_gbps(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, 'detect_cuda', lambda: False)
    monkeypatch.setattr(benchmark.getspecs, 'getgfx', lambda dev, cuda: 'gfx_test')
    spec_file = tmp_path / 'machine_spec.yaml'
    benchmark.write_machine_spec_yaml(str(spec_file), settings={})
    spec = yaml.safe_load(spec_file.read_text())
    assert spec['theo_max'] is False
    assert 'GBps' not in spec


def test_load_machine_file_rejects_unknown_keys(tmp_path):
    good = tmp_path / 'good.yaml'
    good.write_text('gbps: 100\npeak_gflops: {f32: 1}\n')
    assert benchmark.load_machine_file(str(good))['gbps'] == 100
    bad = tmp_path / 'bad.yaml'
    bad.write_text('bandwidth: 100\n')
    with pytest.raises(SystemExit):
        benchmark.load_machine_file(str(bad))


def test_clock_summary():
    summary = benchmark.clock_summary([1.0, 3.0, 2.0])
    assert summary['median'] == 2.0 and summary['count'] == 3
    assert benchmark.clock_summary([]) is None


def test_result_clocks():
    assert perf_model.result_clocks({'lowest-avg-freq': '1500.5', 'avg-MCLK': '1900'}) == (1500.5, 1900.0)
    # hipblas-bench prints 0 when amd-smi had no reading
    assert perf_model.result_clocks({'lowest-avg-freq': '0', 'avg-MCLK': 'nan?'}) == (None, None)
    assert perf_model.result_clocks({'N': '10'}) == (None, None)


# hipblas-bench output with HIPBLAS_BENCH_FREQ: clock columns follow the timing columns
BENCH_WITH_CLOCKS = (
    'function,compute_type,N,alpha,incx,incy,hipblas-Gflops,hipblas-GB/s,hipblas-us,lowest-avg-freq,'
    'lowest-median-freq,avg-MCLK,median-MCLK,\n'
    'axpy,f32_c,1024,(1,0),1,1,10, 20, 30, 1500, 1490, 900, 900, \n'
    'function,compute_type,N,alpha,incx,incy,hipblas-Gflops,hipblas-GB/s,hipblas-us,lowest-avg-freq,'
    'lowest-median-freq,avg-MCLK,median-MCLK,\n'
    'axpy,f32_c,2048,(1,0),1,1,10, 20, 30, 1300, 1290, 0, 0, \n'
)


def test_bench_clocks_reads_every_result(tmp_path):
    out = tmp_path / 'axpy.csv'
    out.write_text(BENCH_WITH_CLOCKS)
    sclk, mclk = benchmark.bench_clocks([str(out), str(tmp_path / 'missing.csv')])
    assert sclk == [1500.0, 1300.0]
    assert mclk == [900.0]          # the second result had no memory clock


def test_machine_ceilings_use_the_clocks_of_each_result():
    import plot
    spec = {'GBps': 2000, 'GBps_effective': 1000, 'peak_gflops': {'f32': 20000},
            'peak_gflops_effective': {'f32': 7500}, 'sclk_peak_mhz': 2000, 'mclk_max_mhz': 1000,
            'partitions': 2, 'per_result_sclk': True, 'per_result_mclk': True}
    row = {'lowest-avg-freq': '1000', 'avg-MCLK': '500'}
    assert plot.machine_ceilings(spec, row) == (pytest.approx(500), {'f32': pytest.approx(5000)})
    # a result without clocks, or no row, uses the run-wide values
    assert plot.machine_ceilings(spec, {'N': '1'}) == (1000, {'f32': 7500})
    assert plot.machine_ceilings(spec) == (1000, {'f32': 7500})
    # an explicit --clock-scale leaves per_result_sclk unset, so the row's engine clock is ignored
    fixed = dict(spec, per_result_sclk=False)
    assert plot.machine_ceilings(fixed, row) == (pytest.approx(500), {'f32': 7500})


def test_flush_yaml_injection(tmp_path):
    src = tmp_path / 'axpy.yaml'
    src.write_text('Tests:\n  - name: axpy_bench\n    function: axpy\n  - name: other\n    function: dot\n')
    out = benchmark.bench_yaml_for_run(str(src), 1048576)
    data = yaml.safe_load(open(out).read())
    assert out.endswith('.flush.yaml')
    assert benchmark.bench_yaml_for_run(str(src), 0) == str(src)
    assert [t['flush_memory_size'] for t in data['Tests']] == [1048576, 1048576]


def test_flush_yaml_injection_matches_sweep_files(tmp_path):
    for level, func in (('blas1', 'axpy'), ('blas2', 'gbmv'), ('blas3', 'gemm')):
        src = os.path.join(MULTIPLOT_DIR, level, func + '.yaml')
        dst = tmp_path / (func + '.yaml')
        shutil.copy(src, dst)
        out = benchmark.bench_yaml_for_run(str(dst), 4096)
        text = open(out).read()
        tests = text.count('- name:')
        assert tests > 0 and text.count('flush_memory_size: 4096') == tests


@pytest.mark.parametrize('prefix', ['s', 'd', 'c', 'z'])
def test_level1_theo_max_covers_the_level1_sweep(prefix):
    factors = benchmark.L1_theo_max(1.0)
    for func in ('axpy', 'dot', 'scal', 'nrm2', 'iamax'):
        assert prefix + func in factors


def test_large_sweeps_include_the_shared_definitions():
    for level in ('blas1', 'blas2'):
        large_dir = os.path.join(MULTIPLOT_DIR, level, 'large')
        files = sorted(f for f in os.listdir(large_dir) if f.endswith('.yaml'))
        assert files
        for f in files:
            text = open(os.path.join(large_dir, f)).read()
            assert 'include: ../../../../../clients/include/hipblas_common.yaml' in text, f


HIPBLAS_DIR = os.path.abspath(os.path.join(MULTIPLOT_DIR, '..', '..', '..'))
SIZE_ARGS = ('M', 'N', 'K', 'KL', 'KU')


def _sweep_files():
    for level in ('blas1', 'blas2', 'blas3'):
        for sub in ('', 'large'):
            d = os.path.join(MULTIPLOT_DIR, level, sub)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.endswith('.yaml'):
                        yield os.path.join(level, sub, f)


def _argument_names():
    common = yaml.safe_load(open(os.path.join(HIPBLAS_DIR, 'clients', 'include', 'hipblas_common.yaml')))
    return {list(entry)[0] for entry in common['Arguments']}


@pytest.mark.parametrize('sweep', list(_sweep_files()))
def test_sweep_sets_the_size_arguments_its_test_reads(sweep):
    # hipblas_gentest.py ignores unknown keys, so a misspelled or missing size argument makes every
    # case quick-return and the CSV comes out empty
    text = open(os.path.join(MULTIPLOT_DIR, sweep)).read()
    text = re.sub(r'^include:.*$', '', text, flags=re.MULTILINE)
    # aliases defined in the included hipblas_common.yaml (precision lists) are not needed here
    local = set(re.findall(r'&(\w+)', text))
    text = re.sub(r'\*(\w+)', lambda m: m.group(0) if m.group(1) in local else 'external', text)
    data = yaml.safe_load(text)
    known = _argument_names()
    for test in data['Tests']:
        sizes = test.get('matrix_size') or [{}]
        keys = set(test) | {k for size in sizes for k in size}
        for key in keys:
            clash = [name for name in known if name.lower() == str(key).lower() and name != key]
            assert not clash, f"{sweep}: '{key}' should be '{clash[0]}'"
        func = test['function']
        hpp = [os.path.join(HIPBLAS_DIR, 'clients', 'include', lvl, f'testing_{func}.hpp')
               for lvl in ('blas1', 'blas2', 'blas3')]
        hpp = [h for h in hpp if os.path.isfile(h)]
        if not hpp:
            continue
        reads = set(re.findall(r'\barg\.(' + '|'.join(SIZE_ARGS) + r')\b', open(hpp[0]).read()))
        for size in sizes:
            missing = reads - set(size) - set(test)
            assert not missing, f"{sweep}: testing_{func}.hpp reads arg.{sorted(missing)} but the sweep does not set it"


# ---------- plot.py end to end on the checked-in gfx90a results ----------

@pytest.fixture
def data_root(tmp_path):
    for level in ('blas1', 'blas2'):
        shutil.copytree(os.path.join(MULTIPLOT_DIR, level, 'gfx90a'), tmp_path / level / 'gfx90a')
    return tmp_path


def _plot(data_root, *args):
    cmd = [sys.executable, os.path.join(MULTIPLOT_DIR, 'plot.py'), '--base', str(data_root),
           '--tag1', 'gfx90a', '--tag2', 'gfx90a'] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True)


def test_plot_theo_max_includes_functions_that_used_to_crash(data_root):
    # nrm2 (level 1) and tbmv (level 2) had no theoretical maximum and raised KeyError
    r = _plot(data_root, '-l', 'blas1', '--theo_max', '-f', 'dot', '-f', 'axpy', '-f', 'scal', '-f', 'nrm2')
    assert r.returncode == 0, r.stderr
    r = _plot(data_root, '-l', 'blas2', '--theo_max', '--label1', 'M', '--label2', 'N',
              '-f', 'gemv', '-f', 'trmv', '-f', 'tpmv', '-f', 'tbmv')
    assert r.returncode == 0, r.stderr
    assert len(list((data_root / 'blas2' / 'gfx90a' / 'plots_vs_theo_max').glob('*.png'))) == 4


def test_plot_roofline_writes_csv_and_figures(data_root):
    spec_file = data_root / 'blas2' / 'gfx90a' / 'machine_spec.yaml'
    spec = yaml.safe_load(spec_file.read_text())
    spec.update({'peak_gflops': {'f32': 23000, 'f64': 300}, 'peak_gflops_effective': {'f32': 23000, 'f64': 300},
                 'l2_mb': 8})
    spec_file.write_text(yaml.dump(spec))
    r = _plot(data_root, '-l', 'blas2', '--roofline', '--label1', 'M', '--label2', 'N',
              '-f', 'gemv', '-f', 'ger', '-f', 'hemv', '-f', 'tbmv', '--achievable-bw-frac', '0.65')
    assert r.returncode == 0, r.stderr
    out = data_root / 'blas2' / 'gfx90a' / 'plots_roofline'
    assert len(list(out.glob('*.png'))) == 4
    lines = (out / 'roofline_gemv_ger_hemv_tbmv.csv').read_text().splitlines()
    header = lines[0].split(',')
    rows = [dict(zip(header, line.split(','))) for line in lines[1:]]
    # a 300 GFLOP/s FP64 peak is below 1600 GB/s x 1/4 FLOP/byte, so dgemv is compute-bound here
    dgemv = [row for row in rows if row['function'] == 'gemv' and row['compute_type'] == 'f64_r']
    assert dgemv and all(float(row['ceiling']) == 300 for row in dgemv)
    assert any(row['cache_resident'] == 'True' for row in rows)
    # results benchmarked without HIPBLAS_BENCH_FREQ have no clocks
    assert all(row['sclk_mhz'] == '' and row['mclk_mhz'] == '' for row in rows)


def test_plot_draws_every_precision_any_function_has(tmp_path):
    # hemm is complex only; listed first it used to hide the real precisions of the other functions
    shutil.copytree(os.path.join(MULTIPLOT_DIR, 'blas3', 'gfx90a'), tmp_path / 'blas3' / 'gfx90a')
    r = _plot(tmp_path, '-l', 'blas3', '--label1', 'M', '--label2', 'N', '-f', 'hemm', '-f', 'symm')
    assert r.returncode == 0, r.stderr
    drawn = {'_'.join(f.stem.split('_')[-2:])
             for f in (tmp_path / 'blas3' / 'gfx90a' / 'plots_gflops').glob('*.png')}
    assert {'f32_r', 'f64_r', 'f32_c', 'f64_c'} <= drawn


def test_cache_resident_only_for_bandwidth_bound_results():
    import plot
    spec = {'GBps': 1000, 'peak_gflops_effective': {'f32': 10000}, 'l2_mb': 1000}
    small_axpy = {'N': 1000}
    small_gemm = {'M': 256, 'N': 256, 'K': 256}
    # axpy: 1000 GB/s x 1/6 FLOP/byte is far below the peak, so it is memory-bound and fits in cache
    assert plot.is_cache_resident('axpy', 'f32_r', small_axpy, spec)
    # gemm: intensity 32 FLOP/byte makes it compute-bound, so cache reuse is expected, not flagged
    assert not plot.is_cache_resident('gemm', 'f32_r', small_gemm, spec)
    # without a bandwidth there is no memory-bound ceiling to compare with
    assert not plot.is_cache_resident('axpy', 'f32_r', small_axpy, {'l2_mb': 1000})


def test_roofline_legend_has_one_entry_per_function(data_root, monkeypatch):
    import matplotlib.figure
    import plot
    labels = []
    def capture(fig, *args, **kwargs):
        legend = fig.axes[0].get_legend()
        labels.append([t.get_text() for t in legend.get_texts()])
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', capture)
    spec = {'GBps': 1600, 'peak_gflops_effective': {'f32': 23000, 'f64': 300}, 'l2_mb': 4096}
    # l2_mb this large makes every gfx90a gemv result cache-resident, the case that used to add a
    # second, empty legend entry
    plot.plot_roofline(str(data_root / 'blas2' / 'gfx90a'), ['gemv', 'ger'], spec, str(data_root / 'out'), 'test')
    assert labels
    for figure_labels in labels:
        functions = [l for l in figure_labels if not l.startswith(('ceiling', 'published'))]
        assert len(functions) == len(set(functions)), figure_labels


def test_roofline_splits_gemm_by_transpose(tmp_path, monkeypatch):
    import csv
    import matplotlib.figure
    import plot
    shutil.copytree(os.path.join(MULTIPLOT_DIR, 'blas3', 'gfx90a'), tmp_path / 'blas3')
    # the checked-in gemm results have one transpose combination (N, T); add the same rows as N, N
    gemm_csv = tmp_path / 'blas3' / 'gemm.csv'
    lines = gemm_csv.read_text().splitlines()
    extra = []
    for i, line in enumerate(lines[:-1]):
        if 'hipblas-Gflops' in line:
            header = line.split(',')
            data = lines[i + 1].split(',')
            data[header.index('transB')] = 'N'
            extra += [line, ','.join(data)]
    gemm_csv.write_text('\n'.join(lines + extra) + '\n')
    labels = {}
    def capture(fig, path, *args, **kwargs):
        labels[os.path.basename(str(path))] = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', capture)
    spec = {'GBps': 1600, 'peak_gflops_effective': {'f32': 23000, 'f64': 11500}}
    out = tmp_path / 'out'
    plot.plot_roofline(str(tmp_path / 'blas3'), ['gemm', 'symm'], spec, str(out), 'test')

    f32 = labels['roofline_gemm_symm_f32_r']
    gemm_series = sorted(l for l in f32 if l.startswith('sgemm'))
    # transposes share a column, so each combination gets its own series
    assert len(gemm_series) > 1 and all(len(l.split()) == 2 for l in gemm_series), f32
    # a function swept with one transpose setting keeps a single, unsuffixed series
    assert 'ssymm' in f32

    rows = list(csv.DictReader(open(out / 'roofline_gemm_symm.csv')))
    gemm_rows = [r for r in rows if r['function'] == 'gemm']
    assert {r['transA'] + r['transB'] for r in gemm_rows} == {l.split()[1] for l in gemm_series}


def test_plot_modes_are_exclusive_and_need_a_ceiling(data_root):
    r = _plot(data_root, '-l', 'blas1', '--theo_max', '--roofline', '-f', 'axpy')
    assert r.returncode != 0
    spec_file = data_root / 'blas1' / 'gfx90a' / 'machine_spec.yaml'
    spec_file.write_text(yaml.dump({'arch': 'gfx90a', 'theo_max': False}))
    r = _plot(data_root, '-l', 'blas1', '--theo_max', '-f', 'axpy')
    assert r.returncode != 0 and 'no theoretical maximum' in (r.stdout + r.stderr)
