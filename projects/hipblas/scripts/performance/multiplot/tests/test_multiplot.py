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
    # explicit scale wins over samples
    settings = benchmark.resolve_settings(_args(sclk_peak_mhz=2000, clock_scale=0.5), {}, sclk, mclk)
    assert settings['clock_scale'] == 0.5
    # samples without a reference clock leave the scale at 1
    settings = benchmark.resolve_settings(_args(), {}, sclk, mclk)
    assert settings['clock_scale'] == 1.0 and settings['mclk_scale'] == 1.0


def test_write_machine_spec_effective_values(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, 'detect_cuda', lambda: False)
    monkeypatch.setattr(benchmark.getspecs, 'getgfx', lambda dev, cuda: 'gfx_test')
    settings = {'gbps': 20000, 'peak_gflops': {'f64': 4000}, 'clock_scale': 0.9,
                'mclk_scale': 0.5, 'partitions': 2, 'l2_mb': 100}
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


def test_clock_sampler_parsing_and_summary():
    assert benchmark.ClockSampler._mhz('1500 MHz') == 1500.0
    assert benchmark.ClockSampler._mhz('1350') == 1350.0
    assert benchmark.ClockSampler._mhz('N/A') is None
    summary = benchmark.ClockSampler.summary([1.0, 3.0, 2.0])
    assert summary['median'] == 2.0 and summary['count'] == 3
    assert benchmark.ClockSampler.summary([]) is None


def test_clock_sampler_keeps_only_busy_samples():
    samples = [(50.0, 900.0, 0.0), (1600.0, 1300.0, 98.0), (1400.0, 1300.0, 75.0), (400.0, 900.0, 10.0)]
    busy, have_activity = benchmark.ClockSampler.busy_samples(samples)
    assert have_activity
    assert [smp[0] for smp in busy] == [1600.0, 1400.0]
    # without activity readings nothing can be filtered
    no_activity = [(50.0, 900.0, None), (1600.0, 1300.0, None)]
    busy, have_activity = benchmark.ClockSampler.busy_samples(no_activity)
    assert not have_activity and busy == no_activity


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


def test_plot_modes_are_exclusive_and_need_a_ceiling(data_root):
    r = _plot(data_root, '-l', 'blas1', '--theo_max', '--roofline', '-f', 'axpy')
    assert r.returncode != 0
    spec_file = data_root / 'blas1' / 'gfx90a' / 'machine_spec.yaml'
    spec_file.write_text(yaml.dump({'arch': 'gfx90a', 'theo_max': False}))
    r = _plot(data_root, '-l', 'blas1', '--theo_max', '-f', 'axpy')
    assert r.returncode != 0 and 'no theoretical maximum' in (r.stdout + r.stderr)
