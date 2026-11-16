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

import os
import shutil
import subprocess
import sys

import pytest
import yaml

MULTIPLOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MULTIPLOT_DIR)

import benchmark  # noqa: E402


@pytest.mark.parametrize('prefix', ['s', 'd', 'c', 'z'])
def test_level1_theo_max_covers_the_level1_sweep(prefix):
    factors = benchmark.L1_theo_max(1.0)
    for func in ('axpy', 'dot', 'scal', 'nrm2', 'iamax'):
        assert prefix + func in factors


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
    assert 'skip stbmv' in r.stdout
    assert len(list((data_root / 'blas2' / 'gfx90a' / 'plots_vs_theo_max').glob('*.png'))) == 4


def test_plot_theo_max_needs_a_ceiling(data_root):
    spec_file = data_root / 'blas1' / 'gfx90a' / 'machine_spec.yaml'
    spec_file.write_text(yaml.dump({'arch': 'gfx90a', 'theo_max': False}))
    r = _plot(data_root, '-l', 'blas1', '--theo_max', '-f', 'axpy')
    assert r.returncode != 0 and 'no theoretical maximum' in (r.stdout + r.stderr)
