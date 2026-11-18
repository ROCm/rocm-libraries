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
import re
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
    assert 'skip stbmv' in r.stdout
    assert len(list((data_root / 'blas2' / 'gfx90a' / 'plots_vs_theo_max').glob('*.png'))) == 4


def test_plot_theo_max_needs_a_ceiling(data_root):
    spec_file = data_root / 'blas1' / 'gfx90a' / 'machine_spec.yaml'
    spec_file.write_text(yaml.dump({'arch': 'gfx90a', 'theo_max': False}))
    r = _plot(data_root, '-l', 'blas1', '--theo_max', '-f', 'axpy')
    assert r.returncode != 0 and 'no theoretical maximum' in (r.stdout + r.stderr)
