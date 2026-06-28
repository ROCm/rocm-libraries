################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#
################################################################################

"""Combined build-then-run test phase for YAML kernel configs (default mode).

This module runs when neither ``--build-only`` nor ``--use-cache`` is passed.
Two mechanisms keep it from running in split-CI mode: the
``pytest_ignore_collect`` hook in ``conftest.py`` excludes it at collection
time, and the test function itself calls ``pytest.skip`` if either flag is
present (fallback for pytest versions or invocation styles where the hook is
not called).

This is the single-machine equivalent of the full split-CI workflow. It drives
the same ``_build`` and ``_run`` helpers that live in ``test_config_build.py``
and ``test_config_run.py``, verifying the full artifact round-trip in one
pytest session:

  1. ``_build``  — compile kernels, compress the output to a temporary artifact
  2. wipe        — delete the build output directory
  3. ``_run``    — extract the artifact, benchmark against the cached kernels
  4. cleanup     — delete the temporary artifact

Wiping the output between steps 1 and 3 confirms the artifact is genuinely
self-contained and not relying on leftover build state.

Each phase is launched in a subprocess so that Tensile's process-level global
state accumulated during the build phase cannot bleed into the run phase.
The helpers are imported by name in the child process, keeping the logic
defined in exactly one place (the build/run modules) rather than duplicated
here.
"""

import contextlib
import glob
import os
import shutil
import subprocess
import sys

import py
import pytest

from artifact_helpers import artifact_name_for_config


def _dump_debug(label, output_dir, artifact_dir):
    """Dump directory tree and cache.yaml contents for split-test debugging."""
    print(f"\n=== DEBUG [{label}] ===", flush=True)
    print(f"  output_dir: {output_dir}", flush=True)
    print(f"  artifact_dir: {artifact_dir}", flush=True)
    print(f"  output_dir exists: {os.path.isdir(output_dir)}", flush=True)
    if os.path.isdir(output_dir):
        for root, dirs, files in os.walk(output_dir):
            depth = root.replace(output_dir, "").count(os.sep)
            if depth > 4:
                dirs.clear()
                continue
            indent = "  " * (depth + 1)
            print(f"  {indent}{os.path.basename(root)}/", flush=True)
            subindent = "  " * (depth + 2)
            for f in files[:20]:
                fpath = os.path.join(root, f)
                sz = os.path.getsize(fpath)
                print(f"  {subindent}{f} ({sz}B)", flush=True)
            if len(files) > 20:
                print(f"  {subindent}... and {len(files) - 20} more", flush=True)
    for cache_yaml in glob.glob(os.path.join(output_dir, "**", "cache.yaml"), recursive=True):
        print(f"  cache.yaml found: {cache_yaml}", flush=True)
        try:
            with open(cache_yaml) as f:
                content = f.read(2000)
            print(f"  cache.yaml content (first 2000 chars):\n{content}", flush=True)
        except Exception as e:
            print(f"  cache.yaml read error: {e}", flush=True)
    tarball = glob.glob(os.path.join(artifact_dir, "*.tar.gz"))
    if tarball:
        print(f"  tarballs in artifact_dir: {tarball}", flush=True)
        for tb in tarball:
            print(f"    {os.path.basename(tb)}: {os.path.getsize(tb)}B", flush=True)
    print(f"=== END DEBUG [{label}] ===\n", flush=True)

_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))


def _call_helper_in_subprocess(
    module: str,
    func: str,
    config: str,
    output_dir: str,
    artifact_dir: str,
    tensile_args: list[str],
) -> None:
    """Call module.func(config, output_dir, artifact_dir, tensile_args) in a subprocess.

    Each phase runs in a clean interpreter so Tensile's global state from the
    build phase cannot bleed into the run phase (uninstalled checkout case).
    PYTHONPATH is forwarded from sys.path so the child can import Tensile.
    """
    script = (
        f"import sys; sys.path.insert(0, {repr(_COMMON_DIR)}); "
        f"from {module} import {func}; "
        f"{func}(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    subprocess.run(
        [sys.executable, "-c", script, config, output_dir, artifact_dir, *tensile_args],
        check=True,
        env=env,
    )


def test_config(tensile_args: list[str], config: str, tmpdir: py.path.local, pytestconfig: pytest.Config) -> None:
    """Pytest wrapper: run the full build→artifact→run round-trip on a single machine.

    Activated in the default mode (no ``--build-only`` / ``--use-cache`` flags).
    Requires a GPU. See the module docstring for a description of the four steps.
    """
    if pytestconfig.getoption("--build-only") or pytestconfig.getoption("--use-cache"):
        pytest.skip("split mode active — use test_config_build or test_config_run")
    artifact_name = artifact_name_for_config(config)
    output_dir = os.path.join(tmpdir.strpath, artifact_name)
    artifact_dir = tmpdir.strpath
    artifact_path = os.path.join(artifact_dir, artifact_name + ".tar.gz")

    print(f"\n=== test_config debug: config={config}", flush=True)
    print(f"  tensile_args={tensile_args}", flush=True)
    print(f"  tmpdir={tmpdir.strpath}", flush=True)
    print(f"  output_dir={output_dir}", flush=True)
    print(f"  artifact_dir={artifact_dir}", flush=True)
    print(f"  artifact_path={artifact_path}", flush=True)

    _call_helper_in_subprocess("test_config_build", "_build", config, output_dir, artifact_dir, tensile_args)
    _dump_debug("after_build", output_dir, artifact_dir)
    shutil.rmtree(output_dir)
    _dump_debug("after_rmtree", output_dir, artifact_dir)
    try:
        _call_helper_in_subprocess("test_config_run", "_run", config, output_dir, artifact_dir, tensile_args)
        _dump_debug("after_run", output_dir, artifact_dir)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(artifact_path)
