# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Phase-1 build driver: delegates to install-host.sh (legacy bash) while ensuring
# the project venv's bin is first on PATH so CMake's Python3 resolves to an
# interpreter with PyYAML (and other build-time Python deps).

from __future__ import annotations

import os
import sys
from pathlib import Path
from shlex import quote
from typing import List

from invoke import task

SCRIPT_DIR = Path(__file__).resolve().parent
HOST_SCRIPT = SCRIPT_DIR / "install-host.sh"


def _venv_bin_dir() -> Path:
    if sys.platform == "win32":
        return SCRIPT_DIR / "build" / "venv" / "Scripts"
    return SCRIPT_DIR / "build" / "venv" / "bin"


def _legacy_argv(
    *,
    install_deps: bool,
    install_pkg: bool,
    clients: bool,
    relocatable: bool,
    debug: bool,
    cuda: bool,
    static: bool,
    address_sanitizer: bool,
    relwithdebinfo: bool,
    codecoverage: bool,
    architecture: str,
    cpu_ref_lib: str,
    install_prefix: str,
    tensile_logic: str,
    tensile_cov: str,
    tensile_fork: str,
    tensile_tag: str,
    tensile_test_local_path: str,
    tensile_version: str,
    no_tensile: bool,
    merge_files: bool,
    no_merge_files: bool,
    msgpack: bool,
    build_dir: str,
    keep_build_tmp: bool,
    disable_hipsparselt_marker: bool,
    enable_tensile_marker: bool,
) -> List[str]:
    argv: List[str] = []
    if install_deps:
        argv.append("-d")
    if install_pkg:
        argv.append("-i")
    if clients:
        argv.append("-c")
    if relocatable:
        argv.append("-r")
    if debug:
        argv.append("-g")
    if cuda:
        argv.append("--cuda")
    if static:
        argv.append("--static")
    if address_sanitizer:
        argv.append("--address-sanitizer")
    if relwithdebinfo:
        argv.append("-k")
    if codecoverage:
        argv.append("--codecoverage")
    if architecture and architecture != "all":
        argv.extend(["-a", architecture])
    if cpu_ref_lib and cpu_ref_lib != "blis":
        argv.extend(["--cpu_ref_lib", cpu_ref_lib])
    if install_prefix:
        argv.extend(["--prefix", install_prefix])
    if tensile_logic:
        argv.extend(["-l", tensile_logic])
    if tensile_cov:
        argv.extend(["-o", tensile_cov])
    if tensile_fork:
        argv.extend(["-f", tensile_fork])
    if tensile_tag:
        argv.extend(["-b", tensile_tag])
    if tensile_test_local_path:
        argv.extend(["-t", tensile_test_local_path])
    if tensile_version:
        argv.extend(["-u", tensile_version])
    if no_tensile:
        argv.append("-n")
    if merge_files:
        argv.append("--merge-files")
    if no_merge_files:
        # Matches legacy getopt option spelling.
        argv.append("-no-merge-files")
    if not msgpack:
        argv.append("--no-msgpack")
    # When msgpack is True, omit flags; legacy defaults match.
    if build_dir and build_dir != "build":
        argv.extend(["--build_dir", build_dir])
    if keep_build_tmp:
        argv.append("--keep-build-tmp")
    if disable_hipsparselt_marker:
        argv.append("--disable-hipsparselt-marker")
    if enable_tensile_marker:
        argv.append("--enable-tensile-marker")
    return argv


@task(
    auto_shortflags=False,
    help={
        "install_deps": "Install build dependencies (-d).",
        "install_pkg": "Build packages and install via distro tool (-i).",
        "clients": "Build clients/tests/benchmarks (-c).",
        "relocatable": "Relocatable ROCm package (-r).",
        "debug": "Debug CMake build (-g).",
        "cuda": "CUDA backend (--cuda).",
        "static": "Static library (--static).",
        "address_sanitizer": "AddressSanitizer (--address-sanitizer).",
        "relwithdebinfo": "RelWithDebInfo (-k).",
        "codecoverage": "Code coverage (--codecoverage).",
        "architecture": "GPU_TARGETS value for -a/--architecture.",
        "cpu_ref_lib": "blis or lapack (--cpu_ref_lib).",
        "install_prefix": "CMAKE install prefix dir name (--prefix), default hipsparselt-install.",
        "tensile_logic": "TensileLite logic path (-l).",
        "tensile_cov": "Tensile coverage (-o).",
        "tensile_fork": "Tensile fork (-f).",
        "tensile_tag": "Tensile tag/branch (-b).",
        "tensile_test_local_path": "Tensile local test path (-t).",
        "tensile_version": "Custom Tensile version (-u).",
        "no_tensile": "Legacy -n (not recommended; build may fail).",
        "merge_files": "Pass legacy --merge-files.",
        "no_merge_files": "Pass legacy -no-merge-files.",
        "msgpack": "Tensile msgpack backend (default True); use --no-msgpack for YAML format.",
        "build_dir": "Build directory name (--build_dir).",
        "keep_build_tmp": "Keep Tensile build tmp (--keep-build-tmp).",
        "disable_hipsparselt_marker": "Disable marker library.",
        "enable_tensile_marker": "Enable Tensile marker.",
    }
)
def build(
    c,
    install_deps=False,
    install_pkg=False,
    clients=False,
    relocatable=False,
    debug=False,
    cuda=False,
    static=False,
    address_sanitizer=False,
    relwithdebinfo=False,
    codecoverage=False,
    architecture="all",
    cpu_ref_lib="blis",
    install_prefix="",
    tensile_logic="",
    tensile_cov="",
    tensile_fork="",
    tensile_tag="",
    tensile_test_local_path="",
    tensile_version="",
    no_tensile=False,
    merge_files=False,
    no_merge_files=False,
    msgpack=True,
    build_dir="build",
    keep_build_tmp=False,
    disable_hipsparselt_marker=False,
    enable_tensile_marker=False,
):
    if not HOST_SCRIPT.is_file():
        print(f"hipSPARSELt: missing legacy driver {HOST_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    if merge_files and no_merge_files:
        print("hipSPARSELt: use only one of --merge-files and --no-merge-files.", file=sys.stderr)
        sys.exit(1)

    legacy = _legacy_argv(
        install_deps=install_deps,
        install_pkg=install_pkg,
        clients=clients,
        relocatable=relocatable,
        debug=debug,
        cuda=cuda,
        static=static,
        address_sanitizer=address_sanitizer,
        relwithdebinfo=relwithdebinfo,
        codecoverage=codecoverage,
        architecture=architecture,
        cpu_ref_lib=cpu_ref_lib,
        install_prefix=install_prefix,
        tensile_logic=tensile_logic,
        tensile_cov=tensile_cov,
        tensile_fork=tensile_fork,
        tensile_tag=tensile_tag,
        tensile_test_local_path=tensile_test_local_path,
        tensile_version=tensile_version,
        no_tensile=no_tensile,
        merge_files=merge_files,
        no_merge_files=no_merge_files,
        msgpack=msgpack,
        build_dir=build_dir,
        keep_build_tmp=keep_build_tmp,
        disable_hipsparselt_marker=disable_hipsparselt_marker,
        enable_tensile_marker=enable_tensile_marker,
    )

    cmd = ["bash", str(HOST_SCRIPT), *legacy]
    env = os.environ.copy()
    vbin = _venv_bin_dir()
    if vbin.is_dir():
        env["PATH"] = str(vbin) + os.pathsep + env.get("PATH", "")
    elif install_deps or clients:
        print(
            "hipSPARSELt: warning: build/venv not found; "
            "run ./install.sh once or create the venv so Python deps resolve.",
            file=sys.stderr,
        )

    line = " ".join(quote(part) for part in cmd)
    c.run(line, env=env, pty=False)
