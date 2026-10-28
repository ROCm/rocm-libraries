# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""hipTensor's C++ kernel-pack builder writes the pack rocKE's Python reference writes.

hipTensor's build writes its rocKE kernel packs with a C++ host tool (hipTensor's
``library/src/backends/rocke/pack/rocke_pack_builder.cpp``, built from rocKE's C++ engine);
``builders/hiptensor/make_rocke_pack.py`` writes the same packs with rocKE's Python engine. For
the same rocKE sources and COMGR library, given the Python writer's two header strings, the C++
builder must write byte-identical ``rocke.htpk`` and ``rocke.index.json`` files. Without them
only those two strings may differ (decoded comparison).

The cross-check needs the built C++ builder and the COMGR library it links; see
``builders/hiptensor/README.md`` for the environment variables. Without
``ROCKE_HIPTENSOR_PACK_BUILDER`` it skips. The table checks run everywhere.
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

_LIBROOT = Path(__file__).resolve().parents[1]
_WRITER = _LIBROOT / "builders" / "hiptensor" / "make_rocke_pack.py"


# ------------------------------------------------------------------------------------------------
# Pack reader (the format of hipTensor's pack/rocke_pack_format.cpp and the writer's write_pack)
# ------------------------------------------------------------------------------------------------


def read_pack(path):
    d = Path(path).read_bytes()
    assert d[:8] == b"HTPKIDX\0", path
    (ver, minr, abi, nk, na, nat, nr, _r) = struct.unpack_from("<8I", d, 8)
    koff, aoff, atoff, roff, soff, slen = struct.unpack_from("<6Q", d, 40)
    boff, _blen = struct.unpack_from("<2Q", d, 88)
    arch, cid, rv, pid = struct.unpack_from("<4I", d, 104)
    strings = d[soff : soff + slen]

    def s(o):
        return strings[o : strings.index(b"\0", o)].decode()

    kernels = {}
    for i in range(nk):
        b = koff + 96 * i
        f = struct.unpack_from("<12I", d, b)
        co_off, co_size = struct.unpack_from("<2Q", d, b + 48)
        args = [
            struct.unpack_from("<4I", d, aoff + 16 * j)
            for j in range(f[3], f[3] + f[4])
        ]
        attrs = []
        for j in range(f[5], f[5] + f[6]):
            k, kind, v = struct.unpack_from("<IIq", d, atoff + 16 * j)
            attrs.append((s(k), v if kind == 0 else s(v)))
        kernels[s(f[0])] = dict(
            family=s(f[1]),
            spec_key=s(f[2]),
            args=[(s(n), o, sz, kind) for n, o, sz, kind in args],
            attrs=attrs,
            md=f[7:12],
            sha=d[b + 64 : b + 96].hex(),
            co=d[boff + co_off : boff + co_off + co_size],
        )
    routes = {}
    for i in range(nr):
        k, v = struct.unpack_from("<2I", d, roff + 8 * i)
        routes[s(k)] = s(v)
    return dict(
        header=(ver, minr, abi, nk, na, nat, nr),
        arch=s(arch),
        compiler_id=s(cid),
        rocke_version=s(rv),
        pack_id=s(pid),
        kernels=kernels,
        routes=routes,
    )


def pack_differences(a_dir, b_dir):
    """Decoded differences of two packs (first, second), the two header strings excluded."""
    a, b = read_pack(Path(a_dir) / "rocke.htpk"), read_pack(Path(b_dir) / "rocke.htpk")
    out = []
    for k in ("header", "arch", "pack_id"):
        if a[k] != b[k]:
            out.append(f"{k}: {a[k]} vs {b[k]}")
    for name in sorted(set(a["kernels"]) | set(b["kernels"])):
        ka, kb = a["kernels"].get(name), b["kernels"].get(name)
        if ka is None or kb is None:
            out.append(
                f"kernel {name}: only in {'the second pack' if ka is None else 'the first pack'}"
            )
            continue
        for f in ("family", "spec_key", "args", "attrs", "md", "sha", "co"):
            if ka[f] != kb[f]:
                out.append(f"kernel {name}: {f} differs")
    for r in sorted(set(a["routes"]) | set(b["routes"])):
        if a["routes"].get(r) != b["routes"].get(r):
            out.append(f"route {r}: {a['routes'].get(r)!r} vs {b['routes'].get(r)!r}")
    ja = json.loads((Path(a_dir) / "rocke.index.json").read_text())
    jb = json.loads((Path(b_dir) / "rocke.index.json").read_text())
    for k in ("compiler_id", "rocke_version"):
        ja.pop(k, None)
        jb.pop(k, None)
    if ja != jb:
        out.append("rocke.index.json differs beyond the header strings")
    return out


# ------------------------------------------------------------------------------------------------
# Always: the writer's tables
# ------------------------------------------------------------------------------------------------


def _load_writer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_hiptensor_make_rocke_pack", _WRITER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_writer_default_source_tree_is_this_tree():
    mod = _load_writer()
    assert Path(mod._DEFAULT_ROCKE_SRC) == _LIBROOT.parent
    for sub in ("library/kernels/common/hiptensor_aot", "platform/python/rocke"):
        assert (Path(mod._DEFAULT_ROCKE_SRC) / sub).is_dir()


def test_writer_descriptor_families():
    """Every family the C++ builder builds has descriptors in the Python writer (gfx950), and the
    family names are the ones both writers' --only filters see."""
    mod = _load_writer()
    from rocke.helpers.split_k import arch_num_cus

    descs = mod.kernel_descriptors("gfx950", arch_num_cus("gfx950"))
    families = {f for f, _ in descs}
    assert families == {
        "ew",
        "reduce",
        "ewt",
        "ewv",
        "gemm",
        "gett_plan",
        "dgett",
        "rgett",
        "gett64",
        "gett32",
    }, families


# ------------------------------------------------------------------------------------------------
# With the built C++ builder: the packs
# ------------------------------------------------------------------------------------------------


def _builder():
    tool = os.environ.get("ROCKE_HIPTENSOR_PACK_BUILDER", "")
    if not tool:
        pytest.skip(
            "ROCKE_HIPTENSOR_PACK_BUILDER (hipTensor's hiptensor_rocke_pack_builder) not set"
        )
    if not os.access(tool, os.X_OK):
        pytest.fail(f"ROCKE_HIPTENSOR_PACK_BUILDER={tool} is not an executable")
    return tool


def _builder_comgr(tool):
    lib = os.environ.get("ROCKE_HIPTENSOR_PACK_COMGR", "")
    if lib:
        return lib
    ldd = shutil.which("ldd")
    if ldd is None:
        pytest.skip(
            "no ldd: set ROCKE_HIPTENSOR_PACK_COMGR to the COMGR library of the builder"
        )
    out = subprocess.run([ldd, tool], capture_output=True, text=True, check=True).stdout
    m = re.search(r"libamd_comgr\.so\S*\s+=>\s+(\S+)", out)
    if m is None:
        pytest.fail(f"{tool} links no COMGR library:\n{out}")
    return m.group(1)


def _run(cmd, env, log):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    log.write_text(r.stdout + r.stderr)
    assert (
        r.returncode == 0
    ), f"{cmd[0]} rc={r.returncode}:\n{(r.stdout + r.stderr)[-4000:]}"
    return r.stdout


def test_cpp_pack_equals_python_pack(tmp_path):
    tool = _builder()
    comgr = _builder_comgr(tool)
    arches = [
        a
        for a in os.environ.get("ROCKE_HIPTENSOR_PACK_ARCHS", "gfx950").split(":")
        if a
    ]
    only = os.environ.get("ROCKE_HIPTENSOR_PACK_ONLY", "")
    models = os.environ.get("ROCKE_HIPTENSOR_PACK_SELECT_MODELS", "")
    jobs = os.environ.get("ROCKE_HIPTENSOR_PACK_JOBS", "")

    common = [a for arch in arches for a in ("--arch", arch)]
    if only:
        # both writers match the filter against "<family> <descriptor>"
        common += ["--only", f"^({only}) "]
    if models:
        common += ["--select-models", models]
    if jobs:
        common += ["--jobs", jobs]

    env = dict(os.environ, ROCKE_COMGR_LIB=comgr, PYTHONDONTWRITEBYTECODE="1")
    env.pop("ROCKE_LLVM_FLAVOR", None)  # both follow the COMGR library's flavor
    py_out, cpp_out, cpp_hdr = tmp_path / "py", tmp_path / "cpp", tmp_path / "cpp_hdr"
    _run(
        [
            sys.executable,
            str(_WRITER),
            "--out",
            str(py_out),
            "--cache-dir",
            str(tmp_path / "pyc"),
        ]
        + common,
        env,
        tmp_path / "py.log",
    )
    _run(
        [tool, "--out", str(cpp_out), "--cache-dir", str(tmp_path / "cppc")] + common,
        env,
        tmp_path / "cpp.log",
    )

    for arch in arches:
        py_idx = json.loads((py_out / arch / "rocke.index.json").read_text())
        cpp_idx = json.loads((cpp_out / arch / "rocke.index.json").read_text())
        assert py_idx["kernels"], f"{arch}: the Python pack has no kernel"

        # 1. the pack the build writes: only the two header strings may differ
        diffs = pack_differences(cpp_out / arch, py_out / arch)
        assert not diffs, f"{arch}: " + "\n".join(diffs[:40])
        assert cpp_idx["compiler_id"] and cpp_idx["rocke_version"]

        # 2. with the Python writer's header strings: byte-identical files (cache of run 1)
        _run(
            [
                tool,
                "--out",
                str(cpp_hdr),
                "--cache-dir",
                str(tmp_path / "cppc"),
                "--compiler-id",
                py_idx["compiler_id"],
                "--rocke-version",
                py_idx["rocke_version"],
                "--arch",
                arch,
            ]
            + common[2 * len(arches) :],
            env,
            tmp_path / f"cpp_hdr_{arch}.log",
        )
        for f in ("rocke.htpk", "rocke.index.json"):
            a = (cpp_hdr / arch / f).read_bytes()
            b = (py_out / arch / f).read_bytes()
            assert a == b, f"{arch}/{f}: the C++ pack differs from the Python pack"
