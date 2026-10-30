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
``ROCKE_HIPTENSOR_PACK_BUILDER`` it skips.

Always checked, with no build: the manifests of hipTensor's pack targets
(``builders/hiptensor/manifests/<arch>.manifest``: every kernel with its attributes and argument
names, the routes and the kernel classes of hipTensor's JIT) are the ones the Python writer
writes for this tree (``make_rocke_pack.py --list``). hipTensor's test rocke_pack_manifest_test
checks that its C++ builder writes the same manifests (``--list --check-manifests``), so a
change of a kernel row, attribute, route or JIT class on either side fails a default test until
both sides and the manifests agree. The kernel ABI names of rocKE's C header must equal the
Python ones.

With the built C++ builder, both writers also write the GEMM selection models of rocKE's tensor
GETT planner next to the packs of the targets it serves (``--gemm-models``,
``<out>/<arch>/gemm_models``, the data hipTensor's contraction JIT reads): the same files and the
same ``index.txt`` byte for byte.

With the built C++ builder, its ``--rreduce-picks`` mode (hipTensor's untuned reduction choice,
``rocke_rreduce_rule.hpp``) must return the candidates of ``rreduce_host.untuned_choices`` for
random problems, compute-unit counts and workspace limits on every pack target.
"""

import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

_LIBROOT = Path(__file__).resolve().parents[1]
_WRITER = _LIBROOT / "builders" / "hiptensor" / "make_rocke_pack.py"
_MANIFESTS = _LIBROOT / "builders" / "hiptensor" / "manifests"
# hipTensor's pack targets (HIPTENSOR_ROCKE_SUPPORTED_TARGETS of its hiptensorRockePack.cmake)
_PACK_TARGETS = ("gfx90a", "gfx942", "gfx950")
# hipTensor's tree in the rocm-libraries monorepo (absent in a standalone rocKE checkout)
_HIPTENSOR = _LIBROOT.parents[3] / "projects" / "hiptensor"


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
        "rreduce",
        "rreduce_final",
        "ewt",
        "ewv",
        "ewf",
        "gemm",
        "gett_plan",
        "dgett",
        "rgett",
        "gett64",
        "gett32",
    }, families


@pytest.mark.parametrize("arch", _PACK_TARGETS)
def test_manifest_is_current(arch, tmp_path):
    """The committed manifest of ``arch`` is the one the Python writer writes for this tree."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(
        [
            sys.executable,
            str(_WRITER),
            "--list",
            "--arch",
            arch,
            "--out",
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    got = (tmp_path / f"{arch}.manifest").read_text().splitlines()
    want = (_MANIFESTS / f"{arch}.manifest").read_text().splitlines()
    if got != want:
        missing = [line for line in want if line not in set(got)]
        extra = [line for line in got if line not in set(want)]
        pytest.fail(
            f"{arch}: builders/hiptensor/manifests/{arch}.manifest is stale; regenerate it with "
            "make_rocke_pack.py --list --arch gfx90a --arch gfx942 --arch gfx950 --out "
            "library/builders/hiptensor/manifests, then make hipTensor's C++ pack builder agree "
            "(its test rocke_pack_manifest_test)\n"
            + "\n".join(f"  committed only: {line}" for line in missing[:20])
            + "\n"
            + "\n".join(f"  written only:   {line}" for line in extra[:20])
            + (
                "\n  (the same lines in another order)"
                if not missing and not extra
                else ""
            )
        )


def test_manifests_cover_the_pack_targets():
    """One manifest per pack target of hipTensor (and its CMake list, in the monorepo)."""
    assert sorted(p.stem for p in _MANIFESTS.glob("*.manifest")) == list(_PACK_TARGETS)
    for arch in _PACK_TARGETS:
        text = (_MANIFESTS / f"{arch}.manifest").read_bytes()
        assert b"\r" not in text, f"{arch}.manifest: LF line endings only"
        lines = text.decode().splitlines()
        assert lines[0] == f"# rocKE kernel pack manifest {arch}"
        kinds = {line.split(" ", 1)[0] for line in lines[1:]}
        assert kinds == {"kernel", "route", "jit"}, kinds
    cmake = _HIPTENSOR / "cmake" / "Functions" / "hiptensorRockePack.cmake"
    if not cmake.is_file():
        pytest.skip(f"no hipTensor tree at {_HIPTENSOR}")
    m = re.search(
        r"set\(HIPTENSOR_ROCKE_SUPPORTED_TARGETS ([^)]*)\)", cmake.read_text()
    )
    assert m is not None, cmake
    assert tuple(m.group(1).split()) == _PACK_TARGETS


def test_jit_classes_are_the_jit_rows_classes():
    """The JIT classes of the manifests: every class of jit_classes() of every row of the pack's
    generic elementwise kernel (32- and 64-bit offsets, mixed-type rows, math rows; the rule
    hipTensor's jitClassAdmitted encodes, whose C++ builder lists the classes that rule admits),
    and every launch of that kernel classifies (jit_class_of, the rule of hipTensor's classify)
    to one of them: packed, gathered and odd layouts of every row."""
    mod = _load_writer()
    from kernels.common.hiptensor_aot import common as C
    from kernels.common.hiptensor_aot.ew import EwSpec
    from kernels.common.hiptensor_aot.host import canonicalize, ew_args
    from kernels.common.hiptensor_aot.jit_ew import jit_class_of, jit_classes

    names = mod.hiptensor_jit_classes()
    assert len(names) == len(set(names)) and names == sorted(names)
    rows = [p for idx64 in (False, True) for p in mod.ew_params(idx64)]
    every = {c.kernel_name() for p in rows for c in jit_classes([EwSpec(**p)])}
    assert set(names) == every
    # the widened kinds are in: mixed-type rows, math rows with vec 1, 64-bit-offset math rows
    mixed = [p for p in rows if p not in mod.ew_params(p["idx64"], mixed=False)]
    assert mixed and all(
        EwSpec(**p).kernel_name().replace("hiptensor_aot_ew", "hiptensor_jit_ew", 1)
        + "_v1"
        in names
        for p in mixed
    )
    for p in rows:
        if p.get("math"):
            # vec 1 and the all-vector class, also with 64-bit offsets
            assert {c.vec for c in jit_classes([EwSpec(**p)])} == {
                1,
                16 // C.DTYPES[p["d"]][1],
            }
            assert all(c.kernel_name() in names for c in jit_classes([EwSpec(**p)])), p
    layouts = {
        "packed": [(64, None), (9, None), (3, None)],
        "gathered": [(64, 0), (9, None), (3, None)],
        "odd": [(13, None), (9, None), (3, None)],
    }
    for p in rows:
        base = EwSpec(**p)
        tensors = base.tensors()
        for kind, modes in layouts.items():
            # column-major packed tensors; "gathered": the first input's mode 0 is its slowest
            dims = []
            for m, (ext, gather) in enumerate(modes):
                strides = []
                for t in tensors:
                    order = (
                        [1, 2, 0]
                        if gather is not None and t == tensors[0]
                        else [0, 1, 2]
                    )
                    st = 1
                    for q in order[: order.index(m)]:
                        st *= modes[q][0]
                    strides.append(st)
                dims.append((ext, strides))
            vals, _, _ = ew_args(canonicalize(dims, len(tensors) - 1), tensors)
            for n in tensors[:-1]:
                vals["op" + n] = C.OP_IDENTITY
            vals["opAB"] = vals["opABC"] = C.OP_ADD
            spec, _, _ = jit_class_of(base, vals)
            assert spec.kernel_name() in names, (p, kind, spec.kernel_name())
    for arch in _PACK_TARGETS:
        lines = (_MANIFESTS / f"{arch}.manifest").read_text().splitlines()
        assert [line[4:] for line in lines if line.startswith("jit ")] == names


def test_kernel_abi_names_match_the_c_header():
    """rocke/instance_hiptensor_aot.h's ROCKE_HTAOT_*_ABI names are the Python ABI constants."""
    from kernels.common.hiptensor_aot import (
        dgett,
        ew,
        ewf,
        ewt,
        jit_ew,
        reduce,
        rgett,
        rreduce,
    )

    header = (
        _LIBROOT.parent
        / "platform"
        / "cpp"
        / "include"
        / "rocke"
        / "instance_hiptensor_aot.h"
    ).read_text()
    macros = dict(re.findall(r'#define ROCKE_HTAOT_(\w+)_ABI "([^"]*)"', header))
    assert macros == {
        "EW": ew.EW_ABI,
        "REDUCE": reduce.REDUCE_ABI,
        "RREDUCE": rreduce.RREDUCE_ABI,
        "EW_JIT": jit_ew.EW_JIT_ABI,
        "EWV": jit_ew.EWV_ABI,
        "EWT": ewt.EWT_ABI,
        "EWF": ewf.EWF_ABI,
        "DGETT": dgett.DGETT_ABI,
        "RGETT": rgett.RGETT_ABI,
    }


def test_writer_gemm_models_and_digest_check(tmp_path):
    """The writer's --gemm-models directory: exactly the files the planner's model loader reads
    for a target the tensor GETT planner serves (none for the others), an index of their SHA-256
    digests by name, and the index's digest, which a v3 selection model reading the planner's
    predicted time must name (another one is refused)."""
    import hashlib

    from rocke.instances.common.tensor.gett import TENSOR_GETT_ARCHS

    w = _load_writer()
    src = _LIBROOT.parent / "platform" / "python" / "rocke" / "dispatch" / "gemm"
    src = src / "selector_models"
    for arch in _PACK_TARGETS:
        digest = w.write_gemm_models(str(src), str(tmp_path / arch), arch)
        d = tmp_path / arch / "gemm_models"
        if arch not in TENSOR_GETT_ARCHS:
            assert digest == "" and not d.exists(), arch
            continue
        files = w.gemm_model_files(str(src), arch)
        assert files and sorted(p.name for p in d.iterdir()) == sorted(
            [*files, "index.txt"]
        )
        index = (d / "index.txt").read_bytes()
        assert digest == hashlib.sha256(index).hexdigest()
        lines = index.decode().splitlines()
        assert [ln.split(" ", 1)[1] for ln in lines] == sorted(files)
        for ln in lines:
            sha, name = ln.split(" ", 1)
            assert sha == hashlib.sha256((src / name).read_bytes()).hexdigest(), name
        # the sidecars name the guard bags and skinny models the loader reads
        rcr = json.loads((src / f"gemm_rcr_{arch}.meta.json").read_text())
        skinny = json.loads((src / f"skinny_cfg_{arch}.meta.json").read_text())
        named = list(rcr.get("guard", {}).get("bags", [])) + [skinny["model"]]
        named += list(skinny.get("bags", []))
        assert named and set(named) <= set(files), sorted(set(named) - set(files))
        head = "v3;schema=3;routes=11;margin=0;batched=1;gemm=%s;features=route,lg_pred_ns;trees=0"
        good = {"select.model.contraction.h": head % digest}
        bad = {"select.model.contraction.h": head % ("0" * 64)}
        none = {"select.model.contraction.h": head % "-"}
        assert w.gemm_digest_errors(good, digest) == []
        assert w.gemm_digest_errors(none, digest) == []
        assert len(w.gemm_digest_errors(bad, digest)) == 1
        assert (
            w.select_model_gemm_digest(
                "v2;schema=2;routes=3;margin=0;features=x;trees=0"
            )
            == ""
        )


# ------------------------------------------------------------------------------------------------
# With the built C++ builder: the manifests and the packs
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


def test_cpp_manifests_equal_the_committed_ones(tmp_path):
    """The C++ builder lists the manifests this tree commits (hipTensor's
    rocke_pack_manifest_test runs the same check from hipTensor's build)."""
    tool = _builder()
    cmd = [tool, "--list", "--check-manifests", str(_MANIFESTS), "--out", str(tmp_path)]
    for arch in _PACK_TARGETS:
        cmd += ["--arch", arch]
    _run(cmd, dict(os.environ), tmp_path / "list.log")


def _rreduce_problem(rng):
    """A random reduction in hipTensor's canonical form (kept: extent, A stride, D stride; red:
    extent, A stride), A stored in a random mode order with padding, D in another order; at
    most 2^40 elements (the counts stay far from int64 overflow in the C++ rule)."""
    from kernels.common.hiptensor_aot.host import canonicalize

    rank = rng.randrange(1, 7)
    shape = [
        rng.choice([1, 2, 3, 5, 8, 16, 33, 64, 128, 1000, 4096, 1 << 16, 3 << 18])
        for _ in range(rank)
    ]
    while math.prod(shape) > 1 << 40:
        i = max(range(rank), key=lambda j: shape[j])
        shape[i] = max(1, shape[i] // 7)
    perm = list(range(rank))
    rng.shuffle(perm)
    sa, acc = [0] * rank, 1
    for i in perm:
        sa[i] = acc
        acc *= shape[i] + (rng.choice([0, 0, 3]) if i == perm[0] else 0)
    kept = sorted(rng.sample(range(rank), rng.randrange(0, rank + 1)))
    sd, acc = {}, 1
    order = list(kept)
    rng.shuffle(order)
    for i in order:
        sd[i] = acc
        acc *= shape[i]
    kd = canonicalize([(shape[i], [sa[i], sd[i]]) for i in kept], 0)
    rd = canonicalize([(shape[i], [sa[i]]) for i in range(rank) if i not in kept], 0)
    return [(e, s[0], s[1]) for e, s in kd], [(e, s[0]) for e, s in rd]


@pytest.mark.parametrize("arch", _PACK_TARGETS)
def test_cpp_rreduce_picks_equal_python(arch, tmp_path):
    """hipTensor's untuned reduction choice is rocKE's: the C++ rule (rocke_rreduce_rule.hpp,
    which hipTensor's planner runs; here through the C++ builder's --rreduce-picks with the
    target's palette and prototypes) returns exactly the candidates (class and split, in order)
    of rreduce_host.untuned_choices, for random problems (data types, operators, opA, layouts),
    compute-unit counts of the targets' devices and workspace limits (none, zero, small).
    """
    from kernels.common.hiptensor_aot import rreduce_host as H
    from kernels.common.hiptensor_aot.rreduce import OPS

    tool = _builder()
    classes = H.pack_classes(arch)
    protos = H.prototypes()
    rng = random.Random(4096 + _PACK_TARGETS.index(arch))
    lines, want = [], []
    capped = split = 0
    for _ in range(1500):
        kept, red = _rreduce_problem(rng)
        a = rng.choice(("f16", "bf16", "f32", "f64"))
        op = rng.choice(tuple(OPS))
        una = rng.random() < 0.25
        cus = rng.choice((80, 104, 110, 120, 220, 228, 256, 304))
        limit = rng.choice((None, None, None, 0, 256, 4096, 1 << 16, 1 << 20))
        ch = H.untuned_choices(
            kept,
            red,
            a,
            a,
            op,
            classes,
            protos,
            una=una,
            cus=cus,
            arch=arch,
            workspace_limit=limit,
        )
        free = H.untuned_choices(
            kept, red, a, a, op, classes, protos, una=una, cus=cus, arch=arch
        )
        if [k.split for k, _ in ch] != [k.split for k, _ in free]:
            capped += 1
        split += any(k.split > 1 for k, _ in ch)
        want.append(" ".join(f"{s.kernel_name()}:split{k.split}" for k, s in ch) or "-")
        lines.append(
            f"cus={cus};limit={-1 if limit is None else limit};una={int(una)};a={a};d={a};"
            f"op={op};k={','.join(f'{e}:{x}:{y}' for e, x, y in kept)};"
            f"r={','.join(f'{e}:{x}' for e, x in red)}"
        )
    # the sample exercises splits and the workspace limit
    assert split > 100 and capped > 50, (split, capped)
    problems = tmp_path / "problems.txt"
    problems.write_text("\n".join(lines) + "\n")
    got = _run(
        [tool, "--rreduce-picks", str(problems), "--arch", arch],
        dict(os.environ),
        tmp_path / "picks.log",
    ).splitlines()
    assert len(got) == len(want)
    bad = [(p, g, w) for p, g, w in zip(lines, got, want) if g != w]
    assert not bad, f"{len(bad)} of {len(want)} differ:\n" + "\n".join(
        f"{p}\n  C++    {g}\n  Python {w}" for p, g, w in bad[:10]
    )


def test_cpp_pack_equals_python_pack(tmp_path):
    tool = _builder()
    comgr = _builder_comgr(tool)
    arches = [
        a
        for a in os.environ.get(
            "ROCKE_HIPTENSOR_PACK_ARCHS", ":".join(_PACK_TARGETS)
        ).split(":")
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
    # the planner's GEMM selection models (no compile: cheap), next to the packs
    gemm_models = (
        _LIBROOT.parent / "platform" / "python" / "rocke" / "dispatch" / "gemm"
    )
    common += ["--gemm-models", str(gemm_models / "selector_models")]

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

        # 3. the planner's GEMM selection models next to the pack: the same files, byte for byte
        # (a directory only for the targets the planner serves, on both sides)
        from rocke.instances.common.tensor.gett import TENSOR_GETT_ARCHS

        cg, pg = cpp_out / arch / "gemm_models", py_out / arch / "gemm_models"
        assert cg.is_dir() == pg.is_dir() == (arch in TENSOR_GETT_ARCHS), arch
        if cg.is_dir():
            cf = sorted(p.name for p in cg.iterdir())
            pf = sorted(p.name for p in pg.iterdir())
            assert cf == pf, f"{arch}/gemm_models: {cf} vs {pf}"
            assert "index.txt" in cf and len(cf) > 1
            for name in cf:
                assert (cg / name).read_bytes() == (
                    pg / name
                ).read_bytes(), (
                    f"{arch}/gemm_models/{name}: the C++ builder's file differs"
                )
