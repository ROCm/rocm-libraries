# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# CI wiring for the portable-IR replay path.
#
# The implementation and its drivers live in python/rocke/portable_ir/; this
# module is what makes them run under the normal pytest invocation instead of
# only when someone remembers the driver command line.
#
# The layering, weakest gate to strongest:
#
#   unit tests      the roller, the CBOR bundle codec, and the recorder, in
#                   isolation. Pure Python, always run.
#   record coverage every production kernel the parity emitters can build is
#                   recorded, and the live recording is checked against an
#                   independent post-hoc walk of the same KernelDef. Catches a
#                   recorder that silently drops ops. Pure Python, always run.
#   parity matrix   the real gate: for every kernel and arch, the .ll produced
#                   by replaying through the C++ engine must be BYTE-IDENTICAL
#                   to the Python lowerer's. Resolves or builds a source-matched
#                   shared librocke.
#   standalone CLI  the same byte-identity, but through a binary with no Python
#                   in the process at all -- the shape this actually deploys in.
#                   It comes from the same source-keyed artifact bundle.

import os
import pkgutil
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

import rocke.portable_ir.tests as _unit_tests

_HERE = Path(__file__).resolve().parent
_PLATFORM = _HERE.parents[1]  # tests/portable_ir -> tests -> platform
_PYROOT = _PLATFORM / "python"
_LIBRARY_ROOTS = (
    _PLATFORM.parent / "library",  # source tree: rocke/library
    _PLATFORM / "tests" / "library",  # installed tree: <test-root>/tests/library
)


def _env():
    """Subprocess env with the engine package (and the library tree) importable."""
    env = dict(os.environ)
    parts = [str(_PYROOT)]
    parts.extend(str(root) for root in _LIBRARY_ROOTS if (root / "kernels").is_dir())
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # The oracle for every byte-identity claim below is the Python lowerer, and
    # the C++ pybind extension is not on this path; say so rather than emitting
    # a fallback warning per kernel.
    env["ROCKE_BACKEND"] = "python"
    return env


@dataclass(frozen=True)
class _PortableIrArtifacts:
    library: str
    replay_cli: str


@cache
def _portable_ir_artifacts() -> _PortableIrArtifacts:
    """Resolve one source-matched artifact bundle for the pytest session."""
    from rocke.portable_ir.src import online

    library = online.resolve_library_path()
    replay_cli = online.resolve_replay_cli_path()
    assert library is not None and replay_cli is not None
    online.validate_artifact_bundle(library, replay_cli)
    return _PortableIrArtifacts(library=library, replay_cli=replay_cli)


def _online_lib():
    return _portable_ir_artifacts().library


@cache
def _replay_cli():
    """Resolve the standalone CLI without imposing a shared-library dependency."""
    from rocke.portable_ir.src import online

    replay_cli = online.resolve_replay_cli_path()
    assert replay_cli is not None
    return replay_cli


def _run(args, **kw):
    return subprocess.run(
        args, cwd=str(_PLATFORM), env=_env(), capture_output=True, text=True, **kw
    )


def test_subprocess_env_includes_an_installed_library_tree(monkeypatch, tmp_path):
    installed = tmp_path / "tests" / "library"
    (installed / "kernels").mkdir(parents=True)
    monkeypatch.setattr(sys.modules[__name__], "_LIBRARY_ROOTS", (installed,))
    monkeypatch.delenv("PYTHONPATH", raising=False)

    assert str(installed) in _env()["PYTHONPATH"].split(os.pathsep)


def test_replay_cli_resolution_does_not_require_a_shared_library(monkeypatch):
    """Installed CLI consumers must not acquire the source-only library lane."""
    from rocke.portable_ir.src import online

    _replay_cli.cache_clear()
    monkeypatch.setattr(online, "resolve_replay_cli_path", lambda: "installed-cli")
    monkeypatch.setattr(
        online,
        "resolve_library_path",
        lambda: pytest.fail("CLI-only resolution requested librocke"),
    )
    try:
        assert _replay_cli() == "installed-cli"
    finally:
        _replay_cli.cache_clear()


# ---------------------------------------------------------------------
# Always-on unit/coverage lanes (never build a native artifact on demand)
# ---------------------------------------------------------------------
# Discover through the importable package rather than a repository path. TheRock
# relocates ``rocke`` beside the installed CTest metadata, where no
# ``platform/python`` source layout exists.
# ``test_hsaco_gate`` imports the HSACO parity driver, whose memory-capped
# subprocess implementation uses Unix-only ``resource`` and ``os.fork``.
# Keep the platform-independent portable-IR tests running on Windows, but do
# not claim this Unix-specific HSACO gate is covered there until it has a
# Windows-compatible subprocess implementation.
_UNIT_MODULES = sorted(
    module.name
    for module in pkgutil.iter_modules(_unit_tests.__path__)
    if module.name.startswith("test_")
    and not (os.name == "nt" and module.name == "test_hsaco_gate")
)
if not _UNIT_MODULES:
    raise RuntimeError("portable-IR unit package contains no test modules")


@pytest.mark.parametrize("mod", _UNIT_MODULES)
def test_unit(mod):
    """Run in-package units that pytest does not otherwise collect.

    Most are pure Python. A few exercise a source-matched engine when the caller
    already supplied one, but this lane never constructs native artifacts.
    """
    r = _run([sys.executable, "-m", "unittest", "-v", f"rocke.portable_ir.tests.{mod}"])
    assert r.returncode == 0, r.stderr[-4000:] or r.stdout[-4000:]


def test_record_coverage():
    """Every buildable production kernel records faithfully.

    This is what keeps the recorder honest as new kernels land: a builder that
    uses an IRBuilder call the recorder does not intercept shows up here as a
    recorder gap rather than as a mysterious parity failure later.
    """
    r = _run([sys.executable, "-m", "rocke.portable_ir.drivers.record_coverage"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]


# ---------------------------------------------------------------------
# Engine-binary lanes
# ---------------------------------------------------------------------
@pytest.mark.integration
def test_parity_matrix():
    """Replayed .ll == Python .ll, byte for byte, for every kernel x arch.

    Covers both replay paths: the concrete-graph importer (ir_export -> C
    import) and the recipe VM (record -> CBOR -> C VM). Byte-identity, not
    just equivalence -- concrete recipes carry the SSA names, so the VM
    reproduces Python's naming exactly.
    """
    lib = _online_lib()
    env_lib = dict(ROCKE_ONLINE_LIB=lib)
    r = subprocess.run(
        [sys.executable, "-m", "rocke.portable_ir.drivers.parity_matrix"],
        cwd=str(_PLATFORM),
        env={**_env(), **env_lib},
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-4000:]


def test_standalone_cli_matches_python(tmp_path):
    """A binary with no Python in it reproduces the Python lowerer's .ll.

    parity_matrix proves the C++ engine agrees, but it drives that engine over
    ctypes from inside a Python process, so it cannot by itself rule out a
    dependency on something the interpreter set up. This runs the artifacts
    through a standalone executable instead, which is how the runtime ships.
    Both artifact forms are checked: the concrete IR JSON and the CBOR recipe.
    """
    cli = _replay_cli()

    # Author the artifacts out-of-process so this test does not import the IR
    # stack into the pytest process (the recorder patches core.ir.IRBuilder).
    author = f"""
import sys
from rocke.core import ir_export
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.elementwise import ElementwiseSpec, build_elementwise
from rocke.portable_ir.src import recipe_bundle
from rocke.portable_ir.src.recording_builder import record_kernel

out = sys.argv[1]
spec = ElementwiseSpec(op="add")
kernel, recipe = record_kernel(lambda: build_elementwise(spec))
open(out + "/k.ir.json", "w").write(ir_export.export_kernel_ir_json(kernel))
open(out + "/k.recipe.cbor", "wb").write(recipe_bundle.cbor_encode(recipe))
open(out + "/py.ll", "w").write(
    lower_kernel_to_llvm(kernel, llvm_flavor="llvm20", arch="gfx950"))
"""
    r = _run([sys.executable, "-c", author, str(tmp_path)])
    assert r.returncode == 0, r.stderr[-4000:]

    expected = (tmp_path / "py.ll").read_text()
    for args, label in (
        (["--ir", str(tmp_path / "k.ir.json")], "portable IR JSON"),
        (["--recipe", str(tmp_path / "k.recipe.cbor"), "--cbor"], "CBOR recipe"),
    ):
        got = subprocess.run(
            [cli, *args, "--arch", "gfx950", "--flavor", "llvm20"],
            capture_output=True,
            text=True,
        )
        assert got.returncode == 0, f"{label}: {got.stderr[-2000:]}"
        assert (
            got.stdout == expected
        ), f"{label}: standalone replay diverged from the Python lowerer"


def _int_list(*items):
    """A list-of-bare-ints attr as ir_export writes it."""
    return {"t": "l", "v": [{"_": {"t": "i", "v": n}} for n in items]}


_ARRAY_ITEMS = [{"_": {"t": "i", "v": n}} for n in [2, 1, 1]]

_MALFORMED_LIST_ATTRS = [
    ("non-array value", {"t": "l", "v": 3}, "must be an array"),
    ("missing value", {"t": "l"}, "must be an array"),
    (
        "non-int item",
        {"t": "l", "v": [{"_": {"t": "s", "v": "x"}}]},
        "item 0 is not a bare int",
    ),
    ("item without a payload", {"t": "l", "v": [{}]}, "item 0 is not a bare int"),
    ("non-integral item", _int_list(2, 1.5), "item 1 is not an integer"),
    ("too many items", _int_list(*([1] * 17)), "at most 16"),
    (
        "wrong tag s with array value",
        {"t": "s", "v": _ARRAY_ITEMS},
        "array value requires tag",
    ),
    (
        "wrong tag b with array value",
        {"t": "b", "v": _ARRAY_ITEMS},
        "array value requires tag",
    ),
    ("missing tag with array value", {"v": _ARRAY_ITEMS}, "array value requires tag"),
]

# agpr_alloc may also be a "min,max" string, so only cluster_dims insists on "l".
_MALFORMED_CLUSTER_ATTRS = [
    ("int tag", {"t": "i", "v": 2}, 'requires tag "l"'),
    ("string tag", {"t": "s", "v": "2,2,1"}, 'requires tag "l"'),
    ("missing tag", {"v": 2}, 'requires tag "l"'),
]


@pytest.mark.parametrize(
    "key,attr,message",
    [
        (k, a, m)
        for k in ("cluster_dims", "agpr_alloc")
        for _, a, m in _MALFORMED_LIST_ATTRS
    ]
    + [("cluster_dims", a, m) for _, a, m in _MALFORMED_CLUSTER_ATTRS],
    ids=[
        f"{k}-{w}"
        for k in ("cluster_dims", "agpr_alloc")
        for w, _, _ in _MALFORMED_LIST_ATTRS
    ]
    + [f"cluster_dims-{w}" for w, _, _ in _MALFORMED_CLUSTER_ATTRS],
)
def test_standalone_importer_rejects_malformed_list_attr(tmp_path, key, attr, message):
    """A list kernel attr the importer cannot read is an error, never dropped.

    Dropping ``cluster_dims`` would compile the kernel without
    "amdgpu-cluster-dims" and launch it unclustered; Python's launch.eval_cluster
    refuses the same input, so the C importer must as well."""
    cli = _replay_cli()
    if cli is None:
        pytest.skip(
            "replay CLI not built; "
            "`cmake --build <build> --target rocke_portable_ir_replay_cli` "
            "or point ROCKE_REPLAY_CLI at it"
        )
    author = """
import sys
from rocke.core import ir_export
from rocke.instances.common.elementwise import ElementwiseSpec, build_elementwise

open(sys.argv[1], "w").write(
    ir_export.export_kernel_ir_json(build_elementwise(ElementwiseSpec(op="add")))
)
"""
    r = _run([sys.executable, "-c", author, str(tmp_path / "k.ir.json")])
    assert r.returncode == 0, r.stderr[-4000:]

    import json

    doc = json.loads((tmp_path / "k.ir.json").read_text())
    doc["kernel"]["attrs"][key] = attr
    bad = tmp_path / "bad.ir.json"
    bad.write_text(json.dumps(doc))

    got = subprocess.run(
        [cli, "--ir", str(bad), "--arch", "gfx1250", "--flavor", "llvm23"],
        capture_output=True,
        text=True,
    )
    assert got.returncode != 0, f"malformed {key} was accepted"
    assert message in got.stderr and key in got.stderr, got.stderr[-2000:]

    # The well-formed shape must not trip the same check.
    doc["kernel"]["attrs"][key] = _int_list(2, 1, 1)
    ok = tmp_path / "ok.ir.json"
    ok.write_text(json.dumps(doc))
    got = subprocess.run(
        [cli, "--ir", str(ok), "--arch", "gfx1250", "--flavor", "llvm23"],
        capture_output=True,
        text=True,
    )
    assert f"kernel attr '{key}'" not in got.stderr, got.stderr[-2000:]
