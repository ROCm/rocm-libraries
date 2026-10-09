"""Packing, desk-check and corpus helpers shared by the tests in `tests/` and
`tests/rocke/`.

A plain module, for the reason `synthesised_objects.py` gives, and so that no
test module imports from another: `tests/rocke/` is not collected in a build
without rocKE, so a helper it borrowed from a shared test module could be renamed
there with no local failure.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hkp_pack import provenance_sidecar
from hkp_pack.pipeline import run_pipeline

PACKAGING = Path(__file__).resolve().parent.parent

#: The arch the hip fixtures and the selection corpus are packed for.
ARCH = "gfx942"
#: The one arch the rocKE fixture's UKD is scoped to.
ROCKE_ARCH = "gfx950"

EXAMPLE_ROOT = PACKAGING / "examples" / "descriptors"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _silent(*_args, **_kwargs):
    pass


# --- Packing ------------------------------------------------------------------
def _load_kpack(rocm_kpack_dir):
    from hkp_pack.kpack_resolver import load_kpack

    kpack, _comp = load_kpack(rocm_kpack_dir)
    return kpack


def _run(source_root, tmp_path, hipcc, rocm_kpack_dir, arches, source_label=None):
    """Pack one root. A root holding an `embedded_source` descriptor needs a label."""
    return run_pipeline(
        source_root=source_root,
        arches=list(arches),
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
        source_label=source_label,
    )


def _nest(root, sub, fixture):
    """Copy a flat fixture into `root/sub`, returning the child folder."""
    dest = root / sub
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(fixture, dest)
    return dest


# --- Authoring descriptors ------------------------------------------------------
# Standard library only, like `_write_corpus` in test_hkp_pack_parallel.py, which
# calls them and is copied with them to run outside pytest.
def _ukd(uid, kernel_source, arch=None):
    doc = {
        "version": "0.1",
        "id": uid,
        "name": uid,
        "kernel_source": kernel_source,
        "metadata": {},
        "priority": 0,
    }
    if arch is not None:
        doc["arch"] = arch
    return doc


def _kdp(kid, arch, entries):
    # matchers/engine/dispatch are authored empty: the loader requires the keys
    # and resolves only non-null references, and these corpora are about variant
    # selection, so carrying generics would add files without adding a case.
    return {
        "version": "0.1",
        "id": kid,
        "name": kid,
        "arch": arch,
        "matchers": [],
        "engine": None,
        "dispatch": None,
        "kernelDescriptors": entries,
    }


def _write_json(dest, name, doc):
    (dest / name).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


# --- Packed descriptors, as the packer ships them -------------------------------
def write_shipped(path, doc):
    """Write `doc` at `path` as the packer ships it: a compact copy with each
    UKD's provenance moved to the sidecar beside it, and the packed marker in its
    directory. Returns the copy as written."""
    path = Path(path)
    doc = json.loads(json.dumps(doc))
    name, data = provenance_sidecar.detach(path.name, doc)
    path.with_name(name).write_bytes(data)
    path.with_name(provenance_sidecar.PACKED_MARKER).write_bytes(b"")
    path.write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
    return doc


def is_compact(text):
    """Whether `text` is one compact JSON value plus a trailing newline. Either
    escaping of non-ASCII text counts; the loader reads both."""
    doc = json.loads(text)
    return any(
        text == json.dumps(doc, separators=(",", ":"), ensure_ascii=ascii_only) + "\n"
        for ascii_only in (True, False)
    )


def read_shipped(path):
    """A packed descriptor as its consumers read it: each UKD's provenance put
    back from the sidecar beside it, which must exist.

    First asserts the layout the runtime loader reads fastest: compact JSON with a
    KDP's `kernelDescriptors` last, which it reads in a single pass. Then the packed
    marker beside it, without which attach reads the descriptor as authored. With the
    sidecar present, attach refuses any UKD that still carries provenance inline.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    doc = json.loads(text)
    assert is_compact(text), f"{path}: not compact"
    if path.name.endswith(".kdp.json"):
        assert list(doc)[-1] == "kernelDescriptors", f"{path}: kernels are not last"
    assert provenance_sidecar.is_packed(path.parent), f"{path}: no packed marker"
    return provenance_sidecar.attach(path, doc)


# --- The desk-check CLI and the bundles it reads --------------------------------
# `tools/hkp_desk_check.py` is what an agent runs at RUNBOOK §4's host boundary.
_TOOL = PACKAGING / "tools" / "hkp_desk_check.py"

# Two roots are wired and both read: `examples/descriptors` is the documented
# sample tree, and the engine root is what a consumer loads.
_EXAMPLES = [
    EXAMPLE_ROOT,
    PACKAGING.parent / "src" / "engines" / "kernel_ingestor_engine" / "descriptors",
]
#: Case ids that name the tree, so a failure or a skip says WHICH root it was.
_ROOT_IDS = [root.parent.name for root in _EXAMPLES]


def _run_cli(*args, mode="structural"):
    """The CLI as an agent runs it. The mode is always explicit, because the tool
    requires it."""
    return subprocess.run(
        [sys.executable, str(_TOOL), "--mode", mode, *args],
        capture_output=True,
        text=True,
    )


def _require_bundles(producer_root):
    """Every `.kdp.json` under one producer subtree of one root, or a NAMED skip
    when it holds none: an emptied root and an absent producer are both legitimate,
    but the skip must name which, or a root that stopped being read looks like one
    that passed."""
    kdps = sorted(producer_root.glob("*/*.kdp.json"))
    if not kdps:
        pytest.skip(
            f"{producer_root.parent} carries no '{producer_root.name}' bundle "
            f"-- nothing to check for this producer in this root"
        )
    return kdps


# --- What a pool worker inherits -------------------------------------------------
def _child_sys_path(_ignored):
    """Run in a pool worker; returns the child's `sys.path`."""
    return list(sys.path)


def _conftest_only_paths(candidates):
    """The candidates on `sys.path` that `PYTHONPATH` does not carry.

    Only such a path tells a child that inherited `sys.path` apart from one that
    re-read `PYTHONPATH`: an entry `PYTHONPATH` carries reaches a child either way.
    """
    env_paths = {
        os.path.abspath(p)
        for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if p
    }
    return [
        str(p)
        for p in candidates
        if p and str(p) in sys.path and os.path.abspath(p) not in env_paths
    ]
