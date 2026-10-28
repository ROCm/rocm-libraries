# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the per-op scope of ``inputs_sha256`` (generators/_manifest.py).

An op's digest covers the generator modules its ``gen_<op>.py`` imports, its kernel
directories, the shared kernel helpers, and its own registry, instance and family rows.
Scoping it that way is what lets one op's change leave the others' objects current; the
risk is the reverse, an input the digest misses, so an object built from other sources
passes ``gen_descriptors.py --check``. These tests refuse the ways that could happen:
an import the closure walk cannot see, a data file a generator reads, a kernel import
outside the op's declared directories. The rest pin the scope itself, both ways.

Host-only: no FlyDSL install and no content tree is needed.
"""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
from generators import _manifest  # noqa: E402
from generators._ops import OP_KERNEL_DIRS  # noqa: E402

GENERATORS = FLYDSL / "generators"
KERNELS = FLYDSL / "kernels_src" / "kernels"
# Data the generators read. The family table is digested as the target's own row.
GENERATOR_DATA = {"arch_families.json"}
# Kernel packages every op may import; digested for every op.
SHARED_KERNEL_PACKAGES = {"common"}


def _arch_for(op: str) -> str:
    from generators._instances import instances_for  # noqa: PLC0415

    for arch in ("gfx11-generic", "gfx12-generic", "gfx950", "gfx942", "gfx90a"):
        try:
            if instances_for(op, arch):
                return arch
        except KeyError:  # an op family that has no kernel for this target
            continue
    pytest.fail(f"{op} has instance rows for no known target")


def _copy_tree(tmp_path: Path) -> Path:
    root = tmp_path / "flydsl"
    ignore = shutil.ignore_patterns("__pycache__")
    shutil.copytree(GENERATORS, root / "generators", ignore=ignore)
    shutil.copytree(FLYDSL / "kernels_src", root / "kernels_src", ignore=ignore)
    return root


def _touch(path: Path) -> None:
    path.write_bytes(path.read_bytes() + b"\n# edited\n")


def test_every_generator_has_a_registry_row_and_every_row_a_generator() -> None:
    generators = {
        path.stem.removeprefix("gen_") for path in GENERATORS.glob("gen_*.py")
    }
    assert generators == set(OP_KERNEL_DIRS)


def test_generators_import_only_what_the_closure_walk_sees() -> None:
    """No importlib, __import__, exec or eval: the walk follows import statements only."""
    hidden = {"importlib", "__import__", "exec", "eval", "runpy"}
    for path in sorted(GENERATORS.glob("*.py")):
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            name = (
                node.id
                if isinstance(node, ast.Name)
                else node.attr if isinstance(node, ast.Attribute) else None
            )
            assert name not in hidden, f"{path.name}:{node.lineno} uses {name}"
            if isinstance(node, ast.Import):
                assert all(a.name not in hidden for a in node.names), path.name
            if isinstance(node, ast.ImportFrom):
                assert node.module not in hidden, path.name


def test_generators_read_no_data_file_the_digest_misses() -> None:
    files = {
        path.name
        for path in GENERATORS.iterdir()
        if path.is_file() and path.suffix != ".py"
    }
    assert files <= GENERATOR_DATA, sorted(files - GENERATOR_DATA)


@pytest.mark.parametrize("op", sorted(OP_KERNEL_DIRS))
def test_an_op_imports_kernels_only_from_its_declared_directories(op: str) -> None:
    """Kernel packages are digested by directory, so a kernel import from a directory
    the op does not declare would be an input the digest misses."""
    allowed = set(OP_KERNEL_DIRS[op]) | SHARED_KERNEL_PACKAGES
    sources = list(_manifest.generator_modules(FLYDSL, op))
    for directory in OP_KERNEL_DIRS[op]:
        sources += sorted((KERNELS / directory).rglob("*.py"))
    for path in sources:
        for level, module, names in _manifest._imports(path):
            if level != 0 or not (module == "kernels" or module.startswith("kernels.")):
                continue
            parts = module.split(".")
            packages = {parts[1]} if len(parts) > 1 else set(names)
            assert packages <= allowed, (
                f"{path.relative_to(FLYDSL)} imports kernels.{sorted(packages - allowed)} "
                f"but {op} declares {sorted(allowed - SHARED_KERNEL_PACKAGES)}"
            )


@pytest.mark.parametrize("op", sorted(OP_KERNEL_DIRS))
def test_the_closure_holds_every_generator_module_the_op_imports(op: str) -> None:
    closure = set(_manifest.generator_modules(FLYDSL, op))
    assert GENERATORS / f"gen_{op}.py" in closure
    assert GENERATORS / "__init__.py" in closure
    for path in closure:
        for level, module, names in _manifest._imports(path):
            if level == 1:
                targets = [module.split(".")[0]] if module else names
                for target in targets:
                    assert GENERATORS / f"{target}.py" in closure, (path.name, target)


def test_registering_an_op_leaves_every_other_digest_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {op: _manifest.inputs_digest(op, _arch_for(op)) for op in OP_KERNEL_DIRS}
    monkeypatch.setitem(_manifest.OP_KERNEL_DIRS, "new_op", ("norm",))
    after = {op: _manifest.inputs_digest(op, _arch_for(op)) for op in before}
    assert after == before


def test_an_ops_own_registry_row_is_in_its_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arch = _arch_for("rmsnorm")
    before = _manifest.inputs_digest("rmsnorm", arch)
    monkeypatch.setitem(_manifest.OP_KERNEL_DIRS, "rmsnorm", ("norm", "common"))
    assert _manifest.inputs_digest("rmsnorm", arch) != before


def test_a_new_generator_module_changes_only_the_ops_that_import_it(
    tmp_path: Path,
) -> None:
    root = _copy_tree(tmp_path)
    before = {
        op: _manifest.inputs_digest(op, _arch_for(op), root) for op in OP_KERNEL_DIRS
    }
    (root / "generators" / "_sdpa_bwd_build.py").write_text("X = 1\n", encoding="utf-8")
    gen = root / "generators" / "gen_sdpa_cdna.py"
    gen.write_text(
        gen.read_text(encoding="utf-8") + "\nfrom ._sdpa_bwd_build import X  # noqa\n",
        encoding="utf-8",
    )
    after = {
        op: _manifest.inputs_digest(op, _arch_for(op), root) for op in OP_KERNEL_DIRS
    }
    changed = {op for op in OP_KERNEL_DIRS if after[op] != before[op]}
    assert changed == {"sdpa_cdna"}
    # The new module is now an input of sdpa_cdna: editing it alone restamps it.
    _touch(root / "generators" / "_sdpa_bwd_build.py")
    assert (
        _manifest.inputs_digest("sdpa_cdna", _arch_for("sdpa_cdna"), root)
        != after["sdpa_cdna"]
    )


@pytest.mark.parametrize(
    ("edited", "expected"),
    [
        (
            "generators/_sdpa_cdna_build.py",
            {"sdpa_cdna", "sdpa_cdna_dualwave", "sdpa_cdna_decode"},
        ),
        ("generators/_sdpa_rdna_build.py", {"sdpa_rdna", "sdpa_rdna_decode"}),
        ("generators/gen_rmsnorm.py", {"rmsnorm"}),
        ("generators/_codeobject.py", set(OP_KERNEL_DIRS)),
        ("generators/_ops.py", set()),
        ("kernels_src/kernels/norm/rmsnorm_kernel.py", {"rmsnorm", "rmsnorm_bwd"}),
        (
            "kernels_src/kernels/attention_rdna4/philox.py",
            {"sdpa_aiw", "sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq"},
        ),
        (
            "kernels_src/kernels/attention_bwd_rdna/fmha_bwd_dkdv_gfx11.py",
            {"sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq"},
        ),
        (
            "generators/_sdpa_bwd_rdna_build.py",
            {"sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq"},
        ),
        ("kernels_src/kernels/common/kernels_common.py", set(OP_KERNEL_DIRS)),
    ],
)
def test_an_edit_restamps_exactly_the_ops_that_read_the_file(
    tmp_path: Path, edited: str, expected: set[str]
) -> None:
    root = _copy_tree(tmp_path)
    before = {
        op: _manifest.inputs_digest(op, _arch_for(op), root) for op in OP_KERNEL_DIRS
    }
    _touch(root / edited)
    after = {
        op: _manifest.inputs_digest(op, _arch_for(op), root) for op in OP_KERNEL_DIRS
    }
    assert {op for op in OP_KERNEL_DIRS if after[op] != before[op]} == expected
