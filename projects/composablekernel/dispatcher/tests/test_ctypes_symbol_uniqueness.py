#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every exported ``dispatcher_*`` C symbol must mean exactly one thing.

The bridges are separate ``.so``s built from separate translation units, so two
of them exporting the same ``extern "C"`` name with *different* parameter
sequences is not a link error -- it is a runtime one, and a silent one.  The
Python side resolves the symbol by name out of whichever handle it opened; if a
caller opens the wrong ``.so``, or a process ends up with both loaded, the
arguments are marshalled against the wrong signature and the call reads garbage
off the stack.

That is exactly the shape the two ``dispatcher_run_bquant_gemm`` exports had --
one taking the grouped BQuant argument list, one the non-grouped -- and nothing
in the suite objected.

The rule enforced here:

  * a symbol exported from more than one ctypes lib must have an **identical**
    parameter type sequence in every file that exports it, and
  * every ``_RUN_SYMBOL`` a Python bridge resolves must name a symbol that
    exists and is unambiguous under that rule.

Names that are *deliberately* shared with one signature (``dispatcher_cleanup``,
``dispatcher_get_kernel_name``, the ``QUANT_BRIDGE_C_API`` block) pass without
an exemption, because they satisfy the rule rather than being excused from it.

Pure source analysis; no GPU, no hipcc, no build.
"""

import re
import sys
from pathlib import Path

import pytest

_DISP = Path(__file__).resolve().parent.parent
_CTYPES = _DISP / "bindings" / "ctypes"
_PYTHON = _DISP / "python"

for _p in (_PYTHON,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# A definition of a free function whose name starts with dispatcher_, at file
# scope (column 0), with its parameter list up to the closing paren.
_DEF = re.compile(
    r"^(?P<ret>[A-Za-z_][\w:*<>\s]*?[\s*])(?P<name>dispatcher_\w+)\s*\((?P<params>[^)]*)\)",
    re.MULTILINE,
)


def _normalize_params(params: str):
    """Parameter *types*, in order, with names and whitespace removed.

    Two declarations of the same symbol are compatible iff this sequence
    matches.  Parameter names are irrelevant to the ABI, so they are stripped;
    types are not.
    """
    params = re.sub(r"//[^\n]*", " ", params)
    params = re.sub(r"/\*.*?\*/", " ", params, flags=re.DOTALL)
    out = []
    for raw in params.split(","):
        item = " ".join(raw.split())
        if not item or item == "void":
            continue
        # Drop the trailing identifier, keeping pointer/reference decoration
        # with the type: "const void* A" -> "const void*",  "int64_t M" -> "int64_t".
        item = re.sub(r"\s*\b[A-Za-z_]\w*\s*$", "", item)
        out.append(" ".join(item.split()).replace(" *", "*"))
    return tuple(out)


def _exports():
    """{symbol: {file_name: param_type_tuple}} across every ctypes lib."""
    table = {}
    for path in sorted(_CTYPES.glob("*.cpp")):
        text = path.read_text()
        for m in _DEF.finditer(text):
            ret = m.group("ret").strip()
            # Skip declarations that are obviously not definitions of an export
            # (e.g. a call inside a function is never at column 0 with a return
            # type in front of it, but be defensive about macros).
            if ret.startswith(("#", "//")):
                continue
            table.setdefault(m.group("name"), {})[path.name] = _normalize_params(
                m.group("params")
            )
    return table


_EXPORTS = _exports()


def test_symbols_were_found_at_all():
    """A detector that finds nothing would pass every other test in this file."""
    assert len(_EXPORTS) >= 5, f"only found {sorted(_EXPORTS)}"
    assert "dispatcher_run_gemm" in _EXPORTS


def test_no_symbol_has_two_different_signatures():
    """One exported name, one parameter sequence -- across every ctypes lib."""
    collisions = []
    for symbol, per_file in sorted(_EXPORTS.items()):
        signatures = set(per_file.values())
        if len(signatures) > 1:
            detail = "\n      ".join(
                f"{fname}: ({', '.join(sig)})" for fname, sig in sorted(per_file.items())
            )
            collisions.append(f"  {symbol} has {len(signatures)} signatures:\n      {detail}")
    assert not collisions, (
        "exported C symbols with conflicting signatures -- two .so files export "
        "the same name meaning different things, which ctypes resolves by name "
        "and marshals against the wrong argument list:\n" + "\n".join(collisions)
    )


# Two spellings are in use: the shared layer declares ``_RUN_SYMBOL`` on the
# lib class, while the grouped bridges bind ``lib.<symbol>.restype`` directly.
_RUN_SYMBOL_RE = re.compile(r'_RUN_SYMBOL\s*=\s*["\'](dispatcher_\w+)["\']')
_RESTYPE_RE = re.compile(r'\blib\.(dispatcher_run_\w+)\.restype\b')


def _python_run_symbols():
    """{module_file: [entry-point symbols]} for every bridge utils module.

    A list, not a scalar: ``gemm_utils.py`` legitimately drives four operators
    (plain, grouped, multi-ABD, multi-D) and binds one entry point for each.
    The one-symbol-per-bridge rule is asserted separately, for the quant
    bridges, where it is the actual invariant.
    """
    out = {}
    for path in sorted(_PYTHON.glob("*_utils.py")):
        text = path.read_text()
        found = sorted(set(_RUN_SYMBOL_RE.findall(text)) | set(_RESTYPE_RE.findall(text)))
        if found:
            out[path.name] = found
    return out


def test_every_python_run_symbol_resolves_to_one_definition():
    """A bridge may only bind a symbol that exists and means one thing."""
    symbols = _python_run_symbols()
    assert symbols, "no _RUN_SYMBOL found in any python/*_utils.py"
    problems = []
    for module, found in sorted(symbols.items()):
        for symbol in found:
            per_file = _EXPORTS.get(symbol)
            if not per_file:
                problems.append(
                    f"{module}: binds {symbol!r}, which no ctypes lib exports")
            elif len(set(per_file.values())) > 1:
                problems.append(
                    f"{module}: binds {symbol!r}, exported with "
                    f"{len(set(per_file.values()))} different signatures by "
                    f"{sorted(per_file)}"
                )
    assert not problems, "\n".join(problems)


def test_quant_bridges_do_not_share_a_run_symbol():
    """The ten quant bridges must each bind their own entry point.

    Stronger than the signature rule and the reason it exists: even if two quant
    bridges happened to agree on the argument list today, sharing one name means
    the wrong kernel can be reached through the right handle.
    """
    symbols = {
        module: found
        for module, found in _python_run_symbols().items()
        if "quant" in module
    }
    assert len(symbols) == 10, f"expected 10 quant bridges, found {sorted(symbols)}"
    seen = {}
    duplicates = []
    for module, found in sorted(symbols.items()):
        assert len(found) == 1, (
            f"{module} binds {len(found)} run entry points {found}; a quant "
            "bridge must have exactly one"
        )
        sym = found[0]
        if sym in seen:
            duplicates.append(f"{sym!r}: {seen[sym]} and {module}")
        seen[sym] = module
    assert not duplicates, (
        "quant bridges sharing one exported symbol:\n  " + "\n  ".join(duplicates)
    )


# ---------------------------------------------------------------------------
# Self-test: the detector must go RED on a real collision
# ---------------------------------------------------------------------------
#
# A source-scanning gate is worthless if the scanner silently matches nothing.
# These reproduce the exact defect the rule exists for -- the two
# dispatcher_run_bquant_gemm exports -- and assert the detector reports it.

_COLLIDING_A = """
extern "C" {
int dispatcher_run_bquant_gemm(const void* A,
                               const void* B,
                               const void* BQ,
                               void* C,
                               int64_t M,
                               float* time_ms)
{
    return 0;
}
}
"""

_COLLIDING_B = """
extern "C" {
int dispatcher_run_bquant_gemm(const void* A,
                               const void* B,
                               const void* BQ,
                               void* C,
                               int64_t M,
                               int64_t N,
                               int64_t QM_B,
                               float* time_ms)
{
    return 0;
}
}
"""


@pytest.fixture
def colliding_ctypes_dir(tmp_path, monkeypatch):
    (tmp_path / "a_ctypes_lib.cpp").write_text(_COLLIDING_A)
    (tmp_path / "b_ctypes_lib.cpp").write_text(_COLLIDING_B)
    monkeypatch.setattr(sys.modules[__name__], "_CTYPES", tmp_path)
    return tmp_path


def test_detector_reports_a_reintroduced_collision(colliding_ctypes_dir):
    """RED-on-demand: two libs, one name, two argument lists -> reported."""
    table = _exports()
    assert "dispatcher_run_bquant_gemm" in table, (
        "the scanner did not even find the colliding symbol; the regex is broken "
        "and every other assertion in this file is vacuous"
    )
    signatures = set(table["dispatcher_run_bquant_gemm"].values())
    assert len(signatures) == 2, (
        f"expected the detector to see two distinct signatures, saw {signatures}"
    )


def test_detector_accepts_identical_signatures(tmp_path, monkeypatch):
    """GREEN control: the same name with the same argument list is not a collision."""
    same = _COLLIDING_A
    (tmp_path / "a_ctypes_lib.cpp").write_text(same)
    (tmp_path / "b_ctypes_lib.cpp").write_text(same.replace("return 0;", "return 1;"))
    monkeypatch.setattr(sys.modules[__name__], "_CTYPES", tmp_path)
    table = _exports()
    assert len(set(table["dispatcher_run_bquant_gemm"].values())) == 1


# ---------------------------------------------------------------------------
# RED on real history, not just on a synthetic fixture
# ---------------------------------------------------------------------------

_PRE_RENAME_REV = "fec98c1c0a"
_CTYPES_REL = "projects/composablekernel/dispatcher/bindings/ctypes"


def _repo_root():
    path = _DISP
    while path != path.parent:
        if (path / ".git").exists():
            return path
        path = path.parent
    return None


def test_detector_is_red_on_the_pre_rename_tree(tmp_path, monkeypatch):
    """The collisions were real and this detector finds them.

    Materializes the ctypes libs as they stood before the grouped exports were
    renamed and asserts the scanner reports the collisions.  A synthetic fixture
    proves the regex works; this proves the rule was violated by shipped code
    and that this file would have said so.
    """
    import subprocess
    root = _repo_root()
    if root is None:
        pytest.skip("not a git checkout")
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "--name-only",
         f"{_PRE_RENAME_REV}:{_CTYPES_REL}"],
        capture_output=True, text=True, timeout=60,
    )
    if listing.returncode != 0:
        pytest.skip(f"{_PRE_RENAME_REV} unavailable in this clone")
    for name in listing.stdout.split():
        if not name.endswith(".cpp"):
            continue
        blob = subprocess.run(
            ["git", "-C", str(root), "show", f"{_PRE_RENAME_REV}:{_CTYPES_REL}/{name}"],
            capture_output=True, timeout=60,
        )
        if blob.returncode == 0:
            (tmp_path / name).write_bytes(blob.stdout)

    monkeypatch.setattr(sys.modules[__name__], "_CTYPES", tmp_path)
    table = _exports()
    colliding = {
        sym: per_file for sym, per_file in table.items()
        if len(set(per_file.values())) > 1
    }
    assert colliding, (
        "the detector found no collision in the pre-rename tree, where at least "
        "dispatcher_run_gemm was exported by four libs with two different "
        "argument lists -- the detector is not detecting"
    )
    assert "dispatcher_run_gemm" in colliding, sorted(colliding)
