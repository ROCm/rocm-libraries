# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The backend coverage record (``traits/data/backend_coverage.json``) equals what rocke's backend does.

For every (catalog row, target) pair this module measures whether rocke's backend issues the row's
instruction there, and requires the measurement to equal the committed record exactly, so a backend
op that is added, removed or rewired cannot change what the tiling layer issues without the record
changing with it. ``traits/README.md`` ("Backend coverage") explains the classes.

How a pair is measured:

1. Its *candidates* are the backend ops on the target with the row's operand key: A, B, C and
   result dtypes, M, N, K, and the scale contract (a scaled row's block size and scale formats; no
   scale fields at all for a dense or sparse row). A selectable integer row also takes the unsigned
   twins of its operands. No candidate: ``no_backend_op``.
2. Each candidate is lowered alone, fed the operand form the tiling layer feeds: the row's logical
   elements per lane (``fragment_length`` of its encodings), as
   ``helpers/mma_io.mma_operand_ir_type`` spells them. Two exceptions take their counts or form from
   the op: a multi-block row, which the encoder does not lay out, and gfx950's scaled MFMA, which
   takes A and B only as register words. The lowered module's matrix intrinsic is read with
   ``tools/check_mma_catalog.intrinsic_base_name``.
3. The candidate that lowers to the row's ``llvm_intrinsic`` is the pair's backend op; two such
   candidates are an error (the record holds one op per pair, or per variant of it).
4. The pair is classified, first match wins: ``other_instruction``, ``multi_block``,
   ``lowering_not_wired``, ``operand_form_words_only``, ``lane_map_differs``, then ``invalid_ir``
   or ``issuable`` by compiling the module for the target (the compile tests).

The classes are measured at the record's LLVM flavor, passed explicitly, so the host's flavor
cannot change them. The compile tests lower at the flavor of the clang that judges the module.

Regenerate the record with a clang that targets every catalog target:
``python -m rocke.tests.helpers.tiling.traits.test_mma_backend_coverage --write`` (from
``platform/``).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import functools
import itertools
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from rocke.assets import platform_root
from rocke.core.arch import ArchTarget, MmaOp
from rocke.core.arch.mfma_scale import gfx950_scaled_mfma, matrix_layout
from rocke.core.dtypes import dtype_info
from rocke.core.ir import I32, IRBuilder, PtrType, VectorType, dtype_to_ir_type
from rocke.core.lower_llvm import LLVM_FLAVORS, _lower_kernel_to_llvm_python
from rocke.helpers.mma_io import mma_operand_ir_type
from rocke.helpers.tiling.fragments import fragment_length
from rocke.helpers.tiling.mma.warp_encoding import (
    a_warp_encoding,
    b_warp_encoding,
    c_warp_encoding,
)
from rocke.helpers.tiling.register_mapper import RegisterMapper
from rocke.helpers.tiling.traits import (
    DEFAULT_COVERAGE_PATH,
    BackendCoverage,
    CoverageEntry,
    MmaTraits,
    load_backend_coverage,
    load_mma_traits,
)

_CATALOG = load_mma_traits()
_ROWS = sorted(_CATALOG.by_op_id.values(), key=lambda row: row.op_id)
_TARGETS = sorted({target for row in _ROWS for target in row.supported_targets})
_UNSIGNED_TWIN = {"i8": "u8", "i4": "u4"}
_COMPILED_CLASSES = ("issuable", "invalid_ir")
_THIS_MODULE = "rocke.tests.helpers.tiling.traits.test_mma_backend_coverage"


def _tools() -> tuple[ModuleType, ModuleType]:
    """``tools/check_mma_catalog.py`` and ``tools/check_ir_validity.py``, imported by path.

    Source tree only: an installed rocke has no ``tools/``, and the measurement is skipped there.
    """
    tools = platform_root() / "tools"
    if not (tools / "check_mma_catalog.py").is_file():
        pytest.skip("platform/tools is not installed beside rocke (source tree only)")
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import check_ir_validity
    import check_mma_catalog

    return check_mma_catalog, check_ir_validity


# ---------------------------------------------------------------------------
# Measuring one pair
# ---------------------------------------------------------------------------


def operand_counts(row: MmaTraits) -> tuple[int, int, int]:
    """A, B and C logical elements per lane for one instruction of ``row``: the counts the TileMma
    driver slices (never core's ``*_frag_len``, which counts register words for some ops).
    """
    return tuple(
        fragment_length(encoding(row))
        for encoding in (a_warp_encoding, b_warp_encoding, c_warp_encoding)
    )


def _variant_of(row: MmaTraits, op: MmaOp) -> tuple[str, str] | None:
    """The variant ``op`` answers for ``row``: its scale formats, or its operand dtypes."""
    if row.family == "scaled":
        return (str(op.a_scale_dtype), str(op.b_scale_dtype))
    if row.operand_signedness == "selectable":
        return (op.a_dtype, op.b_dtype)
    return None


def _variants(row: MmaTraits) -> list[tuple[str, str] | None]:
    """Every variant ``row`` accepts, or ``[None]`` for a row recorded without variants."""
    if row.family == "scaled":
        return list(itertools.product(row.scale.a_dtypes, row.scale.b_dtypes))
    if row.operand_signedness == "selectable":
        return list(
            itertools.product(
                (row.a_dtype, _UNSIGNED_TWIN[row.a_dtype]),
                (row.b_dtype, _UNSIGNED_TWIN[row.b_dtype]),
            )
        )
    return [None]


def _scale_contract_matches(row: MmaTraits, op: MmaOp) -> bool:
    if row.scale is None:
        return op.scale_block_k is None
    return (
        op.scale_block_k is not None
        and int(op.scale_block_k) == row.scale.block_k
        and str(op.a_scale_dtype) in row.scale.a_dtypes
        and str(op.b_scale_dtype) in row.scale.b_dtypes
    )


def candidates(row: MmaTraits, target: str) -> list[MmaOp]:
    """The backend ops on ``target`` with ``row``'s operand key (A, B, C, D, M, N, K, scales)."""
    return [
        op
        for op in ArchTarget.from_gfx(target).mma.ops
        if row.accepts(a=op.a_dtype, b=op.b_dtype, c=op.c_dtype, d=op.dst_dtype)
        and (op.m, op.n, op.k) == (row.m, row.n, row.k)
        and _scale_contract_matches(row, op)
    ]


def _takes_register_words(op: MmaOp) -> bool:
    """gfx950's scaled MFMA takes A and B only as caller-packed register words (core's
    ``scaled_mfma_error``), never in the tiling layer's operand form."""
    return gfx950_scaled_mfma(op.op_id) is not None


def _zero_operand(b: IRBuilder, operand: VectorType):
    """A zero value of ``operand``: built as 32-bit words, since ``zero_vec`` takes neither TF32
    nor a packed sub-byte type, then reinterpreted."""
    words = operand.count * dtype_info(operand.elem.name).encoded_bits // 32
    zero = b.zero_vec(I32, words)
    return zero if operand == zero.type else b.vec_bitcast(zero, operand)


def _scale_operands(b: IRBuilder, op: MmaOp) -> list:
    """Zero scales for a scaled op: one 32-bit word per operand, or 64-bit where one lane's scales
    need more than 32 bits."""
    if op.scale_block_k is None:
        return []
    scales = []
    for dtype, count in (
        (op.a_scale_dtype, op.a_scale_frag_len),
        (op.b_scale_dtype, op.b_scale_frag_len),
    ):
        bits = count * dtype_info(str(dtype)).encoded_bits
        scales.append(b.const_i64(0) if bits > 32 else b.const_i32(0))
    return scales


def _call_kernel(row: MmaTraits, op: MmaOp):
    """A kernel that issues ``op`` once on zero operands and stores one result element.

    A and B take the tiling layer's operand form at the row's counts; a multi-block row takes the
    op's counts, and an op that takes register words gets its declared words.
    """
    if row.b > 1:
        counts = (op.a_frag_len, op.b_frag_len, op.c_frag_len)
    else:
        counts = operand_counts(row)
    b = IRBuilder(f"coverage_{op.op_id}")
    out = b.param("out", PtrType(dtype_to_ir_type(op.dst_dtype), "global"))
    if _takes_register_words(op):
        a_type = VectorType(I32, matrix_layout(op.a_dtype).fragment.carrier_count)
        b_type = VectorType(I32, matrix_layout(op.b_dtype).fragment.carrier_count)
    else:
        a_type = mma_operand_ir_type(op.a_dtype, counts[0])
        b_type = mma_operand_ir_type(op.b_dtype, counts[1])
    c = b.zero_vec(dtype_to_ir_type(row.c_dtype), counts[2])
    d = b.mma(
        op,
        _zero_operand(b, a_type),
        _zero_operand(b, b_type),
        c,
        *_scale_operands(b, op),
    )
    b.global_store(out, b.const_i32(0), b.vec_extract(d, 0))
    return b.kernel


def lower_pair(row: MmaTraits, op: MmaOp, target: str, llvm_flavor: str) -> str:
    """The LLVM IR module of ``op`` issued for ``row`` on ``target``."""
    # Python engine pinned: the record is a fact about rocke's lowering, measured one way.
    return _lower_kernel_to_llvm_python(
        _call_kernel(row, op), arch=target, llvm_flavor=llvm_flavor
    )


class _IntBuilder:
    """Evaluates core's lane-map arithmetic on Python integers."""

    const_i32 = staticmethod(int)
    add = staticmethod(lambda x, y: x + y)
    mul = staticmethod(lambda x, y: x * y)
    div = staticmethod(lambda x, y: x // y)
    mod = staticmethod(lambda x, y: x % y)


def _core_elements(layout, count: int, dtype: str, k_axis: int | None) -> dict:
    """``(lane, logical slot) -> coordinates`` of a core lane map, in logical elements.

    Core counts some operands in 32-bit register words (``frag_len * 32 == count * bits``): slot
    ``s`` then names the first of the word's elements, which hold consecutive K from the low bits.
    """
    builder = _IntBuilder()
    per_slot = count // layout.frag_len
    if count % layout.frag_len or (
        per_slot > 1
        and (k_axis is None or per_slot * dtype_info(dtype).encoded_bits != 32)
    ):
        raise AssertionError(
            f"core lane map of {layout.frag_len} slots cannot be read as {count} {dtype} "
            f"elements (role {layout.role!r})"
        )
    elements = {}
    for lane in range(layout.wave_size):
        for slot in range(layout.frag_len):
            first = list(layout.fn(builder, lane, slot))
            for j in range(per_slot):
                coords = list(first)
                if per_slot > 1:
                    coords[k_axis] += j
                elements[(lane, slot * per_slot + j)] = tuple(coords)
    return elements


def _lane_maps_differ(row: MmaTraits, op: MmaOp, counts: tuple[int, int, int]) -> bool:
    """Whether core's lane maps and the catalog's disagree beyond one K order shared by A and B.

    An op without a core lane map is not compared (README: Backend coverage).
    """
    try:
        layouts = (op.a_layout(), op.b_layout(), op.c_layout())
    except NotImplementedError:
        return False
    # Catalog coordinates: A (m, k), B (n, k), C (m, n). Core: A (m, k), B (k, n), C (m, n).
    roles = (
        (a_warp_encoding, layouts[0], counts[0], row.a_dtype, 1, lambda c: c),
        (
            b_warp_encoding,
            layouts[1],
            counts[1],
            row.b_dtype,
            0,
            lambda c: (c[1], c[0]),
        ),
        (c_warp_encoding, layouts[2], counts[2], row.d_dtype, None, lambda c: c),
    )
    k_relabelling: dict[int, int] = {}
    for encoding, layout, count, dtype, k_axis, to_catalog in roles:
        mapper = RegisterMapper(encoding(row))
        core = _core_elements(layout, count, dtype, k_axis)
        for lane in range(mapper.num_lanes):
            for slot in range(mapper.num_vector_items):
                outer, inner = mapper.matrix_coordinates(lane=lane, register=slot)
                core_outer, core_inner = to_catalog(core[(lane, slot)])
                if core_outer != outer:
                    return True
                if k_axis is None:
                    if core_inner != inner:
                        return True
                elif k_relabelling.setdefault(inner, core_inner) != core_inner:
                    return True
    return len(set(k_relabelling.values())) != len(k_relabelling)


def _measure_variant(
    row: MmaTraits,
    target: str,
    ops: list[MmaOp],
    llvm_flavor: str,
    intrinsics_of,
) -> dict:
    """One record entry (without the variant field) for ``ops``, the candidates of one variant."""
    if not ops:
        return {"class": "no_backend_op"}
    matched, unwired = [], []
    for op in ops:
        try:
            module = lower_pair(row, op, target, llvm_flavor)
        except NotImplementedError as error:
            unwired.append((op, str(error)))
            continue
        if row.llvm_intrinsic in intrinsics_of(module):
            matched.append(op)
    names = [op.op_id for op in matched]
    assert len(matched) <= 1, (
        f"row {row.op_id!r} on {target!r} matches {len(matched)} backend ops by intrinsic and "
        f"key: {names} -- the record holds one op per pair; distinguish them in the row (scale "
        f"block size) or the backend"
    )
    if not matched:
        # Without a lowered match the record names the one op with the row's key, so that op
        # must be unique.
        unmatched = [op for op, _ in unwired] if unwired else ops
        assert len(unmatched) == 1, (
            f"row {row.op_id!r} on {target!r} has {len(unmatched)} backend ops with its key "
            f"and none lowers to {row.llvm_intrinsic}: {[op.op_id for op in unmatched]}"
        )
        if unwired:
            return {
                "class": "lowering_not_wired",
                "backend_op": unwired[0][0].op_id,
                "reason": unwired[0][1],
            }
        return {"class": "other_instruction", "backend_op": ops[0].op_id}
    (op,) = matched
    if row.b > 1:
        status = "multi_block"
    elif _takes_register_words(op):
        status = "operand_form_words_only"
    elif _lane_maps_differ(row, op, operand_counts(row)):
        status = "lane_map_differs"
    else:
        status = "issuable"
    return {"class": status, "backend_op": op.op_id}


def measure_backend_coverage(llvm_flavor: str) -> dict:
    """The record, measured without a compiler: every pair that passes the clang-free checks is
    ``issuable`` here (the compile tests split it into ``issuable`` and ``invalid_ir``).
    """
    catalog_tool, _ = _tools()
    pairs: dict[str, dict[str, list[dict]]] = {}
    matched_ops: dict[str, set[str]] = {target: set() for target in _TARGETS}
    for row in _ROWS:
        for target in row.supported_targets:
            found = candidates(row, target)
            entries = []
            for variant in _variants(row):
                ops = [op for op in found if _variant_of(row, op) == variant]
                entry = _measure_variant(
                    row, target, ops, llvm_flavor, catalog_tool.amdgcn_intrinsics
                )
                if variant is not None:
                    key = "scale_format" if row.family == "scaled" else "operand_dtypes"
                    entry = {key: list(variant), **entry}
                if entry["class"] not in ("no_backend_op", "other_instruction"):
                    matched_ops[target].add(entry["backend_op"])
                entries.append(entry)
            pairs.setdefault(row.op_id, {})[target] = entries
    without_row = {}
    for target in _TARGETS:
        unmatched = sorted(
            op.op_id
            for op in ArchTarget.from_gfx(target).mma.ops
            if op.op_id not in matched_ops[target]
        )
        if unmatched:
            without_row[target] = unmatched
    return {
        "_meta": {"llvm_flavor": llvm_flavor},
        "pairs": pairs,
        "backend_ops_without_row": without_row,
    }


def _without_compile_verdicts(document: dict) -> dict:
    """``document`` as the clang-free measurement sees it: ``invalid_ir`` read as ``issuable``."""
    clang_free = copy.deepcopy(document)
    for by_target in clang_free["pairs"].values():
        for entries in by_target.values():
            for entry in entries:
                if entry["class"] in _COMPILED_CLASSES:
                    entry["class"] = "issuable"
    return clang_free


def _issuable(document: dict) -> set[tuple]:
    found = set()
    for op_id, by_target in document["pairs"].items():
        for target, entries in by_target.items():
            for entry in entries:
                if entry["class"] == "issuable":
                    variant = entry.get("scale_format") or entry.get("operand_dtypes")
                    found.add((op_id, target, *([tuple(variant)] if variant else [])))
    return found


def _assert_same_record(measured: dict, recorded: dict) -> None:
    now, before = _issuable(measured), _issuable(recorded)
    assert now == before, (
        f"backend coverage changed -- now issuable: {sorted(now - before)}; no longer "
        f"issuable: {sorted(before - now)} -- if a backend op was added or removed, update "
        f"helpers/tiling/traits/data/backend_coverage.json"
    )
    changed = sorted(
        (op_id, target)
        for op_id in set(measured["pairs"]) | set(recorded["pairs"])
        for target in set(measured["pairs"].get(op_id, {}))
        | set(recorded["pairs"].get(op_id, {}))
        if measured["pairs"].get(op_id, {}).get(target)
        != recorded["pairs"].get(op_id, {}).get(target)
    )
    assert not changed, (
        f"backend coverage changed -- class or backend op differs for: {changed} -- if a "
        f"backend op was added or removed, update "
        f"helpers/tiling/traits/data/backend_coverage.json"
    )
    assert measured == recorded, (
        "backend coverage changed -- backend_ops_without_row or _meta differs -- update "
        "helpers/tiling/traits/data/backend_coverage.json"
    )


# ---------------------------------------------------------------------------
# The record equals the measurement
# ---------------------------------------------------------------------------


@functools.cache
def _record() -> BackendCoverage:
    return load_backend_coverage()


def _record_document() -> dict:
    return json.loads(DEFAULT_COVERAGE_PATH.read_text())


def test_backend_coverage_classes() -> None:
    # The record's flavor, passed explicitly: the newest flavor expresses every catalog target's
    # ops, and the host's flavor must not change a committed record.
    measured = measure_backend_coverage(_record().llvm_flavor)
    _assert_same_record(measured, _without_compile_verdicts(_record_document()))


@pytest.mark.parametrize("host_flavor", ["llvm20", "llvm23"])
def test_backend_coverage_does_not_depend_on_the_host_flavor(host_flavor: str) -> None:
    # A fresh process with ROCKE_LLVM_FLAVOR set measures the same classes: nothing the host's
    # flavor decides reaches the record.
    _tools()
    env = {**os.environ, "ROCKE_LLVM_FLAVOR": host_flavor}
    proc = subprocess.run(
        [sys.executable, "-m", _THIS_MODULE, "--print-classes"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    _assert_same_record(
        json.loads(proc.stdout), _without_compile_verdicts(_record_document())
    )


def _judging_clang() -> tuple[str, str]:
    """The arbiter's clang and its LLVM flavor; skip when there is none."""
    _, validity_tool = _tools()
    from _llvm_identity import flavor_of_clang, tool_identity

    clang = validity_tool._llvm_tool("clang")
    if clang is None:
        pytest.skip("no clang to compile lowered IR")
    flavor = flavor_of_clang(tool_identity(clang))
    if flavor not in LLVM_FLAVORS:
        pytest.skip(f"clang {clang} reports no LLVM flavor rocke lowers for")
    return clang, flavor


@dataclass(frozen=True)
class _CompileJob:
    """One module the compile decides: ``row``'s backend ``op``, known to the caller as ``key``."""

    key: tuple
    row: MmaTraits
    op: MmaOp


def _compile_target(
    clang: str, flavor: str, target: str, jobs: list[_CompileJob]
) -> dict[tuple, tuple[bool, str]] | None:
    """Compile and link each job's module for ``target`` (``key -> (accepted, diagnostic)``);
    ``None`` when this clang cannot target it."""
    catalog_tool, validity_tool = _tools()
    name = catalog_tool.compiler_names([target], flavor)[target]
    if name is None:
        return None

    def judge(job: _CompileJob) -> tuple[bool, str]:
        try:
            module = lower_pair(job.row, job.op, target, flavor)
        except NotImplementedError as error:
            raise AssertionError(
                f"recorded pair {job.key} does not lower at {flavor}: {error}"
            ) from None
        with tempfile.TemporaryDirectory() as work:
            source = Path(work) / "module.ll"
            source.write_text(module)
            return validity_tool._compile(
                clang, source, name, Path(work) / "module.hsaco"
            )

    with cf.ThreadPoolExecutor(os.cpu_count() or 1) as pool:
        verdicts = dict(zip([job.key for job in jobs], pool.map(judge, jobs)))
    if any(validity_tool._target_unsupported(diag) for _, diag in verdicts.values()):
        return None
    return verdicts


def _entry_key(op_id: str, target: str, entry: CoverageEntry) -> tuple:
    """``(op_id, target)``, plus the entry's variant when it has one."""
    variant = entry.scale_format or entry.operand_dtypes
    return (op_id, target, *([variant] if variant else []))


@pytest.mark.parametrize("target", _TARGETS)
def test_backend_coverage_compiles(target: str) -> None:
    clang, flavor = _judging_clang()
    ops = {op.op_id: op for op in ArchTarget.from_gfx(target).mma.ops}
    recorded = {
        _entry_key(op_id, target, entry): entry
        for (op_id, pair_target), entries in _record().entries.items()
        if pair_target == target
        for entry in entries
        if entry.status in _COMPILED_CLASSES
    }
    jobs = [
        _CompileJob(key, _CATALOG.get(key[0]), ops[entry.backend_op])
        for key, entry in sorted(recorded.items())
    ]
    verdicts = _compile_target(clang, flavor, target, jobs)
    if verdicts is None:
        pytest.skip(f"host clang cannot target {target}")
    _, validity_tool = _tools()
    now_issuable = sorted(
        key
        for key, (accepted, _) in verdicts.items()
        if accepted and recorded[key].status != "issuable"
    )
    assert not now_issuable, (
        f"backend coverage changed -- now issuable: {now_issuable}; no longer issuable: [] "
        f"-- if a backend op was added or removed, update "
        f"helpers/tiling/traits/data/backend_coverage.json"
    )
    for key, (accepted, diagnostic) in verdicts.items():
        assert accepted or recorded[key].status != "issuable", (
            f"backend op {recorded[key].backend_op!r} on {target!r} lowers to IR clang "
            f"rejects for the target: {validity_tool._first_diagnostic(diagnostic)}"
        )


# ---------------------------------------------------------------------------
# Facts the record pins
# ---------------------------------------------------------------------------


def test_record_covers_every_row_and_target() -> None:
    expected = {
        (row.op_id, target) for row in _ROWS for target in row.supported_targets
    }
    assert set(_record().entries) == expected


def test_record_names_real_backend_ops() -> None:
    for (op_id, target), entries in _record().entries.items():
        ops = {op.op_id for op in ArchTarget.from_gfx(target).mma.ops}
        for entry in entries:
            assert entry.backend_op is None or entry.backend_op in ops, (op_id, target)


@pytest.mark.parametrize(
    "op_id, counts",
    [
        ("wmma_i32_16x16x16_iu8_w32", (16, 16)),
        ("wmma_i32_16x16x16_iu4_w32", (16, 16)),
        ("wmma_f32_16x16x64_fp8_fp8", (32, 32)),
    ],
)
def test_operand_counts_are_logical_elements(
    op_id: str, counts: tuple[int, int]
) -> None:
    # Logical elements per lane, as the driver slices them; core's frag lengths for these ops
    # count 32-bit words (4, 2 and 8).
    assert operand_counts(_CATALOG.get(op_id))[:2] == counts


def test_gfx950_scaled_mfma_rows() -> None:
    # gfx950's scaled MFMA ops exist at 32x32x64 only, and take A and B only as register words.
    for row in _ROWS:
        if not row.op_id.startswith("mfma_scale_f32_"):
            continue
        entry = _record().entry(row.op_id, "gfx950")
        if row.k == 128:
            assert entry.status == "no_backend_op", row.op_id
            continue
        assert entry.status == "operand_form_words_only", row.op_id
        op = ArchTarget.from_gfx("gfx950").mma.by_op_id(entry.backend_op)
        assert (op.family, op.a_dtype, op.b_dtype) == (
            "mfma_scaled",
            row.a_dtype,
            row.b_dtype,
        )


def test_gfx1250_scaled_pairs_with_a_backend_op_are_issuable() -> None:
    # Each recorded scale format names the op with the row's block size and that scale format.
    ops = {op.op_id: op for op in ArchTarget.from_gfx("gfx1250").mma.ops}
    for row in _ROWS:
        if row.family != "scaled" or not row.supports("gfx1250"):
            continue
        for entry in _record().entries[(row.op_id, "gfx1250")]:
            if entry.backend_op is None:
                continue
            op = ops[entry.backend_op]
            assert entry.status == "issuable", (row.op_id, entry)
            assert int(op.scale_block_k) == row.scale.block_k
            assert (str(op.a_scale_dtype), str(op.b_scale_dtype)) == entry.scale_format


@pytest.mark.parametrize(
    "op_id, target, status, backend_op",
    [
        (
            "wmma_bf16f32_16x16x32_bf16",
            "gfx1250",
            "issuable",
            "wmma_gfx1250_bf16f32_16x16x32_bf16",
        ),
        (
            "wmma_f32_16x16x32_bf16",
            "gfx1250",
            "issuable",
            "wmma_gfx1250_f32_16x16x32_bf16",
        ),
        ("mfma_f64_4x4x4f64", "gfx942", "multi_block", "mfma_f64_4x4x4_f64"),
        (
            "mfma_f32_32x32x4bf16_1k",
            "gfx90a",
            "other_instruction",
            "mfma_f32_32x32x4_bf16",
        ),
        ("mfma_f32_32x32x8f16", "gfx90a", "issuable", "mfma_f32_32x32x8_f16"),
    ],
)
def test_record_pins(op_id: str, target: str, status: str, backend_op: str) -> None:
    # The result dtype tells the two gfx1250 bf16 instructions apart; a 4-block row with a backend
    # op is classed, not laid out; the single-block bf16 op is another instruction than the
    # 2-block _1k row with its operand key.
    assert _record().status(op_id, target) == status
    assert _record().backend_op(op_id, target) == backend_op


@pytest.mark.parametrize(
    "op_id, target, own_op",
    [
        ("wmma_i32_16x16x16_iu8_w32", "gfx1100", "wmma_i32_16x16x16_iu8"),
        ("wmma_i32_16x16x64_iu8", "gfx1250", "wmma_gfx1250_i32_16x16x64_iu8"),
    ],
)
def test_record_answers_a_selectable_integer_row_for_its_own_dtypes(
    op_id: str, target: str, own_op: str
) -> None:
    # A selectable integer row is recorded per operand dtype pair, but the row as the catalog
    # lists it (signed A and B) is looked up by (op_id, target) alone; only an unsigned twin is
    # named.
    coverage = _record()
    assert coverage.status(op_id, target) == "issuable"
    assert coverage.backend_op(op_id, target) == own_op
    assert coverage.backend_op(op_id, target, operand_dtypes=("i8", "i8")) == own_op
    assert (
        coverage.backend_op(op_id, target, operand_dtypes=("u8", "i8"))
        == f"{own_op}_u8_i8"
    )


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "backend_coverage.json"
    path.write_text(json.dumps(document))
    return path


def _minimal() -> dict:
    return {
        "_meta": {"llvm_flavor": "llvm23"},
        "pairs": {
            "x": {
                "gfx942": [{"class": "issuable", "backend_op": "op_x"}],
                "gfx90a": [{"class": "no_backend_op"}],
            },
            "s": {
                "gfx1250": [
                    {
                        "scale_format": ["e8m0", "e8m0"],
                        "class": "issuable",
                        "backend_op": "op_s_e8m0",
                    },
                    {
                        "scale_format": ["e8m0", "e5m3"],
                        "class": "issuable",
                        "backend_op": "op_s_e5m3",
                    },
                ],
                "gfx950": [
                    {"scale_format": ["e8m0", "e8m0"], "class": "no_backend_op"}
                ],
            },
        },
        "backend_ops_without_row": {"gfx942": ["op_y"]},
    }


def _with_integer_pair(entries: list[dict]) -> dict:
    """``_minimal()`` plus a selectable integer row ``n`` on gfx1100 recorded as ``entries``."""
    document = _minimal()
    document["pairs"]["n"] = {"gfx1100": entries}
    return document


def _integer_entry(a: str, b: str) -> dict:
    return {
        "operand_dtypes": [a, b],
        "class": "issuable",
        "backend_op": f"op_n_{a}_{b}",
    }


def test_reader_answers_status_and_backend_op(tmp_path: Path) -> None:
    coverage = load_backend_coverage(_write(tmp_path, _minimal()))
    assert coverage.llvm_flavor == "llvm23"
    assert coverage.status("x", "gfx942") == "issuable"
    assert coverage.backend_op("x", "gfx942") == "op_x"
    assert coverage.backend_ops_without_row("gfx942") == ("op_y",)
    assert coverage.backend_ops_without_row("gfx90a") == ()


def test_reader_refuses_an_unknown_class(tmp_path: Path) -> None:
    document = _minimal()
    document["pairs"]["x"]["gfx942"][0]["class"] = "issueable"
    with pytest.raises(ValueError) as error:
        load_backend_coverage(_write(tmp_path, document))
    assert str(error.value) == (
        "backend_coverage.json: unknown class 'issueable' -- expected one of "
        "['no_backend_op', 'other_instruction', 'multi_block', 'lowering_not_wired', "
        "'operand_form_words_only', 'lane_map_differs', 'invalid_ir', 'issuable']"
    )


def test_reader_refuses_an_issuable_pair_without_a_backend_op(tmp_path: Path) -> None:
    document = _minimal()
    del document["pairs"]["x"]["gfx942"][0]["backend_op"]
    with pytest.raises(ValueError) as error:
        load_backend_coverage(_write(tmp_path, document))
    assert str(error.value) == (
        "backend_coverage.json: ('x', 'gfx942') is 'issuable' but records no backend op"
    )


def test_reader_refuses_an_unknown_flavor(tmp_path: Path) -> None:
    document = _minimal()
    document["_meta"]["llvm_flavor"] = "llvm19"
    with pytest.raises(ValueError) as error:
        load_backend_coverage(_write(tmp_path, document))
    assert str(error.value) == (
        "backend_coverage.json: _meta.llvm_flavor 'llvm19' is not an LLVM flavor -- "
        f"expected one of {list(LLVM_FLAVORS)}"
    )


def test_reader_names_an_unrecorded_pair(tmp_path: Path) -> None:
    coverage = load_backend_coverage(_write(tmp_path, _minimal()))
    with pytest.raises(KeyError) as error:
        coverage.status("x", "gfx950")
    assert error.value.args[0] == (
        "no backend coverage recorded for ('x', 'gfx950') -- the gap test records every "
        "(row, target) pair; re-run it"
    )


def test_reader_has_no_backend_op_for_a_pair_without_one(tmp_path: Path) -> None:
    coverage = load_backend_coverage(_write(tmp_path, _minimal()))
    assert coverage.status("x", "gfx90a") == "no_backend_op"
    with pytest.raises(KeyError) as error:
        coverage.backend_op("x", "gfx90a")
    assert error.value.args[0] == (
        "('x', 'gfx90a') matched no backend op (backend coverage: no_backend_op)"
    )


def test_reader_needs_the_scale_format_of_a_pair_recorded_per_format(
    tmp_path: Path,
) -> None:
    coverage = load_backend_coverage(_write(tmp_path, _minimal()))
    assert (
        coverage.backend_op("s", "gfx1250", scale_format=("e8m0", "e5m3"))
        == "op_s_e5m3"
    )
    with pytest.raises(ValueError, match="is recorded per scale_format"):
        coverage.status("s", "gfx1250")
    with pytest.raises(KeyError, match=r"scale_format=\('e4m3', 'e4m3'\)"):
        coverage.status("s", "gfx1250", scale_format=("e4m3", "e4m3"))
    # One recorded format: naming it is optional.
    assert coverage.status("s", "gfx950") == "no_backend_op"
    assert (
        coverage.status("s", "gfx950", scale_format=("e8m0", "e8m0")) == "no_backend_op"
    )
    # A pair recorded without variants takes no scale format.
    with pytest.raises(
        ValueError, match="is recorded without variants -- pass no scale_format"
    ):
        coverage.status("x", "gfx942", scale_format=("e8m0", "e8m0"))


def test_reader_answers_a_pair_recorded_per_operand_dtypes_for_the_signed_pair(
    tmp_path: Path,
) -> None:
    entries = [_integer_entry(a, b) for a in ("i8", "u8") for b in ("i8", "u8")]
    coverage = load_backend_coverage(_write(tmp_path, _with_integer_pair(entries)))
    assert coverage.backend_op("n", "gfx1100") == "op_n_i8_i8"
    assert (
        coverage.backend_op("n", "gfx1100", operand_dtypes=("u8", "i8")) == "op_n_u8_i8"
    )
    with pytest.raises(KeyError, match=r"operand_dtypes=\('i4', 'i4'\)"):
        coverage.status("n", "gfx1100", operand_dtypes=("i4", "i4"))
    with pytest.raises(
        ValueError, match="is recorded without variants -- pass no operand_dtypes"
    ):
        coverage.status("x", "gfx942", operand_dtypes=("i8", "i8"))


def test_reader_refuses_a_pair_recorded_per_operand_dtypes_without_the_signed_pair(
    tmp_path: Path,
) -> None:
    entries = [_integer_entry("u8", "i8"), _integer_entry("u8", "u8")]
    with pytest.raises(ValueError) as error:
        load_backend_coverage(_write(tmp_path, _with_integer_pair(entries)))
    assert str(error.value) == (
        "backend_coverage.json: ('n', 'gfx1100') records operand_dtypes "
        "[('u8', 'i8'), ('u8', 'u8')] -- expected exactly one pair of signed dtypes "
        "['i4', 'i8'] (the row's own), found []"
    )


# ---------------------------------------------------------------------------
# Regenerating the record
# ---------------------------------------------------------------------------


def _write_record(llvm_flavor: str) -> None:
    """Measure the record and compile every pair the compile decides with the arbiter's clang."""
    document = measure_backend_coverage(llvm_flavor)
    _, validity_tool = _tools()
    clang = validity_tool._llvm_tool("clang")
    from _llvm_identity import flavor_of_clang, tool_identity

    clang_flavor = flavor_of_clang(tool_identity(clang)) if clang else None
    if clang_flavor is None:
        raise SystemExit("no clang with a known LLVM flavor to judge the record")
    for target in _TARGETS:
        ops = {op.op_id: op for op in ArchTarget.from_gfx(target).mma.ops}
        jobs = [
            _CompileJob((op_id, index), _CATALOG.get(op_id), ops[entry["backend_op"]])
            for op_id, by_target in document["pairs"].items()
            for index, entry in enumerate(by_target.get(target, []))
            if entry["class"] == "issuable"
        ]
        verdicts = _compile_target(clang, clang_flavor, target, jobs)
        if verdicts is None:
            raise SystemExit(f"clang {clang} cannot target {target}; use one that can")
        for (op_id, index), (ok, _) in verdicts.items():
            if not ok:
                document["pairs"][op_id][target][index]["class"] = "invalid_ir"
    DEFAULT_COVERAGE_PATH.write_text(_dump(document))


def _dump(document: dict) -> str:
    """The record as JSON with one line per (row, target) pair, for review as data."""
    lines = ["{", f' "_meta": {json.dumps(document["_meta"])},', ' "pairs": {']
    rows = list(document["pairs"].items())
    for row_index, (op_id, by_target) in enumerate(rows):
        lines.append(f"  {json.dumps(op_id)}: {{")
        targets = list(by_target.items())
        for index, (target, entries) in enumerate(targets):
            comma = "," if index < len(targets) - 1 else ""
            lines.append(f"   {json.dumps(target)}: {json.dumps(entries)}{comma}")
        lines.append("  }" + ("," if row_index < len(rows) - 1 else ""))
    lines.append(" },")
    without_row = json.dumps(document["backend_ops_without_row"])
    lines.append(f' "backend_ops_without_row": {without_row}')
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="regenerate the record")
    group.add_argument(
        "--print-classes",
        action="store_true",
        help="print the clang-free measurement at the record's flavor as JSON",
    )
    parser.add_argument("--llvm-flavor", help="the flavor to measure at (--write)")
    args = parser.parse_args()
    if args.write:
        _write_record(args.llvm_flavor or _record().llvm_flavor)
    else:
        json.dump(measure_backend_coverage(_record().llvm_flavor), sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
