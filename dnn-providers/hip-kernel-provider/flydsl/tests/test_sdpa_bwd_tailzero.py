# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The tail-zero kernels (padded-row zeroing of dense gradients): addressing, declaration and companion wiring.

The addressing model (``fmha_bwd_tailzero_addressing``) is the arithmetic the kernels run on FlyDSL values; here it runs on Python
integers over the whole grid, and the set of elements it writes is held to an independent statement of the contract: for each
output whose ``mask`` bit is set, exactly the elements of the logical rows ``seq_len[b] <= row < S``, at the output's own strides.
Host-only: no FlyDSL install and no compiled object is needed (the object itself is proven by the spike that builds it).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
import gen_descriptors as gd  # noqa: E402
from generators import (
    _instances,
    _sdpa_bwd_cdna_build as cdna,
    gen_sdpa_bwd_tailzero as tz,
)  # noqa: E402
from generators._codeobject import GeneratorError  # noqa: E402
from generators._ops import OP_KERNEL_DIRS  # noqa: E402

KERNELS = FLYDSL / "kernels_src" / "kernels"
CDNA = KERNELS / "attention_bwd_cdna"
RDNA = KERNELS / "attention_bwd_rdna"
HELPER = "fmha_bwd_tailzero_addressing.py"
KERNEL_FILES = {
    "gfx950": CDNA / "fmha_bwd_tailzero_gfx950.py",
    "gfx11-generic": RDNA / "fmha_bwd_tailzero_gfx11.py",
}


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_tz_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


addr = _load(CDNA / HELPER)


# ---------------------------------------------------------------------------
# Addressing model
# ---------------------------------------------------------------------------


def _expected(lens_q, lens_kv, hq, hkv, sq, skv, dqk, dv, strides, mask, hv=None):
    """The contract, stated without the grid: the elements of the padded rows of each selected output (dV has `hv` heads)."""
    hv = hkv if hv is None else hv
    shapes = ((sq, dqk, hq, lens_q), (skv, dqk, hkv, lens_kv), (skv, dv, hv, lens_kv))
    want = set()
    for tensor, (seq, dim, heads, lens) in enumerate(shapes):
        if not mask & (1 << tensor):
            continue
        sb, sh, ss = strides[tensor]
        for b, length in enumerate(lens):
            for h, row, col in itertools.product(
                range(heads), range(length, seq), range(dim)
            ):
                want.add((tensor, b * sb + h * sh + row * ss + col))
    return want


def _contiguous(hq, hkv, sq, skv, dqk, dv, hv=None):
    hv = hkv if hv is None else hv
    return (
        (hq * sq * dqk, sq * dqk, dqk),
        (hkv * skv * dqk, skv * dqk, dqk),
        (hv * skv * dv, skv * dv, dv),
    )


def _shared_view(hq, hkv, s, d):
    """dQ, dK, dV as the three slices of one ``[B, S, 3, H, D]`` buffer (MHA: every head count equal)."""
    assert hq == hkv
    batch, head, seq = s * 3 * hq * d, d, 3 * hq * d
    return tuple((batch, head, seq) for _ in range(3))


CASES = {
    "contiguous": dict(
        lens_q=[3, 0, 8], lens_kv=[5, 2, 8], hq=2, hkv=2, sq=8, skv=8, dqk=16, dv=16
    ),
    "gqa": dict(lens_q=[1, 7], lens_kv=[2, 0], hq=4, hkv=2, sq=7, skv=5, dqk=24, dv=8),
    "unequal head dims": dict(
        lens_q=[2, 4], lens_kv=[3, 1], hq=2, hkv=1, sq=6, skv=4, dqk=40, dv=24
    ),
    "partial vector": dict(
        lens_q=[0, 5], lens_kv=[3, 3], hq=1, hkv=1, sq=6, skv=6, dqk=37, dv=21
    ),
    "all valid": dict(
        lens_q=[6, 6], lens_kv=[4, 4], hq=2, hkv=1, sq=6, skv=4, dqk=16, dv=16
    ),
    "all padded": dict(
        lens_q=[0, 0], lens_kv=[0, 0], hq=2, hkv=1, sq=6, skv=4, dqk=16, dv=16
    ),
    "single row": dict(lens_q=[0], lens_kv=[0], hq=1, hkv=1, sq=1, skv=1, dqk=8, dv=8),
    "grid larger than one block": dict(
        lens_q=[2], lens_kv=[2], hq=2, hkv=2, sq=70, skv=70, dqk=64, dv=64
    ),
}


@pytest.mark.parametrize("elem_bytes", [2, 4], ids=["16-bit", "fp32"])
@pytest.mark.parametrize("mask", range(8))
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_written_set_is_the_padded_rows_of_the_masked_outputs(
    case: str, mask: int, elem_bytes: int
) -> None:
    c = CASES[case]
    strides = _contiguous(c["hq"], c["hkv"], c["sq"], c["skv"], c["dqk"], c["dv"])
    got = addr.written_elements(
        c["lens_q"],
        c["lens_kv"],
        c["hq"],
        c["hkv"],
        c["sq"],
        c["skv"],
        c["dqk"],
        c["dv"],
        strides,
        mask,
        elem_bytes,
    )
    assert got == _expected(
        c["lens_q"],
        c["lens_kv"],
        c["hq"],
        c["hkv"],
        c["sq"],
        c["skv"],
        c["dqk"],
        c["dv"],
        strides,
        mask,
    )


HV_CASES = {
    "fewer value heads": dict(
        lens_q=[3, 0, 8],
        lens_kv=[5, 0, 8],
        hq=4,
        hkv=2,
        hv=1,
        sq=8,
        skv=8,
        dqk=16,
        dv=24,
    ),
    "more value heads": dict(
        lens_q=[1, 7], lens_kv=[2, 4], hq=2, hkv=1, hv=3, sq=7, skv=5, dqk=24, dv=8
    ),
    "value heads equal key heads": dict(
        lens_q=[2, 6], lens_kv=[3, 5], hq=2, hkv=2, hv=2, sq=6, skv=5, dqk=16, dv=16
    ),
    "partial vector": dict(
        lens_q=[0, 5], lens_kv=[3, 3], hq=1, hkv=2, hv=1, sq=6, skv=6, dqk=37, dv=21
    ),
    "all valid": dict(
        lens_q=[6, 6], lens_kv=[4, 4], hq=2, hkv=2, hv=1, sq=6, skv=4, dqk=16, dv=16
    ),
    "all padded": dict(
        lens_q=[0, 0], lens_kv=[0, 0], hq=2, hkv=1, hv=2, sq=6, skv=4, dqk=16, dv=16
    ),
    "grid larger than one block": dict(
        lens_q=[2], lens_kv=[2], hq=2, hkv=2, hv=1, sq=70, skv=70, dqk=64, dv=64
    ),
}


@pytest.mark.parametrize("elem_bytes", [2, 4], ids=["16-bit", "fp32"])
@pytest.mark.parametrize("mask", range(8))
@pytest.mark.parametrize("case", sorted(HV_CASES))
def test_dv_extent_follows_its_own_head_count(
    case: str, mask: int, elem_bytes: int
) -> None:
    c = HV_CASES[case]
    args = (
        c["lens_q"],
        c["lens_kv"],
        c["hq"],
        c["hkv"],
        c["sq"],
        c["skv"],
        c["dqk"],
        c["dv"],
    )
    strides = _contiguous(*args[2:], hv=c["hv"])
    got = addr.written_elements(*args, strides, mask, elem_bytes, hv=c["hv"])
    assert got == _expected(*args, strides, mask, hv=c["hv"])


def test_planes_per_batch_count_each_head_set_once() -> None:
    assert addr.heads_total(4, 2) == 8
    assert addr.heads_total(4, 2, 2) == 8
    assert addr.heads_total(4, 2, 1) == 7
    assert addr.heads_total(2, 1, 3) == 6


def test_omitting_hv_is_hv_equal_to_hkv() -> None:
    c = CASES["gqa"]
    strides = _contiguous(c["hq"], c["hkv"], c["sq"], c["skv"], c["dqk"], c["dv"])
    args = (
        c["lens_q"],
        c["lens_kv"],
        c["hq"],
        c["hkv"],
        c["sq"],
        c["skv"],
        c["dqk"],
        c["dv"],
        strides,
        0b111,
        2,
    )
    assert addr.written_elements(*args) == addr.written_elements(*args, hv=c["hkv"])


@pytest.mark.parametrize("elem_bytes", [2, 4], ids=["16-bit", "fp32"])
def test_a_shared_gradient_view_is_written_only_in_the_padded_rows(
    elem_bytes: int,
) -> None:
    """dQ, dK, dV are slices of one ``[B, S, 3, H, D]`` buffer: each tensor's offsets are its own slice's padded rows."""
    hq = hkv = 2
    s, d = 6, 16
    lens_q, lens_kv = [2, 6, 0], [3, 1, 6]
    strides = _shared_view(hq, hkv, s, d)
    got = addr.written_elements(
        lens_q, lens_kv, hq, hkv, s, s, d, d, strides, 0b111, elem_bytes
    )
    assert got == _expected(lens_q, lens_kv, hq, hkv, s, s, d, d, strides, 0b111)
    # In the shared buffer the slices interleave per row; with the slice bases `0, H*D, 2*H*D`, the three tensors' padded rows never
    # meet each other's live rows.
    row_width = 3 * hq * d
    for tensor, lens in ((0, lens_q), (1, lens_kv), (2, lens_kv)):
        base = tensor * hq * d
        for _, offset in (w for w in got if w[0] == tensor):
            absolute = base + offset
            b, rest = divmod(absolute, s * row_width)
            row = rest // row_width
            assert row >= lens[b]


def test_an_output_whose_bit_is_clear_is_not_written() -> None:
    c = CASES["contiguous"]
    strides = _contiguous(c["hq"], c["hkv"], c["sq"], c["skv"], c["dqk"], c["dv"])
    args = (
        c["lens_q"],
        c["lens_kv"],
        c["hq"],
        c["hkv"],
        c["sq"],
        c["skv"],
        c["dqk"],
        c["dv"],
        strides,
    )
    assert addr.written_elements(*args, 0, 2) == set()
    only_dk = addr.written_elements(*args, addr.MASK_DK, 2)
    assert only_dk and {t for t, _ in only_dk} == {1}


def test_the_grid_covers_every_tail_unit_of_the_largest_tensor() -> None:
    """``grid_blocks`` of the launch's ``units`` reaches the last unit of the widest tail."""
    for elem_bytes in (2, 4):
        vec = addr.vec_of(elem_bytes)
        for sq, skv, dqk, dv in ((7, 5, 24, 8), (1, 1, 8, 8), (300, 17, 129, 7)):
            units = max(
                addr.units_of(sq, dqk, vec),
                addr.units_of(skv, dqk, vec),
                addr.units_of(skv, dv, vec),
            )
            assert addr.grid_blocks(units) * addr.BLOCK_THREADS >= units
            assert (addr.grid_blocks(units) - 1) * addr.BLOCK_THREADS < units


def test_a_vector_is_sixteen_bytes() -> None:
    assert (addr.vec_of(2), addr.vec_of(4)) == (8, 4)


def _helper_uses(path: Path) -> set[str]:
    """The names a kernel reads off the helper module (`addr.<name>`), without importing FlyDSL."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "addr"
    }


def test_both_arches_kernels_take_their_addressing_from_the_one_helper() -> None:
    """Every `addr.<name>` a kernel evaluates is a function or constant the helper defines (the one
    `written_elements` models), the two kernels evaluate the same helper functions, and neither defines a
    function of the helper's names itself."""
    helper = {n for n in vars(addr) if not n.startswith("_")}
    functions = {n for n in helper if callable(getattr(addr, n))}
    used = {arch: _helper_uses(path) for arch, path in KERNEL_FILES.items()}
    for arch, names in used.items():
        assert names, arch
        assert names <= helper, (arch, sorted(names - helper))
    assert {a: n & functions for a, n in used.items()}[
        "gfx950"
    ] == {a: n & functions for a, n in used.items()}["gfx11-generic"]
    for arch, path in KERNEL_FILES.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        own = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert not own & functions, (arch, sorted(own & functions))


# ---------------------------------------------------------------------------
# The declaration, the kernels and the registries
# ---------------------------------------------------------------------------


def _kernel_parameters(path: Path, hv: bool = False) -> list[tuple[str, str]]:
    """The ``(name, annotation)`` list of the ``@flyc.kernel`` function of @p path (the V-head one when @p hv), without importing
    FlyDSL."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    (kernel,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any("kernel" in ast.unparse(d) for d in node.decorator_list)
        and ("_hv_" in node.name) == hv
    ]
    return [(a.arg, ast.unparse(a.annotation)) for a in kernel.args.args]


KIND = {"ptr": "fx.Pointer", "i32": "fx.Int32", "i64": "fx.Int64"}


@pytest.mark.parametrize("arch", sorted(KERNEL_FILES))
def test_the_declared_parameter_list_is_the_kernels(arch: str) -> None:
    declared = [(name, KIND[kind]) for name, kind in tz.PARAMS]
    assert _kernel_parameters(KERNEL_FILES[arch]) == declared


@pytest.mark.parametrize("arch", sorted(KERNEL_FILES))
def test_the_declared_v_head_parameter_list_is_the_v_head_kernels(arch: str) -> None:
    declared = [(name, KIND[kind]) for name, kind in tz.PARAMS_HV]
    assert _kernel_parameters(KERNEL_FILES[arch], hv=True) == declared
    names = [name for name, _ in tz.PARAMS_HV]
    assert names.index("Hv") == names.index("Hkv") + 1
    assert [n for n in names if n != "Hv"] == [name for name, _ in tz.PARAMS]


@pytest.mark.parametrize("arch", sorted(KERNEL_FILES))
def test_the_launch_signature_is_the_kernels_plus_the_stream(arch: str) -> None:
    tree = ast.parse(KERNEL_FILES[arch].read_text(encoding="utf-8"))
    (launch,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("launch_fmha_bwd_tailzero")
        and "_hv_" not in node.name
    ]
    names = [a.arg for a in launch.args.args]
    assert names == [name for name, _ in tz.PARAMS] + ["stream"]


@pytest.mark.parametrize("arch", sorted(KERNEL_FILES))
def test_the_v_head_launch_signature_is_its_kernels_plus_the_stream(arch: str) -> None:
    tree = ast.parse(KERNEL_FILES[arch].read_text(encoding="utf-8"))
    (launch,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("launch_fmha_bwd_tailzero_hv_")
    ]
    names = [a.arg for a in launch.args.args]
    assert names == [name for name, _ in tz.PARAMS_HV] + ["stream"]


@pytest.mark.parametrize("arch", sorted(KERNEL_FILES))
def test_the_kernel_builds_the_symbol_and_dtypes_the_generator_declares(
    arch: str,
) -> None:
    text = KERNEL_FILES[arch].read_text(encoding="utf-8")
    family = "gfx950" if arch == "gfx950" else "gfx11"
    assert f'KERNEL_NAME = "fmha_bwd_tailzero_{family}"' in text
    assert tz.SYMBOL[arch] == f"fmha_bwd_tailzero_{family}_kernel_0"
    for dtype in tz.SDPA_BWD_TAILZERO_DTYPES:
        assert f'"{dtype}":' in text
    assert f'KERNEL_NAME_HV = "fmha_bwd_tailzero_hv_{family}"' in text
    assert tz.SYMBOL_HV[arch] == f"fmha_bwd_tailzero_hv_{family}_kernel_0"


def test_the_instance_rows_are_three_dtypes_per_arch_family() -> None:
    for arch in ("gfx950", "gfx11-generic"):
        rows = _instances.instances_for("sdpa_bwd_tailzero", arch)
        assert [r.dtype for r in rows] == ["f16", "bf16", "fp32"]
        assert {r.op for r in rows} == {"sdpa_bwd_tailzero"}
    assert _instances.instances_for("sdpa_bwd_tailzero", "gfx12-generic") == []


def test_the_op_is_registered_with_the_split_ops_kernel_directories() -> None:
    assert OP_KERNEL_DIRS["sdpa_bwd_tailzero"] == OP_KERNEL_DIRS["sdpa_bwd_dq"]


def test_the_generator_refuses_an_arch_it_does_not_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert tz.main(["--arch", "gfx90a", "--out-dir", str(tmp_path)]) == 1
    assert "gfx950" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_a_companion_spec_names_its_role_and_dtype() -> None:
    for arch in tz.ARCHES:
        for dtype in tz.SDPA_BWD_TAILZERO_DTYPES:
            companion = {"spec": tz.spec_key(arch, dtype)}
            assert gd.companion_identity(companion) == ("tailzero", dtype)
    reduce = {"spec": "dbias_reduce,bf16,32,plain,bk,0,0,tail,0,1;arch=gfx11-generic"}
    assert gd.companion_identity(reduce) == ("dbias_reduce", "bf16")


def test_a_v_head_spec_names_its_role_and_dtype() -> None:
    for arch in tz.ARCHES:
        for dtype in tz.HV_DTYPES:
            companion = {"spec": tz.spec_key(arch, dtype, True)}
            assert gd.companion_identity(companion) == ("tailzero_hv", dtype)
    assert tz.spec_key("gfx950", "bf16", True) == "tailzero_hv,bf16"
    assert (
        tz.spec_key("gfx11-generic", "f16", True)
        == "tailzero_hv,f16;arch=gfx11-generic"
    )
    assert tz.HV_DTYPES == ("f16", "bf16")


def test_the_existing_spec_keys_are_unchanged() -> None:
    assert tz.spec_key("gfx950", "f16") == "tailzero,f16"
    assert tz.spec_key("gfx11-generic", "fp32") == "tailzero,fp32;arch=gfx11-generic"


def test_the_ten_objects_have_distinct_names() -> None:
    names = {
        tz.object_name(arch, dtype, hv)
        for arch in tz.ARCHES
        for hv, dtypes in ((False, tz.SDPA_BWD_TAILZERO_DTYPES), (True, tz.HV_DTYPES))
        for dtype in dtypes
    }
    assert len(names) == 10


def test_the_six_objects_have_distinct_names() -> None:
    names = {
        tz.object_name(arch, dtype)
        for arch in tz.ARCHES
        for dtype in tz.SDPA_BWD_TAILZERO_DTYPES
    }
    assert len(names) == 6


# ---------------------------------------------------------------------------
# Companions through ``gen_descriptors.collect``
# ---------------------------------------------------------------------------

BLOB = b"\x7fELF-object"
SHA = __import__("hashlib").sha256(BLOB).hexdigest()


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A synthetic content tree: a gfx950 dQ and merged op whose companions are written by ``write``; objects are not inspected."""
    monkeypatch.setattr(gd, "describe", lambda data, where, symbol=None: {})
    monkeypatch.setattr(gd, "verify_arch", lambda *a, **k: None)
    monkeypatch.setattr(gd, "verify_generic", lambda *a, **k: None)
    monkeypatch.setattr(gd, "inputs_digest", lambda op, arch: "digest")

    def write(op: str, rows: list[dict], companions: list[dict], extra=()) -> None:
        folder = tmp_path / op / "gfx950"
        folder.mkdir(parents=True)
        for record in rows:
            (folder / record["file"]).write_bytes(BLOB)
        for companion in extra:
            (folder / companion).parent.mkdir(parents=True, exist_ok=True)
            (folder / companion).write_bytes(BLOB)
        manifest = {
            "arch": "gfx950",
            "op": op,
            "toolchain": {"inputs_sha256": "digest"},
            "instances": rows,
        }
        if companions:
            manifest["companions"] = companions
        (folder / "manifest.json").write_text(json.dumps(manifest))

    return tmp_path, write


def _record(name: str, dtype: str, **knobs) -> dict:
    return {
        "name": name,
        "file": f"{name}.hsaco",
        "symbol": f"sym_{name}",
        "dtype": dtype,
        "priority": 100,
        "sha256": SHA,
        "knobs": knobs,
    }


def _companion(role: str, dtype: str, file: str, symbol: str = "tz") -> dict:
    return {
        "hash": "h",
        "spec": f"{role},{dtype}",
        "file": file,
        "symbol": symbol,
        "sha256": SHA,
    }


def test_each_dtype_row_gets_its_own_dtypes_tailzero_companion(tree) -> None:
    root, write = tree
    write(
        "sdpa_bwd_dq",
        [
            _record("dq_f16", "f16", reduce_symbol=""),
            _record("dq_bf16", "bf16", reduce_symbol=""),
        ],
        [
            _companion("tailzero", "f16", "tz_f16.hsaco"),
            _companion("tailzero", "bf16", "tz_bf16.hsaco"),
        ],
        extra=["tz_f16.hsaco", "tz_bf16.hsaco"],
    )
    entries = {e.record["name"]: e for e in gd.collect(root, "gfx950")}
    assert entries["dq_f16"].companions["tailzero"]["file"] == "tz_f16.hsaco"
    assert entries["dq_bf16"].companions["tailzero"]["file"] == "tz_bf16.hsaco"
    assert set(entries["dq_f16"].companions) == {"tailzero"}


def test_a_dq_row_without_a_tailzero_companion_is_refused(tree) -> None:
    root, write = tree
    write("sdpa_bwd_dq", [_record("dq_f16", "f16", reduce_symbol="")], [])
    with pytest.raises(ValueError, match="tailzero"):
        gd.collect(root, "gfx950")


def test_a_companion_of_another_dtype_does_not_serve_the_row(tree) -> None:
    root, write = tree
    write(
        "sdpa_bwd_dq",
        [_record("dq_f16", "f16", reduce_symbol="")],
        [_companion("tailzero", "bf16", "tz_bf16.hsaco")],
        extra=["tz_bf16.hsaco"],
    )
    with pytest.raises(ValueError, match="tailzero"):
        gd.collect(root, "gfx950")


def test_two_companions_of_one_role_and_dtype_are_refused(tree) -> None:
    root, write = tree
    write(
        "sdpa_bwd_dq",
        [_record("dq_f16", "f16", reduce_symbol="")],
        [
            _companion("tailzero", "f16", "a.hsaco"),
            _companion("tailzero", "f16", "b.hsaco"),
        ],
        extra=["a.hsaco", "b.hsaco"],
    )
    with pytest.raises(ValueError, match="two companions"):
        gd.collect(root, "gfx950")


def test_a_dtype_may_hold_a_reduce_and_a_tailzero_companion(tree) -> None:
    root, write = tree
    write(
        "sdpa_bwd_dq",
        [
            _record("dq_bias", "bf16", reduce_symbol="red"),
            _record("dq_plain", "bf16", reduce_symbol=""),
        ],
        [
            _companion("dbias_reduce", "bf16", "red.hsaco", symbol="red"),
            _companion("tailzero", "bf16", "tz.hsaco"),
        ],
        extra=["red.hsaco", "tz.hsaco"],
    )
    entries = {e.record["name"]: e for e in gd.collect(root, "gfx950")}
    assert set(entries["dq_bias"].companions) == {"dbias_reduce", "tailzero"}
    # A row that stores no dB through a standalone reduce does not launch one.
    assert set(entries["dq_plain"].companions) == {"tailzero"}


def test_a_reduce_symbol_the_reduce_companion_does_not_hold_is_refused(tree) -> None:
    root, write = tree
    write(
        "sdpa_bwd_dq",
        [_record("dq_bias", "bf16", reduce_symbol="other")],
        [
            _companion("dbias_reduce", "bf16", "red.hsaco", symbol="red"),
            _companion("tailzero", "bf16", "tz.hsaco"),
        ],
        extra=["red.hsaco", "tz.hsaco"],
    )
    with pytest.raises(ValueError, match="reduce symbol"):
        gd.collect(root, "gfx950")


def test_prep_and_dkdv_rows_take_no_companion(tree) -> None:
    root, write = tree
    write("sdpa_bwd_dkdv", [_record("dkdv_f16", "f16", gqa_pack=0)], [])
    (entry,) = gd.collect(root, "gfx950")
    assert entry.companions is None


def test_the_merged_manifest_reaches_the_dq_folders_object_across_folders(tree) -> None:
    root, write = tree
    write(
        "sdpa_bwd_dq",
        [_record("dq_f16", "f16", reduce_symbol="")],
        [_companion("tailzero", "f16", "tz_f16.hsaco")],
        extra=["tz_f16.hsaco"],
    )
    # The merged folder holds the dq folder's object by relative path, not a copy of it.
    shared = "../../sdpa_bwd_dq/gfx950/tz_f16.hsaco"
    write(
        "sdpa_bwd_merged",
        [_record("merged_f16", "f16", gqa_pack=0, block_q=-1, reduce_symbol="")],
        [_companion("tailzero", "f16", shared)],
    )
    entries = {e.record["name"]: e for e in gd.collect(root, "gfx950")}
    merged = entries["merged_f16"]
    assert merged.companions["tailzero"]["file"] == shared
    assert (
        merged.companions["tailzero"]["path"].resolve()
        == (root / "sdpa_bwd_dq" / "gfx950" / "tz_f16.hsaco").resolve()
    )


def test_a_cross_folder_companion_that_is_missing_is_refused(tree) -> None:
    root, write = tree
    write(
        "sdpa_bwd_merged",
        [_record("merged_f16", "f16", gqa_pack=0, block_q=-1, reduce_symbol="")],
        [_companion("tailzero", "f16", "../../sdpa_bwd_dq/gfx950/absent.hsaco")],
    )
    with pytest.raises(FileNotFoundError, match="absent"):
        gd.collect(root, "gfx950")


# ---------------------------------------------------------------------------
# The companion writers of the dQ and merged generators (the objects are built by ``tz.build_objects``; faked here)
# ---------------------------------------------------------------------------


def _fake_entries(arch: str, folder: Path) -> list[dict]:
    """What ``tz.companion_entries`` returns and writes: one object per dtype."""
    out = []
    for dtype in tz.SDPA_BWD_TAILZERO_DTYPES:
        blob = f"{arch}-{dtype}".encode()
        name = tz.object_name(arch, dtype)
        (folder / name).write_bytes(blob)
        out.append(
            {
                "hash": tz.spec_hash(arch, dtype),
                "spec": tz.spec_key(arch, dtype),
                "file": name,
                "symbol": tz.SYMBOL[arch],
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
    return out


def _op_dir(root: Path, op: str, arch: str = "gfx950") -> Path:
    op_dir = root / op / arch
    op_dir.mkdir(parents=True)
    (op_dir / "manifest.json").write_text(
        json.dumps({"op": op, "arch": arch, "instances": []}), encoding="utf-8"
    )
    (root / op / f"{arch}.SOURCE.md").write_text("# source\n", encoding="utf-8")
    return op_dir


@pytest.fixture
def dq_folder(tmp_path: Path) -> Path:
    """A generated dQ folder whose manifest lists the tail-zero companions per dtype."""
    dq = _op_dir(tmp_path, "sdpa_bwd_dq")
    cdna.write_companions(dq, _fake_entries("gfx950", dq))
    return dq


def test_the_dq_manifest_lists_one_tailzero_companion_per_dtype_and_role(
    dq_folder: Path,
) -> None:
    listed = json.loads((dq_folder / "manifest.json").read_text(encoding="utf-8"))[
        "companions"
    ]
    assert [gd.companion_identity(c) for c in listed] == [
        ("tailzero", dtype) for dtype in tz.SDPA_BWD_TAILZERO_DTYPES
    ]
    assert all((dq_folder / c["file"]).is_file() for c in listed)
    assert "Companion objects" in (dq_folder.parent / "gfx950.SOURCE.md").read_text(
        encoding="utf-8"
    )


def test_the_merged_build_lists_the_dq_folders_objects_by_path_and_keeps_none(
    dq_folder: Path,
) -> None:
    merged = _op_dir(dq_folder.parent.parent, "sdpa_bwd_merged")
    entries = cdna.dq_tailzero_companions("gfx950", merged)
    assert not list(merged.glob("*.hsaco"))
    assert [gd.companion_identity(c) for c in entries] == [
        ("tailzero", dtype) for dtype in tz.SDPA_BWD_TAILZERO_DTYPES
    ]
    for entry in entries:
        name = Path(entry["file"]).name
        assert entry["file"] == f"../../sdpa_bwd_dq/gfx950/{name}"
        assert (merged / entry["file"]).resolve() == (dq_folder / name).resolve()


def test_the_merged_build_refuses_a_dq_folder_that_is_missing_or_altered(
    tmp_path: Path, dq_folder: Path
) -> None:
    merged = _op_dir(tmp_path, "sdpa_bwd_merged", "gfx950")
    # (the dq folder is under tmp_path too: merged's sibling)
    (dq_folder / tz.object_name("gfx950", "f16")).write_bytes(b"another build")
    with pytest.raises(GeneratorError, match="generate sdpa_bwd_dq gfx950 first"):
        cdna.dq_tailzero_companions("gfx950", merged)
    (dq_folder / tz.object_name("gfx950", "f16")).unlink()
    with pytest.raises(GeneratorError, match="is missing: generate sdpa_bwd_dq"):
        cdna.dq_tailzero_companions("gfx950", merged)
    (dq_folder / "manifest.json").write_text(
        json.dumps({"op": "sdpa_bwd_dq", "instances": []}), encoding="utf-8"
    )
    with pytest.raises(GeneratorError, match="lists no tailzero companion"):
        cdna.dq_tailzero_companions("gfx950", merged)
    (dq_folder / "manifest.json").unlink()
    with pytest.raises(GeneratorError, match="generate sdpa_bwd_dq gfx950 first"):
        cdna.dq_tailzero_companions("gfx950", merged)


def test_the_shared_builds_import_no_tailzero_generator() -> None:
    # The digest closure follows imports: an op that reaches the tail-zero generator would take the rdna kernels into its scope.
    for name in ("_sdpa_bwd_cdna_build", "_sdpa_bwd_rdna_build", "gen_sdpa_bwd_merged"):
        tree = ast.parse((FLYDSL / "generators" / f"{name}.py").read_text("utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
            assert not any("gen_sdpa_bwd_tailzero" in n for n in names), name
