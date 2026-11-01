# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-R3: the Rust direct code-object backend in the true-mono builds (CPU only).

* the direct backend consumes WP1's named lever scopes (``attrs["rocke.scope"]`` on ops, ``attrs["direct_scopes"]``
  on the kernel) as the compiler's ``dlevers`` op scopes, strips every ``rocke.*`` marker and the kernel's
  ``direct_scopes`` / ``direct_levers``, and leaves the kernel itself (the COMGR fallback) untouched;
* the backend's default levers (``acqfence``, ``bufoffany``; D0.6) and ``ROCKE_DIRECT_DEFAULT_LEVERS``;
* D0.6's per-family default: ``persist.knobs.rust_family`` picks direct only for the standalone families that won
  on both rocke nodes (B3, fused MoE rows 1); ``FrameworkKnobs(code_object_backend="auto")`` only for the families
  whose framework-hosted builds won (``RUST_FRAMEWORK_DIRECT_FAMILIES``: none yet, so COMGR);
* nothing of this reaches the LLVM IR, the serialized IR of a build without the knobs, the name or a pair id.
"""

from __future__ import annotations

import dataclasses
import os

import pytest
from kernels.gfx950.qwen38_mono import mailbox_ir  # noqa: F401  (registers "ir" / "irf")
from kernels.gfx950.qwen38_mono import moe_fused as MF
from kernels.gfx950.qwen38_mono.framework import FrameworkKnobs
from kernels.gfx950.qwen38_mono.persist import knobs as K
from kernels.gfx950.qwen38_mono.persist import lint as L
from kernels.gfx950.qwen38_mono.persist import ptoy as T
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers import compile as C


def _ll(k):
    return _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor="llvm22")


def _scoped(stages, **kw):
    return T.build_static(FrameworkKnobs(rust_scope=K.RustScope.of(stages), **kw))


# ------------------------------------------------------------ direct_source


def test_direct_source_turns_named_scopes_into_dlevers():
    k = _scoped({"g": "b3"})
    before = serialize(k)
    src = C.direct_source(k)
    assert serialize(k) == before  # the kernel (COMGR fallback) is untouched
    assert "direct_scopes" in k.attrs and "direct_scopes" not in src.attrs
    ops = list(L.ops_in_order(src))
    assert not any(a.startswith("rocke.") for op in ops for a in op.attrs)
    want = ",".join(K.RUST_LEVER_SETS["b3"])
    got = {op.attrs.get("dlevers") for op in ops}
    assert got == {None, want}
    # every op of stage g (constants included) carries the set; the others do not
    tagged = [op for op in L.ops_in_order(k) if op.attrs.get("rocke.scope") == "g"]
    assert any(op.name == "arith.constant" for op in tagged)
    assert sum(1 for op in ops if op.attrs.get("dlevers") == want) == len(tagged)
    text = serialize(src)
    assert "rocke." not in text and f'dlevers = s:"{want}"' in text


def test_direct_source_explicit_dlevers_win_and_empty_sets_add_nothing():
    k = _scoped({"g": "b3", "r": "head"}, stage_levers="g=minmax")
    src = C.direct_source(k)
    levers = {op.attrs.get("dlevers") for op in L.ops_in_order(src)}
    assert "minmax" in levers and ",".join(K.RUST_LEVER_SETS["b3"]) not in levers
    # the head set is empty: its ops get no scope (the kernel's levers)
    assert levers == {None, "minmax"}


def test_direct_source_is_the_kernel_when_nothing_to_change():
    k = T.build_static()
    assert C.direct_source(k) is k


def test_direct_source_strips_markers_and_kernel_levers():
    k = T.build_static()
    k.attrs["direct_levers"] = "minmax"
    src = C.direct_source(k)
    assert "direct_levers" not in src.attrs and k.attrs["direct_levers"] == "minmax"


def test_parse_direct_scopes():
    assert C.parse_direct_scopes("a:x+y;b:-z;c:") == {"a": ["x", "y"], "b": ["-z"], "c": []}
    with pytest.raises(ValueError):
        C.parse_direct_scopes("nocolon")


# ------------------------------------------------------- defaults and levers


def test_backend_defaults_match_the_plan(monkeypatch):
    assert tuple(C.DIRECT_DEFAULT_LEVERS) == K.RUST_BACKEND_DEFAULTS == ("acqfence", "bufoffany")
    monkeypatch.delenv("ROCKE_DIRECT_DEFAULT_LEVERS", raising=False)
    assert C.direct_default_levers() == ["acqfence", "bufoffany"]
    monkeypatch.setenv("ROCKE_DIRECT_DEFAULT_LEVERS", "")
    assert C.direct_default_levers() == []
    monkeypatch.setenv("ROCKE_DIRECT_DEFAULT_LEVERS", "acqfence")
    assert C.direct_default_levers() == ["acqfence"]


class _Captured(Exception):
    pass


def _capture_direct(monkeypatch):
    """Patch the direct compiler: record (serialized IR, ROCKE_DIRECT_LEVERS) and stop."""
    from rocke.runtime import code_object

    seen = {}

    def fake(serialized, isa):
        seen["ir"], seen["levers"] = serialized, os.environ.get("ROCKE_DIRECT_LEVERS")
        raise _Captured()

    monkeypatch.setattr(code_object, "build_direct_rocke_ir_hsaco", fake)
    return seen


def test_direct_compile_reads_scopes_defaults_and_kernel_levers(monkeypatch):
    seen = _capture_direct(monkeypatch)
    monkeypatch.delenv("ROCKE_DIRECT_DEFAULT_LEVERS", raising=False)
    monkeypatch.setenv("ROCKE_DIRECT_LEVERS", "storewait")
    k = _scoped({"g": "moe_r1"}, direct_levers="minmax")
    with pytest.raises(_Captured):
        C.compile_kernel(k, arch="gfx950", backend="python", code_object_backend="direct")
    assert seen["levers"] == "storewait,acqfence,bufoffany,minmax"
    assert f'dlevers = s:"{",".join(K.RUST_LEVER_SETS["moe_r1"])}"' in seen["ir"]
    assert "rocke." not in seen["ir"] and "direct_scopes" not in seen["ir"] and "direct_levers" not in seen["ir"]
    assert os.environ["ROCKE_DIRECT_LEVERS"] == "storewait"  # restored
    monkeypatch.setenv("ROCKE_DIRECT_DEFAULT_LEVERS", "")
    monkeypatch.delenv("ROCKE_DIRECT_LEVERS")
    with pytest.raises(_Captured):
        C.compile_kernel(T.build_static(), arch="gfx950", backend="python", code_object_backend="direct")
    assert seen["levers"] is None and "ROCKE_DIRECT_LEVERS" not in os.environ


def test_backend_selection(monkeypatch):
    monkeypatch.delenv("ROCKE_CODEOBJ_BACKEND", raising=False)
    k = T.build_static()
    assert C._selected_code_object_backend(None, k) == "comgr"
    C.request_code_object_backend(k, "direct")
    assert C._selected_code_object_backend(None, k) == "direct"
    assert C._selected_code_object_backend("comgr", k) == "comgr"
    monkeypatch.setenv("ROCKE_CODEOBJ_BACKEND", "direct")
    assert C._selected_code_object_backend(None, T.build_static()) == "direct"
    with pytest.raises(ValueError):
        C.request_code_object_backend(k, "llvm")


# ---------------------------------------------------- D0.6 per-family default


def test_rust_scope_backend_picks_direct_only_for_winning_families():
    assert K.RUST_DIRECT_FAMILIES == ("b3", "moe_r1")
    # framework kernels: no framework-hosted family has won yet (fix round: Q2 in the framework layer loses direct)
    assert K.RUST_FRAMEWORK_DIRECT_FAMILIES == ()
    assert K.rust_scope_backend(K.RustScope.of({"g": "b3", "r": "moe_r1"}), ("g", "r")) == "comgr"
    assert K.rust_scope_backend(K.RustScope.of({"g": "b3", "r": "b3"}), ("g", "r")) == "comgr"
    w = K.RUST_DIRECT_FAMILIES
    assert K.rust_scope_backend(K.RustScope.of({"g": "b3", "r": "moe_r1"}), ("g", "r"), w) == "direct"
    assert K.rust_scope_backend(K.RustScope.of({"g": "b3", "r": "head"}), ("g", "r"), w) == "comgr"
    assert K.rust_scope_backend(K.RustScope.of({"g": "b3"}), ("g", "r"), w) == "comgr"  # r unscoped
    assert K.rust_scope_backend(K.RustScope.of({"g": "moe_r4", "r": "b3"}), ("g", "r"), w) == "comgr"
    assert K.rust_scope_backend(K.RustScope.of({"g": "minmax+selxor", "r": "b3"}), ("g", "r"), w) == "comgr"
    assert K.rust_scope_backend(K.RustScope.of({"g": "minmax", "r": "b3"}, {"g": "b3"}), ("g", "r"), w) == "direct"
    assert K.rust_scope_backend(None, ("g",), w) == "comgr"


def test_framework_auto_backend_marks_without_changing_ir(monkeypatch):
    plain = T.build_static(FrameworkKnobs(rust_scope=K.RustScope.of({"g": "b3", "r": "b3"})))
    auto = T.build_static(FrameworkKnobs(rust_scope=K.RustScope.of({"g": "b3", "r": "b3"}), code_object_backend="auto"))
    assert C.requested_code_object_backend(plain) is None
    assert C.requested_code_object_backend(auto) == "comgr"  # no framework-hosted family has won yet
    monkeypatch.setattr(K, "RUST_FRAMEWORK_DIRECT_FAMILIES", ("b3",))
    auto = T.build_static(FrameworkKnobs(rust_scope=K.RustScope.of({"g": "b3", "r": "b3"}), code_object_backend="auto"))
    assert C.requested_code_object_backend(auto) == "direct"
    assert serialize(auto) == serialize(plain) and auto.name == plain.name
    head = T.build_static(FrameworkKnobs(rust_scope=K.RustScope.of({"g": "b3", "r": "head"}), code_object_backend="auto"))
    assert C.requested_code_object_backend(head) == "comgr"
    with pytest.raises(ValueError, match="rust_scope"):
        FrameworkKnobs(code_object_backend="auto")
    with pytest.raises(ValueError):
        FrameworkKnobs(code_object_backend="llvm")


def test_rust_family_for_standalone_kernels():
    k = T.build_static()
    before = serialize(k)
    assert K.rust_family(k, "b3") is k
    assert C.requested_code_object_backend(k) == "direct"
    assert k.attrs["direct_levers"] == "+".join(K.RUST_LEVER_SETS["b3"])
    assert serialize(C.direct_source(k)) == before  # the attribute never reaches the direct compiler's IR
    h = T.build_static()
    K.rust_family(h, "head")
    K.rust_family(h, "moe_r4")
    assert C.requested_code_object_backend(h) is None and "direct_levers" not in h.attrs
    with pytest.raises(ValueError):
        K.rust_family(h, "nosuch")


# ------------------------------------------------- nothing reaches COMGR builds


def test_knobs_are_invisible_to_comgr_builds():
    base = T.build_static()
    for fk in (
        FrameworkKnobs(code_object_backend="direct"),
        FrameworkKnobs(code_object_backend="comgr"),
        FrameworkKnobs(stage_levers="g=nofence;r=-minmax"),
        FrameworkKnobs(direct_levers="minmax"),
        FrameworkKnobs(rust_scope=K.RustScope.of({"g": "b3", "r": "b3"}), code_object_backend="auto"),
    ):
        k = T.build_static(fk)
        assert _ll(k) == _ll(base) and k.name == base.name
    # stage_levers mark the stage's ops (explicit dlevers); empty knobs are byte-identical IR
    sl = T.build_static(FrameworkKnobs(stage_levers="g=nofence"))
    assert any(op.attrs.get("dlevers") == "nofence" for op in L.ops_in_order(sl))
    assert serialize(T.build_static(FrameworkKnobs(stage_levers=""))) == serialize(base)


def test_new_fields_stay_out_of_repr_and_pair_ids():
    for f in dataclasses.fields(FrameworkKnobs):
        if f.name in ("code_object_backend", "irops", "stage_levers", "direct_levers"):
            assert f.repr is False, f.name
    fk = MF.FusedKnobs()
    assert "phase_levers" not in repr(fk)
    assert repr(dataclasses.replace(fk, phase_levers="dn=nofence")) == repr(fk)


# ------------------------------------------- with the direct library (cluster)


def _direct_lib():
    from rocke.runtime import code_object as CO

    if CO._direct_library_path() is None:
        pytest.skip("no direct library (ROCKE_CODEOBJ_DIRECT_LIB)")
    if "gfx950" not in CO.direct_supported_targets():
        pytest.skip("the direct library does not compile for gfx950")


def _direct(k):
    art = C.compile_kernel(k, arch="gfx950", backend="python", code_object_backend="direct")
    assert art.code_object_backend == "direct", art.direct_fallback
    return bytes(art.hsaco)


def test_named_scopes_build_what_explicit_scopes_build(monkeypatch):
    """rust_scope (rocke.scope -> dlevers in the direct backend) and stage_levers (dlevers from the builder) put the
    same levers on the same ops: the same object; and the scope reaches the compiler (it changes the object)."""
    _direct_lib()
    monkeypatch.delenv("ROCKE_DIRECT_LEVERS", raising=False)
    monkeypatch.setenv("ROCKE_DIRECT_STRICT", "1")
    lv = "nofence+nopause+storewait+wgfence+vmemfirst+loadfirst+litreuse+litshare+minmax"
    named = _direct(T.build_static(FrameworkKnobs(rust_scope=K.RustScope.of({"g": lv, "r": lv}))))
    explicit = _direct(T.build_static(FrameworkKnobs(stage_levers=f"g={lv};r={lv}")))
    plain = _direct(T.build_static())
    assert named == explicit
    assert named != plain


def test_policy_object_is_the_env_lever_object(monkeypatch):
    """rust_family(kernel, "b3") compiles (backend from the mark) to what ROCKE_DIRECT_LEVERS=<the B3 set> builds."""
    _direct_lib()
    monkeypatch.delenv("ROCKE_DIRECT_LEVERS", raising=False)
    monkeypatch.delenv("ROCKE_CODEOBJ_BACKEND", raising=False)
    art = C.compile_kernel(K.rust_family(T.build_static(), "b3"), arch="gfx950", backend="python")
    assert art.code_object_backend == "direct", art.direct_fallback
    monkeypatch.setenv("ROCKE_DIRECT_LEVERS", ",".join(K.RUST_LEVER_SETS["b3"]))
    assert bytes(art.hsaco) == _direct(T.build_static())
