# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Rust direct code-object compiler pilot on the Qwen3.8 builds
(``kernels.gfx950.qwen38_mono.direct_pilot``; framework knob
``FrameworkKnobs.code_object_backend``).

CPU: the knob defaults off, validates, changes no IR and no kernel name, and
only marks the built ``KernelDef``; ``compile_kernel`` takes the mark when the
call names no backend, an explicit argument wins.

Direct library (``ROCKE_CODEOBJ_DIRECT_LIB``): every representative build the
direct compiler does not lower is refused with ``ROCKE_DIRECT_UNSUPPORTED``
at the expected first construct (never a parse / malformed error); past the
first refusal (placeholders, ``neutralize_census``) every later construct is
refused cleanly too, down to an object or the recorded end state. Since R2
the MXFP4 head's logits builds compile as they are, and the R2 pilots
(``direct_pilot.pilot_builds``) compile in their ``irops`` form, the IR-only
form of the shipped builds.

gfx950 + COMGR: ``compile_kernel(..., code_object_backend="direct")`` and the
framework knob fall back to COMGR, say so (``code_object_backend == "comgr"``,
``direct_fallback``), and build the same code object byte for byte as the
knob-off compile.

``Q38_DIRECT_STRICT=1`` turns the skips (no library, no gfx950) into failures.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from kernels.gfx950.qwen38_mono import direct_pilot as DP
from kernels.gfx950.qwen38_mono import toy_chain as T
from kernels.gfx950.qwen38_mono.framework import CODE_OBJECT_BACKENDS, FrameworkKnobs
from rocke.core.ir import I32, IRBuilder
from rocke.core.ir_print import print_ir
from rocke.core.ir_serialize import serialize
from rocke.helpers import compile as C

STRICT = os.environ.get("Q38_DIRECT_STRICT") == "1"
ISA = "amdgcn-amd-amdhsa--gfx950"
UNSUPPORTED = "direct rocKE IR compiler rejected the kernel: ROCKE_DIRECT_UNSUPPORTED: "

ASM = 'tile.inline_asm other than an empty or comment-only "=v,0" / "=s,0" pin'
# The first construct the direct compiler refuses in each build; None: the
# build compiles as it is.
FIRST_REFUSAL = {
    "toy_default": ASM,
    "toy_best_d4w8_l2": ASM,
    "toy_best_d8w0": ASM,
    "b3_r1_default": 'op "math.exp" is not lowered',
    "b3_r1_tuned": 'op "math.exp" is not lowered',
    "b3_r4s_tuned": 'op "math.exp" is not lowered',
    "moe_fused_r1": 'op "math.exp" is not lowered',
    "moe_fused_r4": 'kernel symbol "qwen38_fmoe_e512_r4_dninflight0_combinebase-1_',
    "moe_suffix_r1": 'op "math.exp" is not lowered',
    "moe_suffix_r4": 'op "math.exp" is not lowered',
    "head_logits": None,
    "head_fused_argmax": ASM,
    "head_nt2": None,
    "mx_mfma_probe": 'a pointer to vec<i32x8> ("ptr<vec<i32x8>,global>")',
}
REFUSED = [label for label, first in FIRST_REFUSAL.items() if first is not None]
COMPILED = [label for label, first in FIRST_REFUSAL.items() if first is None]
# Where ``neutralize_census`` ends: "compiled" (every construct the direct
# compiler cannot lower was refused cleanly, and the rest compiles), or the
# known end state.
CENSUS_FINAL = {label: "compiled" for label in REFUSED}
# A pointer-to-vector kernel argument has no placeholder (the probe kernel
# stays on COMGR: research rust-direct.md 2.2).
CENSUS_FINAL["mx_mfma_probe"] = "stuck: no rule neutralizes the last refusal"
# The kernel symbol is the only refusal the census cannot neutralize either.
CENSUS_FINAL["moe_fused_r4"] = "stuck: no rule neutralizes the last refusal"

BUILDS = dict(DP.representative_builds())
PILOTS = dict(DP.pilot_builds())
IROPS = [label for label in DP.R2_PILOTS if label.endswith("_irops")]


def _skip(why: str):
    if STRICT:
        pytest.fail(why)
    pytest.skip(why)


def _direct_lib():
    from rocke.runtime import code_object as CO

    if CO._direct_library_path() is None:
        _skip("no direct library (ROCKE_CODEOBJ_DIRECT_LIB)")
    if "gfx950" not in CO.direct_supported_targets():
        _skip("the direct library does not compile for gfx950")
    return CO


def _require_gfx950():
    if not Path("/dev/kfd").exists():
        _skip("requires a gfx950 GPU node (COMGR)")
    from rocke.runtime.hip_module import get_device_arch

    if get_device_arch(0) != "gfx950":
        _skip("requires gfx950")


# ----------------------------------------------------------------- CPU


def test_tables_cover_every_build():
    assert set(FIRST_REFUSAL) == set(BUILDS)
    assert set(DP.FRAMEWORK_BUILDS) <= set(BUILDS)


def test_knob_defaults_off_and_validates():
    assert FrameworkKnobs().code_object_backend == ""
    assert CODE_OBJECT_BACKENDS == ("", "comgr", "direct", "auto")  # "auto": D0.6 (WP-R3, test_qwen38_wpr3.py)
    for value in ("comgr", "direct"):
        assert FrameworkKnobs(code_object_backend=value).suffix() == FrameworkKnobs().suffix()
    with pytest.raises(ValueError, match="code_object_backend"):
        FrameworkKnobs(code_object_backend="llvm")


@pytest.mark.parametrize("label", DP.FRAMEWORK_BUILDS)
def test_knob_changes_no_ir(label):
    off = BUILDS[label]()
    assert C.requested_code_object_backend(off) is None
    for value, build in (
        ("direct", dict(DP.with_code_object_backend("direct"))[label]),
        ("comgr", dict(DP.with_code_object_backend("comgr"))[label]),
    ):
        on = build()
        assert C.requested_code_object_backend(on) == value
        assert on.name == off.name
        assert on.attrs == off.attrs
        assert print_ir(on) == print_ir(off)
        assert serialize(on) == serialize(off)


def _tiny(name="rk_direct_pilot_tiny"):
    b = IRBuilder(name)
    b.param("n", I32)
    b.ret()
    return b.kernel


def test_request_mark_is_not_ir():
    k = _tiny()
    text = serialize(k)
    assert C.request_code_object_backend(k, "direct") is k
    assert C.requested_code_object_backend(k) == "direct"
    assert serialize(k) == text and k == _tiny()
    with pytest.raises(ValueError):
        C.request_code_object_backend(k, "")
    with pytest.raises(ValueError):
        C.request_code_object_backend(k, "hipcc")


def test_compile_kernel_takes_the_mark(monkeypatch):
    """The call's backend, else the kernel's mark, else ROCKE_CODEOBJ_BACKEND, else COMGR (WP-R3: the selection is
    ``compile._selected_code_object_backend``; an unmarked kernel never loads the code-object facade)."""
    monkeypatch.delenv("ROCKE_CODEOBJ_BACKEND", raising=False)
    marked = C.request_code_object_backend(_tiny(), "direct")
    for kernel, backend, want in (
        (_tiny(), None, "comgr"),
        (marked, None, "direct"),
        (marked, "comgr", "comgr"),
        (C.request_code_object_backend(_tiny(), "comgr"), None, "comgr"),
    ):
        assert C._selected_code_object_backend(backend, kernel) == want
    monkeypatch.setenv("ROCKE_CODEOBJ_BACKEND", "direct")
    assert C._selected_code_object_backend(None, _tiny()) == "direct"
    monkeypatch.setenv("ROCKE_CODEOBJ_BACKEND", "llvm")
    with pytest.raises(ValueError):
        C._selected_code_object_backend(None, _tiny())


def test_toy_knob_survives_spec_replace():
    spec = T.best_spec(4, 8, layers=2)
    on = dataclasses.replace(spec, knobs=dataclasses.replace(spec.knobs, code_object_backend="direct"))
    assert on.name() == spec.name()
    assert C.requested_code_object_backend(T.build_toy_chain(on)) == "direct"


# ------------------------------------------------------ direct library


# WP-R3: R1's per-build refusal tables (FIRST_REFUSAL / CENSUS_FINAL) describe the R1 crate, which refused these
# builds at their first unsupported construct. The rust track's crate lowers them all (math.exp, the scaled MFMAs,
# the asm through direct_lift + the inline-asm assembler, symbols with "-"), so R1's census tests are superseded by
# the coverage test below and by the rust-track census (rust-track/rt_census*.py, the WP-R3 direct census).
# Pointer-to-vector kernel arguments are still refused (the scaled-MFMA probe stays on COMGR).
STILL_REFUSED = {"mx_mfma_probe": "ptr<vec<i32x8>,global>"}


@pytest.mark.parametrize("label", sorted(BUILDS))
def test_every_representative_build_compiles_direct_or_is_refused_cleanly(label, monkeypatch):
    """compile_kernel(direct) in strict mode: every representative build compiles with the direct backend (the asm
    lifted), except the known refusals, which raise DirectUnsupported (never a parse / malformed error)."""
    CO = _direct_lib()
    monkeypatch.setenv("ROCKE_DIRECT_STRICT", "1")
    kernel = BUILDS[label]()
    if label in STILL_REFUSED:
        with pytest.raises(CO.DirectUnsupported) as refused:
            C.compile_kernel(kernel, arch="gfx950", backend="python", code_object_backend="direct")
        assert str(refused.value).startswith(UNSUPPORTED) and STILL_REFUSED[label] in str(refused.value)
        return
    art = C.compile_kernel(kernel, arch="gfx950", backend="python", code_object_backend="direct")
    assert art.code_object_backend == "direct" and art.direct_fallback == ""
    assert bytes(art.hsaco).startswith(b"ELF")


# ------------------------------------------------------ gfx950 + COMGR


def _compile_pair(build_off, build_on, **kw):
    off = C.compile_kernel(build_off(), arch="gfx950", backend="python")
    on = C.compile_kernel(build_on(), arch="gfx950", backend="python", **kw)
    return off, on


@pytest.mark.parametrize("label", COMPILED + IROPS)
def test_build_compiles_directly(label):
    CO = _direct_lib()
    kernel = (BUILDS.get(label) or PILOTS[label])()
    hsaco = CO.build_direct_rocke_ir_hsaco(serialize(kernel), isa=ISA)[0]
    assert hsaco[:4] == b"\x7fELF" and kernel.name.encode() in hsaco


@pytest.mark.parametrize("label", REFUSED)
def test_direct_falls_back_to_the_same_object(label):
    _direct_lib()
    _require_gfx950()
    # One KernelDef for both arms: a rebuild in the same process is not always the same IR text
    # (``moe_fused._guarded`` names values from a module-global counter).
    kernel = BUILDS[label]()
    off, on = _compile_pair(lambda: kernel, lambda: kernel, code_object_backend="direct")
    assert off.code_object_backend == "comgr" and off.direct_fallback == ""
    assert on.code_object_backend == "comgr"
    assert on.direct_fallback.startswith(UNSUPPORTED), on.direct_fallback
    assert FIRST_REFUSAL[label] in on.direct_fallback
    assert on.llvm_text == off.llvm_text
    assert on.hsaco == off.hsaco


@pytest.mark.parametrize("value", ["direct", "comgr"])
@pytest.mark.parametrize("label", DP.FRAMEWORK_BUILDS)
def test_framework_knob_compiles_the_same_object(label, value):
    _direct_lib()
    _require_gfx950()
    off, on = _compile_pair(BUILDS[label], dict(DP.with_code_object_backend(value))[label])
    assert on.code_object_backend == "comgr"
    if value == "direct":
        assert on.direct_fallback.startswith(UNSUPPORTED), on.direct_fallback
        assert FIRST_REFUSAL[label] in on.direct_fallback
    else:
        assert on.direct_fallback == ""
    assert on.hsaco == off.hsaco


# ------------------------------------------------------------ R2: irops


def _asm_templates(kernel) -> list[str]:
    import re

    return re.findall(r'tile\.inline_asm [^\n]*template = s:"([^"]*)"', serialize(kernel))


def test_irops_knob_defaults_off():
    assert FrameworkKnobs().irops is False
    assert "irops" not in FrameworkKnobs().suffix()
    assert FrameworkKnobs(irops=True).suffix().endswith("irops")


@pytest.mark.parametrize("label", IROPS)
def test_irops_pilots_keep_only_fresh_pins(label):
    """The IR-only forms hold no inline asm but ``device_ops.fresh``'s pins;
    the shipped forms are named differently and do hold device-op asm."""
    on, off = PILOTS[label](), PILOTS[label[: -len("_irops")]]()
    assert on.name != off.name
    assert set(_asm_templates(on)) == {"; fresh $0"}
    assert set(_asm_templates(off)) > {"; fresh $0"}


def test_irops_head_lowers_to_the_asm_heads_llvm():
    """The head's ``irops`` (scoped atomics as IR ops) is the asm build's LLVM
    text, the kernel name aside: a COMGR build of either is the same code."""
    from kernels.gfx950 import mx_decode_gemv as M
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    for fsync in M.FSYNCS:
        spec = M.MxDecodeGemvSpec(**DP.HYBRID_HEAD, fsync=fsync, ctas=256 if fsync == "counter" else 243)
        off = M.build_mx_decode_gemv(spec)
        on = M.build_mx_decode_gemv(dataclasses.replace(spec, irops=True))
        assert on.name == off.name.replace("_pyonly", "_irTrue_pyonly")
        a = lower_kernel_to_llvm(off, arch="gfx950", llvm_flavor="llvm22")
        b = lower_kernel_to_llvm(on, arch="gfx950", llvm_flavor="llvm22")
        assert b.replace(on.name, off.name) == a, fsync


@pytest.mark.parametrize("label", ["toy_best_d4w8_l2_irops", "toy_best_d8w0_irops"])
def test_irops_toy_differs_from_the_asm_toy_only_in_pointers(label):
    """The toy chain's ``irops`` LLVM text differs from the shipped build's only
    where a runtime pointer is made (``inttoptr`` for the ``s_mov_b64`` asm):
    the XCD id, sleeps and the L2-touch prefetch lower to the same asm."""
    import re

    from rocke.core.lower_llvm import lower_kernel_to_llvm

    on, off = PILOTS[label](), PILOTS[label[: -len("_irops")]]()
    a = lower_kernel_to_llvm(off, arch="gfx950", llvm_flavor="llvm22").splitlines()
    b = lower_kernel_to_llvm(on, arch="gfx950", llvm_flavor="llvm22").replace(on.name, off.name).splitlines()
    assert len(a) == len(b)
    asm_ptr = re.compile(r'%aptr(\d+) = call ptr addrspace\(1\) asm "s_mov_b64 \$0, \$1", "=s,s"\(i64 (%\w+)\)$')
    int_ptr = re.compile(r"%iptr(\d+) = inttoptr i64 (%\w+) to ptr addrspace\(1\)$")
    differ = 0
    for x, y in zip(a, b):
        if x == y:
            continue
        mx, my = asm_ptr.search(x), int_ptr.search(y)
        if mx and my:
            assert mx.groups() == my.groups(), (x, y)
            differ += 1
            continue
        assert x.replace("%aptr", "%iptr") == y, (x, y)
    assert differ > 0
