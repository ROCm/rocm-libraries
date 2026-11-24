# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which dense attention bodies read their 12 stride kernargs, on the IR (no GPU).

Every gfx942 and gfx950 dense body declares the 12 trailing Q/K/V/O stride kernargs
(``DENSE_STRIDE_NAMES``), and ``body_serves_strides(spec)`` says whether the body
reads them: ``run_attention_dense_torch`` admits a strided view only when it does.
That predicate is written by hand, apart from the code that forms the addresses, so
these tests hold it to the lowered kernel:

* on a body that serves strides, each stride of tensor T reaches the address of a
  load or store of T, and of no other tensor. A KV-segment body is the exception for
  O: the combine kernel writes O packed (the stride admission refuses a strided O
  there), so its O strides reach no access;
* on a body that does not, no stride reaches any access.

A served body that dropped, say, ``stride_v_head`` from its V addresses would
otherwise launch a head-major V at the wrong addresses with no error, and only a GPU
lane would see it.

The probe follows SSA def-use edges forward from each kernel parameter. A load's
result is data, not an address, so the walk stops at every memory access; an access
reaches a parameter when one of its operands is derived from it. The record count of
a buffer resource is a size, not an address, so of ``make.buffer.rsrc`` the walk
follows only the base pointer: a served body sizes each K and V slab resource with
the tensor's token stride, and that use alone must not count as addressing a row.
:func:`test_the_probe_sees_a_token_stride_missing_from_the_offsets` keeps the
probe honest about it.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pytest

from kernels.common._attention_dense_strides import (
    DENSE_STRIDE_NAMES,
    DENSE_STRIDE_TENSORS,
    packed_dense_strides,
)
from kernels.gfx942 import attention_dense as k942
from kernels.gfx950 import attention_dense as k950

_MODULES = {"gfx942": k942, "gfx950": k950}
_SPEC_TYPES = {
    "gfx942": k942.Gfx942AttentionDenseSpec,
    "gfx950": k950.Gfx950AttentionDenseSpec,
}

_BASE = {
    "batch": 1,
    "seqlen_q": 512,
    "seqlen_kv": 512,
    "num_query_heads": 8,
    "num_kv_heads": 2,
    "head_size": 128,
    "causal": True,
    "dtype": "bf16",
}
_PAGED = {"dtype": "fp16", "paged": True, "block_size": 16, "num_kv_blocks": 32}
_GQA_32_8_FP16 = {"dtype": "fp16", "num_query_heads": 32, "num_kv_heads": 8}
_PERSIST_950 = {"persistent": True, "num_persistent": 256}
_PERSIST_942 = {"persistent": True, "num_persistent": 304}
_KDB_942 = {**_PERSIST_942, "persist_decode": "bt_hkv_minor"}

# (arch, body) -> spec fields on top of _BASE: the bodies that serve strides and the
# ones that keep packed addressing, on both architectures.
_BODIES = {
    ("gfx950", "grid"): {},
    ("gfx950", "grid_d64_fp16"): {"head_size": 64, "dtype": "fp16"},
    ("gfx950", "noncausal"): {"causal": False},
    ("gfx950", "sliding_window"): {"sliding_window": 128},
    ("gfx950", "bottom_right"): {"seqlen_kv": 1024, "causal_bottom_right": True},
    ("gfx950", "lse"): {"emit_lse": True},
    ("gfx950", "sinks"): {"use_sinks": True},
    ("gfx950", "kv_segments"): {"kv_segments": 2},
    ("gfx950", "persistent"): _PERSIST_950,
    ("gfx950", "persistent_hkv_major"): {**_PERSIST_950, "persist_decode": "hkv_major"},
    ("gfx950", "persistent_window_lse"): {
        **_PERSIST_950,
        "sliding_window": 128,
        "emit_lse": True,
    },
    ("gfx950", "wide_dma"): {
        **_GQA_32_8_FP16,
        "persistent": True,
        "num_persistent": 16,
        "persist_decode": "qb_major",
        "wide_lds_dma": True,
    },
    ("gfx950", "gqa_pair"): {
        **_GQA_32_8_FP16,
        "persistent": True,
        "num_persistent": 16,
        "persist_decode": "gqa_pair",
        "wide_lds_dma": True,
    },
    ("gfx950", "bias"): {"use_additive_bias": True},
    ("gfx950", "ragged"): {"seqlen_q": 500, "seqlen_kv": 500, "ragged": True},
    ("gfx950", "varlen"): {"varlen": True},
    ("gfx950", "paged"): _PAGED,
    ("gfx950", "fp8_kv"): {"kv_storage_dtype": "fp8e4m3"},
    ("gfx942", "grid"): {},
    ("gfx942", "grid_d64_fp16"): {"head_size": 64, "dtype": "fp16"},
    # BF16 D128 without a mask: the body with the per-row LDS destination.
    ("gfx942", "noncausal"): {"causal": False},
    ("gfx942", "sliding_window"): {"sliding_window": 128},
    ("gfx942", "bottom_right"): {"seqlen_kv": 1024, "causal_bottom_right": True},
    ("gfx942", "lse"): {"emit_lse": True},
    ("gfx942", "sinks"): {"use_sinks": True},
    ("gfx942", "kv_segments"): {"kv_segments": 2},
    ("gfx942", "persistent"): _PERSIST_942,
    ("gfx942", "persistent_hkv_major"): {**_PERSIST_942, "persist_decode": "hkv_major"},
    ("gfx942", "persistent_window_lse"): {
        **_PERSIST_942,
        "sliding_window": 128,
        "emit_lse": True,
    },
    ("gfx942", "bias"): {"use_additive_bias": True},
    ("gfx942", "bias_persistent"): {**_PERSIST_942, "use_additive_bias": True},
    ("gfx942", "kdb"): {**_KDB_942, "lds_num_buffers": 2},
    ("gfx942", "vpf"): {**_KDB_942, "v_reg_prefetch": True},
    ("gfx942", "paged"): _PAGED,
}

_DEF = re.compile(r"^\s+(%[-\w.]+) = (.+)$")
_VALUE = re.compile(r"%[-\w.]+")
_MEMORY = re.compile(
    r"^(load|store|atomicrmw|cmpxchg)\b"
    r"|\bcall\b[^@]*@llvm\.amdgcn\.[\w.]*(load|store|atomic|ds\.read)"
)
# The first argument of a buffer resource call: its base pointer. The others are the
# stride, the record count and the flags.
_RESOURCE_BASE = re.compile(
    r"\bcall\b[^@]*@llvm\.amdgcn\.make\.buffer\.rsrc\.[\w.]*\(([^,]*),"
)


def _spec(arch: str, body: str):
    return _SPEC_TYPES[arch](**{**_BASE, **_BODIES[(arch, body)]})


def _lower(arch: str, spec) -> str:
    """The LLVM IR text of ``spec``'s kernel (the Python lowerer, a fixed flavor)."""
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    kernel = _MODULES[arch].build_attention_dense(spec, arch=arch)
    return _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor="llvm22")


def _address_reach(ir: str) -> tuple[dict[str, set[str]], set[str]]:
    """For each stride kernarg, the tensors (``q``, ``k``, ``v``, ``o``) whose
    pointer kernarg feeds a memory access that the stride also feeds; and the
    tensors whose pointer feeds any access at all. A buffer resource carries only
    its base pointer's edge: its record count sizes the accesses, it does not
    address them."""
    users: dict[str, set[str]] = defaultdict(set)
    accesses: list[list[str]] = []
    for line in ir.splitlines():
        if not line.startswith((" ", "\t")):
            continue
        match = _DEF.match(line)
        rhs = match.group(2) if match else line.strip()
        operands = _VALUE.findall(rhs)
        if _MEMORY.search(rhs):
            accesses.append(operands)
        elif match:
            resource = _RESOURCE_BASE.search(rhs)
            if resource:
                operands = _VALUE.findall(resource.group(1))
            for operand in operands:
                users[operand].add(match.group(1))

    def touched_by(param: str) -> set[int]:
        seen, todo = {param}, [param]
        while todo:
            for user in users.get(todo.pop(), ()):
                if user not in seen:
                    seen.add(user)
                    todo.append(user)
        return {i for i, ops in enumerate(accesses) if seen.intersection(ops)}

    by_tensor = {t: touched_by(f"%{t}_ptr") for t in DENSE_STRIDE_TENSORS}
    reach = {
        name: {t for t, hits in by_tensor.items() if hits & touched_by(f"%{name}")}
        for name in DENSE_STRIDE_NAMES
    }
    return reach, {t for t, hits in by_tensor.items() if hits}


def _tensor(stride_name: str) -> str:
    """``stride_k_head`` -> ``k``."""
    return stride_name.split("_")[1]


def test_the_matrix_has_both_kinds_of_body_on_each_arch():
    """Guard against vacuity: each architecture has bodies on both sides."""
    for arch, module in _MODULES.items():
        verdicts = {
            module.body_serves_strides(_spec(a, body))
            for a, body in _BODIES
            if a == arch
        }
        assert verdicts == {True, False}, arch


@pytest.mark.parametrize(
    ("arch", "body"), [pytest.param(a, b, id=f"{a}-{b}") for a, b in _BODIES]
)
def test_a_body_reads_its_strides_iff_it_serves_them(arch, body):
    """``body_serves_strides(spec)`` matches the IR: each stride feeds the accesses
    of its own tensor and no other when it is True, and no access when it is False;
    the 12 strides close the kernel ABI either way."""
    spec = _spec(arch, body)
    served = _MODULES[arch].body_serves_strides(spec)
    ir = _lower(arch, spec)
    (define,) = [ln for ln in ir.splitlines() if ln.startswith("define ")]
    params = re.findall(r"%([-\w.]+)", define)
    assert tuple(params[-12:]) == DENSE_STRIDE_NAMES, "the strides do not close the ABI"
    assert tuple(re.findall(r"\bi32 %(stride_\w+)", define)) == DENSE_STRIDE_NAMES
    # The combine kernel writes O of a KV-segment body, packed.
    addressed = set(DENSE_STRIDE_TENSORS)
    if spec.kv_segments > 1:
        addressed.discard("o")
    reach, reached = _address_reach(ir)
    assert addressed <= reached, f"the probe sees no access of {addressed - reached}"
    want = {
        name: {_tensor(name)} if served and _tensor(name) in addressed else set()
        for name in DENSE_STRIDE_NAMES
    }
    wrong = {name: sorted(got) for name, got in reach.items() if got != want[name]}
    assert not wrong, (
        f"body_serves_strides is {served}, but these strides feed the accesses of "
        f"the tensors shown (want {'its own tensor' if served else 'none'}): {wrong}"
    )


def _sizing_only(ir: str, stride: str, value: int) -> tuple[str, int, int]:
    """``ir`` with each read of ``%<stride>`` replaced by the constant ``value``,
    except the reads that compute a buffer resource's record count. Returns the
    edited IR and the number of reads kept and replaced."""
    defs = {m.group(1): m.group(2) for m in map(_DEF.match, ir.splitlines()) if m}
    sizing: set[str] = set()
    todo = [
        operand
        for rhs in defs.values()
        if (resource := _RESOURCE_BASE.search(rhs))
        for operand in _VALUE.findall(rhs[resource.end() :])
    ]
    while todo:
        name = todo.pop()
        if name in defs and name not in sizing:
            sizing.add(name)
            todo.extend(_VALUE.findall(defs[name]))
    read = re.compile(rf"%{stride}\b")
    lines, kept, replaced = [], 0, 0
    for line in ir.splitlines():
        if line.startswith((" ", "\t")) and read.search(line):
            match = _DEF.match(line)
            if match and match.group(1) in sizing:
                kept += 1
            else:
                line = read.sub(str(value), line)
                replaced += 1
        lines.append(line)
    return "\n".join(lines), kept, replaced


@pytest.mark.parametrize("stride", ["stride_k_token", "stride_v_token"])
@pytest.mark.parametrize(
    ("arch", "body"),
    [
        pytest.param(a, b, id=f"{a}-{b}")
        for a in _MODULES
        for b in ("grid", "persistent")
    ],
)
def test_the_probe_sees_a_token_stride_missing_from_the_offsets(arch, body, stride):
    """A served body whose K or V offsets step rows by the packed token stride,
    while the slab's record count still reads the stride kernarg, reads a head-major
    K or V at the wrong rows. Edited into the lowered IR, that stride must reach no
    access, so the main test fails on such a body: the record count alone does not
    address the tensor."""
    spec = _spec(arch, body)
    assert _MODULES[arch].body_serves_strides(spec), "not a body that serves strides"
    ir = _lower(arch, spec)
    assert _address_reach(ir)[0][stride] == {_tensor(stride)}
    mutant, kept, replaced = _sizing_only(
        ir, stride, packed_dense_strides(spec)[stride]
    )
    assert kept and replaced, (
        f"the edit kept {kept} reads of %{stride} (want the record count's) and "
        f"replaced {replaced} (want the offsets'); it no longer models the defect"
    )
    reach = _address_reach(mutant)[0][stride]
    assert not reach, (
        f"%{stride} feeds no load offset in the edited IR, yet the probe says it "
        f"reaches {sorted(reach)}: it follows the buffer resource's record count"
    )
