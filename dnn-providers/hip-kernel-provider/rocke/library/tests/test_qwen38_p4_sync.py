# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P4 kernels (``kernels.gfx950.qwen38_mono.probes.p4_sync``): CPU checks.

Every builder variant lowers with the Python engine (the probe family's
exemption: ``q38p4_`` names, ``backend="python"``); the emitted LLVM text
carries the pieces the GPU bench relies on: the LDS pad that keeps one CTA per
CU (an asm ``ds_write`` through the escaped address), the bounded poll loops,
agent-scope atomics, ``sc1`` data words. The GPU proofs (numerics, negative
controls, mutants, timing) run in ``hackathon-prep/qwen38/truemono/P4``.
"""

from __future__ import annotations

import pytest
from kernels.gfx950.qwen38_mono import mailbox_ir  # noqa: F401  (registers "ir")
from kernels.gfx950.qwen38_mono.probes import p4_sync as S
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers import mailbox as M

FLAVOR = "llvm22"
KN_IR2 = M.MailboxKnobs(impl="ir", numa_copies=2)
KN_ASM2 = M.MailboxKnobs(impl="asm", numa_copies=2)


def _ll(k):
    return _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor=FLAVOR)


def _common(ll):
    assert "ds_write_b32" in ll  # LDS pad kept
    assert "s_memrealtime" in ll


def test_counter_kernel():
    ll = _ll(S.build_counter_kernel())
    _common(ll)
    assert "global_atomic_add" in ll
    assert "global_store_dword $0, $1, $2 sc1" in ll
    assert "v_cmp_lt_u32_e64" in ll and "s_cmp_ge_u32" in ll  # bounded ballot poll
    assert S.build_counter_kernel(S.CounterSpec(tag="rc")).name.endswith("_rc")


@pytest.mark.parametrize("P", [160, 256])
@pytest.mark.parametrize(
    "kn,poll,mutant",
    [
        (KN_IR2, "mailbox", "none"),
        (KN_ASM2, "mailbox", "none"),
        (M.MailboxKnobs(impl="ir"), "mailbox", "none"),
        (KN_ASM2, "deadline", "none"),
        (KN_IR2, "mailbox", "skiplast"),
    ],
)
def test_a2a_kernel(P, kn, poll, mutant):
    k = S.build_a2a_kernel(S.A2ASpec(knobs=kn, producers=P, poll=poll, mutant=mutant))
    assert k.name.startswith("q38p4_a2a")
    ll = _ll(k)
    _common(ll)
    if poll == "deadline":
        assert "global_load_dwordx2 $0, $4, $5 sc1" in ll


def test_a2a_rejects():
    with pytest.raises(ValueError):
        S.build_a2a_kernel(S.A2ASpec(knobs=KN_IR2, poll="deadline"))
    with pytest.raises(ValueError):
        S.build_a2a_kernel(S.A2ASpec(knobs=KN_IR2, poll="bogus"))


def test_ticket_kernel():
    ll = _ll(S.build_ticket_kernel())
    _common(ll)
    assert "sc0 sc1" not in ll  # agent scope
    assert "offset:2048 sc1" in ll  # eighth XCD rank word


def test_ticket_kernel_xcd():
    k = S.build_ticket_kernel(S.TicketSpec(start="xcd"))
    assert k.name.endswith("_stxcd")
    ll = _ll(k)
    _common(ll)
    assert "s_mov_b64 exec, 1" in ll  # one returning atomic per CTA
    assert ll.count("v_readlane_b32") >= 8  # lane-parallel rank-word sum
    with pytest.raises(ValueError):
        S.build_ticket_kernel(S.TicketSpec(start="bogus"))


@pytest.mark.parametrize("rows", [1, 4, 16])
@pytest.mark.parametrize(
    "mode,kn,mutant",
    [
        ("pairs", KN_IR2, "none"),
        ("pairs", KN_ASM2, "none"),
        ("pairs", M.MailboxKnobs(impl="ir"), "none"),
        ("bulk", KN_IR2, "none"),
        ("pairs", KN_IR2, "tagoff"),
        ("bulk", KN_IR2, "noready"),
    ],
)
def test_xl_kernel(rows, mode, kn, mutant):
    k = S.build_xl_kernel(S.XlSpec(rows=rows, mode=mode, knobs=kn, mutant=mutant))
    ll = _ll(k)
    _common(ll)
    if mode == "bulk" and mutant == "none":
        assert "buffer_wbl2 sc1" in ll and "buffer_inv sc1" in ll
    if mutant == "noready":
        assert "buffer_inv sc1" not in ll


def test_xl_rejects_and_sizes():
    with pytest.raises(ValueError):
        S.build_xl_kernel(S.XlSpec(mode="pairs", mutant="noready"))
    with pytest.raises(ValueError):
        S.build_xl_kernel(S.XlSpec(mode="bulk", mutant="tagoff"))
    assert S.xl_words_per_producer(1) == 32
    assert S.xl_chunks_per_thread(1) == 5 and S.xl_chunks_per_thread(16) == 80
    assert S.numa_phys_pair(1023) == 1023 and S.numa_phys_pair(1024) == 2048


@pytest.mark.parametrize("rows", [1, 16])
def test_boundary_kernel(rows):
    ll = _ll(S.build_boundary_kernel(S.BoundarySpec(rows=rows)))
    assert "ds_write_b32" in ll
