# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-C5 fix round (``kernels.gfx950.qwen38_mono.large_batch.insitu_moe``): CPU checks of the MoE-down in-situ
all-reduce. Every bench build (TP2 buckets 16-128, default MoE arm, noreset twin, the MoE's slot mutant) builds, lowers,
passes the CCL peer-address lint; the combine pushes to every rank (no ``y`` store left), the consumer polls / resets /
sums; the launch's last CTA advances the exchange epoch; ``hz`` pads exactly the SGPR-operand VMEM asm; building the
in-situ kernel leaves the plain MoE's IR unchanged. GPU proofs: ``hackathon-prep/qwen38/truemono-lb/wp-c5-fix``."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from kernels.gfx950.qwen38_mono.ccl import lint as LI
from kernels.gfx950.qwen38_mono.large_batch import insitu as X
from kernels.gfx950.qwen38_mono.large_batch import insitu_moe as XM
from kernels.gfx950.qwen38_mono.large_batch import moe_lb as M
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python


def _ll(k):
    return _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor="llvm22")


CASES = [XM.MoeInsituKnobs(moe=M.MoeKnobs(tp=2, bucket=bk), ranks=n) for bk in (16, 32, 64, 128) for n in (2, 4)]
CASES += [XM.mutant(CASES[0]), replace(CASES[0], moe=replace(CASES[0].moe, mutant="slot")),
          XM.MoeInsituKnobs(moe=M.MoeKnobs(tp=2, bucket=64, combine="gbar"))]
# fix-round levers: the combine-unit-major arena layout and the poll pauses
CASES += [XM.MoeInsituKnobs(moe=M.MoeKnobs(tp=2, bucket=bk), lay="tok") for bk in (16, 128)]
CASES += [XM.MoeInsituKnobs(moe=M.MoeKnobs(tp=2, bucket=16), pause=p) for p in ("s_sleep1", "s_sleep16")]


def test_layout_default_and_names():
    """lay=auto (default) is the unit layout under combine=phase (dt even) and tok under combine=gbar; the name lists
    only a non-default layout."""
    d = XM.MoeInsituKnobs(moe=M.MoeKnobs(tp=2, bucket=16))
    assert d.layout == "unit" and XM.moe_insitu_name(d) == XM.moe_insitu_name(replace(d, lay="unit"))
    assert XM.moe_insitu_name(replace(d, lay="tok")).endswith("_ltok")
    g = XM.MoeInsituKnobs(moe=M.MoeKnobs(tp=2, bucket=16, combine="gbar"))
    assert g.layout == "tok" and "_l" not in XM.moe_insitu_name(g).split("gbar")[-1]
    assert _ll(XM.build_moe_insitu_kernel(d)) == _ll(XM.build_moe_insitu_kernel(replace(d, lay="unit")))
CASES += [XM.mutant(CASES[-4])]


def test_unit_layout_producer_and_consumer_cover_the_same_bytes():
    """lay=unit address algebra (the formulas of combine_cols / consume): every consumer lane's 16 bytes are exactly
    the two 8-byte pieces lane groups 2 gp and 2 gp + 1 of the same token and tile pushed, and every pushed piece
    is read once."""
    k = M.MoeKnobs(tp=2, bucket=32)
    dt, nrb = k.dt, k.nrb
    prod = {}
    for tt in range(2):
        for rb in range(nrb):
            cu = tt * nrb + rb
            for l16 in range(16):
                for g in range(4):
                    for d in range(dt):
                        off = cu * dt * 512 + l16 * 32 + g * 8 + d * 512
                        prod[off] = (tt * 16 + l16, rb * dt * 16 + d * 16 + g * 4)
    seen = set()
    for u in range(2 * nrb * dt // 2):
        cu, dp = divmod(u, dt // 2)
        for lane in range(64):
            d = dp * 2 + lane // 32
            l16, gp = (lane >> 1) & 15, lane & 1
            tt, rb = divmod(cu, nrb)
            tok, col = tt * 16 + l16, rb * dt * 16 + d * 16 + gp * 8
            inner = cu * dt * 512 + (d * 16 + l16) * 32 + gp * 16
            assert prod[inner] == (tok, col) and prod[inner + 8] == (tok, col + 4)
            seen.update((inner, inner + 8))
    assert seen == set(prod)


@pytest.mark.parametrize("kn", CASES, ids=[XM.moe_insitu_name(k) for k in CASES])
def test_builds_lowers_and_lints(kn):
    k = XM.build_moe_insitu_kernel(kn)
    assert k.name == XM.moe_insitu_name(kn)
    assert LI.lint_peer_access(k) == []
    text = _ll(k)
    N = kn.ranks
    assert f"global_store_dwordx2 $0, $1, ${N + 1} sc0 sc1" in text  # the combine's push to every rank
    assert ("global_store_dwordx4" in text) == (kn.mutant != "noreset")  # the consumer's lamport reset
    names = [p.name for p in k.params]
    assert names[: len(M.moe_params(kn.moe))] == M.moe_params(kn.moe)
    assert names[-(N + 6):] == [f"peer{r}" for r in range(N)] + ["ctrl", "arena", "dst", "rank", "dl_first", "dl"]


def test_combine_pushes_instead_of_storing_y():
    """The plain kernel's y stores (bf16 <2 x i32> buffer stores of the combine) are gone in the in-situ build."""
    kn = CASES[0]
    plain = _ll(M.build_moe_kernel(kn.moe))
    ins = _ll(XM.build_moe_insitu_kernel(kn))
    st = "@llvm.amdgcn.raw.ptr.buffer.store.v2i32("
    assert plain.count(st) > 0 and ins.count(st) == 0


def test_epoch_advance_appended_and_names():
    kn = CASES[0]
    text = _ll(XM.build_moe_insitu_kernel(kn))
    # the last CTA's stores of the next epoch / slot base (global_store_dword ... sc0 sc1 on the arena)
    assert text.count("global_store_dword $0, $1, $2 sc0 sc1") >= 3
    names = [XM.moe_insitu_name(k) for k in CASES]
    assert len(names) == len(set(names))
    assert XM.moe_insitu_name(replace(kn, hz=False)).endswith("_nohz")
    with pytest.raises(ValueError):
        XM.build_moe_insitu_kernel(XM.MoeInsituKnobs(moe=M.MoeKnobs(tp=2, bucket=16, combine="last")))
    with pytest.raises(ValueError):
        XM.MoeInsituKnobs(moe=M.MoeKnobs(tp=2, bucket=16, null="oob")).check()
    with pytest.raises(ValueError):
        XM.MoeInsituKnobs(moe=M.MoeKnobs(tp=2, bucket=16), mutant="slot").check()
    with pytest.raises(ValueError):  # the unit layout follows combine=phase's units
        XM.MoeInsituKnobs(moe=M.MoeKnobs(tp=2, bucket=16, combine="gbar"), lay="unit").check()


def test_hz_pads_and_plain_moe_unchanged():
    kn = CASES[0]
    before = hashlib.sha256(_ll(M.build_moe_kernel(kn.moe)).encode()).hexdigest()
    on, off = XM.build_moe_insitu_kernel(kn), XM.build_moe_insitu_kernel(replace(kn, hz=False))
    after = hashlib.sha256(_ll(M.build_moe_kernel(kn.moe)).encode()).hexdigest()
    assert before == after
    off.name = on.name
    assert X.pad_asm_vmem(off) > 0
    assert _ll(off) == _ll(on)
    assert X.pad_asm_vmem(on) == 0  # idempotent: everything already padded (moe_lb's own hz pads included)


def test_arena_map():
    for kn in CASES[:8]:
        lay = XM.arena_map(kn)
        assert lay["pay"] >= X.JUNK_OFF + X.JUNK_BYTES
        assert lay["pay"] + lay["region"] <= lay["nbytes"]
        assert lay["region"] == 2 * kn.ranks * kn.moe.bucket * M.H * 2


COMB = [XM.MoeInsituKnobs(moe=M.MoeKnobs(tp=2, bucket=bk), ranks=n, cons="combine", lay=ly)
        for bk in (16, 128) for n in (2, 4) for ly in ("tok", "unit")]


@pytest.mark.parametrize("kn", COMB + [XM.mutant(COMB[0])], ids=[XM.moe_insitu_name(k) for k in COMB + [XM.mutant(COMB[0])]])
def test_combine_consumes_its_own_unit(kn):
    """cons=combine: the combine pushes 8-byte pieces to the N - 1 OTHER ranks, polls them back as 8-byte pieces, resets
    them with 8-byte stores (absent in the noreset twin) and has no 16-byte consumer pass."""
    k = XM.build_moe_insitu_kernel(kn)
    assert LI.lint_peer_access(k) == []
    text = _ll(k)
    N = kn.ranks
    assert f"global_store_dwordx2 $0, $1, ${N} sc0 sc1" in text and f"global_store_dwordx2 $0, $1, ${N + 1} sc0 sc1" not in text
    assert "global_load_dword $0, $" in text and "global_load_dwordx4" not in text
    resets = f"global_store_dwordx2 $0, ${4 * (N - 1) if N > 1 else 0}"
    assert (resets in text) == (kn.mutant != "noreset")
    with pytest.raises(ValueError):
        XM.MoeInsituKnobs(moe=M.MoeKnobs(tp=2, bucket=16, combine="gbar"), cons="combine").check()
