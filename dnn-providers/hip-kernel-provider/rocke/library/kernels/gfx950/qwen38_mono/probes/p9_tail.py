# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P9: the MTP round tail as ONE persistent launch with real work (DESIGN.md 8.2, P9).

Question: what does the true mono gain by running the non-layer work between two layer passes of a K=3
round -- final HC mix, MXFP4 head + greedy argmax, the rejection-sampler accept, the drafter's front end
(embedding norm + ``fc_embedding`` + hidden norm + ``fc_hidden``) and the draft meta -- as stages of
one launch instead of ATOM's kernel sequence (two Triton HC-mix kernels, the head, index selects,
``topk_select`` x 2, the greedy rejection kernel, gather / scatter / add, ``DraftFront``, the draft meta
kernel)?

Two chains, one builder (:class:`P9Spec`):

* ``mode="verify"`` (the K=3 verify tail, 4 rows): HC final mix of the target's last-layer streams
  ``h [4, 4 * 2560]`` -> head (fused argmax, ATOM's greedy-sampler NaN rule) -> accept (bitwise
  ``rejection_greedy_sample_kernel`` + ``next_token_ids`` gather + the drafter's ``input_ids.scatter_``)
  -> draft front (``DraftFront`` of the 4 rows) -> draft meta (``positions + 1`` and
  ``write_draft_meta``).
* ``mode="draft"`` (between two draft layers, 1 row): row select (``index_select`` of the anchor row)
  -> HC final mix -> head (fused argmax, the mono rule) -> draft token -> draft front (1 row) -> meta.

Numerics: every stage is built to be bitwise ATOM's op: the Triton kernels' arithmetic order is
reproduced from their ttgir / amdgcn (triton 3.8, gfx950; ``hackathon-prep/qwen38/truemono/P9``):
per-thread products with the backend's FMA contraction, Triton's AMD DPP warp-reduction tree
(``row_shr`` 8 / 4 / 2 / 1 then ``row_bcast`` 15 / 31), the cross-warp ``(w0 + w2) + (w1 + w3)``
combine, the chunk order of each sum, bf16 rounding points, ``exp2`` / precise division, and the
MFMA ``16x16x32`` chain of ``tl.dot`` (``isTransposed``: weights as the A operand). The head is P2's
:func:`p2_head.emit_head` (``mx_decode_gemv``'s body, bitwise).

Placement (grid 256 x 512 threads, one CTA per CU): waves 0-3 of CTAs ``0 .. 242`` run the head (its
first weight panels issued before the x poll); waves 4-7 of CTAs ``toff .. toff + 159`` run task ``t``:
PRE (``_hc_pre_kernel`` program set ``(n-tile, stream)``), then MIX (``_hc_mix_kernel`` block of 16
columns), then FRONT (``_draft_front_kernel`` block of 16 columns; its hidden half runs while the head
streams, its embedding half after the token ids arrive). CTA 243 (waves 0-3) computes the front's
bundle norms, CTA 244 wave 0 the meta words. Stage hand-offs are tagged pair mailboxes
(``rocke.helpers.mailbox``); wave groups of one CTA never use ``s_barrier`` after the entry barrier (a
hardware barrier counts every wave of the workgroup): they synchronise through LDS counters, one per
barrier site.

Launch epoch: a never-reset 64-bit counter at ``scr[0:8]``; every wave reads it at entry and adds 1 at
its end, so every wave of launch ``e`` reads a value in ``[e W, e W + W)`` (``W = 2048`` waves) and
``tag = e + 1`` needs no host-written step word (graph replays).

Latency knobs (all default to the first gated structure, ``lat=v1``): ``lat`` v2 / v2w (PRE / MIX loads issued
up front, MIX row polls once per row-wave, gate shared through LDS), ``prer`` (PRE store regions), ``mloop``
(MIX row loop), ``opq`` (pure asm guards), ``xbt`` (head x poll batch), ``fold=direct`` (the head CTAs put
their own argmax keys; every FRONT task folds them and runs the accept, task 0 stores the outputs: no
last-arriver atomic, slot poll or ids hop), ``fnorm=pre`` (the FRONT embedding rows normalized once into
LDS instead of inside the MFMA chain).

Mutants (must fail the GPU gate): ``accept_ooo`` (accepted count one too high), ``stale_ids`` (the front
embeds the speculative ids, ignoring the accept), ``sumsq_order`` (sequential five-chunk sum in the
HC norm; masked by the data: equal sums on the gate's inputs), ``acc_order`` (PRE chunk sums reversed),
``tree_order`` (left-to-right warp combine), ``mma_order`` (MFMA chain over k reversed), ``hn_order`` /
``en_order`` (hidden / embedding norm products re-associated), ``stale_tag`` (polls with the previous
launch's tag).

Python engine only (``buffer_*_cp``, ``mov_dpp``, the head). Not imported by any served build.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from kernels.gfx950 import mx_decode_gemv as MX
from kernels.gfx950.qwen38_mono.probes import p2_head as P2
from rocke.core.ir import (
    BF16,
    CPOL_SC1,
    F32,
    I8,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
    VectorType,
)
from rocke.helpers import device_ops as dops
from rocke.helpers import mailbox as MB

__all__ = [
    "MODES",
    "MUTANTS",
    "P9Spec",
    "build_p9",
    "build_p9_parts",
    "is_valid_p9",
    "p9_signature",
    "parts_signature",
    "SCR_BYTES",
    "SCR",
]

# ---------------------------------------------------------------- geometry (Qwen3.8-Flash-Next)

H = 2560  # hidden
HC = 4  # HC streams
R = 320  # HC low rank (hc_pre's NOUT for the final mixers: no inject)
HCW = HC * H  # 10240
VOCAB = 248320
NTASK = 160  # PRE: 40 n-tiles x 4 streams; MIX: 2560 / 16; FRONT: 2560 / 16
HEAD_CTAS = 243
NEG_LOG2E = 0xBFB8AA3B  # f32 bits of -log2(e) (Triton's sigmoid: exp2(x * -log2e))
QSA_WORDS = 12  # len(ATOM kernels/stages/qsa.STATE): the draft-layer qsa table row (step_meta.QSA_WORDS)
QSA_PAGES = 1  # STATE.index("pages") (step_meta.QSA_PAGES); the harness asserts both against ATOM

MODES = ("verify", "draft")
MUTANTS = ("none", "accept_ooo", "stale_ids", "sumsq_order", "acc_order", "tree_order", "mma_order", "hn_order",
           "stale_tag", "en_order")

# scratch layout (bytes); pair mailboxes are 8-byte (payload, tag) pairs
WAVES_PER_LAUNCH_ = 256 * 8
SCR = {
    "epoch": 0,  # u64 launch counter
    "part": 4096,  # pairs [4 rows][4 streams][320]
    "rstd": 4096 + 4 * HC * R * 8,  # pairs [4][4]
    "xbox": 4096 + 4 * HC * R * 8 + 256,  # P2 x layout: pairs [4 rows][H / 2]
    "rstdh": 4096 + 4 * HC * R * 8 + 256 + 4 * (H // 2) * 8,  # pairs [4]
    "ids": 4096 + 4 * HC * R * 8 + 256 + 4 * (H // 2) * 8 + 256,  # pairs [8]
}
SCR["keys"] = SCR["ids"] + 8 * 8  # fold=direct: pairs [4 rows][243 head CTAs][2] (key, column) per CTA
SCR["err"] = 1 << 17  # [64 sites][256 CTAs] x 16 B: (tag, wave + 1, site + 1, first observed tag) of a timeout
SCR["dbg"] = SCR["err"] + 64 * 256 * 16  # tmo builds: [2048 waves] (tag, raw counter lo, raw hi, 0)
SCR["dbgw"] = SCR["dbg"] + WAVES_PER_LAUNCH_ * 16  # tmo builds: PRE (p = 0) [stream][row][wave][chunk] + ss, rstd
SCR_BYTES = SCR["dbgw"] + 2 * 4 * 4 * 32 * 4
TMO_ITERS = 400000  # poll / barrier retries before a tmo build gives up (each retry is at least one sc1 round trip)
WAVES_PER_LAUNCH = 256 * 8
SITE_NAMES: Dict[str, List[str]] = {}  # kernel name -> timeout-site names (tmo builds)
LATS = ("v1", "v2", "v2w")
_NL = chr(10)
NBAR = 64  # LDS barrier counters (one per barrier site)


@dataclass(frozen=True)
class P9Spec:
    """One P9 build.

    ``mode``: ``verify`` (4 rows, accept) or ``draft`` (1 row selected from ``rows_in``).
    ``rows_in``: rows of the chain input ``h`` (verify: 4; draft: 4 at step 0, 1 later).
    ``toff``: first CTA of the 160 PRE/MIX/FRONT tasks (task ``t`` on CTA ``toff + t``).
    ``fpre``: ``after_mix`` (FRONT's weights issued after the task's MIX) or ``early`` (W_h streamed
    and W_e staged as soon as the task's PRE is done, MIX after; contention knob).
    ``espec``: 1 = prefetch the speculative ids' embedding rows (and their norms) before the accept.
    ``mutant``: see the module docstring.
    """

    mode: str = "verify"
    rows_in: int = 4
    toff: int = 0
    fpre: str = "after_mix"
    espec: int = 1
    head_dg: int = 2
    stamps: bool = False
    reject: int = 0
    dbg: int = 0  # bisect knob (0 in every gated / timed build): 1 no meta, 2 no accept, 4 no front
    tmo: int = 0  # 1: every poll / barrier gives up after TMO_ITERS retries and records an error (debug)
    pause: str = "s_sleep1"  # poll retry pause (rocke.helpers.device_ops.PAUSES)
    xwait: str = "hint"  # head x wait: "hint" (wave 0 spins on one pair per MIX task, then the payload) or "spin"
    idwait: str = "wave0"  # FRONT ids: "wave0" (one wave polls, LDS broadcast) or "all" (every wave polls)
    prex: str = "lds"  # PRE norm partials: "lds" (one barrier), "lds2" (two barriers), "redund" (every wave all)
    #: PRE / MIX structure: "v1" (per-row load-use chains, every wave polls every row's parts row by row), "v2"
    #: (every independent load issued up front; MIX wave ``r`` polls row ``r``'s parts and rstd once and shares
    #: the gate through LDS), "v2w" (v2 with each MIX row poll in one batch)
    lat: str = "v1"
    mloop: int = 0  # lat v2 / v2w: 1 = MIX's per-row compute in a runtime loop (code size knob)
    #: lat v2 / v2w PRE store regions: "one" (all rows' LDS partials, then all rows' parts, each in one lane-0
    #: region) or "row" (a region per row: bounds the scheduler's live ranges; row r's parts leave as soon as done)
    prer: str = "one"
    #: head argmax delivery: "fused" (the head's last-arriver fold: atomic, slot poll, accept on the last CTA,
    #: ids mailbox) or "direct" (every head CTA puts its own (key, column) per row; FRONT wave 0 of every task
    #: polls the 243 slots, folds, and runs the accept itself; task 0 stores the outputs)
    fold: str = "fused"
    #: FRONT embedding norm: "loop" (wave 0 normalizes every 32-k chunk of every column inside the MFMA chain)
    #: or "pre" (the wave staging a token row normalizes it once into LDS; the chain reads bf16 operands)
    fnorm: str = "loop"
    #: head waves sleep ``hdelay`` x ``s_sleep 63`` (63 x 64 clocks each) before their first weight
    #: panels: the PRE / MIX loads reach HBM ahead of the head's prefetch (the head waits for x far longer)
    hdelay: int = 0
    #: LDS barrier spin pause: ``s_sleep bsleep`` per retry (0: ``s_sleep 0``). Waiting waves share SIMDs with the
    #: other wave group (head waves 1-3 wait for x while PRE / MIX run; FRONT waves 1-3 wait for the ids while
    #: the head streams), so a tight spin takes their issue slots
    bsleep: int = 0
    #: launch-epoch arrivals: "wave" (every wave adds 1 at its end: ~430 waves with no work add at launch start,
    #: all on one address) or "cta" (the CTA's last wave, found with an LDS counter, adds 8)
    ectr: str = "wave"
    opq: str = "se"  # lat v2 / v2w PRE / MIX asm guards (opaque copies, LDS settles): "se" side-effecting or "pure"
    xbt: int = 4  # head x staging: specs-pairs groups per payload poll batch (4: two batches at 4 rows; 8: one)
    mutant: str = "none"
    #: draft only, default off (consolidated-stack tail, ``truemono_tail``): the head is the N8 draft-vocab subset
    #: (``ATOM_Q38_DRAFT_VOCAB``): ``vsub`` rows of the re-packed MXFP4 head (``hw`` / ``hs`` = the subset's codes and
    #: scales), the folded column mapped to the token id through the extra parameter ``vmap`` (i32 ``[vsub]``) before
    #: the export and the FRONT (ATOM's ``DraftVocab.argmax``: subset argmax, then ``index_select(map)``). 0 = full head
    vsub: int = 0
    #: draft meta: rows of the step positions bumped by ``qsa_draft_decode_metadata``'s ``bumped_positions`` (one
    #: request: 1 = ``[rows]``, 3 = mRoPE's ``[3, 1]``, consecutive i64). Default 1 (the gated build)
    pbump: int = 1
    #: draft, fold=direct, default off: no FRONT; task 0's wave 0 alone folds the head CTAs' slots and stores the
    #: token (and ``y``): the last draft step of a round, whose token feeds no further draft step
    tokonly: int = 0

    @property
    def rows(self) -> int:
        return 4 if self.mode == "verify" else 1

    def head_spec(self) -> MX.MxDecodeGemvSpec:
        return MX.MxDecodeGemvSpec(
            n=self.vsub or VOCAB, k=H, rows_max=4, epilogue="fused", waves=4, waves_per_eu=0, pro="w_first", dg=self.head_dg
        )

    def kernel_name(self) -> str:
        d = P9Spec()
        parts = [f"p9_{self.mode}", f"ri{self.rows_in}"]
        if self.toff != d.toff:
            parts.append(f"to{self.toff}")
        if self.fpre != d.fpre:
            parts.append(f"fp{self.fpre}")
        if self.espec != d.espec:
            parts.append(f"es{self.espec}")
        if self.head_dg != d.head_dg:
            parts.append(f"dg{self.head_dg}")
        if self.reject:
            parts.append("rej")
        if self.dbg:
            parts.append(f"DBG{self.dbg}")
        if self.tmo:
            parts.append("TMO")
        if self.pause != d.pause:
            parts.append(f"pz{self.pause}")
        if self.xwait != d.xwait:
            parts.append(f"xw{self.xwait}")
        if self.idwait != d.idwait:
            parts.append(f"iw{self.idwait}")
        if self.prex != d.prex:
            parts.append(f"px{self.prex}")
        if self.lat != d.lat:
            parts.append(f"lt{self.lat}")
        if self.mloop != d.mloop:
            parts.append(f"ml{self.mloop}")
        if self.prer != d.prer:
            parts.append(f"pr{self.prer}")
        if self.fold != d.fold:
            parts.append(f"fo{self.fold}")
        if self.opq != d.opq:
            parts.append(f"oq{self.opq}")
        if self.fnorm != d.fnorm:
            parts.append(f"fn{self.fnorm}")
        if self.hdelay != d.hdelay:
            parts.append(f"hd{self.hdelay}")
        if self.bsleep != d.bsleep:
            parts.append(f"bs{self.bsleep}")
        if self.ectr != d.ectr:
            parts.append(f"ec{self.ectr}")
        if self.xbt != d.xbt:
            parts.append(f"xb{self.xbt}")
        if self.vsub:
            parts.append(f"vs{self.vsub}")
        if self.pbump != d.pbump:
            parts.append(f"pb{self.pbump}")
        if self.tokonly:
            parts.append("tk")
        if self.mutant != "none":
            parts.append(f"MUT{self.mutant}")
        if self.stamps:
            parts.append("STAMP")
        return "_".join(parts) + "_pyonly"


#: the measured best knobs (P9 report): lat v2w, fold direct, fnorm pre, ectr cta; verify also xbt 8, hdelay 8
BEST_KNOBS = {"lat": "v2w", "fold": "direct", "fnorm": "pre", "ectr": "cta"}
BEST_VERIFY_KNOBS = {"xbt": 8, "hdelay": 8}


def best_spec(mode: str, rows_in: int = 4, reject: int = 0, **kw) -> P9Spec:
    """The P9 build with the measured best knobs (``P9Spec()``'s defaults stay the first gated build)."""
    extra = dict(BEST_KNOBS, **(BEST_VERIFY_KNOBS if mode == "verify" else {}))
    extra.update(kw)
    return P9Spec(mode=mode, rows_in=rows_in, reject=reject, **extra)


def is_valid_p9(spec: P9Spec) -> Tuple[bool, str]:
    checks = [
        (spec.mode in MODES, f"mode must be one of {MODES}"),
        (spec.mutant in MUTANTS, f"mutant must be one of {MUTANTS}"),
        (spec.mode == "draft" or spec.rows_in == 4, "verify takes 4 input rows"),
        (spec.rows_in in (1, 4), "rows_in must be 1 or 4"),
        (0 <= spec.toff and spec.toff + NTASK <= 256 and not (spec.toff <= 243 < spec.toff + NTASK)
         and not (spec.toff <= 244 < spec.toff + NTASK), "tasks must avoid CTAs 243 (norm) and 244 (meta)"),
        (spec.fpre in ("after_mix", "early"), "fpre must be after_mix or early"),
        (spec.espec in (0, 1), "espec must be 0 or 1"),
        (spec.mutant != "accept_ooo" or spec.mode == "verify", "accept_ooo needs the accept"),
        (spec.mutant != "stale_ids" or spec.mode == "verify", "stale_ids needs the accept"),
        (1 <= spec.head_dg <= 4, "head_dg must be 1..4"),
        (spec.pause in dops.PAUSES, f"pause must be one of {tuple(dops.PAUSES)}"),
        (spec.xwait in ("hint", "spin"), "xwait must be hint or spin"),
        (spec.idwait in ("wave0", "all"), "idwait must be wave0 or all"),
        (spec.prex in ("lds", "lds2", "redund"), "prex must be lds, lds2 or redund"),
        (spec.lat in LATS, f"lat must be one of {LATS}"),
        (spec.lat == "v1" or spec.prex == "lds", "lat v2 / v2w take prex lds"),
        (spec.lat == "v1" or not spec.tmo, "tmo is a lat v1 debug knob"),
        (spec.mloop in (0, 1) and (spec.mloop == 0 or spec.lat != "v1"), "mloop is a lat v2 / v2w knob (0 or 1)"),
        (spec.xbt in (4, 8), "xbt must be 4 or 8"),
        (spec.prer in ("one", "row") and (spec.prer == "one" or spec.lat != "v1"), "prer is a lat v2 / v2w knob"),
        (spec.fold in ("fused", "direct"), "fold must be fused or direct"),
        (spec.opq in ("se", "pure") and (spec.opq == "se" or spec.lat != "v1"), "opq is a lat v2 / v2w knob"),
        (spec.fnorm in ("loop", "pre"), "fnorm must be loop or pre"),
        (0 <= spec.hdelay <= 16, "hdelay must be 0..16"),
        (0 <= spec.bsleep <= 16 and not (spec.bsleep and spec.tmo), "bsleep must be 0..16 (not with tmo)"),
        (spec.ectr in ("wave", "cta"), "ectr must be wave or cta"),
        (spec.fold == "fused" or (spec.idwait == "wave0" and not spec.dbg & 2), "fold=direct needs idwait=wave0, accept"),
        (spec.reject in (0, 1) and (spec.reject == 0 or spec.mode == "draft"), "reject is a draft-step knob"),
        (spec.vsub == 0 or (spec.mode == "draft" and spec.fold == "direct" and 0 < spec.vsub < VOCAB
                            and spec.vsub % 16 == 0), "vsub: draft, fold=direct, a multiple of 16 below the vocab"),
        (spec.pbump in (1, 2, 3) and (spec.pbump == 1 or spec.mode == "draft"), "pbump: 1..3, draft only"),
        (spec.tokonly in (0, 1) and (not spec.tokonly or (spec.mode == "draft" and spec.fold == "direct")),
         "tokonly: draft, fold=direct"),
    ]
    for ok, why in checks:
        if not ok:
            return False, why
    return MX.is_valid_spec(spec.head_spec())


# kernel parameters: (name, kind); every build takes all of them (unused ones are ignored)
_SIG = [
    ("h", "ptr"),  # bf16 [rows_in, 10240] chain input (target / draft layer HC streams)
    ("normw", "ptr"),  # bf16 [10240] hc_norm weight (Gemma offset applied in kernel)
    ("wcat", "ptr"),  # bf16 [320, 10240] final mixer down weight
    ("wup", "ptr"),  # bf16 [10240, 320] final mixer up weight
    ("hw", "ptr"),  # u8 [N, K/2] MXFP4 head codes
    ("hs", "ptr"),  # u8 [N, K/32] E8M0 scales
    ("hws", "ptr"),  # head fused workspace (counter64 + slots)
    ("y", "ptr"),  # i32 [4] head argmax per row
    ("emb", "ptr"),  # bf16 [V, 2560] embeddings
    ("ge", "ptr"),  # bf16 [2560] pre_fc_norm_embedding weight
    ("gh", "ptr"),  # bf16 [10240] pre_fc_norm_hidden weight
    ("we", "ptr"),  # bf16 [2560, 2560] fc_embedding
    ("wh", "ptr"),  # bf16 [2560, 2560] fc_hidden
    ("hin", "ptr"),  # bf16 [rows, 10240] front output (the draft layer's input)
    ("ids", "ptr"),  # i32 [8]: verify: the step's input ids (t0, d1, d2, d3); draft: unused
    ("dids", "ptr"),  # i32 [4]: the drafter's ids buffer (input_ids.gpu[1:5]); scattered by accept
    ("outtok", "ptr"),  # i32 [4]: verify: output_token_ids [1, 4]; draft: the draft token export
    ("verdict", "ptr"),  # i32 [3]: verify: (accepted, rejected, anchor); draft: read anchor (row select)
    ("nexttok", "ptr"),  # i32 [1]: next_token_ids
    ("pos", "ptr"),  # i64 [rows]: verify: the target positions; draft: the step positions (bumped in place)
    ("posout", "ptr"),  # i64 [rows]: verify: positions + 1 (EagleProposer.propose)
    ("logical", "ptr"),  # i64 [rows]: qsa logical positions (verify: input; draft: decode-metadata output)
    ("lens", "ptr"),  # i32 [1]: draft: context_lens (in place)
    ("tables", "ptr"),  # i32 [TS]: draft: the request's block-table row
    ("slots", "ptr"),  # i64 [1]: draft: slot_mapping
    ("reqs", "ptr"),  # i32 [1]: draft: qsa_token_to_req
    ("comp", "ptr"),  # i64 [1]: draft: qsa_compressed_slots
    ("rej", "ptr"),  # i32 [1]: draft (reject=1): num_reject_tokens
    ("mpos", "ptr"),  # i32 [rows]: draft layer pos table
    ("mtab", "ptr"),  # i64 [rows, 12]: draft layer qsa table
    ("mctr", "ptr"),  # i32 [1]: draft layer step counter
    ("scr", "ptr"),  # scratch (SCR_BYTES): epoch + mailboxes
    ("btab", "i64"),  # block_tables row pointer (the value write_draft_meta stores)
    ("eps", "f32"),  # hc_norm eps
    ("epse", "f32"),
    ("epsh", "f32"),
    ("st", "ptr"),  # stamps (stamped builds) else unused
]


def p9_signature(spec: Optional[P9Spec] = None) -> List[Tuple[str, str]]:
    """The kernel parameters of ``spec`` (``_SIG``; a ``vsub`` build also takes ``vmap``)."""
    return list(_SIG) + ([("vmap", "ptr")] if spec is not None and spec.vsub else [])


# ---------------------------------------------------------------- small helpers


def _ci(b: IRBuilder, v: int) -> Value:
    v = int(v)
    if v >= 1 << 31:
        v -= 1 << 32
    return b.const_i32(v)


def _f(b: IRBuilder, bits: int) -> Value:
    return b.bitcast(_ci(b, bits), F32)


def _i2f(b: IRBuilder, v: Value) -> Value:
    return b.bitcast(v, F32)


def _f2i(b: IRBuilder, v: Value) -> Value:
    return b.bitcast(v, I32)


def bf16r(b: IRBuilder, x: Value) -> Value:
    """f32 -> bf16 (RNE, ``fptrunc``) -> f32."""
    return b.cast_to_f32(b.trunc_f32_to_bf16(x))


def _bf_words(b: IRBuilder, words: Sequence[Value]) -> List[Value]:
    """i32 words holding two bf16 each (low first) -> f32 values."""
    out = []
    for w in words:
        out.append(_i2f(b, b.shl(w, _ci(b, 16))))
        out.append(_i2f(b, b.land(w, _ci(b, -65536))))
    return out


def _load8(b: IRBuilder, rsrc: Value, off: Value) -> List[Value]:
    """Eight consecutive bf16 at byte ``off`` -> eight f32."""
    v = b.buffer_load_cp(rsrc, off, _ci(b, 0), dwords=4, aux=0)
    return _bf_words(b, [b.vec_extract(v, i) for i in range(4)])


def _load8_raw(b: IRBuilder, rsrc: Value, off: Value) -> Value:
    """Eight consecutive bf16 as <4 x i32>."""
    return b.buffer_load_cp(rsrc, off, _ci(b, 0), dwords=4, aux=0)


def _pack8(b: IRBuilder, vals: Sequence[Value]) -> Value:
    """Eight f32 -> <8 x bf16> (RNE)."""
    words = [MB.bf16_pair(b, vals[2 * i], vals[2 * i + 1]) for i in range(4)]
    return b.vec_bitcast(b.vec_pack(words, I32), VectorType(BF16, 8))


def tri_tree(b: IRBuilder, v: Value, mutant: str = "none") -> Value:
    """Triton's AMD warp sum (``row_shr`` 8, 4, 2, 1 with ``bound_ctrl``, then ``row_bcast`` 15 / 31):
    lane 63 ends with ``(R3 + R2) + (R1 + R0)`` where ``R_k`` is row ``k``'s 16-lane tree. Returned
    as a wave-uniform f32 (read from lanes 15 / 31 / 47 / 63; fadd is commutative, so the bits are
    lane 63's)."""
    x = v
    for sh in (8, 4, 2, 1):
        y = _i2f(b, b.mov_dpp(_f2i(b, x), row_shr=sh, bound_ctrl=True))
        x = b.fadd(x, y)
    r = [b.readlane(x, _ci(b, 16 * k + 15)) for k in range(4)]
    if mutant == "tree_order":
        return b.fadd(b.fadd(b.fadd(r[0], r[1]), r[2]), r[3])
    return b.fadd(b.fadd(r[3], r[2]), b.fadd(r[1], r[0]))


def dot8(b: IRBuilder, a: Sequence[Value], w: Sequence[Value]) -> Value:
    """Triton's per-thread sum of eight products (8 contiguous k per thread; amdgcn of ``_hc_pre`` /
    ``_hc_mix`` / ``_draft_front``): ``(fma(a0, w0, a2 w2) + fma(a4, w4, a6 w6)) + (fma(a1, w1, a3 w3)
    + fma(a5, w5, a7 w7))`` with the products not fused into an FMA rounded to f32."""
    def p(j):
        return b.fmul(a[j], w[j])

    e = b.fadd(b.fma(a[0], w[0], p(2)), b.fma(a[4], w[4], p(6)))
    o = b.fadd(b.fma(a[1], w[1], p(3)), b.fma(a[5], w[5], p(7)))
    return b.fadd(e, o)


def _lds_settle(b: IRBuilder, vals: Sequence[Value], pure: bool = False) -> List[Value]:
    """The f32 ``vals`` (results of LDS loads) through one asm that first waits ``lgkmcnt(0)``: no use of any
    of them can be scheduled before every one of the loads has returned. (P9 hit a deterministic 1-ulp
    norm mismatch when the compiler consumed mixed-width ``ds_read`` results under partial ``lgkmcnt(N)``
    waits; with this settle, or a second barrier, or no LDS exchange, it is gone.)"""
    n = len(vals)
    cons = ",".join(["=v"] * n + [str(i) for i in range(n)])
    kw = {"sideeffect": False} if pure else {}  # pure: no ordering against other side effects (opq knob)
    outs = b.inline_asm_multi("s_waitcnt lgkmcnt(0)", cons, [_f2i(b, v) for v in vals], result_types=[I32] * n,
                              result_name_hint="ldsset", **kw)
    if n == 1:
        outs = [outs] if not isinstance(outs, list) else outs
    return [_i2f(b, o) for o in outs]


def _opaque(b: IRBuilder, x: Value, pure: bool = False) -> Value:
    """``x`` (f32) through an empty asm: the backend cannot fuse its producing fmul into a later fadd.
    Needed where a rounded f32 product feeds an add (products of two bf16 values are exact in f32, so
    contraction cannot change them; Triton's dot / sum-of-squares orders hold either way). ``pure``: the
    asm without side effects (still opaque to contraction; free to schedule)."""
    if pure:
        return b.bitcast(b.inline_asm("; opq $0", "=v,0", [b.bitcast(x, I32)], I32, sideeffect=False,
                                      result_name_hint="opq"), F32)
    return b.bitcast(dops.fresh(b, b.bitcast(x, I32)), F32)


def sigmoid_tr(b: IRBuilder, x: Value) -> Value:
    """``tl.sigmoid``: ``1 / (1 + exp2(x * -log2 e))`` (precise division)."""
    e = b.exp2(b.fmul(x, _f(b, NEG_LOG2E)))
    return b.fdiv(b.const_f32(1.0), b.fadd(e, b.const_f32(1.0)))


# ---------------------------------------------------------------- polls with a timeout (tmo builds)


def _tpoll_asm(npairs: Sequence[int], nbases: int, base_of: Sequence[int], iters: int,
               pause: str = "s_nop 0") -> Tuple[str, str]:
    """``rocke.helpers.mailbox.asm_poll_template`` plus a retry counter: after ``iters`` failed retries the
    loop exits with ``$flag = 1``. Outputs: one ``<2 x i32>`` per pair, an i64 VGPR temp, an i64 SGPR mask,
    the i32 SGPR counter and the i32 SGPR flag; inputs as the mailbox template."""
    p_total = sum(npairs)
    tmp, mask, cnt, flag = p_total, p_total + 1, p_total + 2, p_total + 3
    voff0 = p_total + 4
    base0 = voff0 + len(npairs)
    tag = base0 + nbases
    top = "${:private}p9tp${:uid}_top"
    done = "${:private}p9tp${:uid}_done"
    lines = [f"s_mov_b32 ${cnt}, 0", f"s_mov_b32 ${flag}, 0", f"{top}:"]
    p = 0
    for k, n in enumerate(npairs):
        for q in range(n):
            off = f" offset:{q * 8}" if q else ""
            lines.append(f"global_load_dwordx2 ${p}, ${voff0 + k}, ${base0 + base_of[k]}{off} sc1")
            p += 1
    lines.append("s_waitcnt vmcnt(0)")
    for q in range(p_total):
        lines.append(f"v_lshrrev_b64 ${tmp}, 32, ${q}")
        dst = f"${mask}" if q == 0 else "vcc"
        lines.append(f"v_cmp_ne_u64_e64 {dst}, ${tmp}, ${tag}")
        if q:
            lines.append(f"s_or_b64 ${mask}, ${mask}, vcc")
    lines.append(f"s_and_b64 ${mask}, ${mask}, exec")
    lines.append(f"s_cbranch_scc0 {done}")
    lines.append(f"s_add_u32 ${cnt}, ${cnt}, 1")
    lines.append(f"s_cmp_gt_u32 ${cnt}, {iters}")
    lines.append(f"s_cselect_b32 ${flag}, 1, 0")
    lines.append(f"s_cbranch_scc1 {done}")
    if pause:
        lines.append(pause)
    lines.append(f"s_branch {top}")
    lines.append(f"{done}:")
    cons = ["=&v"] * p_total + ["=&v", "=&s", "=&s", "=&s"]
    cons += ["v"] * len(npairs) + ["s"] * nbases + ["s"]
    cons += ["~{vcc}", "~{scc}"]
    return "\n".join(lines), ",".join(cons)


def _record(c: "_Ctx", site: int, flag: Value, seen: Value) -> None:
    """A timed-out poll / barrier: lane 0 stores (tag, wave + 1, site + 1, seen) at err[site][cta]."""
    b = c.b
    hit = b.land(b.cmp_ne(flag, _ci(b, 0)), b.cmp_eq(b.lane_id(), _ci(b, 0)))
    with b.scf_if(hit):
        wave = b.lshr(b.thread_id_x(), _ci(b, 6))
        off = b.add(_ci(b, SCR["err"] + site * 256 * 16), b.mul(b.block_id_x(), _ci(b, 16)))
        rs = b.buffer_rsrc(c.p["scr"], _ci(b, SCR_BYTES))
        b.buffer_store_cp(b.vec_pack([c.tag, b.add(wave, _ci(b, 1)), _ci(b, site + 1), seen], I32), rs, off,
                          _ci(b, 0), aux=CPOL_SC1)


def poll(c: "_Ctx", specs, tag: Value, name: str = "poll", pmax: int = 0) -> List[List[Value]]:
    """``rocke.helpers.mailbox.poll`` (``pmax``: specs per batch, 0 = the knobs' default); ``tmo`` builds give
    up after ``TMO_ITERS`` retries per batch and record the site."""
    b = c.b
    if not c.spec.tmo:
        return MB.poll(b, specs, tag, replace(c.mbk, poll_max=pmax) if pmax else c.mbk)
    out: List[List[Value]] = []
    for first in range(0, len(specs), MB.POLL_MAX):
        batch = specs[first: first + MB.POLL_MAX]
        bases: List[Value] = []
        base_of: List[int] = []
        for mb_, _, _ in batch:
            for j, x in enumerate(bases):
                if x is mb_.addr:
                    base_of.append(j)
                    break
            else:
                base_of.append(len(bases))
                bases.append(mb_.addr)
        npairs = [n for _, _, n in batch]
        tpl, cons = _tpoll_asm(npairs, len(bases), base_of, TMO_ITERS, dops.PAUSES[c.spec.pause])
        voffs = [MB.pair_offset(b, i) for _, i, _ in batch]
        pt = sum(npairs)
        res = b.inline_asm_multi(tpl, cons, voffs + bases + [b.zext(tag, I64)],
                                 result_types=[VectorType(I32, 2)] * pt + [I64, I64, I32, I32], result_name_hint="tpoll")
        site = c.sites
        c.sites += 1
        c.site_names.append(f"{name}#{first // MB.POLL_MAX}")
        _record(c, site, res[pt + 3], b.vec_extract(res[0], 1))
        q = 0
        for n in npairs:
            out.append([b.vec_extract(res[q + j], 0) for j in range(n)])
            q += n
    return out


def fold_poll_tmo(c: "_Ctx"):
    """The head fold's slot poll (``(rsrc, byte offset, 4)`` specs: 16-byte slots ``(key, tag, col, tag)``
    in the head workspace) as timed pair polls; returns the ``<4 x i32>`` slots."""
    b = c.b
    mb = MB.mailbox_from_ptr(b, c.p["hws"], _ci(b, 1 << 20), c.mbk)

    def fp(tag: Value, specs) -> List[Value]:
        got = poll(c, [(mb, b.lshr(off, _ci(b, 3)), 2) for _, off, _ in specs], tag, "fold")
        return [b.vec_pack([g[0], tag, g[1], tag], I32) for g in got]

    return fp


# ---------------------------------------------------------------- LDS wave-group barriers


_BAR_ASM = "\n".join(
    [
        "s_waitcnt lgkmcnt(0)",
        "ds_add_u32 $2, $3",
        "${:private}p9bar${:uid}_top:",
        "ds_read_b32 $0, $2",
        "s_waitcnt lgkmcnt(0)",
        "v_cmp_ne_u32_e64 $1, $0, $4",
        "s_and_b64 $1, $1, exec",
        "s_cbranch_scc0 ${:private}p9bar${:uid}_done",
        "s_sleep 0",
        "s_branch ${:private}p9bar${:uid}_top",
        "${:private}p9bar${:uid}_done:",
    ]
)


_BAR_ASM_TMO = "\n".join(
    [
        "s_mov_b32 $2, 0",
        "s_mov_b32 $3, 0",
        "s_waitcnt lgkmcnt(0)",
        "ds_add_u32 $4, $5",
        "${:private}p9bt${:uid}_top:",
        "ds_read_b32 $0, $4",
        "s_waitcnt lgkmcnt(0)",
        "v_cmp_ne_u32_e64 $1, $0, $6",
        "s_and_b64 $1, $1, exec",
        "s_cbranch_scc0 ${:private}p9bt${:uid}_done",
        "s_add_u32 $2, $2, 1",
        f"s_cmp_gt_u32 $2, {TMO_ITERS * 4}",
        "s_cselect_b32 $3, 1, 0",
        "s_cbranch_scc1 ${:private}p9bt${:uid}_done",
        "s_sleep 0",
        "s_branch ${:private}p9bt${:uid}_top",
        "${:private}p9bt${:uid}_done:",
    ]
)


class Bars:
    """One LDS counter per barrier site (zeroed at launch entry, used once per launch). A site's
    barrier is executed by every lane of ``nwaves`` waves (each wave adds 64): call it only in
    wave-uniform control flow, never inside a runtime loop."""

    def __init__(self, b: IRBuilder, smem: Value):
        self.b = b
        self.base = b.trunc(b.smem_addr_of(smem), I32)
        self.next = 0
        self.ctx: Optional["_Ctx"] = None

    def site(self, nwaves: int) -> Callable[[], None]:
        idx = self.next
        self.next += 1
        if self.next > NBAR - 1:  # the last counter is the CTA's exit count (ectr=cta)
            raise ValueError("P9: too many barrier sites")
        b = self.b
        base = self.base

        def bar() -> None:
            addr = b.add(base, _ci(b, 4 * idx))
            c = self.ctx
            if c is None or not c.spec.tmo:
                b.inline_asm_multi(
                    _BAR_ASM if c is None or not c.spec.bsleep else _BAR_ASM.replace("s_sleep 0", f"s_sleep {c.spec.bsleep}"),
                    "=&v,=&s,v,v,v,~{vcc},~{scc},~{memory}",
                    [addr, _ci(b, 1), _ci(b, 64 * nwaves)],
                    result_types=[I32, I64],
                    result_name_hint="p9bar",
                )
                return
            res = b.inline_asm_multi(
                _BAR_ASM_TMO,
                "=&v,=&s,=&s,=&s,v,v,v,~{vcc},~{scc},~{memory}",
                [addr, _ci(b, 1), _ci(b, 64 * nwaves)],
                result_types=[I32, I64, I32, I32],
                result_name_hint="p9bar",
            )
            site = c.sites
            c.sites += 1
            c.site_names.append(f"bar{idx}")
            _record(c, site, res[3], res[0])

        return bar


# ---------------------------------------------------------------- the context of one build


class _LazyRsrc:
    """``rs[name]``: a buffer resource over parameter ``name`` (bounds 2 GiB - 16), emitted at the use."""

    def __init__(self, b: IRBuilder, p: Dict[str, Value]):
        self.b, self.p = b, p

    def __getitem__(self, name: str) -> Value:
        return self.b.buffer_rsrc(self.p[name], _ci(self.b, 0x7FFFFFF0))


class _LazyMb:
    """``mb[region]``: the scratch mailbox ``region`` (``SCR``), its base computed at the use."""

    def __init__(self, b: IRBuilder, p: Dict[str, Value], knobs: MB.MailboxKnobs):
        self.b, self.p, self.k = b, p, knobs

    def __getitem__(self, region: str) -> MB.Mailbox:
        b = self.b
        base = b.add(b.global_ptr_to_i64(self.p["scr"]), b.const_i64(SCR[region]))
        return MB.mailbox_from_i64(b, base, _ci(b, 1 << 20), self.k)


@dataclass
class _Ctx:
    b: IRBuilder
    spec: P9Spec
    p: Dict[str, Value]
    rs: "_LazyRsrc"  # buffer resources
    tag: Value
    ptag: Value  # previous launch's tag (stale_tag mutant)
    mbk: MB.MailboxKnobs
    mb: "_LazyMb"
    lds: Dict[str, Value]
    bars: Bars
    sel: Value  # first input row (draft: the anchor row)
    stamp: Callable[[int], None] = lambda i: None
    dids0: Optional[List[Value]] = None  # fold=direct verify: the drafter's ids pinned before MIX published
    dr0: Optional[List[Value]] = None  # fold=direct verify: the step's draft ids (ids[1..3]), pinned likewise
    sites: int = 0  # timed-poll / barrier error sites handed out so far (tmo builds)
    site_names: List[str] = field(default_factory=list)


def _row_off(c: _Ctx, r: int) -> Value:
    """Byte offset of input row ``r`` (after the draft row select) in ``h``."""
    b = c.b
    return b.mul(b.add(c.sel, _ci(b, r)), _ci(b, HCW * 2))


# ---------------------------------------------------------------- PRE: _hc_pre_kernel


def emit_pre(c: _Ctx, t: Value, gw: Value, lane: Value, bar1: Callable, bar2: Callable) -> None:
    """Task ``t`` = (stream ``s = t // 40``, n-tile ``p = t % 40``) of ``_hc_pre_kernel`` for every row;
    group wave ``gw`` plays Triton warp ``gw``: outputs ``n = 8 p + gw`` and ``8 p + gw + 4``."""
    b, spec = c.b, c.spec
    s = b.readfirstlane(b.div(t, _ci(b, 40)))
    pt = b.readfirstlane(b.mod(t, _ci(b, 40)))
    sb = b.mul(s, _ci(b, H))  # stream base (elements)
    # weights first (the issue-before-poll rule): rows n0, n1, five 512-chunks, 8 k per lane
    ns = [b.add(b.mul(pt, _ci(b, 8)), gw), b.add(b.add(b.mul(pt, _ci(b, 8)), gw), _ci(b, 4))]
    wraw = {}
    for j, n in enumerate(ns):
        for ch in range(5):
            k0 = b.add(sb, b.add(_ci(b, ch * 512), b.mul(lane, _ci(b, 8))))
            off = b.mul(b.add(b.mul(n, _ci(b, HCW)), k0), _ci(b, 2))
            wraw[(j, ch)] = _load8_raw(b, c.rs["wcat"], off)
    lw = c.lds["prew"]  # f32 [4 waves][4 rows][8]
    rows = spec.rows
    # sum of squares, Triton warp gw's elements: 128 gw + 2 lane + {0, 1} of each chunk
    def wred(r: int, ch: int, w: int) -> Value:
        """Triton warp ``w``'s partial sum of squares of chunk ``ch``, computed by this wave itself."""
        e = b.add(sb, b.add(_ci(b, ch * 512 + 128 * w), b.mul(lane, _ci(b, 2))))
        wd = b.buffer_load_cp(c.rs["h"], b.add(_row_off(c, r), b.mul(e, _ci(b, 2))), _ci(b, 0), dwords=1, aux=0)
        x0, x1 = _bf_words(b, [wd])
        return tri_tree(b, b.fadd(b.fmul(x0, x0), b.fmul(x1, x1)), spec.mutant)

    for r in range(rows if spec.prex != "redund" else 0):
        ro = _row_off(c, r)
        for ch in range(5):
            e = b.add(sb, b.add(_ci(b, ch * 512), b.add(b.mul(gw, _ci(b, 128)), b.mul(lane, _ci(b, 2)))))
            wd = b.buffer_load_cp(c.rs["h"], b.add(ro, b.mul(e, _ci(b, 2))), _ci(b, 0), dwords=1, aux=0)
            x0, x1 = _bf_words(b, [wd])
            tt = b.fadd(b.fmul(x0, x0), b.fmul(x1, x1))
            wsum = tri_tree(b, tt, spec.mutant)
            with b.scf_if(b.cmp_eq(lane, _ci(b, 0))):
                b.smem_store_vN(lw, [gw, _ci(b, r), _ci(b, ch)], wsum, 1)
                if spec.tmo:
                    with b.scf_if(b.cmp_eq(pt, _ci(b, 0))):
                        o = b.add(_ci(b, SCR["dbgw"]), b.mul(b.add(b.mul(b.add(b.mul(s, _ci(b, 4)), _ci(b, r)),
                                                                          _ci(b, 32)), b.add(b.mul(gw, _ci(b, 5)), _ci(b, ch))), _ci(b, 4)))
                        b.buffer_store_cp(b.bitcast(wsum, I32), b.buffer_rsrc(c.p["scr"], _ci(b, SCR_BYTES)), o, _ci(b, 0),
                                          aux=CPOL_SC1)
    bar_x = c.bars.site(4) if spec.prex == "lds2" else None
    bar1()
    if bar_x is not None:
        bar_x()
    rstd = []
    g1 = None
    for r in range(rows):
        sc = []
        for ch in range(5):
            if spec.prex == "redund":
                wv = [wred(r, ch, w) for w in range(4)]
            else:
                wv = [b.vec_extract(b.smem_load_vN(lw, _ci(b, w), _ci(b, r), _ci(b, ch), dtype=F32, n=1), 0)
                      for w in range(4)]
                wv = _lds_settle(b, wv)
            sc.append(b.fadd(b.fadd(wv[0], wv[2]), b.fadd(wv[1], wv[3])))
            if spec.tmo:
                with b.scf_if(b.land(b.land(b.cmp_eq(pt, _ci(b, 0)), b.cmp_eq(gw, _ci(b, 0))),
                                     b.cmp_eq(lane, _ci(b, 0)))):
                    o = b.add(_ci(b, SCR["dbgw"] + 4 * 4 * 32 * 4),
                              b.mul(b.add(b.mul(b.add(b.mul(s, _ci(b, 4)), _ci(b, r)), _ci(b, 32)), _ci(b, 4 * ch)),
                                    _ci(b, 4)))
                    b.buffer_store_cp(b.vec_pack([b.bitcast(v, I32) for v in wv], I32),
                                      b.buffer_rsrc(c.p["scr"], _ci(b, SCR_BYTES)), o, _ci(b, 0), aux=CPOL_SC1)
        if spec.mutant == "sumsq_order":
            ss = b.fadd(b.fadd(b.fadd(b.fadd(sc[0], sc[1]), sc[2]), sc[3]), sc[4])
        else:
            ss = b.fadd(b.fadd(b.fadd(sc[0], sc[1]), sc[2]), b.fadd(sc[3], sc[4]))
        rstd.append(b.rsqrt(b.fadd(b.fdiv(ss, b.const_f32(float(H))), c.p["eps"])))
        if spec.tmo:
            with b.scf_if(b.land(b.land(b.cmp_eq(pt, _ci(b, 0)), b.cmp_eq(gw, _ci(b, 0))), b.cmp_eq(lane, _ci(b, 0)))):
                o = b.add(_ci(b, SCR["dbgw"]), b.mul(b.add(b.mul(b.add(b.mul(s, _ci(b, 4)), _ci(b, r)), _ci(b, 32)),
                                                         _ci(b, 20)), _ci(b, 4)))
                qq = b.fadd(b.fdiv(ss, b.const_f32(float(H))), c.p["eps"])
                b.buffer_store_cp(b.vec_pack([b.bitcast(ss, I32), b.bitcast(qq, I32), b.bitcast(rstd[-1], I32),
                                              _ci(b, 0)], I32), b.buffer_rsrc(c.p["scr"], _ci(b, SCR_BYTES)), o,
                                  _ci(b, 0), aux=CPOL_SC1)
    # the norm weight (Gemma offset) for this lane's 8 k of each chunk
    g1 = {}
    for ch in range(5):
        k0 = b.add(sb, b.add(_ci(b, ch * 512), b.mul(lane, _ci(b, 8))))
        gv = _load8(b, c.rs["normw"], b.mul(k0, _ci(b, 2)))
        g1[ch] = [b.fadd(g, b.const_f32(1.0)) for g in gv]
    for r in range(rows):
        ro = _row_off(c, r)
        acc = [None, None]
        dsum: Dict[Tuple[int, int], Value] = {}
        for ch in range(5):
            k0 = b.add(sb, b.add(_ci(b, ch * 512), b.mul(lane, _ci(b, 8))))
            xv = _load8(b, c.rs["h"], b.add(ro, b.mul(k0, _ci(b, 2))))
            xn = [bf16r(b, b.fmul(g1[ch][i], b.fmul(rstd[r], xv[i]))) for i in range(8)]
            for j in range(2):
                wv = _bf_words(b, [b.vec_extract(wraw[(j, ch)], q) for q in range(4)])
                d = tri_tree(b, dot8(b, xn, wv), spec.mutant)
                dsum[(j, ch)] = d
                if spec.mutant != "acc_order":
                    acc[j] = d if acc[j] is None else b.fadd(acc[j], d)
        if spec.mutant == "acc_order":  # mutant: the five chunk sums in reverse order
            for j in range(2):
                for ch in (4, 3, 2, 1, 0):
                    acc[j] = dsum[(j, ch)] if acc[j] is None else b.fadd(acc[j], dsum[(j, ch)])
        base = (r * HC) * R
        with b.scf_if(b.cmp_eq(lane, _ci(b, 0))):
            for j in range(2):
                idx = b.add(_ci(b, base), b.add(b.mul(s, _ci(b, R)), ns[j]))
                MB.put(b, c.mb["part"], idx, acc[j], c.tag, c.mbk)
            with b.scf_if(b.land(b.cmp_eq(pt, _ci(b, 0)), b.cmp_eq(gw, _ci(b, 0)))):
                MB.put(b, c.mb["rstd"], b.add(_ci(b, r * HC), s), rstd[r], c.tag, c.mbk)
    bar2()


# ---------------------------------------------------------------- MIX: _hc_mix_kernel


def emit_mix(c: _Ctx, t: Value, gw: Value, lane: Value, bar1: Callable) -> None:
    """Task ``t`` = column block ``16 t .. 16 t + 15`` of ``_hc_mix_kernel`` for every row; group wave
    ``gw`` plays Triton warp ``gw``: columns ``16 t + gw + 4 i``. Lanes ``>= 40`` hold k >= 320
    (Triton's masked zeros) and contribute +0."""
    b, spec = c.b, c.spec
    rows = spec.rows
    kl = b.smin(lane, _ci(b, 39))  # clamp the k index of the masked lanes
    live = b.cmp_lt(lane, _ci(b, 40))
    hs = [b.add(b.mul(t, _ci(b, 16)), b.add(gw, _ci(b, 4 * i))) for i in range(4)]
    # w_up rows s * 2560 + h, k = 8 lane .. + 7 (issued before the polls)
    wraw = {}
    for s in range(HC):
        for i in range(4):
            off = b.mul(b.add(b.mul(b.add(_ci(b, s * H), hs[i]), _ci(b, R)), b.mul(kl, _ci(b, 8))), _ci(b, 2))
            wraw[(s, i)] = _load8_raw(b, c.rs["wup"], off)

    def bf_at(rsrc: Value, row_off: Value, e: Value) -> Value:
        """The bf16 element ``e`` (uniform) at ``row_off`` as f32 (load its word, pick the half)."""
        wd = b.buffer_load_cp(rsrc, b.add(row_off, b.mul(b.lshr(e, _ci(b, 1)), _ci(b, 4))), _ci(b, 0),
                              dwords=1, aux=0)
        lo, hi = _bf_words(b, [wd])
        return b.select(b.cmp_eq(b.land(e, _ci(b, 1)), _ci(b, 0)), lo, hi)

    # rstd(r, s) from PRE (task p = 0 of stream s)
    specs = [(c.mb["rstd"], _ci(b, r * HC + s), 1) for r in range(rows) for s in range(HC)]
    got = poll(c, specs, c.tag, "mix.rstd")
    rstd = {}
    q = 0
    for r in range(rows):
        for s in range(HC):
            rstd[(r, s)] = _i2f(b, got[q][0])
            q += 1
    lm = c.lds["mix"]  # f32 [4 rows][16]
    for r in range(rows):
        # d = ((((p0 + 0) + p1) + p2) + p3) over the four streams' partials (one stream's 8 at a time)
        d = None
        for s in range(HC):
            base = b.add(_ci(b, (r * HC + s) * R), b.mul(kl, _ci(b, 8)))
            pv = poll(c, [(c.mb["part"], b.add(base, _ci(b, 2 * qq)), 2) for qq in range(4)], c.tag, f"mix.part{r}{s}")
            if r == rows - 1 and s == HC - 1:
                if c.spec.stamps:
                    with b.scf_if(b.cmp_eq(gw, _ci(b, 0))):
                        c.stamp(3)
            ps = [_i2f(b, pv[qq][h2]) for qq in range(4) for h2 in range(2)]
            if d is None:
                d = [b.fadd(x, b.const_f32(0.0)) for x in ps]
            else:
                d = [b.fadd(d[k], ps[k]) for k in range(8)]
        gate = []
        for k in range(8):
            dk = bf16r(b, d[k])
            dk = bf16r(b, b.fmul(dk, b.const_f32(0.25)))
            gk = bf16r(b, b.fmul(sigmoid_tr(b, dk), dk))
            gate.append(b.select(live, gk, b.const_f32(0.0)))
        for i in range(4):
            tot = None
            for s in range(HC):
                wv = _bf_words(b, [b.vec_extract(wraw[(s, i)], qq) for qq in range(4)])
                wv = [b.select(live, x, b.const_f32(0.0)) for x in wv]
                u = bf16r(b, tri_tree(b, dot8(b, gate, wv), spec.mutant))
                e = b.add(_ci(b, s * H), hs[i])
                g1 = b.fadd(bf_at(c.rs["normw"], _ci(b, 0), e), b.const_f32(1.0))
                xh = bf_at(c.rs["h"], _row_off(c, r), e)
                nm = bf16r(b, b.fmul(g1, b.fmul(rstd[(r, s)], xh)))
                # an opaque copy: the backend contracts fmul + fadd (fp-contract) and Triton's sum of these
                # f32 products is not contracted (pk_mul then pk_add in its amdgcn)
                ts = _opaque(b, b.fmul(sigmoid_tr(b, u), nm))
                tot = ts if tot is None else b.fadd(tot, ts)
            mx = bf16r(b, b.fmul(tot, b.const_f32(0.25)))
            with b.scf_if(b.cmp_eq(lane, _ci(b, 0))):
                b.smem_store_vN(lm, [_ci(b, r), b.add(gw, _ci(b, 4 * i))], mx, 1)
    # gather the 16 columns of each row in LDS, wave 0 publishes the x pairs (P2's layout)
    bar1()
    with b.scf_if(b.land(b.cmp_eq(gw, _ci(b, 0)), b.cmp_lt(lane, _ci(b, 4 * rows)))):
        r = b.lshr(lane, _ci(b, 2))
        q4 = b.land(lane, _ci(b, 3))
        vals = [b.vec_extract(b.smem_load_vN(lm, r, b.add(b.mul(q4, _ci(b, 4)), _ci(b, j)), dtype=F32, n=1), 0)
                for j in range(4)]
        vals = _lds_settle(b, vals)
        col = b.add(b.mul(t, _ci(b, 16)), b.mul(q4, _ci(b, 4)))
        MB.put_bf(b, c.mb["xbox"], b.add(b.mul(r, _ci(b, H)), col), vals, c.tag, c.mbk)


def emit_pre_v2(c: _Ctx, t: Value, gw: Value, lane: Value, bar1: Callable, bar2: Callable) -> None:
    """``emit_pre`` (same task split, same arithmetic) with every load independent of the norm issued up
    front: weights, the sum-of-squares words, the 8-k slices of every row and the norm weight; the
    per-warp partials go to LDS in one region (lane 0) and one barrier."""
    b, spec = c.b, c.spec
    s = b.readfirstlane(b.div(t, _ci(b, 40)))
    pt = b.readfirstlane(b.mod(t, _ci(b, 40)))
    sb = b.mul(s, _ci(b, H))
    rows = spec.rows
    ns = [b.add(b.mul(pt, _ci(b, 8)), gw), b.add(b.add(b.mul(pt, _ci(b, 8)), gw), _ci(b, 4))]
    k0s = [b.add(sb, b.add(_ci(b, ch * 512), b.mul(lane, _ci(b, 8)))) for ch in range(5)]
    sqw = {}
    for r in range(rows):
        ro = _row_off(c, r)
        for ch in range(5):
            e = b.add(sb, b.add(_ci(b, ch * 512), b.add(b.mul(gw, _ci(b, 128)), b.mul(lane, _ci(b, 2)))))
            sqw[(r, ch)] = b.buffer_load_cp(c.rs["h"], b.add(ro, b.mul(e, _ci(b, 2))), _ci(b, 0), dwords=1, aux=0)
    wraw = {}
    for j, n in enumerate(ns):
        for ch in range(5):
            off = b.mul(b.add(b.mul(n, _ci(b, HCW)), k0s[ch]), _ci(b, 2))
            wraw[(j, ch)] = _load8_raw(b, c.rs["wcat"], off)
    xraw = {}
    for r in range(rows):
        ro = _row_off(c, r)
        for ch in range(5):
            xraw[(r, ch)] = _load8_raw(b, c.rs["h"], b.add(ro, b.mul(k0s[ch], _ci(b, 2))))
    graw = [_load8_raw(b, c.rs["normw"], b.mul(k0s[ch], _ci(b, 2))) for ch in range(5)]
    lw = c.lds["prew"]  # f32 [4 waves][4 rows][8]
    wsum = {}
    per_row = spec.prer == "row"

    def store_w(rs_: Sequence[int]) -> None:
        with b.scf_if(b.cmp_eq(lane, _ci(b, 0))):
            for r in rs_:
                for ch in range(5):
                    b.smem_store_vN(lw, [gw, _ci(b, r), _ci(b, ch)], wsum[(r, ch)], 1)

    for r in range(rows):
        for ch in range(5):
            x0, x1 = _bf_words(b, [sqw[(r, ch)]])
            wsum[(r, ch)] = tri_tree(b, b.fadd(b.fmul(x0, x0), b.fmul(x1, x1)), spec.mutant)
        if per_row:
            store_w([r])
    if spec.stamps:
        _gw0(b, gw, lambda: c.stamp(16))
    if not per_row:
        store_w(range(rows))
    bar1()
    if spec.stamps:
        _gw0(b, gw, lambda: c.stamp(17))
    rstd = []
    for r in range(rows):
        sc = []
        for ch in range(5):
            wv = [b.vec_extract(b.smem_load_vN(lw, _ci(b, w), _ci(b, r), _ci(b, ch), dtype=F32, n=1), 0)
                  for w in range(4)]
            wv = _lds_settle(b, wv, spec.opq == "pure")
            sc.append(b.fadd(b.fadd(wv[0], wv[2]), b.fadd(wv[1], wv[3])))
        if spec.mutant == "sumsq_order":
            ss = b.fadd(b.fadd(b.fadd(b.fadd(sc[0], sc[1]), sc[2]), sc[3]), sc[4])
        else:
            ss = b.fadd(b.fadd(b.fadd(sc[0], sc[1]), sc[2]), b.fadd(sc[3], sc[4]))
        rstd.append(b.rsqrt(b.fadd(b.fdiv(ss, b.const_f32(float(H))), c.p["eps"])))
    g1 = {ch: [b.fadd(g, b.const_f32(1.0)) for g in _bf_words(b, [b.vec_extract(graw[ch], q) for q in range(4)])]
          for ch in range(5)}
    accs = []

    def put_rows(rs_: Sequence[int]) -> None:
        with b.scf_if(b.cmp_eq(lane, _ci(b, 0))):
            for r in rs_:
                base = (r * HC) * R
                for j in range(2):
                    idx = b.add(_ci(b, base), b.add(b.mul(s, _ci(b, R)), ns[j]))
                    MB.put(b, c.mb["part"], idx, accs[r][j], c.tag, c.mbk)
            with b.scf_if(b.land(b.cmp_eq(pt, _ci(b, 0)), b.cmp_eq(gw, _ci(b, 0)))):
                for r in rs_:
                    MB.put(b, c.mb["rstd"], b.add(_ci(b, r * HC), s), rstd[r], c.tag, c.mbk)

    for r in range(rows):
        acc = [None, None]
        dsum: Dict[Tuple[int, int], Value] = {}
        for ch in range(5):
            xv = _bf_words(b, [b.vec_extract(xraw[(r, ch)], q) for q in range(4)])
            xn = [bf16r(b, b.fmul(g1[ch][i], b.fmul(rstd[r], xv[i]))) for i in range(8)]
            for j in range(2):
                wv = _bf_words(b, [b.vec_extract(wraw[(j, ch)], q) for q in range(4)])
                d = tri_tree(b, dot8(b, xn, wv), spec.mutant)
                dsum[(j, ch)] = d
                if spec.mutant != "acc_order":
                    acc[j] = d if acc[j] is None else b.fadd(acc[j], d)
        if spec.mutant == "acc_order":
            for j in range(2):
                for ch in (4, 3, 2, 1, 0):
                    acc[j] = dsum[(j, ch)] if acc[j] is None else b.fadd(acc[j], dsum[(j, ch)])
        accs.append(acc)
        if per_row:
            put_rows([r])
    if spec.stamps:
        _gw0(b, gw, lambda: c.stamp(18))
    if not per_row:
        put_rows(range(rows))
    bar2()


def _gw0(b: IRBuilder, gw: Value, fn: Callable[[], None]) -> None:
    """``fn`` on group wave 0 only (task-group stamps)."""
    with b.scf_if(b.cmp_eq(gw, _ci(b, 0))):
        fn()


def emit_mix_v2(c: _Ctx, t: Value, gw: Value, lane: Value, bar0: Callable, bar1: Callable,
                pin: Optional[Callable[[], None]] = None) -> None:
    """``emit_mix`` (same task split, same arithmetic) with the w_up rows and the task's 16 norm-weight /
    input elements issued before the polls, and the gate computed once per row: wave ``r`` polls row
    ``r``'s 1280 parts and its four rstd in 16 specs (lanes >= 40, which hold no k, poll the rstd pairs in
    place of their first two specs; ``rstd`` follows ``part`` in the scratch), computes the gate, and shares
    gate + rstd through LDS (one barrier). ``v2`` polls in two batches, ``v2w`` in one."""
    b, spec = c.b, c.spec
    rows = spec.rows
    kl = b.smin(lane, _ci(b, 39))
    live = b.cmp_lt(lane, _ci(b, 40))
    hs = [b.add(b.mul(t, _ci(b, 16)), b.add(gw, _ci(b, 4 * i))) for i in range(4)]
    wraw = {}
    for s in range(HC):
        for i in range(4):
            off = b.mul(b.add(b.mul(b.add(_ci(b, s * H), hs[i]), _ci(b, R)), b.mul(kl, _ci(b, 8))), _ci(b, 2))
            wraw[(s, i)] = _load8_raw(b, c.rs["wup"], off)

    def word_at(rsrc: Value, row_off: Value, e: Value) -> Value:
        return b.buffer_load_cp(rsrc, b.add(row_off, b.mul(b.lshr(e, _ci(b, 1)), _ci(b, 4))), _ci(b, 0),
                                dwords=1, aux=0)

    def half_of(wd: Value, e: Value) -> Value:
        lo, hi = _bf_words(b, [wd])
        return b.select(b.cmp_eq(b.land(e, _ci(b, 1)), _ci(b, 0)), lo, hi)

    # the task's 16 norm-weight and 16 x rows elements, one per lane: lane q = (r, s, i) = (q // 16, q // 4 % 4,
    # q % 4) loads x[r][s H + 16 t + gw + 4 i] and the norm weight of (s, i); used through readlane
    qr = b.lshr(lane, _ci(b, 4))
    qs = b.land(b.lshr(lane, _ci(b, 2)), _ci(b, 3))
    qi = b.land(lane, _ci(b, 3))
    qe = b.add(b.mul(qs, _ci(b, H)), b.add(b.mul(t, _ci(b, 16)), b.add(gw, b.mul(qi, _ci(b, 4)))))
    qrow = b.mul(b.add(c.sel, b.smin(qr, _ci(b, rows - 1))), _ci(b, HCW * 2))
    gq = b.fadd(half_of(word_at(c.rs["normw"], _ci(b, 0), qe), qe), b.const_f32(1.0))
    xq = half_of(word_at(c.rs["h"], qrow, qe), qe)
    lg = c.lds["gate"]  # i32 [4 rows][320] (f32 bits)
    lr = c.lds["mrstd"]  # f32 [4 rows][4 streams]
    with b.scf_if(b.cmp_lt(gw, _ci(b, rows))):
        r = gw
        mbp = c.mb["part"]
        hi = b.cmp_ge(lane, _ci(b, 40))
        rrel = (SCR["rstd"] - SCR["part"]) // MB.PAIR_BYTES
        specs = []
        for s in range(HC):
            for qq in range(4):
                idx = b.add(b.add(b.mul(b.add(b.mul(r, _ci(b, HC)), _ci(b, s)), _ci(b, R)), b.mul(kl, _ci(b, 8))),
                            _ci(b, 2 * qq))
                if s == 0 and qq < 2:
                    idx = b.select(hi, b.add(b.mul(r, _ci(b, HC)), _ci(b, rrel + 2 * qq)), idx)
                specs.append((mbp, idx, 2))
        got = poll(c, specs, c.tag, "mix.row", pmax=len(specs) if spec.lat == "v2w" else 0)
        rsv = [b.readlane(got[q][h2], _ci(b, 40)) for q in range(2) for h2 in range(2)]
        if c.spec.stamps:
            with b.scf_if(b.cmp_eq(gw, _ci(b, 0))):
                c.stamp(3)
        d = None
        for s in range(HC):
            ps = [_i2f(b, got[s * 4 + qq][h2]) for qq in range(4) for h2 in range(2)]
            d = [b.fadd(x, b.const_f32(0.0)) for x in ps] if d is None else [b.fadd(d[k], ps[k]) for k in range(8)]
        gate = []
        for k in range(8):
            dk = bf16r(b, d[k])
            dk = bf16r(b, b.fmul(dk, b.const_f32(0.25)))
            gate.append(_f2i(b, bf16r(b, b.fmul(sigmoid_tr(b, dk), dk))))
        with b.scf_if(live):
            for h4 in range(2):
                b.smem_store_vN(lg, [r, b.add(b.mul(lane, _ci(b, 8)), _ci(b, 4 * h4))],
                                b.vec_pack(gate[4 * h4: 4 * h4 + 4], I32), 4)
        with b.scf_if(b.cmp_eq(lane, _ci(b, 0))):
            for s in range(HC):
                b.smem_store_vN(lr, [r, _ci(b, s)], _i2f(b, rsv[s]), 1)
    bar0()
    if spec.stamps:
        _gw0(b, gw, lambda: c.stamp(19))
    lm = c.lds["mix"]  # f32 [4 rows][16]

    def row(r: Value) -> None:
        gv = []
        for h4 in range(2):
            v = b.smem_load_vN(lg, r, b.add(b.mul(kl, _ci(b, 8)), _ci(b, 4 * h4)), dtype=I32, n=4)
            gv += [_i2f(b, b.vec_extract(v, q)) for q in range(4)]
        rv = [b.vec_extract(b.smem_load_vN(lr, r, _ci(b, s), dtype=F32, n=1), 0) for s in range(HC)]
        st = _lds_settle(b, gv + rv, spec.opq == "pure")
        gate = [b.select(live, x, b.const_f32(0.0)) for x in st[:8]]
        rstd = st[8:]
        for i in range(4):
            tot = None
            for s in range(HC):
                wv = _bf_words(b, [b.vec_extract(wraw[(s, i)], qq) for qq in range(4)])
                wv = [b.select(live, x, b.const_f32(0.0)) for x in wv]
                u = bf16r(b, tri_tree(b, dot8(b, gate, wv), spec.mutant))
                g1 = b.readlane(gq, _ci(b, s * 4 + i))
                xh = b.readlane(xq, b.add(b.mul(r, _ci(b, 16)), _ci(b, s * 4 + i)))
                nm = bf16r(b, b.fmul(g1, b.fmul(rstd[s], xh)))
                ts = _opaque(b, b.fmul(sigmoid_tr(b, u), nm), spec.opq == "pure")
                tot = ts if tot is None else b.fadd(tot, ts)
            mx = bf16r(b, b.fmul(tot, b.const_f32(0.25)))
            with b.scf_if(b.cmp_eq(lane, _ci(b, 0))):
                b.smem_store_vN(lm, [r, b.add(gw, _ci(b, 4 * i))], mx, 1)

    if spec.mloop and rows > 1:
        with b.scf_for(_ci(b, 0), _ci(b, rows), _ci(b, 1), iv_name="p9mr") as r:
            row(r)
    else:
        for r in range(rows):
            row(_ci(b, r))
            if r == 0 and spec.stamps:
                _gw0(b, gw, lambda: c.stamp(20))
    if spec.stamps:
        _gw0(b, gw, lambda: c.stamp(21))
    if pin is not None:
        pin()
    bar1()
    with b.scf_if(b.land(b.cmp_eq(gw, _ci(b, 0)), b.cmp_lt(lane, _ci(b, 4 * rows)))):
        r = b.lshr(lane, _ci(b, 2))
        q4 = b.land(lane, _ci(b, 3))
        vals = [b.vec_extract(b.smem_load_vN(lm, r, b.add(b.mul(q4, _ci(b, 4)), _ci(b, j)), dtype=F32, n=1), 0)
                for j in range(4)]
        vals = _lds_settle(b, vals, spec.opq == "pure")
        col = b.add(b.mul(t, _ci(b, 16)), b.mul(q4, _ci(b, 4)))
        MB.put_bf(b, c.mb["xbox"], b.add(b.mul(r, _ci(b, H)), col), vals, c.tag, c.mbk)


# ---------------------------------------------------------------- FRONT norm (pre_fc_norm_hidden bundle)


def _ss_rows(c: _Ctx, rsrc: Value, row_off: Value, kbase: Value, lane: Value, mutant: str) -> Value:
    """Triton blocked [*, 512] sum of squares over 2560 elements at ``row_off + 2 (kbase + k)``:
    ``((((0 + s0) + s1) + s2) + s3) + s4`` of five warp trees of per-thread 8-element sums."""
    b = c.b
    acc = b.const_f32(0.0)
    for ch in range(5):
        k0 = b.add(kbase, b.add(_ci(b, ch * 512), b.mul(lane, _ci(b, 8))))
        xv = _load8(b, rsrc, b.add(row_off, b.mul(k0, _ci(b, 2))))
        acc = b.fadd(acc, tri_tree(b, dot8(b, xv, xv), mutant))
    return acc


def emit_norm(c: _Ctx, gw: Value, lane: Value, bar1: Callable) -> None:
    """CTA 243, waves 0-3: ``pre_fc_norm_hidden``'s bundle rstd per token row (DraftFront: ss over each
    stream, then ``(s0 + s2) + (s1 + s3)``, ``rsqrt(eps + ss / 10240)``); wave ``gw`` = stream ``gw``."""
    b, spec = c.b, c.spec
    ln = c.lds["norm"]  # f32 [4 streams][4 tokens]
    for r in range(spec.rows):
        ss = _ss_rows(c, c.rs["h"], _row_off(c, r), b.mul(gw, _ci(b, H)), lane, spec.mutant)
        with b.scf_if(b.cmp_eq(lane, _ci(b, 0))):
            b.smem_store_vN(ln, [gw, _ci(b, r)], ss, 1)
    bar1()
    with b.scf_if(b.cmp_eq(gw, _ci(b, 0))):
        for r in range(spec.rows):
            v = [b.vec_extract(b.smem_load_vN(ln, _ci(b, s), _ci(b, r), dtype=F32, n=1), 0) for s in range(HC)]
            v = _lds_settle(b, v)
            bundle = b.fadd(b.fadd(v[0], v[2]), b.fadd(v[1], v[3]))
            rs = b.rsqrt(b.fadd(c.p["epsh"], b.fdiv(bundle, b.const_f32(float(HCW)))))
            with b.scf_if(b.cmp_eq(lane, _ci(b, 0))):
                MB.put(b, c.mb["rstdh"], _ci(b, r), rs, c.tag, c.mbk)


# ---------------------------------------------------------------- FRONT: _draft_front_kernel


def emit_front(c: _Ctx, t: Value, gw: Value, lane: Value, bar_we: Callable, bar_ss: Callable,
               bar_spec: Callable, get_ids: Callable[[], List[Value]]) -> None:
    """Task ``t`` = output columns ``n0 = 16 t .. + 15`` of ``_draft_front_kernel`` (all rows, all four
    streams: 16 MFMA columns ``m = 4 token + stream``). Wave 0 of the group runs both MFMA chains
    (``tl.dot`` order: ten 256-k tiles, eight ``16x16x32`` steps each, weights as the A operand); waves
    0-3 stage ``fc_embedding``'s 16 rows in LDS; the embedding rows and norms are one token per wave
    (``espec=1``: the speculative ids' rows and norms before the accept, only changed rows after)."""
    b, spec = c.b, c.spec
    rows = spec.rows
    c0, c1 = _ci(b, 0), _ci(b, 1)
    n0 = b.mul(t, _ci(b, 16))
    m = b.land(lane, _ci(b, 15))  # MFMA column = row m = 4 token + stream
    kq = b.mul(b.lshr(lane, _ci(b, 4)), _ci(b, 8))  # k offset of this lane inside a 32-chunk
    tok = b.lshr(m, _ci(b, 2))
    strm = b.land(m, _ci(b, 3))
    mlive = b.cmp_lt(tok, _ci(b, rows))
    tokc = b.smin(tok, _ci(b, rows - 1))
    lwe = c.lds["we"]  # <4 x i32> [80 chunks][64 lanes]
    ler = c.lds["erow"]  # i32 [4 tokens][1280] (bf16 pairs)
    lge = c.lds["ge"]  # i32 [1280]
    rev = spec.mutant == "mma_order"

    def chunk_of(i: Value) -> Value:
        return b.sub(_ci(b, 79), i) if rev else i

    # ---- stage fc_embedding rows n0 .. n0 + 15 (A fragments) in LDS: wave gw takes chunks gw mod 4
    staged = []
    for cc in range(20):
        ch = b.add(_ci(b, 4 * cc), gw)
        off = b.mul(b.add(b.mul(b.add(n0, m), _ci(b, H)), b.add(b.mul(ch, _ci(b, 32)), kq)), _ci(b, 2))
        staged.append((ch, _load8_raw(b, c.rs["we"], off)))
    for ch, v in staged:
        b.smem_store_vN(lwe, [ch, lane], v, 4)
    # ---- hidden half (wave 0): needs the bundle rstd of every token
    lacc = c.lds["acch"]  # f32 [4][64] the hidden accumulators
    with b.scf_if(b.cmp_eq(gw, c0)):
        got = poll(c, [(c.mb["rstdh"], _ci(b, r), 1) for r in range(rows)], c.tag, "front.rstdh")
        rsh = [_i2f(b, g[0]) for g in got]
        rlane = rsh[0]
        for r in range(1, rows):
            rlane = b.select(b.cmp_eq(tokc, _ci(b, r)), rsh[r], rlane)
        xro = b.add(_row_off(c, 0), b.mul(tokc, _ci(b, HCW * 2)))
        loop = b.scf_for_iter(c0, _ci(b, 10), c1, [("acch", b.zero_vec_f32(4))], iv_name="p9th")
        with loop as (it, vals):
            acc = vals[0]
            ld = []
            for j in range(8):
                ch = chunk_of(b.add(b.mul(it, _ci(b, 8)), _ci(b, j)))
                k = b.add(b.mul(ch, _ci(b, 32)), kq)
                woff = b.mul(b.add(b.mul(b.add(n0, m), _ci(b, H)), k), _ci(b, 2))
                ke = b.add(b.mul(strm, _ci(b, H)), k)
                ld.append((_load8_raw(b, c.rs["wh"], woff),
                           _load8_raw(b, c.rs["h"], b.add(xro, b.mul(ke, _ci(b, 2)))),
                           _load8_raw(b, c.rs["gh"], b.mul(ke, _ci(b, 2)))))
            for wr, xr, gr in ld:
                xv = _bf_words(b, [b.vec_extract(xr, q) for q in range(4)])
                gv = _bf_words(b, [b.vec_extract(gr, q) for q in range(4)])
                if spec.mutant == "hn_order":  # mutant: x * (rstd * g) instead of (x * rstd) * g
                    hn = [bf16r(b, b.fmul(xv[i], b.fmul(rlane, b.fadd(gv[i], b.const_f32(1.0))))) for i in range(8)]
                else:
                    hn = [bf16r(b, b.fmul(b.fmul(xv[i], rlane), b.fadd(gv[i], b.const_f32(1.0)))) for i in range(8)]
                hn = [b.select(mlive, x, b.const_f32(0.0)) for x in hn]
                acc = b.mfma_f32_16x16x32_bf16(b.vec_bitcast(wr, VectorType(BF16, 8)), _pack8(b, hn), acc)
            b.scf_yield(acc)
        acc_h = loop.results[0]
        c.stamp(5)
        for i in range(4):
            b.smem_store_vN(lacc, [_ci(b, i), lane], b.vec_extract(acc_h, i), 1)

    # ---- embedding rows (LDS) and norms: wave gw = token gw
    lss = c.lds["ss"]  # f32 [4]

    pre = spec.fnorm == "pre"

    def stage_row(r: int, idv: Value) -> None:
        eo = b.mul(idv, _ci(b, H * 2))
        acc = b.const_f32(0.0)
        raws = []
        for ch in range(5):
            k0 = b.add(_ci(b, ch * 512), b.mul(lane, _ci(b, 8)))
            raws.append((k0, _load8_raw(b, c.rs["emb"], b.add(eo, b.mul(k0, _ci(b, 2))))))
        if pre:
            graws = [_load8_raw(b, c.rs["ge"], b.mul(k0, _ci(b, 2))) for k0, _ in raws]
        for k0, raw in raws:
            if not pre:
                b.smem_store_vN(ler, [_ci(b, r), b.lshr(k0, c1)], raw, 4)
            xv = _bf_words(b, [b.vec_extract(raw, q) for q in range(4)])
            acc = b.fadd(acc, tri_tree(b, dot8(b, xv, xv), spec.mutant))
        if not pre:
            with b.scf_if(b.cmp_eq(lane, c0)):
                b.smem_store_vN(lss, [_ci(b, r)], acc, 1)
            return
        # fnorm=pre: the chain's operand bf16((e rstd_e) (g + 1)), once per element, rstd_e as wave 0 computes it
        rse_ = b.rsqrt(b.fadd(c.p["epse"], b.fdiv(acc, b.const_f32(float(H)))))
        for (k0, raw), graw in zip(raws, graws):
            ev = _bf_words(b, [b.vec_extract(raw, q) for q in range(4)])
            gv = _bf_words(b, [b.vec_extract(graw, q) for q in range(4)])
            if spec.mutant == "en_order":  # mutant: e * (rstd * (g + 1)) instead of (e * rstd) * (g + 1)
                en = [bf16r(b, b.fmul(ev[i], b.fmul(rse_, b.fadd(gv[i], b.const_f32(1.0))))) for i in range(8)]
            else:
                en = [bf16r(b, b.fmul(b.fmul(ev[i], rse_), b.fadd(gv[i], b.const_f32(1.0)))) for i in range(8)]
            words = [MB.bf16_pair(b, en[2 * i], en[2 * i + 1]) for i in range(4)]
            b.smem_store_vN(ler, [_ci(b, r), b.lshr(k0, c1)], b.vec_pack(words, I32), 4)

    if not pre:
        with b.scf_if(b.cmp_eq(gw, c0)):  # pre_fc_norm_embedding's weight
            for ch in range(5):
                k0 = b.add(_ci(b, ch * 512), b.mul(lane, _ci(b, 8)))
                b.smem_store_vN(lge, [b.lshr(k0, c1)], _load8_raw(b, c.rs["ge"], b.mul(k0, _ci(b, 2))), 4)
    spec_ids = None
    if spec.mode == "verify" and spec.espec:
        if c.dids0 is not None:
            spec_ids = list(c.dids0[:rows])
        else:
            spec_ids = [b.buffer_load_cp(c.rs["dids"], _ci(b, 4 * j), c0, dwords=1, aux=0) for j in range(rows)]
            if spec.lat != "v1":
                # one value per id for the staging AND the re-stage test: the accept rewrites dids during the
                # launch, and a re-materialized load (register pressure) would read the new id at the test
                spec_ids = [dops.fresh(b, v) for v in spec_ids]
        for r in range(rows):
            with b.scf_if(b.cmp_eq(gw, _ci(b, r))):
                stage_row(r, spec_ids[r])
    if spec.idwait == "wave0" and not (spec.mode == "verify" and (spec.mutant == "stale_ids" or spec.dbg & 2)):
        lid = c.lds["ids"]
        with b.scf_if(b.cmp_eq(gw, c0)):
            got_ids = get_ids()
            with b.scf_if(b.cmp_eq(lane, c0)):
                for r in range(rows):
                    b.smem_store_vN(lid, [_ci(b, r)], got_ids[r], 1)
        bar_spec()
        ids = [b.vec_extract(b.smem_load_vN(lid, _ci(b, r), dtype=I32, n=1), 0) for r in range(rows)]
        if c.spec.stamps:
            with b.scf_if(b.cmp_eq(gw, c0)):
                c.stamp(6)
    else:
        ids = get_ids()  # uniform i32 per token row
    for r in range(rows):
        cond = b.cmp_eq(gw, _ci(b, r))
        if spec_ids is not None:
            cond = b.land(cond, b.cmp_ne(ids[r], spec_ids[r]))
        with b.scf_if(cond):
            stage_row(r, ids[r])
    bar_we()  # W_e, ge, the rows and their norms are in LDS
    if c.spec.stamps:
        with b.scf_if(b.cmp_eq(gw, c0)):
            c.stamp(7)
    with b.scf_if(b.cmp_eq(gw, c0)):
        rlane = None
        if not pre:
            rse = []
            for r in range(rows):
                ss = _lds_settle(b, [b.vec_extract(b.smem_load_vN(lss, _ci(b, r), dtype=F32, n=1), 0)])[0]
                rse.append(b.rsqrt(b.fadd(c.p["epse"], b.fdiv(ss, b.const_f32(float(H))))))
            rlane = rse[0]
            for r in range(1, rows):
                rlane = b.select(b.cmp_eq(tokc, _ci(b, r)), rse[r], rlane)
        loop = b.scf_for_iter(c0, _ci(b, 10), c1, [("acce", b.zero_vec_f32(4))], iv_name="p9te")
        with loop as (it, vals):
            acc = vals[0]
            ld = []
            for j in range(8):
                ch = chunk_of(b.add(b.mul(it, _ci(b, 8)), _ci(b, j)))
                kw = b.lshr(b.add(b.mul(ch, _ci(b, 32)), kq), c1)  # word index of k
                ld.append((b.smem_load_vN(lwe, ch, lane, dtype=I32, n=4),
                           b.smem_load_vN(ler, tokc, kw, dtype=I32, n=4),
                           None if pre else b.smem_load_vN(lge, kw, dtype=I32, n=4)))
            if pre:
                for wr, er, _ in ld:
                    en = b.vec_pack([b.select(mlive, b.vec_extract(er, q), c0) for q in range(4)], I32)
                    acc = b.mfma_f32_16x16x32_bf16(b.vec_bitcast(wr, VectorType(BF16, 8)),
                                                   b.vec_bitcast(en, VectorType(BF16, 8)), acc)
                ld = []
            for wr, er, gr in ld:
                ev = _bf_words(b, [b.vec_extract(er, q) for q in range(4)])
                gv = _bf_words(b, [b.vec_extract(gr, q) for q in range(4)])
                if spec.mutant == "en_order":
                    en = [bf16r(b, b.fmul(ev[i], b.fmul(rlane, b.fadd(gv[i], b.const_f32(1.0))))) for i in range(8)]
                else:
                    en = [bf16r(b, b.fmul(b.fmul(ev[i], rlane), b.fadd(gv[i], b.const_f32(1.0)))) for i in range(8)]
                en = [b.select(mlive, x, b.const_f32(0.0)) for x in en]
                acc = b.mfma_f32_16x16x32_bf16(b.vec_bitcast(wr, VectorType(BF16, 8)), _pack8(b, en), acc)
            b.scf_yield(acc)
        acc_e = loop.results[0]
        c.stamp(8)
        # y = bf16(bf16(acc_h) + bf16(acc_e)); lane holds n = 4 (lane / 16) + i of column m
        nb = b.mul(b.lshr(lane, _ci(b, 4)), _ci(b, 4))
        with b.scf_if(mlive):
            for i in range(4):
                ah = b.vec_extract(b.smem_load_vN(lacc, _ci(b, i), lane, dtype=F32, n=1), 0)
                y = b.fadd(bf16r(b, ah), bf16r(b, b.vec_extract(acc_e, i)))
                col = b.add(b.add(b.mul(strm, _ci(b, H)), n0), b.add(nb, _ci(b, i)))
                b.global_store(c.p["hin"], b.add(b.mul(tokc, _ci(b, HCW)), col), b.trunc_f32_to_bf16(y))


# ---------------------------------------------------------------- ACCEPT / draft token (fold's last CTA)


def _w0(b: IRBuilder, fn: Callable[[], None]) -> None:
    """``fn`` on wave 0 only (stamp hooks inside a wave group)."""
    with b.scf_if(b.cmp_eq(b.readfirstlane(b.lshr(b.thread_id_x(), _ci(b, 6))), _ci(b, 0))):
        fn()


#: stamp slots (i64 ``s_memrealtime`` per CTA): 0 entry, 2 PRE done, 3 MIX parts polled, 4 MIX published, 5 FRONT
#: hidden chain done, 6 FRONT ids received, 7 FRONT rows / norms staged, 8 FRONT stored, 9 head first panels
#: issued, 10 head x staged, 11 head loop end, 12 head end (last CTA: fold + accept), 13 NORM published, 14 META
#: done, 15 task CTA wave 4 end
STAMP_SLOTS = 32
STAMP_POINTS = {16: "pre_sq", 17: "pre_bar1", 18: "pre_comp", 19: "mix_gate", 20: "mix_row0", 21: "mix_comp",
                22: "head_xhint", 23: "head_xpoll0"}
STAMP_POINTS.update({0: "entry", 1: "accept_done", 2: "pre_done", 3: "mix_parts_in", 4: "mix_pub", 5: "front_hidden_done",
                6: "front_ids_in", 7: "front_staged", 8: "front_store", 9: "head_issue", 10: "head_x_in",
                11: "head_loop_end", 12: "head_end", 13: "norm_pub", 14: "meta_done", 15: "task_end"})


def emit_accept(c: _Ctx, cf: Value, rr: Value) -> None:
    """Verify: ``rejection_greedy_sample_kernel`` for one request with 3 drafts on the folded argmax
    (rows 0..2 the targets, row 3 the bonus), ``next_token_ids = gather(out, a)`` and the drafter's
    ``input_ids.scatter_(anchor, next)``; publishes the drafter ids. Draft: the token. Runs on every
    head wave of the last-arriving head CTA (wave 0 publishes)."""
    b, spec = c.b, c.spec
    wave = b.readfirstlane(b.lshr(b.thread_id_x(), _ci(b, 6)))
    lane = b.lane_id()
    if spec.mode == "draft":
        tokv = b.readlane(cf, _ci(b, 0))
        with b.scf_if(b.land(b.cmp_eq(wave, _ci(b, 0)), b.cmp_eq(lane, _ci(b, 0)))):
            b.global_store(c.p["outtok"], _ci(b, 0), tokv)
            MB.put(b, c.mb["ids"], _ci(b, 0), tokv, c.tag, c.mbk)
        if c.spec.stamps:
            with b.scf_if(b.cmp_eq(wave, _ci(b, 0))):
                c.stamp(1)
        return
    tg = [b.readlane(cf, _ci(b, j)) for j in range(4)]  # rows 0..3 (target rows 0..2, bonus row 3)
    with b.scf_if(b.cmp_eq(wave, _ci(b, 0))):
        # drafts: input_ids[target_logits_indices[r] + 1] = ids[1 + r]
        dr = [b.buffer_load_cp(c.rs["ids"], _ci(b, 4 * (1 + j)), _ci(b, 0), dwords=1, aux=0) for j in range(3)]
        rejected = b.const_i32(0)
        nb = _ci(b, -1)
        outs = []
        for pos in range(3):
            notrej = b.cmp_eq(rejected, _ci(b, 0))
            outs.append(b.select(notrej, tg[pos], _ci(b, -1)))
            mism = b.cmp_ne(dr[pos], tg[pos])
            rejected = b.select(b.land(notrej, mism), _ci(b, 1), rejected)
            nb = b.select(notrej, b.add(nb, _ci(b, 1)), nb)
        rej = b.cmp_ne(rejected, _ci(b, 0))
        acc_n = b.select(rej, nb, b.add(nb, _ci(b, 1)))
        if spec.mutant == "accept_ooo":
            acc_n = b.smin(b.add(acc_n, _ci(b, 1)), _ci(b, 3))
        outs.append(b.select(rej, _ci(b, -1), tg[3]))
        anchor = acc_n  # bonus_logits_indices[0] - num_draft_tokens (= 3 - 3) + accepted
        # next_token_ids = out[accepted]
        nxt = outs[0]
        for j in range(1, 4):
            nxt = b.select(b.cmp_eq(acc_n, _ci(b, j)), outs[j], nxt)
        # the drafter's ids: input_ids.gpu[1:5] with [anchor] := next
        dids = [b.buffer_load_cp(c.rs["dids"], _ci(b, 4 * j), _ci(b, 0), dwords=1, aux=0) for j in range(4)]
        nids = [b.select(b.cmp_eq(anchor, _ci(b, j)), nxt, dids[j]) for j in range(4)]
        c.stamp(1)
        with b.scf_if(b.cmp_eq(lane, _ci(b, 0))):
            for j in range(4):
                b.global_store(c.p["outtok"], _ci(b, j), outs[j])
                b.global_store(c.p["dids"], _ci(b, j), nids[j])
            b.global_store(c.p["verdict"], _ci(b, 0), acc_n)
            b.global_store(c.p["verdict"], _ci(b, 1), b.sub(_ci(b, 3), acc_n))
            b.global_store(c.p["verdict"], _ci(b, 2), anchor)
            b.global_store(c.p["nexttok"], _ci(b, 0), nxt)
            for j in range(4):
                MB.put(b, c.mb["ids"], _ci(b, j), nids[j], c.tag, c.mbk)


def fold_direct(c: _Ctx, tag: Value) -> List[Value]:
    """fold=direct, one wave: poll the 243 head CTAs' (key, column) slots of every row (lane ``l`` takes
    CTAs ``l + 64 j``, clamped), fold them with the head's argmax order (a strict total order on (key,
    column), so any fold order gives the fused fold's answer) and return the uniform column per row."""
    b, rows = c.b, c.spec.rows
    lane = b.lane_id()
    mbk = c.mb["keys"]
    specs = []
    for r in range(rows):
        for j in range(4):
            q = b.smin(b.add(lane, _ci(b, 64 * j)), _ci(b, HEAD_CTAS - 1))
            specs.append((mbk, b.mul(b.add(_ci(b, r * HEAD_CTAS), q), _ci(b, 2)), 2))
    got = poll(c, specs, tag, "front.keys", pmax=len(specs) if len(specs) > MB.POLL_MAX else 0)

    def pick(k1, c1, k2, c2):
        w = P2._better_low(b, k2, c2, k1, c1)
        return b.select(w, k2, k1), b.select(w, c2, c1)

    out = []
    for r in range(rows):
        k, cc = got[4 * r]
        for j in range(1, 4):
            k, cc = pick(k, cc, *got[4 * r + j])
        for m in (1, 2, 4, 8, 16, 32):
            k, cc = pick(k, cc, b.warp_shuffle_xor(k, m), b.warp_shuffle_xor(cc, m))
        out.append(b.readfirstlane(cc))
    return out


def accept_direct(c: _Ctx, tg: List[Value], dids: List[Value], writer: Value) -> List[Value]:
    """fold=direct: ``emit_accept``'s logic on the folded columns ``tg`` (uniform) in one wave of every FRONT
    task; ``dids`` and ``c.dr0`` (the step's draft ids) are read before this task's MIX published, so before
    any task can have rewritten them (the served layout aliases them: ATOM's ``input_ids`` buffer holds both,
    ``ids[1 + j]`` = ``dids[j]``, and P9 found the race: a task reading them after task 0's scatter took the
    new ids as drafts); ``writer`` (task 0) stores y and the outputs. Returns the ids the FRONT embeds."""
    b, spec = c.b, c.spec
    lane = b.lane_id()
    wl = b.land(writer, b.cmp_eq(lane, _ci(b, 0)))
    if spec.mode == "draft":
        tok = tg[0]
        if spec.vsub:  # N8: subset column -> token id (uniform: every lane loads the same word)
            tok = b.readfirstlane(b.global_load(c.p["vmap"], tg[0], I32))
        with b.scf_if(wl):
            b.global_store(c.p["y"], _ci(b, 0), tg[0])
            b.global_store(c.p["outtok"], _ci(b, 0), tok)
        if spec.stamps:
            with b.scf_if(writer):
                c.stamp(1)
        return [tok]
    dr = c.dr0
    rejected = b.const_i32(0)
    nb = _ci(b, -1)
    outs = []
    for pos in range(3):
        notrej = b.cmp_eq(rejected, _ci(b, 0))
        outs.append(b.select(notrej, tg[pos], _ci(b, -1)))
        mism = b.cmp_ne(dr[pos], tg[pos])
        rejected = b.select(b.land(notrej, mism), _ci(b, 1), rejected)
        nb = b.select(notrej, b.add(nb, _ci(b, 1)), nb)
    rej = b.cmp_ne(rejected, _ci(b, 0))
    acc_n = b.select(rej, nb, b.add(nb, _ci(b, 1)))
    if spec.mutant == "accept_ooo":
        acc_n = b.smin(b.add(acc_n, _ci(b, 1)), _ci(b, 3))
    outs.append(b.select(rej, _ci(b, -1), tg[3]))
    anchor = acc_n
    nxt = outs[0]
    for j in range(1, 4):
        nxt = b.select(b.cmp_eq(acc_n, _ci(b, j)), outs[j], nxt)
    nids = [b.select(b.cmp_eq(anchor, _ci(b, j)), nxt, dids[j]) for j in range(4)]
    if spec.stamps:
        with b.scf_if(writer):
            c.stamp(1)
    with b.scf_if(wl):
        for j in range(4):
            b.global_store(c.p["y"], _ci(b, j), tg[j])
            b.global_store(c.p["outtok"], _ci(b, j), outs[j])
            b.global_store(c.p["dids"], _ci(b, j), nids[j])
        b.global_store(c.p["verdict"], _ci(b, 0), acc_n)
        b.global_store(c.p["verdict"], _ci(b, 1), b.sub(_ci(b, 3), acc_n))
        b.global_store(c.p["verdict"], _ci(b, 2), anchor)
        b.global_store(c.p["nexttok"], _ci(b, 0), nxt)
    return nids


# ---------------------------------------------------------------- META


def emit_meta(c: _Ctx) -> None:
    """The draft step's meta words (one lane, plain stores).

    verify (draft step 0 runs on the target's metadata): ``positions + 1`` (``EagleProposer.propose``)
    and ``write_draft_meta`` (counter += 1, ``pos[r] = int32(logical[r])``, ``qsa_tab[r, QSA_PAGES] =
    block_tables row``).
    draft: ``qsa_draft_decode_metadata`` for one request (``bumped_positions`` = the step positions;
    ``reject``: ``num_reject_tokens``) then ``write_draft_meta`` on its logical positions."""
    b, spec = c.b, c.spec
    c0, c1 = _ci(b, 0), _ci(b, 1)

    def ld64(name: str, i: int) -> Value:
        v = b.buffer_load_cp(c.rs[name], _ci(b, 8 * i), c0, dwords=2, aux=0)
        return b.lor(b.zext(b.vec_extract(v, 0), I64), b.shl(b.zext(b.vec_extract(v, 1), I64), b.const_i64(32)))

    def ld32(name: str, i) -> Value:
        off = b.mul(i, _ci(b, 4)) if isinstance(i, Value) else _ci(b, 4 * i)
        return b.buffer_load_cp(c.rs[name], off, c0, dwords=1, aux=0)

    with b.scf_if(b.cmp_eq(b.lane_id(), c0)):
        if spec.mode == "verify":
            for r in range(spec.rows):
                b.global_store(c.p["posout"], _ci(b, r), b.add(ld64("pos", r), b.const_i64(1)))
            logical = [ld64("logical", r) for r in range(spec.rows)]
        else:
            # _draft_decode_metadata, one live request (N = REAL = 1), BUMP_ROWS = 1, PAGE 16, RATIO 4
            length = b.add(ld32("lens", 0), c1)
            for r in range(spec.pbump):
                b.global_store(c.p["pos"], _ci(b, r) if r else c0, b.add(ld64("pos", r), b.const_i64(1)))
            if spec.reject:
                length = b.sub(length, ld32("rej", 0))
            posv = b.sub(length, c1)
            valid = b.cmp_ge(posv, c0)
            pg = b.select(valid, b.div(posv, _ci(b, 16)), c0)
            page = b.select(valid, ld32("tables", pg), c0)
            slot = b.select(valid, b.add(b.mul(b.sext(page, I64), b.const_i64(16)),
                                         b.sext(b.mod(posv, _ci(b, 16)), I64)), b.const_i64(-1))
            b.global_store(c.p["lens"], c0, length)
            b.global_store(c.p["logical"], c0, b.sext(posv, I64))
            b.global_store(c.p["reqs"], c0, c0)
            b.global_store(c.p["slots"], c0, slot)
            cmpok = b.land(valid, b.cmp_eq(b.mod(length, _ci(b, 4)), c0))
            b.global_store(c.p["comp"], c0, b.select(cmpok, b.div(slot, b.const_i64(4)), b.const_i64(-1)))
            logical = [b.sext(posv, I64)]
        # write_draft_meta
        ctr = ld32("mctr", 0)
        b.global_store(c.p["mctr"], c0, b.add(ctr, c1))
        for r in range(spec.rows):
            b.global_store(c.p["mpos"], _ci(b, r), b.trunc(logical[r], I32))
            b.global_store(c.p["mtab"], _ci(b, r * QSA_WORDS + QSA_PAGES), c.p["btab"])


# ---------------------------------------------------------------- the build


def build_p9(spec: P9Spec) -> KernelDef:
    ok, why = is_valid_p9(spec)
    if not ok:
        raise ValueError(f"invalid P9Spec: {why}")
    hs = spec.head_spec()
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = 512
    p: Dict[str, Value] = {}
    for name, kind in p9_signature(spec):
        if kind == "ptr":
            p[name] = b.param(name, PtrType(I8 if name in ("hw", "hs", "scr") else I32, "global"), align=4)
        elif kind == "i64":
            p[name] = b.param(name, I64)
        elif kind == "f32":
            p[name] = b.param(name, F32)
    rs = _LazyRsrc(b, p)  # buffer resources built at each use: short SGPR live ranges
    scr64 = b.global_ptr_to_i64(p["scr"])
    scr_rsrc = b.buffer_rsrc(p["scr"], _ci(b, SCR_BYTES))
    mbk = MB.MailboxKnobs(pause=spec.pause)
    mb = _LazyMb(b, p, mbk)
    # LDS (all exclusive and allocated first: wave groups run concurrently, so nothing may alias)
    lds = {
        "bars": b.smem_alloc(I32, [NBAR], name_hint="p9bars", exclusive=True),
        "prew": b.smem_alloc(F32, [4, 4, 8], name_hint="p9prew", exclusive=True),
        "mix": b.smem_alloc(F32, [4, 16], name_hint="p9mix", exclusive=True),
        "norm": b.smem_alloc(F32, [4, 4], name_hint="p9norm", exclusive=True),
        "we": b.smem_alloc(I32, [80, 64, 4], name_hint="p9we", exclusive=True),
        "acch": b.smem_alloc(F32, [4, 64], name_hint="p9acch", exclusive=True),
        "ss": b.smem_alloc(F32, [4], name_hint="p9ss", exclusive=True),
        "ids": b.smem_alloc(I32, [4], name_hint="p9ids", exclusive=True),
        "erow": b.smem_alloc(I32, [4, H // 2], name_hint="p9erow", exclusive=True),
        "ge": b.smem_alloc(I32, [H // 2], name_hint="p9ge", exclusive=True),
    }
    if spec.lat != "v1":  # (v1 builds keep their LDS plan)
        lds["gate"] = b.smem_alloc(I32, [4, R], name_hint="p9gate", exclusive=True)
        lds["mrstd"] = b.smem_alloc(F32, [4, 4], name_hint="p9mrstd", exclusive=True)
    blk = b.block_id_x()
    tid = b.thread_id_x()
    wave = b.readfirstlane(b.lshr(tid, _ci(b, 6)))
    lane = b.lane_id()
    # ---- launch epoch (every wave reads; adds 1 at its end)
    raw = b.buffer_load_cp(scr_rsrc, _ci(b, 0), _ci(b, 0), dwords=2, aux=CPOL_SC1)
    cnt = b.lor(b.zext(b.vec_extract(raw, 0), I64), b.shl(b.zext(b.vec_extract(raw, 1), I64), b.const_i64(32)))
    ep = b.readfirstlane(b.trunc(b.div(cnt, b.const_i64(WAVES_PER_LAUNCH)), I32))
    tag = b.add(ep, _ci(b, 1))
    if spec.tmo:
        with b.scf_if(b.cmp_eq(lane, _ci(b, 0))):
            dbo = b.add(_ci(b, SCR["dbg"]), b.mul(b.add(b.mul(blk, _ci(b, 8)), wave), _ci(b, 16)))
            b.buffer_store_cp(b.vec_pack([tag, b.vec_extract(raw, 0), b.vec_extract(raw, 1), _ci(b, 0)], I32),
                              scr_rsrc, dbo, _ci(b, 0), aux=CPOL_SC1)
    # draft row select: the anchor row of the previous verify (read once; written by an earlier launch)
    if spec.mode == "draft" and spec.rows_in > 1:
        sel = b.readfirstlane(b.smin(b.smax(b.global_load(p["verdict"], _ci(b, 2), I32), _ci(b, 0)),
                                     _ci(b, spec.rows_in - 1)))
    else:
        sel = _ci(b, 0)
    # zero the LDS barrier counters, then the only hardware barrier of the launch
    with b.scf_if(b.cmp_eq(wave, _ci(b, 0))):
        b.smem_store_vN(lds["bars"], [lane], _ci(b, 0), 1)
    b.s_waitcnt(lgkmcnt=0)
    b.s_barrier_bare(fence="compiler")
    bars = Bars(b, lds["bars"])
    c = _Ctx(b=b, spec=spec, p=p, rs=rs, tag=tag, ptag=ep, mbk=mbk, mb=mb, lds=lds, bars=bars, sel=sel)
    bars.ctx = c
    if spec.stamps:
        def stamp(i: int) -> None:
            """lane 0: ``s_memrealtime`` into ``st[cta][i]`` (STAMP_SLOTS i64 slots per CTA; see STAMP_POINTS)."""
            with b.scf_if(b.cmp_eq(b.lane_id(), _ci(b, 0))):
                b.global_store(p["st"], b.add(b.mul(b.block_id_x(), _ci(b, STAMP_SLOTS)), _ci(b, i)),
                               dops.memrealtime(b))

        c.stamp = stamp
        with b.scf_if(b.cmp_eq(wave, _ci(b, 0))):
            stamp(0)

    # ---- waves 4-7 of the task CTAs: PRE -> MIX -> FRONT
    t = b.sub(blk, _ci(b, spec.toff))
    is_task = b.land(b.land(b.cmp_ge(blk, _ci(b, spec.toff)), b.cmp_lt(blk, _ci(b, spec.toff + NTASK))),
                     b.cmp_ge(wave, _ci(b, 4)))
    bp1, bp2, bm1 = bars.site(4), bars.site(4), bars.site(4)
    bwe, bss, bsp = bars.site(4), bars.site(4), bars.site(4)
    bm0 = bars.site(4) if spec.lat != "v1" else None
    with b.scf_if(is_task):
        gw = b.readfirstlane(b.sub(wave, _ci(b, 4)))
        if spec.lat == "v1":
            emit_pre(c, t, gw, lane, bp1, bp2)
        else:
            emit_pre_v2(c, t, gw, lane, bp1, bp2)
        if c.spec.stamps:
            with b.scf_if(b.cmp_eq(gw, _ci(b, 0))):
                c.stamp(2)
        direct = spec.fold == "direct"
        dids0: List[Value] = []

        def pin() -> None:
            # the drafter's ids, read and waited for before this task's x leaves (task 0 rewrites them only
            # after every head CTA, so after every task's x, has been consumed)
            ld = [b.buffer_load_cp(rs["dids"], _ci(b, 4 * j), _ci(b, 0), dwords=1, aux=0) for j in range(4)]
            ld += [b.buffer_load_cp(rs["ids"], _ci(b, 4 * (1 + j)), _ci(b, 0), dwords=1, aux=0) for j in range(3)]
            pinned = [dops.fresh(b, v) for v in ld]
            dids0.extend(pinned[:4])
            c.dids0 = dids0
            c.dr0 = pinned[4:]

        need_pin = direct and spec.mode == "verify"
        if spec.lat == "v1":
            if need_pin:
                pin()
            emit_mix(c, t, gw, lane, bm1)
        else:
            emit_mix_v2(c, t, gw, lane, bm0, bm1, pin=pin if need_pin else None)
        if c.spec.stamps:
            with b.scf_if(b.cmp_eq(gw, _ci(b, 0))):
                c.stamp(4)

        def get_ids() -> List[Value]:
            if spec.mode == "verify" and (spec.mutant == "stale_ids" or spec.dbg & 2):
                return [b.buffer_load_cp(rs["dids"], _ci(b, 4 * j), _ci(b, 0), dwords=1, aux=0) for j in range(4)]
            if direct:
                tg = fold_direct(c, ep if spec.mutant == "stale_tag" else tag)
                return accept_direct(c, tg, dids0, b.cmp_eq(t, _ci(b, 0)))
            got = poll(c, [(mb["ids"], _ci(b, j), 1) for j in range(spec.rows)], tag, "front.ids")
            return [g[0] for g in got]

        if not (spec.dbg & 4 or spec.tokonly):
            emit_front(c, t, gw, lane, bwe, bss, bsp, get_ids)
        elif spec.tokonly:
            with b.scf_if(b.land(b.cmp_eq(t, _ci(b, 0)), b.cmp_eq(gw, _ci(b, 0)))):
                get_ids()

    # ---- CTA 243 waves 0-3: the front's bundle norms; CTA 244 wave 0: meta
    bn1 = bars.site(4)
    with b.scf_if(b.land(b.cmp_eq(blk, _ci(b, 243)), b.cmp_lt(wave, _ci(b, 4)))):
        emit_norm(c, wave, lane, bn1)
        if c.spec.stamps:
            with b.scf_if(b.cmp_eq(wave, _ci(b, 0))):
                c.stamp(13)
    with b.scf_if(b.land(b.cmp_eq(blk, _ci(b, 244)), b.cmp_eq(wave, _ci(b, 0)))):
        if not spec.dbg & 1:
            emit_meta(c)
            c.stamp(14)

    # ---- waves 0-3 of CTAs 0..242: the head
    hbar = [bars.site(4) for _ in range(5)]
    hb_i = [0]

    def head_sync() -> None:
        hbar[hb_i[0]]()
        hb_i[0] += 1

    poll_tag = ep if spec.mutant == "stale_tag" else tag
    K = H

    def x_stage(ctx):
        lane_ = ctx["lane"]
        XS = K + hs.xpad
        xs = b.smem_alloc(BF16, [hs.rows_max, XS], name_hint="xs")
        per_row = K // 8
        T = hs.waves * 64
        nvalid = per_row * spec.rows
        Bt = spec.xbt
        if spec.xwait == "hint":
            # wave 0: one pair per (MIX task, row) -- the task's last 16-byte put -- then the group barrier
            nh = NTASK * spec.rows
            with b.scf_if(b.cmp_eq(ctx["wave"], _ci(b, 0))):
                hs_ = []
                for rr in range(-(-nh // 64)):
                    q = b.smin(b.add(lane_, _ci(b, 64 * rr)), _ci(b, nh - 1))
                    row = b.div(q, _ci(b, NTASK))
                    tt = b.mod(q, _ci(b, NTASK))
                    pi = b.lshr(b.add(b.mul(row, _ci(b, K)), b.add(b.mul(tt, _ci(b, 16)), _ci(b, 12))), _ci(b, 1))
                    hs_.append((mb["xbox"], pi, 1))
                poll(c, hs_, poll_tag, "head.xhint")
                c.stamp(22)
            head_sync()
        for bt in range(-(-nvalid // (T * Bt))):
            specs, meta = [], []
            for j in range(Bt):
                idx = b.add(ctx["tid"], _ci(b, (bt * Bt + j) * T))
                okk = b.cmp_lt(idx, _ci(b, nvalid))
                idc = b.smin(idx, _ci(b, nvalid - 1))
                row = b.div(idc, _ci(b, per_row))
                col = b.mul(b.mod(idc, _ci(b, per_row)), _ci(b, 8))
                pi = b.lshr(b.add(b.mul(row, _ci(b, K)), col), _ci(b, 1))
                specs += [(mb["xbox"], pi, 2), (mb["xbox"], b.add(pi, _ci(b, 2)), 2)]
                meta.append((okk, row, col))
            got = poll(c, specs, poll_tag, f"head.x{bt}", pmax=len(specs) if len(specs) > MB.POLL_MAX else 0)
            if bt == 0 and spec.stamps:
                _w0(b, lambda: c.stamp(23))
            for j, (okk, row, col) in enumerate(meta):
                words = [got[2 * j][0], got[2 * j][1], got[2 * j + 1][0], got[2 * j + 1][1]]
                v = b.vec_bitcast(b.vec_pack(words, I32), VectorType(BF16, 8))
                with b.scf_if(okk):
                    b.smem_store_vN(xs, [row, col], v, 8)
        head_sync()
        return xs

    io = P2.HeadIO(xp=p["h"], wp=p["hw"], sp=p["hs"], yp=p["y"], m_rows=_ci(b, spec.rows), sxm=_ci(b, K),
                   sym=_ci(b, 0), rev=_ci(b, 0), wsp=p["hws"])
    def on_keys(kb: Value, cb: Value, r: Value, mine: Value) -> None:
        with b.scf_if(mine):
            MB.put2(b, mb["keys"], b.mul(b.add(b.mul(r, _ci(b, HEAD_CTAS)), blk), _ci(b, 2)), kb, cb, tag, mbk)

    is_head = b.land(b.cmp_lt(blk, _ci(b, HEAD_CTAS)), b.cmp_lt(wave, _ci(b, 4)))
    with b.scf_if(is_head):
        if spec.hdelay:
            b.inline_asm(_NL.join(["s_sleep 63"] * spec.hdelay), "", [], None)
        P2.emit_head(
            b, hs, io, cta=blk, x_stage=x_stage,
            key_fn=P2._bf16_key_sampler if spec.mode == "verify" else MX._bf16_key,
            sync=head_sync, on_fold=None if spec.dbg & 2 else (lambda cf, rr: emit_accept(c, cf, rr)),
            fold_poll=fold_poll_tmo(c) if spec.tmo else None,
            on_keys=on_keys if spec.fold == "direct" else None,
            hooks=({"after_issue": lambda: _w0(b, lambda: c.stamp(9)), "after_x": lambda: _w0(b, lambda: c.stamp(10)),
                    "after_loop": lambda: _w0(b, lambda: c.stamp(11)), "end": lambda: _w0(b, lambda: c.stamp(12))}
                   if spec.stamps else None),
        )
    if spec.stamps:
        if c.spec.stamps:
            with b.scf_if(b.cmp_eq(wave, _ci(b, 4))):
                c.stamp(15)
    # ---- end: every wave adds its arrival to the launch counter (ectr=cta: the CTA's last wave adds all 8)
    if spec.ectr == "cta":
        with b.scf_if(b.cmp_eq(lane, _ci(b, 0))):
            ea = b.add(bars.base, _ci(b, 4 * (NBAR - 1)))
            old = b.inline_asm("ds_add_rtn_u32 $0, $1, $2" + _NL + "s_waitcnt lgkmcnt(0)", "=&v,v,v,~{memory}",
                               [ea, _ci(b, 1)], I32, result_name_hint="exitc")
            with b.scf_if(b.cmp_eq(old, _ci(b, 7))):
                b.inline_asm("global_atomic_add_x2 $0, $1, $2, $3 sc0 sc1" + _NL + "s_waitcnt vmcnt(0)", "=&v,v,v,s,~{memory}",
                             [b.const_i32(0), b.const_i64(8), scr64], I64, result_name_hint="aold64")
    else:
        with b.scf_if(b.cmp_eq(lane, _ci(b, 0))):
            MX._atomic_inc64(b, scr64)
    SITE_NAMES[b.kernel.name] = list(c.site_names)
    return b.kernel


# ---------------------------------------------------------------- stage-only builds (numerics debug)


def parts_signature() -> List[Tuple[str, str]]:
    return list(_SIG)


def build_p9_parts(spec: P9Spec) -> KernelDef:
    """Debug build: the same kernel (alias kept for the harness)."""
    return build_p9(spec)
