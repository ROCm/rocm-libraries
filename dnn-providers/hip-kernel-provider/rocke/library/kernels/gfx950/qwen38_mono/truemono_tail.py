# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The Path B tail provider for ATOM-truemono's WP4h hook (``ATOM_Q38_TRUE_MONO=tail``).

ATOM loads ``provider`` (the hook's default ``kernels.gfx950.qwen38_mono.truemono_tail:provider``) once
per decode width and asks it for a *tail* (``atom/models/qwen4_exp_mono/truemono/tail.py``: ``owns_logits``,
``verify``, ``sample``, ``propose``). This module serves:

* **K=3, one request (``r4s``)**: probe P9's round tail as ONE persistent launch (``probes/p9_tail.py``,
  ``best_spec("verify")``: ``lat=v2w``, ``fold=direct``, ``fnorm=pre``, ``ectr=cta``, ``xbt=8``; served with
  ``hdelay=0``, see ``SERVED_VERIFY``), reused unchanged: final HC mix of the target's last-layer streams, the MXFP4 head with the
  fused greedy argmax, the rejection accept (sampled tokens, verdict, the drafter's ``input_ids`` scatter),
  the draft front of the 4 rows into the draft layer's input and the draft meta (``write_draft_meta``).
  The decode graph then holds no head (``owns_logits``). ``propose`` runs ATOM's drafter unchanged except
  that its step 0 forward skips the front and the meta the launch already wrote (the draft runner's
  ``forward`` is overridden for exactly that one call and restored).
* **K=0, one request (``r1``)**: the rocKE fused-argmax head (``rocke_head``'s argmax build, P2's lever):
  one launch instead of the logits head in the graph + ``topk_select``.

Every other width is refused (the hybrid serves it, per width, logged once by the hook). A width whose
inputs do not match what the launch needs (shapes, dtypes, the mono draft runner) is disabled at its first
step, before anything is written, and serves the hybrid from then on.

Env (all optional; read once per width):

* ``ATOM_Q38_TM_TAIL_K3`` = ``p9`` (default) / ``off``; ``ATOM_Q38_TM_TAIL_K0`` = ``argmax`` (default) /
  ``off``.
* ``ATOM_Q38_TM_TAIL_P9`` = JSON of ``P9Spec`` knob overrides on top of ``best_spec("verify")`` (the lever
  space of P9; the P9 mutants too, e.g. ``{"mutant": "accept_ooo"}``).
* ``ATOM_Q38_TM_TAIL_DRAFT`` = ``hybrid`` (default: ATOM's drafter after step 0's front) / ``p9``: the draft
  steps' tails as P9 draft launches too (see "Draft chain" below); ``ATOM_Q38_TM_TAIL_P9D`` = JSON overrides of
  the draft builds (``best_spec("draft")`` + ``vsub`` / ``pbump`` from the served state).
* ``ATOM_Q38_TM_TAIL_MUTANT`` = provider-level must-fail mutants (gates only): ``nofront`` (skip the
  launch's front: the draft layer reads a stale input), ``nometa`` (a wrong meta word: the draft rows' ``pos`` one on),
  ``k0_off1`` (K=0 token + 1), ``alias`` (the drafter ids scattered before the accept reads them: P9's
  race, WP4h's ``mutant_alias`` on this tail), ``dmap`` (draft chain: the N8 id map one entry off), ``dpos`` (draft
  chain: the step positions' bump undone), fids (the launch's FRONT reads a rotated copy of the drafter's ids).
* ``ATOM_Q38_TM_TAIL_DEEPCHECK`` = n: for the first n K=3 steps, also check what the hook's check mode
  cannot see: the launch's front output and meta words against ATOM's own ``draft_front`` /
  ``write_draft_meta`` on the same served inputs, bit for bit (synchronises; gates only). Logged as
  ``tm tail deepcheck <what> equal|DIFF``; with ``ATOM_Q38_TRUE_MONO_STRICT=1`` a DIFF raises.
* ``ATOM_Q38_TM_TAIL_STATS`` = n: a stats line every n steps.
* ``ATOM_Q38_TM_TAIL_HOSTUS`` = us (diagnostics): a host busy-wait of that long after each verify launch; how much of
  it shows in the round time says whether the host is on the round's critical path.

Draft chain (``ATOM_Q38_TM_TAIL_DRAFT=p9``): ``propose`` runs a restricted copy of ATOM's
``EagleProposer.propose`` (one request, no draft graph pass, no aux / PCP / MHA draft, mRoPE or plain positions:
checked once, else the hybrid's drafter) in which, after draft step ``i``'s layer, ONE P9 draft launch does the
anchor-row select (step 0), the drafter's final HC mix, the MXFP4 head with the fused argmax (the N8 subset head and
its id map when ``ATOM_Q38_DRAFT_VOCAB`` is on: P9 knob ``vsub``), the draft token export,
``qsa_draft_decode_metadata`` (P9 knob ``pbump`` for mRoPE's three position rows), the next step's front and
``write_draft_meta``; the last step's launch is mix + head + the token only (``dbg=1``, ``tokonly=1``). ATOM's own host-side metadata
(``_enter_decode_metadata``, ``max_seqlen_k``, the QSA metadata object, the FlyDSL plan refresh) runs as before;
each next layer forward skips the front and the meta the launch wrote. The deep check covers every draft step
(token vs a rocKE head of P9's spec on the same mixed row, the QSA meta vs ``qsa_draft_decode_metadata``, front,
draft meta). Draft-side numerics only: the target verifies every token.

Numerics: the P9 build is bitwise ATOM's ops (P9 gate: 96 / 96 / 24 cases plus mutants, against
``hc_rows``, the rocKE head, ``topk_select``, ``rejection_sample``, ``DraftFront``, ``write_draft_meta``);
the hook's check mode (``ATOM_Q38_TRUE_MONO_CHECK``) and the deep check here gate it on served inputs.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import time
from typing import Optional

import torch

logger = logging.getLogger("atom")

ENV = "ATOM_Q38_TM_TAIL"
MUTANTS = ("none", "nofront", "nometa", "k0_off1", "alias", "dmap", "dpos", "fids")
H, HC, VOCAB, QSA_WORDS = 2560, 4, 248320, 12

__all__ = ["provider", "layer_passthrough", "P9VerifyTail", "ArgmaxTail", "MUTANTS", "settings"]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(f"{ENV}_{name}", "").strip() or default


def settings() -> dict:
    """The provider's knobs (``ATOM_Q38_TM_TAIL_*``), validated."""
    k3 = _env("K3", "p9")
    k0 = _env("K0", "argmax")
    if k3 not in ("p9", "off"):
        raise ValueError(f"{ENV}_K3={k3!r}: p9 or off")
    if k0 not in ("argmax", "off"):
        raise ValueError(f"{ENV}_K0={k0!r}: argmax or off")
    p9 = json.loads(_env("P9", "{}"))
    if not isinstance(p9, dict):
        raise ValueError(f"{ENV}_P9 must be a JSON object")
    mutant = _env("MUTANT", "none")
    if mutant not in MUTANTS:
        raise ValueError(f"{ENV}_MUTANT={mutant!r}: one of {MUTANTS}")
    draft = _env("DRAFT", "hybrid")
    if draft not in ("hybrid", "p9", "lean"):
        raise ValueError(f"{ENV}_DRAFT={draft!r}: hybrid, p9 or lean")
    p9d = json.loads(_env("P9D", "{}"))
    if not isinstance(p9d, dict):
        raise ValueError(f"{ENV}_P9D must be a JSON object")
    return {
        "k3": k3,
        "k0": k0,
        "p9": p9,
        "draft": draft,
        "p9d": p9d,
        "mutant": mutant,
        "deepcheck": int(_env("DEEPCHECK", "0")),
        "hostus": int(_env("HOSTUS", "0")),
        "stats": int(_env("STATS", "0")),
    }


#: served default on top of ``best_spec("verify")``: ``hdelay`` 0 (P9's isolated chain preferred 8; served end to end on
#: the full stack, 0 was lower in round time on both rocke nodes: jobs qk-cs-rt3-n01, qk-cs-rt4-n01 / -n07). Timing
#: only (s_sleep before the head's first panels): the numerics are the gated build's. ``ATOM_Q38_TM_TAIL_P9`` overrides.
SERVED_VERIFY = {"hdelay": 0}


def p9_spec(overrides: dict):
    from kernels.gfx950.qwen38_mono.probes import p9_tail as P

    spec = P.best_spec("verify", **dict(SERVED_VERIFY, **overrides))
    ok, why = P.is_valid_p9(spec)
    if not ok:
        raise ValueError(f"{ENV}_P9: invalid P9Spec: {why}")
    return spec


class _Launch:
    """One compiled rocKE kernel launched on torch's current stream; the ctypes argument objects are built once
    (a call sets their values), as ATOM's ``rocke_head._Kernel``."""

    def __init__(self, kdef, sig, grid: int, block: int):
        from rocke.helpers import compile_kernel
        from rocke.runtime import hip_module as HM

        self.HM = HM
        art = compile_kernel(kdef, arch="gfx950", backend="python")
        self.module = HM.Runtime().load_module(art.hsaco)
        self.fn = self.module.get_function(art.kernel_name)
        self.name = art.kernel_name
        self.sig = list(sig)
        mk = {"ptr": ctypes.c_uint64, "i64": ctypes.c_int64, "f32": ctypes.c_float}
        self.vals = {n: mk.get(kind, ctypes.c_int32)() for n, kind in self.sig}
        self.params = (ctypes.c_void_p * len(self.sig))(
            *[ctypes.cast(ctypes.pointer(self.vals[n]), ctypes.c_void_p) for n, _ in self.sig]
        )
        self.geom = tuple(ctypes.c_uint(v) for v in (grid, 1, 1, block, 1, 1, 0))

    def set(self, **vals) -> None:
        for n, v in vals.items():
            if isinstance(v, torch.Tensor):
                v = v.data_ptr()
            self.vals[n].value = v

    def __call__(self) -> None:
        rc = self.HM._hipModuleLaunchKernel(
            self.fn, *self.geom, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), self.params, None
        )
        if rc != 0:
            raise RuntimeError(f"tm tail {self.name}: hipModuleLaunchKernel error {rc}")


def _unwrap(m):
    return getattr(m, "_orig_mod", m)


def _target(runner):
    """(target CausalLM, its inner model) under the mono wrapper."""
    top = _unwrap(runner.model)
    lm = getattr(top, "model", top)  # MonoDecodeModel.model = the loaded CausalLM
    inner = getattr(lm, "model", None)
    return lm, inner


def _why_tensor(t, name, dtype, shape=None, contiguous=True) -> Optional[str]:
    if not isinstance(t, torch.Tensor):
        return f"{name} is not a tensor"
    if t.dtype != dtype:
        return f"{name} is {t.dtype}, not {dtype}"
    if shape is not None and tuple(t.shape) != tuple(shape):
        return f"{name} is {tuple(t.shape)}, not {tuple(shape)}"
    if contiguous and not t.is_contiguous():
        return f"{name} is not contiguous"
    return None


class _Stats:
    def __init__(self, every: int):
        self.every, self.n, self.c = every, 0, {}

    def count(self, what: str) -> None:
        self.c[what] = self.c.get(what, 0) + 1
        self.n += 1
        if self.every and self.n % self.every == 0:
            logger.info("tm tail stats %s", ", ".join(f"{k}={v}" for k, v in sorted(self.c.items())))


class ArgmaxTail:
    """K=0, one request: the greedy token from the rocKE fused-argmax head (``Mxfp4LMHead.rk.argmax``) of the
    mixed hidden state; the graph holds no head."""

    owns_logits = True

    def __init__(self, request, hybrid, cfg):
        self.request, self.hybrid, self.cfg = request, hybrid, cfg
        self.off_why: Optional[str] = None
        self.head = None
        self.stats = _Stats(cfg["stats"])

    def _bind(self) -> None:
        lm, _ = _target(self.hybrid.runner)
        head = getattr(lm, "lm_head", None)
        if head is None or getattr(head, "rk", None) is None or not getattr(head.rk, "argmax_k", None):
            self.off_why = "no rocKE argmax head (ATOM_Q38_ROCKE_HEAD with the argmax part)"
        else:
            self.head = head
        if self.off_why:
            logger.warning("tm tail %s off: %s; the hybrid serves", self.request.variant, self.off_why)
        else:
            logger.info("tm tail %s: rocKE fused argmax head", self.request.variant)

    def sample(self, step):
        if self.head is None and self.off_why is None:
            self._bind()
        if self.off_why is not None:
            self.stats.count("hybrid")
            return self.hybrid.sample(step)
        bs = step.bs
        x = step.hidden[:bs]
        out = torch.empty(bs, dtype=torch.int32, device=x.device)
        h = self.head
        got = h.rk.argmax(x, h.fp4_weight, h.fp4_scales, h.argmax_keys, out)
        if got is None:  # wider than the rocKE builds: the mono's own argmax head
            from atom.models.qwen4_exp_mono import head as fp4

            fp4.mxfp4_argmax(x, h.fp4_weight, h.fp4_scales, h.argmax_keys, out)
        if self.cfg["mutant"] == "k0_off1":
            out.add_(1)
        self.stats.count("argmax")
        return out

    def verify(self, step):  # never routed (k = 0); the hook checks the attribute exists only for MTP
        return self.hybrid.verify(step)

    def propose(self, step, **kw):
        return self.hybrid.propose(step, **kw)


class P9VerifyTail:
    """K=3, one request: P9's verify chain as one launch (see the module docstring)."""

    owns_logits = True

    def __init__(self, request, hybrid, cfg):
        from kernels.gfx950.qwen38_mono.probes import p9_tail as P

        self.P = P
        self.request, self.hybrid, self.cfg = request, hybrid, cfg
        self.spec = p9_spec(cfg["p9"])
        self.mutant = cfg["mutant"]
        self.kern: Optional[_Launch] = None
        self.off_why: Optional[str] = None
        self.bound = False
        self.pending = False  # the launch wrote the draft step 0's front + meta; propose must skip them
        self.deep_left = cfg["deepcheck"]
        self.stats = _Stats(cfg["stats"])
        self.scr = None
        self.dchain = None  # the draft chain (``_DraftChain``) when ATOM_Q38_TM_TAIL_DRAFT=p9 binds
        self.dchain_why: Optional[str] = None
        self.deep_round = False
        if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
            self._compile()

    # ------------------------------------------------------------ build / bind
    def _alloc(self) -> None:
        """The launch's own buffers: scratch (never reset: the launch epoch lives in it), the head workspace and
        the unused outputs / draft-only arguments of the verify build."""
        from kernels.gfx950 import mx_decode_gemv as MX

        dev = torch.device("cuda", torch.cuda.current_device())
        i32, i64 = torch.int32, torch.int64
        self.scr = torch.zeros(self.P.SCR_BYTES // 4, dtype=i32, device=dev)
        self.hws = torch.zeros(MX.fused_workspace_bytes(self.spec.head_spec()) // 4, dtype=i32, device=dev)
        self.y = torch.zeros(8, dtype=i32, device=dev)
        self.nexttok = torch.zeros(8, dtype=i32, device=dev)
        self.pos = torch.zeros(8, dtype=i64, device=dev)  # positions + 1 stays ATOM's own (mRoPE) op
        self.posout = torch.zeros(8, dtype=i64, device=dev)
        self.d32 = torch.zeros(64, dtype=i32, device=dev)  # draft-only arguments (unused by the verify build)
        self.d64 = torch.zeros(64, dtype=i64, device=dev)

    def _compile(self) -> None:
        if self.scr is None:
            self._alloc()
        if self.kern is None:
            P = self.P
            self.kern = _Launch(P.build_p9(self.spec), P.p9_signature(self.spec), 256, 512)
            logger.info("tm tail %s: P9 launch %s", self.request.variant, self.kern.name)

    def _off(self, why: str) -> None:
        self.off_why = why
        logger.warning("tm tail %s off: %s; the hybrid serves", self.request.variant, why)

    def _bind(self, step) -> None:
        """Resolve every weight and buffer once (first step); disable the tail (nothing written yet) if any is
        not as the launch needs."""
        self.bound = True
        try:
            self._compile()
        except Exception as e:  # noqa: BLE001 -- a build failure turns this width back to the hybrid
            return self._off(f"P9 build failed: {type(e).__name__}: {e}")
        runner = self.hybrid.runner
        lm, inner = _target(runner)
        mixer = getattr(inner, "hyper_connection_mixer", None)
        head = getattr(lm, "lm_head", None)
        if mixer is None or head is None or not hasattr(head, "fp4_weight"):
            return self._off("the target has no HC final mixer or no MXFP4 lm_head")
        drafter = getattr(runner, "drafter", None)
        dmodel = _unwrap(getattr(drafter, "model", None))
        lazy = getattr(dmodel, "_mono", None)
        draft = getattr(dmodel, "draft", None)
        if lazy is None or draft is None:
            return self._off("the drafter is not the mono draft model (MonoDraftModel)")
        k = self.request.k
        if not (lazy.ready(k + 1) and lazy.ready(1)):
            return self._off("the mono draft runner does not serve the 4-row / 1-row steps")
        dr = lazy.runner
        w = dr.single.get(k + 1)
        if w is None or w.group != k + 1:
            return self._off("no single-request step-0 width in the draft runner")
        pred = dr.predictor
        front = dr.front
        if front is None:
            return self._off("the draft runner's front end is ATOM's ops (no DraftFront)")
        checks = [
            _why_tensor(mixer.hc_norm.weight, "hc_norm", torch.bfloat16, (HC * H,)),
            _why_tensor(mixer.fused_w_cat, "w_cat", torch.bfloat16, (320, HC * H)),
            _why_tensor(mixer.input_mix_weight_up.weight, "w_up", torch.bfloat16, (HC * H, 320)),
            _why_tensor(head.fp4_weight, "head codes", torch.uint8, (VOCAB, H // 2)),
            _why_tensor(head.fp4_scales, "head scales", torch.uint8, (VOCAB, H // 32)),
            _why_tensor(front.embed, "embeddings", torch.bfloat16, (VOCAB, H)),
            _why_tensor(front.g_e, "embedding norm", torch.bfloat16, (H,)),
            _why_tensor(front.g_h, "hidden norm", torch.bfloat16, (HC * H,)),
            _why_tensor(front.w_e, "fc_embedding", torch.bfloat16, (H, H)),
            _why_tensor(front.w_h, "fc_hidden", torch.bfloat16, (H, H)),
            _why_tensor(w.h_in, "draft input", torch.bfloat16, (k + 1, HC * H)),
            _why_tensor(w.pos, "draft pos", torch.int32, (k + 1,)),
            _why_tensor(w.qsa, "draft qsa table", torch.int64, (k + 1, QSA_WORDS)),
            _why_tensor(dr.counter, "draft counter", torch.int32, (1,)),
        ]
        for why in checks:
            if why:
                return self._off(why)
        if mixer.block_inject_weight is not None:
            return self._off("the final mixer has a block injection")
        if (front.hidden, front.hc) != (H, HC):
            return self._off("front geometry")
        self.dr, self.w, self.pred, self.front_op = dr, w, pred, front
        self.kern.set(
            normw=mixer.hc_norm.weight,
            wcat=mixer.fused_w_cat,
            wup=mixer.input_mix_weight_up.weight,
            hw=head.fp4_weight,
            hs=head.fp4_scales,
            hws=self.hws,
            y=self.y,
            emb=front.embed,
            ge=front.g_e,
            gh=front.g_h,
            we=front.w_e,
            wh=front.w_h,
            hin=w.h_in,
            nexttok=self.nexttok,
            pos=self.pos,
            posout=self.posout,
            lens=self.d32,
            tables=self.d32,
            slots=self.d64,
            reqs=self.d32,
            comp=self.d64,
            rej=self.d32,
            mpos=w.pos,
            mtab=w.qsa,
            mctr=dr.counter,
            scr=self.scr,
            eps=float(mixer.hc_norm.variance_epsilon),
            epse=float(front.eps_e),
            epsh=float(front.eps_h),
            st=self.d32,
        )
        logger.info(
            "tm tail %s: bound (P9 %s, draft runner step-0 width %d rows)",
            self.request.variant,
            self.kern.name,
            w.rows,
        )
        if self.cfg["draft"] in ("p9", "lean"):
            try:
                self.dchain = _DraftChain(self, runner, drafter, dr, head, front)
            except _Refuse as e:
                self.dchain_why = str(e)
            except Exception as e:  # noqa: BLE001 -- a build failure keeps ATOM's drafter
                self.dchain_why = f"draft builds failed: {type(e).__name__}: {e}"
            if self.dchain is None:
                logger.warning(
                    "tm tail %s draft chain off: %s; ATOM's drafter serves", self.request.variant, self.dchain_why
                )
            else:
                logger.info("tm tail %s: draft chain bound (%s)", self.request.variant, ", ".join(self.dchain.names))

    def _step_why(self, step) -> Optional[str]:
        """Per-step preconditions (a failure serves this step on the hybrid; nothing written yet)."""
        k = self.request.k
        if step.bs != 1 or step.ids is None:
            return "batch"
        h = step.hidden
        if h is None or h.dtype != torch.bfloat16 or h.numel() < (k + 1) * HC * H or not h.is_contiguous():
            return "hidden"
        sid, did = step.ids.step, step.ids.draft
        if sid.dtype != torch.int32 or did.dtype != torch.int32 or sid.numel() < k + 1 or did.numel() < k + 1:
            return "ids"
        if not (sid.is_contiguous() and did.is_contiguous()):
            return "ids"
        spec = step.spec
        if spec is None:
            return "spec"
        md = self._md()
        qsa = getattr(md, "qsa_metadata", None)
        if qsa is None or qsa.logical_positions.dtype != torch.int64:
            return "qsa"
        return None

    @staticmethod
    def _md():
        from atom.utils.forward_context import get_forward_context

        return get_forward_context().attn_metadata

    # ------------------------------------------------------------ the tail
    def verify(self, step):
        if not self.bound:
            self._bind(step)
        why = self.off_why or self._step_why(step)
        if why is not None:
            self.stats.count(f"hybrid.{why if self.off_why is None else 'off'}")
            self.pending = False
            return self.hybrid.verify(step)
        k = self.request.k
        dev = step.hidden.device
        out = torch.empty(1, k + 1, dtype=torch.int32, device=dev)
        verdict = torch.empty(3, 1, dtype=torch.int32, device=dev)
        qsa = self._md().qsa_metadata
        deep = self.deep_left > 0
        self.deep_round = deep  # the draft chain checks this round's draft steps too
        if deep:
            self.deep_left -= 1
            self._deep_pre(step, qsa)
        if self.mutant == "alias":
            # P9's race, provoked: the drafter's ids rewritten before the accept reads the drafts
            step.ids.draft[: k + 1].copy_(step.ids.draft[: k + 1].roll(1))
        dids = step.ids.draft
        if self.mutant == "fids":
            # the FRONT embeds ids from a rotated copy of the drafter's ids (the real buffer keeps ATOM's own
            # scatter): the launch's front input ids are wrong where the rotation moved an id
            dids = dids[: k + 1].roll(1).contiguous()
        self.kern.set(
            h=step.hidden,
            ids=step.ids.step,
            dids=dids,
            outtok=out,
            verdict=verdict,
            logical=qsa.logical_positions,
            btab=qsa.block_tables.data_ptr(),
        )
        self.kern()
        if self.mutant == "nofront":
            self.w.h_in.zero_()  # the launch's front output lost: the draft layer reads zeros
        if self.mutant == "nometa":
            self.w.pos.add_(1)  # a wrong meta word: the draft layer's rows one cache row on
        if self.cfg["hostus"]:  # diagnostics: a host busy-wait after the launch (is the host on the critical path?)
            t_end = time.perf_counter() + self.cfg["hostus"] * 1e-6
            while time.perf_counter() < t_end:
                pass
        self.pending = True
        self.verdict = verdict  # the draft chain's step-0 anchor (row select)
        self.stats.count("p9")
        return out, verdict

    def sample(self, step):  # never routed at k = 3 (every MTP decode step is a verify); the hook requires it
        return self.hybrid.sample(step)

    def propose(self, step, **kw):
        drafter = self.hybrid.runner.drafter
        if not self.pending:
            return drafter.propose(**kw)
        self.pending = False
        dr = self.dr
        if self.cfg["draft"] == "lean" and self.dchain is not None and self.dchain.ready_now():
            deep = getattr(self, "_deep_ref", None)
            self._deep_ref = None
            self.stats.count("draft.lean")
            return self.dchain.propose_lean(kw, self.verdict, deep)
        own = dr.__dict__.get("forward")
        n0 = self.request.k + 1
        deep = getattr(self, "_deep_ref", None)
        self._deep_ref = None
        state = {"first": True}
        chain = self.dchain
        if chain is not None and not chain.ready_now():
            chain = None
            self.stats.count("draft.hybrid")

        def forward(input_ids, hidden_states):
            if state["first"] and input_ids.numel() == n0:
                state["first"] = False
                return self._step0(input_ids, hidden_states, deep)
            if chain is not None and chain.prewritten(input_ids.numel()):
                return chain.layer_only(input_ids, hidden_states)
            return type(dr).forward(dr, input_ids, hidden_states)

        dr.forward = forward
        try:
            if chain is not None:
                self.stats.count("draft.p9")
                return chain.propose(kw, self.verdict)
            return drafter.propose(**kw)
        finally:
            if own is None:
                del dr.forward
            else:
                dr.forward = own

    def _step0(self, input_ids, hidden_states, deep):
        """The draft runner's ``forward`` for step 0 without the front end and the meta (the launch wrote them):
        the layer launch only."""
        from atom.models.qwen4_exp_mono import q38_hook

        dr, w = self.dr, self.w
        g = dr.geometry
        n = input_ids.numel()
        if deep is not None:
            self._deep_post(input_ids, hidden_states, deep)
        stream = torch.cuda.current_stream(dr.device)
        w.launch(dr.counter.data_ptr(), dr.layer_id, w.scratch, *w.args, stream=stream)
        hook = q38_hook.active()
        if hook:
            hook.after_draft(dr, w)
        return w.h_out.view(n, g.hc_count, g.hidden)

    # ------------------------------------------------------------ deep check (gates)
    def _deep_pre(self, step, qsa) -> None:
        """Before the launch: the meta inputs (counter, logical rows, the block-table row)."""
        k = self.request.k
        self._deep_ref = {
            "counter": self.dr.counter.clone(),
            "logical": qsa.logical_positions.clone(),
            "tables": qsa.block_tables,
            "rows": k + 1,
        }

    def _deep_post(self, input_ids, hidden_states, ref) -> None:
        """At step 0's forward (after ATOM's scatter): ATOM's own front end and meta on the served inputs vs
        what the launch wrote, bitwise."""
        from atom.models.qwen4_exp_mono.front import draft_front
        from atom.models.qwen4_exp_mono.step_meta import write_draft_meta

        from atom.models.qwen4_exp_mono.truemono.hook import TrueMonoRefused, strict

        dr, w = self.dr, self.w
        n = ref["rows"]
        hin = torch.empty_like(w.h_in)
        draft_front(dr.front, dr.predictor, input_ids.reshape(-1), hidden_states.reshape(n, -1), hin)
        ctr = ref["counter"].clone()
        pos = torch.empty_like(w.pos)
        tab = w.qsa.clone()
        tab[:, 1].fill_(-1)
        write_draft_meta(ctr, ref["logical"], ref["tables"], pos, tab)
        got_tab = w.qsa
        res = {
            "front": torch.equal(hin.view(torch.int16), w.h_in.view(torch.int16)),
            "counter": torch.equal(ctr, dr.counter),
            "pos": torch.equal(pos, w.pos),
            "qsa": torch.equal(tab, got_tab),
        }
        for what, ok in res.items():
            logger.info("tm tail deepcheck %s %s", what, "equal" if ok else "DIFF")
            if not ok and strict():
                raise TrueMonoRefused(f"tm tail deepcheck {what} differs from ATOM's op")


class _Refuse(Exception):
    """A served-state precondition of the draft chain that does not hold (ATOM's drafter serves)."""


def _need(cond, why: str) -> None:
    if not cond:
        raise _Refuse(why)


def lazy_ready(drafter, rows: int) -> bool:
    lazy = getattr(_unwrap(getattr(drafter, "model", None)), "_mono", None)
    return lazy is not None and lazy.ready(rows)


def _log_deep(what: str, ok: bool) -> None:
    from atom.models.qwen4_exp_mono.truemono.hook import TrueMonoRefused, strict

    logger.info("tm tail deepcheck %s %s", what, "equal" if ok else "DIFF")
    if not ok and strict():
        raise TrueMonoRefused(f"tm tail deepcheck {what} differs from ATOM's op")


class _DraftChain:
    """The draft steps' tails as P9 draft launches (``ATOM_Q38_TM_TAIL_DRAFT=p9``; module docstring)."""

    def __init__(self, tail, runner, drafter, dr, head, front):
        from atom.distributed.pcp_utils import get_pcp_world_size
        from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpMetadataBuilder, Qwen4ExpQSAMetadata
        from atom.spec_decode.eagle_proposer import _pcp_active_for_draft_model
        from atom.utils import envs
        from kernels.gfx950 import mx_decode_gemv as MX

        P = tail.P
        self.tail, self.P, self.runner, self.drafter, self.dr = tail, P, runner, drafter, dr
        k = tail.request.k
        self.k = k
        _need(k >= 2 and getattr(drafter, "mtp_k", None) == k, f"mtp_k {getattr(drafter, 'mtp_k', None)} != {k}")
        _need(getattr(drafter, "step", 1) is None, "the drafter declares a draft graph pass")
        _need(not getattr(drafter, "_share_mtp_indices", True), "MTP index sharing")
        _need(not getattr(drafter, "_captures_aux", True), "the drafter captures aux hidden states")
        _need(not hasattr(runner, "draft_kv_builder"), "an MHA draft KV builder")
        # flat kv_indices (present here) only matter to ATOM's MLA rebase and to a backend whose
        # prepare_mtp_decode returns no slot_mapping; Qwen3.8's returns it, and its target is not MLA
        _need(not getattr(getattr(runner, "attn_family", None), "is_mla", True), "an MLA target")
        _need(get_pcp_world_size() == 1 and not _pcp_active_for_draft_model(drafter.model), "PCP")
        _need(not envs.ATOM_DEBUG_FORCE_SKIP_DRAFT_MODEL, "ATOM_DEBUG_FORCE_SKIP_DRAFT_MODEL")
        builder = runner.attn_metadata_builder
        _need(
            type(builder).prepare_mtp_decode is Qwen4ExpMetadataBuilder.prepare_mtp_decode,
            f"metadata builder {type(builder).__name__} is not Qwen3.8's",
        )
        _need(builder.fuse_mtp_decode_position_update and builder.fuse_mtp_decode_mrope_positions, "unfused positions")
        _need(builder.block_size == 16 and builder._compress_ratio == 4, "QSA page 16 / ratio 4")
        self.builder, self.QM = builder, Qwen4ExpQSAMetadata
        self.pbump = 3 if getattr(runner, "use_mrope", False) else 1
        mixer = getattr(dr.predictor, "hyper_connection_mixer", None)
        _need(mixer is not None and mixer.block_inject_weight is None, "the drafter's final mixer")
        dv = getattr(head, "dv", None)
        if dv is not None:  # N8: the subset head and its id map
            hw, hs, vmap, vsub = dv.weight, dv.scales, dv.map, dv.n
            for why in (
                _why_tensor(hw, "subset codes", torch.uint8, (vsub, H // 2)),
                _why_tensor(hs, "subset scales", torch.uint8, (vsub, H // 32)),
                _why_tensor(vmap, "subset map", torch.int32, (vsub,)),
            ):
                _need(why is None, str(why))
            if tail.mutant == "dmap":
                vmap = vmap.roll(1)
        else:
            hw, hs, vmap, vsub = head.fp4_weight, head.fp4_scales, None, 0
        self.dv, self.vmap, self.vsub, self.hw, self.hs = dv, vmap, vsub, hw, hs
        _need(lazy_ready(drafter, 1), "the 1-row draft step")
        w1 = dr.single.get(1)
        _need(w1 is not None, "no 1-row draft width")
        for why in (
            _why_tensor(mixer.hc_norm.weight, "draft hc_norm", torch.bfloat16, (HC * H,)),
            _why_tensor(mixer.fused_w_cat, "draft w_cat", torch.bfloat16, (320, HC * H)),
            _why_tensor(mixer.input_mix_weight_up.weight, "draft w_up", torch.bfloat16, (HC * H, 320)),
            _why_tensor(w1.h_in, "1-row draft input", torch.bfloat16, (1, HC * H)),
            _why_tensor(w1.pos, "1-row draft pos", torch.int32, (1,)),
            _why_tensor(w1.qsa, "1-row draft qsa table", torch.int64, (1, QSA_WORDS)),
        ):
            _need(why is None, str(why))
        self.w1, self.mixer = w1, mixer
        common = dict(tail.cfg["p9d"])
        common.update(vsub=vsub, pbump=self.pbump)
        self.specs = {
            "d0": P.best_spec("draft", rows_in=k + 1, reject=1, **common),
            "d1": P.best_spec("draft", rows_in=1, **common),
            "dl": P.best_spec("draft", rows_in=1, **dict(common, dbg=1, tokonly=1)),
        }
        for nm, sp in self.specs.items():
            ok, why = P.is_valid_p9(sp)
            _need(ok, f"{nm}: {why}")
        dev = torch.device("cuda", torch.cuda.current_device())
        i32, i64 = torch.int32, torch.int64
        self.d32 = torch.zeros(64, dtype=i32, device=dev)
        self.d64 = torch.zeros(64, dtype=i64, device=dev)
        self.k_ = {}
        for nm, sp in self.specs.items():
            kern = _Launch(P.build_p9(sp), P.p9_signature(sp), 256, 512)
            scr = torch.zeros(P.SCR_BYTES // 4, dtype=i32, device=dev)
            hws = torch.zeros(MX.fused_workspace_bytes(sp.head_spec()) // 4, dtype=i32, device=dev)
            y = torch.zeros(8, dtype=i32, device=dev)
            kern.keep = (scr, hws, y)
            args = dict(
                normw=mixer.hc_norm.weight,
                wcat=mixer.fused_w_cat,
                wup=mixer.input_mix_weight_up.weight,
                hw=hw,
                hs=hs,
                hws=hws,
                y=y,
                emb=front.embed,
                ge=front.g_e,
                gh=front.g_h,
                we=front.w_e,
                wh=front.w_h,
                hin=w1.h_in,
                ids=self.d32,
                dids=self.d32,
                verdict=self.d32,
                nexttok=self.d32,
                posout=self.d64,
                rej=self.d32,
                mpos=w1.pos,
                mtab=w1.qsa,
                mctr=dr.counter,
                scr=scr,
                eps=float(mixer.hc_norm.variance_epsilon),
                epse=float(front.eps_e),
                epsh=float(front.eps_h),
                st=self.d32,
                h=self.d32,
                outtok=self.d32,
                pos=self.d64,
                logical=self.d64,
                lens=self.d32,
                tables=self.d32,
                slots=self.d64,
                reqs=self.d32,
                comp=self.d64,
                btab=0,
            )
            if vsub:
                args["vmap"] = vmap
            kern.set(**args)
            self.k_[nm] = kern
        self.names = [kk.name for kk in self.k_.values()]
        self.pre = False
        self.ref_head = None

    def ready_now(self) -> bool:
        """Per propose: the QSA pools exist (they do once a request is served) and the step is one request."""
        from atom.utils.forward_context import get_forward_context

        c = get_forward_context().context
        return getattr(self.builder, "qsa_arena", None) is not None and getattr(c, "scheduled_bs", 0) == 1

    def prewritten(self, rows: int) -> bool:
        return self.pre and rows == 1

    def layer_only(self, input_ids, hidden_states):
        """A later draft step's forward: the layer launch only (the previous P9 launch wrote front + meta)."""
        from atom.models.qwen4_exp_mono import q38_hook

        self.pre = False
        dr, w = self.dr, self.w1
        g = dr.geometry
        stream = torch.cuda.current_stream(dr.device)
        w.launch(dr.counter.data_ptr(), dr.layer_id, w.scratch, *w.args, stream=stream)
        hook = q38_hook.active()
        if hook:
            hook.after_draft(dr, w)
        return w.h_out.view(1, g.hc_count, g.hidden)

    def propose(self, kw, verdict):
        """``EagleProposer.propose`` for one request (the preconditions checked at bind), the draft tails as P9
        launches. Keeps ATOM's order of every host-side metadata step."""
        from atom.utils.forward_context import get_forward_context

        fc = get_forward_context()
        context, md = fc.context, fc.attn_metadata
        drafter, k = self.drafter, self.k
        scheduled_bs = context.scheduled_bs
        running_bs = scheduled_bs  # no draft graph pass: ATOM's running_bs is the scheduled one
        context.is_draft = True
        input_ids = kw["target_token_ids"]
        next_token_ids = kw["next_token_ids"]
        lti = kw["last_token_indices"]
        num_reject = kw["num_reject_tokens"]
        input_ids.scatter_(0, lti, next_token_ids)  # what the verify launch wrote: idempotent, ATOM's own op
        positions = kw["target_positions"] + 1
        hidden_states = kw["target_hidden_states"]
        draft_token_ids = torch.empty(k, scheduled_bs, dtype=next_token_ids.dtype, device=next_token_ids.device)
        if md.slot_mapping is not None:
            md.slot_mapping = md.slot_mapping[: len(input_ids)]
        for i in range(k):
            drafter._publish_draft_shape(
                fc,
                scheduled_tokens=scheduled_bs if i else input_ids.shape[0],
                running_tokens=running_bs if i else input_ids.shape[0],
                running_tokens_are_unified=False,
            )
            self.pre = i > 0
            ret = drafter.model(input_ids=input_ids, positions=positions, hidden_states=hidden_states)
            self.pre = False
            exported = draft_token_ids[i]
            if i < k - 1:
                if i == 0:
                    _i0, positions = drafter._enter_decode_metadata(
                        scheduled_bs, running_bs, positions, lti, num_reject
                    )
                for key, v in self._advance(i, md, positions, ret, exported, lti, num_reject, verdict).items():
                    md.__dict__[key] = v
                input_ids = exported
                hidden_states = ret[:1]  # the next front is the launch's (its row select): only the shape is read
            else:
                self._last(i, ret, exported)
        return draft_token_ids.t().contiguous()

    def _advance(self, i, md, positions, ret, exported, lti, num_reject, verdict) -> dict:
        """``_advance_decode_metadata`` + ``prepare_mtp_decode`` (fused) with ``qsa_draft_decode_metadata`` inside
        the P9 launch (with this step's head and the next step's front + meta)."""
        md.max_seqlen_k += 1
        var = self.runner.forward_vars
        n = positions.shape[-1]
        slots, logical, req_ids, compressed, lengths, tables = self._launch(
            i, positions, ret, exported, lti, num_reject, verdict
        )
        b = self.builder
        mml = b.model_runner.config.max_model_len
        return {
            "slot_mapping": slots,
            "flydsl_work_plan": b._refreshed_flydsl_plan(var, n),
            "qsa_metadata": self.QM(tables, slots, compressed, req_ids, logical, lengths, mml),
        }

    def _launch(self, i, positions, ret, exported, lti, num_reject, verdict):
        """Draft step ``i``'s P9 launch (head of step i, ``qsa_draft_decode_metadata``, step i + 1's front and meta)
        on the step's forward_vars buffers; the deep check after it in check rounds."""
        var = self.runner.forward_vars
        n = positions.shape[-1]
        assert n == 1 and positions.is_contiguous() and positions.numel() == self.pbump
        assert positions.dtype == torch.int64
        slots = var["slot_mapping"].gpu[:n]
        logical = var["qsa_logical_positions"].gpu[:n]
        req_ids = var["qsa_token_to_req"].gpu[:n]
        compressed = var["qsa_compressed_slots"][:n]
        lengths = var["context_lens"].gpu[:n]
        tables = var["block_tables"].gpu[:n]
        kern = self.k_["d0" if i == 0 else "d1"]
        snap = None
        if self.tail.deep_round:
            hrow = torch.index_select(ret, 0, lti.long()) if i == 0 else ret[:1]
            snap = {
                "h": hrow.clone(),
                "lens": lengths.clone(),
                "pos": positions.clone(),
                "rej": num_reject.clone() if i == 0 else None,
                "ctr": self.dr.counter.clone(),
            }
        kern.set(
            h=ret,
            verdict=verdict if i == 0 else self.d32,
            rej=num_reject if i == 0 else self.d32,
            outtok=exported,
            pos=positions,
            logical=logical,
            lens=lengths,
            tables=tables,
            slots=slots,
            reqs=req_ids,
            comp=compressed,
            btab=tables.data_ptr(),
        )
        kern()
        if self.tail.mutant == "dpos":
            positions.sub_(1)
        if snap is not None:
            got = dict(lens=lengths, pos=positions, slots=slots, logical=logical, reqs=req_ids, comp=compressed)
            self._deep(i, snap, exported, tables, got)
        return slots, logical, req_ids, compressed, lengths, tables

    def propose_lean(self, kw, verdict, deep):
        """``ATOM_Q38_TM_TAIL_DRAFT=lean``: the draft round as GPU launches only, no host-side ATOM drafter code.
        The served round is host-bound around the tail (cs_round / cs_gaps traces: the GPU idles while ATOM's eager
        postprocess / propose Python runs), so the launches' GPU savings only pay if the host work goes too. Per
        round: the step positions (``index_select`` + 1, ATOM's ``target_positions + 1`` then
        ``_enter_decode_metadata``'s anchor column), the 4-row draft layer (its input and meta written by the verify
        launch), then per step a P9 draft launch and the 1-row layer, the last step's head-only launch. ATOM's
        host-only metadata (``max_seqlen_k``, the QSA metadata object, the FlyDSL plan, ``cu_seqlens_q``) is not
        rebuilt: no draft kernel reads it, and the next step's metadata is built from the scheduler again. The
        drafter ids scatter is the verify launch's (ATOM's ``scatter_`` writes the same values)."""
        from atom.models.qwen4_exp_mono import q38_hook

        k, dr, w4, w1 = self.k, self.dr, self.tail.w, self.w1
        tp = kw["target_positions"]
        lti = kw["last_token_indices"]
        num_reject = kw["num_reject_tokens"]
        if getattr(self, "posb", None) is None or self.posb.device != tp.device:
            shape = (self.pbump, 1) if self.pbump > 1 else (1,)
            self.posb = torch.empty(shape, dtype=torch.int64, device=tp.device)
        torch.index_select(tp, tp.dim() - 1, lti, out=self.posb)
        self.posb.add_(1)
        dt = torch.empty(1, k, dtype=kw["next_token_ids"].dtype, device=tp.device)
        g = dr.geometry
        stream = torch.cuda.current_stream(dr.device)
        hook = q38_hook.active()
        if deep is not None:
            n0 = k + 1
            self.tail._deep_post(kw["target_token_ids"][:n0], kw["target_hidden_states"], deep)
        w4.launch(dr.counter.data_ptr(), dr.layer_id, w4.scratch, *w4.args, stream=stream)
        if hook:
            hook.after_draft(dr, w4)
        ret = w4.h_out.view(k + 1, g.hc_count, g.hidden)
        for i in range(k - 1):
            self._launch(i, self.posb, ret, dt[0, i], lti, num_reject, verdict)
            w1.launch(dr.counter.data_ptr(), dr.layer_id, w1.scratch, *w1.args, stream=stream)
            if hook:
                hook.after_draft(dr, w1)
            ret = w1.h_out.view(1, g.hc_count, g.hidden)
        self._last(k - 1, ret, dt[0, k - 1])
        return dt

    def _last(self, i, ret, exported) -> None:
        kern = self.k_["dl"]
        snap = ret[:1].clone() if self.tail.deep_round else None
        kern.set(h=ret, outtok=exported)
        kern()
        if snap is not None:
            self._deep_token(i, snap, exported)

    # ------------------------------------------------------------ deep check (gates)
    def _ref_token(self, hrow):
        """The drafter's mix (ATOM's ``mix_final``) and a rocKE fused-argmax head of P9's own head spec on the
        same (subset) weights, mapped to the token id; and the served N8 head's token (informational)."""
        from atom.models.qwen4_exp_mono.rocke_head import _Kernel
        from kernels.gfx950 import mx_decode_gemv as MX

        if self.ref_head is None:
            hs_ = self.specs["d1"].head_spec()
            self.ref_head = _Kernel(MX.MxDecodeGemvSpec(**{f: getattr(hs_, f) for f in hs_.__dataclass_fields__}))
        mixed = self.mixer.mix_final(hrow.flatten(1))
        col = torch.empty(1, dtype=torch.int32, device=hrow.device)
        self.ref_head(mixed, self.hw, self.hs, col, 0, 0)
        tok = self.dv.map[col.long()] if self.vsub else col  # the true map (the dmap mutant rotates self.vmap)
        served = None
        if self.dv is not None:
            served = torch.empty(1, dtype=torch.int32, device=hrow.device)
            self.dv.argmax(mixed, served)
        return tok.to(torch.int32), served

    def _deep_token(self, i, hrow, exported):
        tok, served = self._ref_token(hrow)
        _log_deep(f"d{i}.token", torch.equal(tok, exported.view(-1).to(torch.int32)))
        if served is not None:
            same = torch.equal(served, exported.view(-1).to(torch.int32))
            logger.info("tm tail deepinfo d%d.n8 %s", i, "equal" if same else "DIFF")
        return tok

    def _deep(self, i, snap, exported, tables, got) -> None:
        from atom.model_ops.qwen4_exp.ops.qsa import qsa_draft_decode_metadata
        from atom.models.qwen4_exp_mono.front import draft_front
        from atom.models.qwen4_exp_mono.step_meta import write_draft_meta

        tok = self._deep_token(i, snap["h"], exported)
        L, Pp = snap["lens"].clone(), snap["pos"].clone()
        S, G = torch.empty_like(got["slots"]), torch.empty_like(got["logical"])
        Q, C = torch.empty_like(got["reqs"]), torch.empty_like(got["comp"])
        qsa_draft_decode_metadata(L, tables, snap["rej"], S, G, Q, C, 1, 16, 4, bumped_positions=Pp)
        for nm, ref in (("lens", L), ("pos", Pp), ("slots", S), ("logical", G), ("reqs", Q), ("comp", C)):
            _log_deep(f"d{i}.qsa.{nm}", torch.equal(ref, got[nm]))
        w, dr = self.w1, self.dr
        hin = torch.empty_like(w.h_in)
        draft_front(dr.front, dr.predictor, tok, snap["h"].reshape(1, -1), hin)
        _log_deep(f"d{i}.front", torch.equal(hin.view(torch.int16), w.h_in.view(torch.int16)))
        ctr = snap["ctr"].clone()
        pos1 = torch.empty_like(w.pos)
        tab = w.qsa.clone()
        tab[:, 1].fill_(-1)
        write_draft_meta(ctr, G, tables, pos1, tab)
        _log_deep(f"d{i}.counter", torch.equal(ctr, dr.counter))
        _log_deep(f"d{i}.pos", torch.equal(pos1, w.pos))
        _log_deep(f"d{i}.tab", torch.equal(tab, w.qsa))


def provider(request, hybrid):
    """WP4h's tail provider: a tail for ``request`` (``TailRequest``) or None (the hybrid serves that width)."""
    cfg = settings()
    if request.k == 3 and request.requests == 1 and cfg["k3"] == "p9":
        return P9VerifyTail(request, hybrid, cfg)
    if request.k == 0 and request.requests == 1 and cfg["k0"] == "argmax":
        return ArgmaxTail(request, hybrid, cfg)
    return None


def layer_passthrough(request, hybrid):
    """``ATOM_Q38_TRUE_MONO_LAYER`` for ``tail`` mode until rocKE's layer kernels exist (WP3): every layer build
    keeps the hybrid's launcher (served without a "no layer provider" fallback line, and strict-safe)."""
    return hybrid
