# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Subprocess bodies for the GPU SDPA-backward cases that need a fresh process:

  * case (l): the autograd worker-thread teardown hazard (M10) is only
    observable by actually letting the interpreter tear down -- a foreground
    ``pytest`` process never exercises that.
  * case (o): ``HIPDNN_TORCH_SELECT=force`` must be set before the first
    ``bootstrap()`` call (``bootstrap.py`` reads it once and caches the
    result), so each force-mode scenario needs its own process.
  * case (r): the benchmarking guard needs ``HIPDNN_FORCE_BENCHMARKING`` and
    ``HIPDNN_CACHE_DIR`` set before bootstrap too, and a process-exclusive
    cache root.
  * case (s): the mixed-engine-pair ``HIPDNN_HEUR_CONFIG_PATH`` rule is a
    process-global preference; isolating it in its own process keeps it from
    leaking into any other test's engine selection.

Not a test module itself (the leading underscore keeps pytest from collecting
it); ``tests/test_sdpa_bwd_parity.py`` launches it with ``sys.executable`` and
reads its stdout (one JSON object) and exit code. Never detached/backgrounded:
every mode here runs to completion and exits.

MHA shapes here use D=128 (``64 < D <= 128``, no mask, no GQA), the one
head-dim range ASM serves on every build state (with or without FlyDSL, with
or without hip-kernel-provider#16337) -- so the backward graph always reaches
a real ``_execute`` call regardless of which build runs this script.
"""

import json
import os
import sys


def _census_json(ov):
    """``ov.census()`` with its ``engines`` sets made JSON-serializable."""
    rows = {}
    for key, row in ov.census().items():
        row = dict(row)
        if "engines" in row:
            row["engines"] = sorted(row["engines"])
        rows[key] = row
    return rows


def _qkv(torch, b, s, hq, d, dtype, hkv=None, requires_grad=True):
    hkv = hkv or hq
    q = torch.randn(b, s, hq, d, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(b, s, hkv, d, device="cuda", dtype=dtype).transpose(1, 2)
    v = torch.randn(b, s, hkv, d, device="cuda", dtype=dtype).transpose(1, 2)
    if requires_grad:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
    return q, k, v


def _route_outcome(ov, sq, skv, d, dtype_tok):
    """``("served", "")`` or ``("native", <first fallback reason>)`` for a
    single-call grad-eligible scenario."""
    row = ov.census().get(f"grad Sq={sq},Skv={skv},D={d},dtype={dtype_tok}")
    if row and row["native"] and not row["aot"]:
        reasons = ov.fallback_reasons()
        reason = next(iter(reasons), "")
        return "native", reason
    return "served", ""


# --------------------------------------------------------------------------- #
# case (l): autograd worker-thread teardown                                   #
# --------------------------------------------------------------------------- #
def _run_backward():
    import torch
    import torch.nn.functional as F

    import hipdnn_torch

    hipdnn_torch.install(["sdpa"])
    ov = hipdnn_torch.overrides()["sdpa"]
    q, k, v = _qkv(torch, 1, 16, 8, 128, torch.bfloat16)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out.sum().backward()
    print(json.dumps({"census": _census_json(ov)}))


_HELD = []  # keeps a forward-only output (and the autograd ctx it holds) alive


def _run_forward_only():
    import torch
    import torch.nn.functional as F

    import hipdnn_torch

    hipdnn_torch.install(["sdpa"])
    ov = hipdnn_torch.overrides()["sdpa"]
    q, k, v = _qkv(torch, 1, 16, 8, 128, torch.bfloat16)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    _HELD.append(out)  # never call .backward(); the process exits with this held
    print(json.dumps({"census": _census_json(ov)}))


# --------------------------------------------------------------------------- #
# case (o): force mode                                                        #
# --------------------------------------------------------------------------- #
def _run_force(engine, shape):
    os.environ["HIPDNN_TORCH_SELECT"] = "force"
    os.environ["HIPDNN_TORCH_ENGINE"] = engine

    import torch
    import torch.nn.functional as F

    import hipdnn_torch

    hipdnn_torch.install(["sdpa"])
    ov = hipdnn_torch.overrides()["sdpa"]
    sq = skv = 16
    d = 128
    if shape == "mha":
        q, k, v = _qkv(torch, 1, sq, 2, d, torch.bfloat16)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    elif shape == "gqa":
        q, k, v = _qkv(torch, 1, sq, 8, d, torch.bfloat16, hkv=2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
    elif shape == "mask":
        q, k, v = _qkv(torch, 1, sq, 2, d, torch.bfloat16)
        mask = torch.zeros(1, 1, sq, skv, device="cuda", dtype=torch.bfloat16)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    else:
        raise ValueError(shape)
    out.sum().backward()
    route, reason = _route_outcome(ov, sq, skv, d, "bf16")
    print(json.dumps({"census": _census_json(ov), "route": route, "reason": reason}))


# --------------------------------------------------------------------------- #
# case (r): benchmarking guard (M17)                                          #
# --------------------------------------------------------------------------- #
def _run_benchmark_guard():
    import torch
    import torch.nn.functional as F

    import hipdnn_torch

    b, hq, sq, skv, d = 1, 2, 8, 8, 128
    torch.manual_seed(0)
    mask = torch.zeros(b, 1, sq, skv, device="cuda", dtype=torch.bfloat16)
    native_sdpa = F.scaled_dot_product_attention  # captured before install patches it

    def _native_grads(qq, kk, vv):
        qq = qq.detach().clone().requires_grad_(True)
        kk = kk.detach().clone().requires_grad_(True)
        vv = vv.detach().clone().requires_grad_(True)
        out = native_sdpa(qq, kk, vv, attn_mask=mask, is_causal=False)
        out.sum().backward()
        return out, qq.grad, kk.grad, vv.grad

    def _pack(out, dq, dk, dv):
        return {
            "o": out.detach().float().cpu().tolist(),
            "dq": dq.float().cpu().tolist(),
            "dk": dk.float().cpu().tolist(),
            "dv": dv.float().cpu().tolist(),
        }

    q1, k1, v1 = _qkv(torch, b, sq, hq, d, torch.bfloat16)
    native1 = _pack(*_native_grads(q1, k1, v1))

    hipdnn_torch.install(["sdpa"])
    ov = hipdnn_torch.overrides()["sdpa"]
    out1 = F.scaled_dot_product_attention(q1, k1, v1, attn_mask=mask, is_causal=False)
    out1.sum().backward()
    routed1 = _pack(out1, q1.grad, k1.grad, v1.grad)

    q2, k2, v2 = _qkv(torch, b, sq, hq, d, torch.bfloat16)
    native2 = _pack(*_native_grads(q2, k2, v2))
    out2 = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=mask, is_causal=False)
    out2.sum().backward()
    routed2 = _pack(out2, q2.grad, k2.grad, v2.grad)

    print(
        json.dumps(
            {
                "census": _census_json(ov),
                "step1": {"routed": routed1, "native": native1},
                "step2": {"routed": routed2, "native": native2},
            }
        )
    )


# --------------------------------------------------------------------------- #
# case (s): mixed engine pair (Q4)                                            #
# --------------------------------------------------------------------------- #
def _run_mixed_pair():
    """Force mode pins one engine for every graph of a call (M13), so it
    cannot build a mixed pair. Instead, a ``HIPDNN_HEUR_CONFIG_PATH`` rule
    (``SelectionHeuristic::Config``, docs/Environment.md#HIPDNN_HEUR_CONFIG_PATH)
    that matches ``sdpa_fwd`` nodes moves ``ASM_SDPA_ENGINE`` to the front
    of the FORWARD graph's ranking only; the rule never names ``sdpa_bwd``,
    so the backward graph still ranks normally and, for a GQA shape, ASM's
    bwd builder declines it (SdpaBwdPlanBuilder.cpp:1137-1141), leaving some
    other (FlyDSL) engine to serve. ``OperationRule::matches`` requires
    ``tensors.size() == inputs.size()`` (EngineOverrideConfig.hpp:94), and a
    plain SDPA forward's input set is exactly Q, K, V (ConfigBuiltIn.cpp
    :217-220 keys them in that order); ``"version": 2`` switches the parser
    to named ``tensor_id`` matching (order-independent, any dims via the
    ``-1`` wildcard) instead of positional, with ids
    ``q_tensor_uid``/``k_tensor_uid``/``v_tensor_uid``
    (AutotuneConfigNames.hpp) -- confirmed against
    backend/tests/heuristics/TestConfigBuiltIn.cpp's own
    ``FinalizeMatchesRuleNamingEveryCuSeqLenTensorOfTheNode`` fixture."""
    import tempfile

    import torch
    import torch.nn.functional as F

    rule_path = os.path.join(tempfile.mkdtemp(), "heur_override.json")
    wildcard = [-1, -1, -1, -1]
    with open(rule_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "version": 2,
                "engine_overrides": [
                    {
                        "op": "sdpa_fwd",
                        "engine_name": "ASM_SDPA_ENGINE",
                        "tensors": [
                            {"tensor_id": "q_tensor_uid", "dim": wildcard},
                            {"tensor_id": "k_tensor_uid", "dim": wildcard},
                            {"tensor_id": "v_tensor_uid", "dim": wildcard},
                        ],
                    }
                ],
            },
            fh,
        )
    os.environ["HIPDNN_HEUR_CONFIG_PATH"] = rule_path

    b, hq, hkv, sq, skv, d = 1, 8, 2, 16, 16, 128
    q, k, v = _qkv(torch, b, sq, hq, d, torch.bfloat16, hkv=hkv)
    native_sdpa = F.scaled_dot_product_attention  # captured before install patches it
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    native_out = native_sdpa(q2, k2, v2, enable_gqa=True)
    native_out.sum().backward()

    import hipdnn_torch

    hipdnn_torch.install(["sdpa"])
    ov = hipdnn_torch.overrides()["sdpa"]
    out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    out.sum().backward()

    def _pack(o, dq, dk, dv):
        return {
            "o": o.detach().float().cpu().tolist(),
            "dq": dq.float().cpu().tolist(),
            "dk": dk.float().cpu().tolist(),
            "dv": dv.float().cpu().tolist(),
        }

    print(
        json.dumps(
            {
                "census": _census_json(ov),
                "routed": _pack(out, q.grad, k.grad, v.grad),
                "native": _pack(native_out, q2.grad, k2.grad, v2.grad),
            }
        )
    )


def main(argv):
    mode = argv[1]
    if mode == "backward":
        _run_backward()
    elif mode == "forward_only":
        _run_forward_only()
    elif mode == "force_asm":
        _run_force("ASM_SDPA_ENGINE", "mha")
    elif mode == "force_flydsl_bwd":
        # A FlyDSL engine that serves only the backward: forcing it for a call
        # that also needs a fwd-with-stats graph makes THAT probe decline
        # (M13: "any other pin declines grad-eligible calls"). Name from the
        # DESIGN.md M8 census example (line 241).
        _run_force("hipkernel:flydsl_sdpa_bwd_split", "mha")
    elif mode == "force_asm_gqa":
        _run_force("ASM_SDPA_ENGINE", "gqa")
    elif mode == "force_asm_mask":
        _run_force("ASM_SDPA_ENGINE", "mask")
    elif mode == "benchmark_guard":
        _run_benchmark_guard()
    elif mode == "mixed_pair":
        _run_mixed_pair()
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
