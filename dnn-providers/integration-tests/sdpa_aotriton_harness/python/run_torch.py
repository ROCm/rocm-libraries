"""Compute selectable SDPA references via PyTorch on ROCm.

For each (non-skipped) case in a run directory this:
  * Loads Q/K/V (and optional mask) per the dtype contract onto CUDA.
  * math_hp_o: the MATH backend with inputs upcast to float32 (independent fp32
    reference; with math_lp_o it sets the precision-gap budget, and it also
    sanity-checks the gpu_ref candidate).
  * math_lp_o: the MATH backend on the native low-precision (bf16/fp16) inputs
    (fp16 for fp8 cases).
  * reference_o: either math_hp_o (``pytorch-math``) or AOTriton via
    ``SDPBackend.FLASH_ATTENTION`` then ``SDPBackend.EFFICIENT_ATTENTION``.
    fp8 always falls back to ``pytorch-math`` because torch SDPA rejects fp8 on
    every backend.

Outputs are always saved upcast to float32 ('<f4'), matching the gpu_ref output
contract.

torch is imported lazily inside ``run`` so the module passes ``py_compile`` and
imports without a torch install.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Optional, Tuple

import numpy as np

import manifest as mf
import sdpa_cases


def _torch_fp8_dtype(dtype: str) -> Any:
    """Map an fp8 case dtype string to the corresponding torch fp8 dtype."""
    import torch

    return {
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
        "fp8_e4m3_fnuz": torch.float8_e4m3fnuz,
        "fp8_e5m2_fnuz": torch.float8_e5m2fnuz,
    }[dtype]


def _load_qkv(path: str, dtype: str, device: Any) -> Any:
    """Load a Q/K/V tensor from ``.npy`` per the contract onto ``device``."""
    import torch

    arr = np.load(path)
    if dtype == "bf16":
        # uint16 raw bits -> bfloat16 view.
        t = torch.from_numpy(arr).view(torch.bfloat16)
    elif dtype == "fp16":
        t = torch.from_numpy(arr)  # float16 already
    elif dtype in sdpa_cases.FP8_DTYPES:
        # uint8 raw bits -> fp8 view.
        t = torch.from_numpy(arr).view(_torch_fp8_dtype(dtype))
    else:
        raise ValueError(f"unsupported dtype for qkv load: {dtype!r}")
    return t.to(device)


def _load_mask(path: str, device: Any) -> Any:
    """Load an additive fp32 mask onto ``device``."""
    import torch

    arr = np.load(path).astype(np.float32, copy=False)
    return torch.from_numpy(arr).to(device)


def _build_attn_mask(man: dict, device: Any) -> Optional[Any]:
    """Return the additive attn_mask tensor for this case, or None.

    * mode "mask": the saved fp32 bias.
    * mode "window": the synthesized -inf additive mask, broadcast to rank 4.
    * plain / causal: None (causal uses is_causal, not a mask).
    """
    import torch

    mode = man["mode"]
    if mode == "mask":
        return _load_mask(man["files"]["mask"], device)
    if mode == "window":
        win = mf.synthesize_window_mask(
            man["Sq"], man["Skv"], man["left"], man["right"], man["top_left"]
        )  # [Sq, Skv] float32
        t = torch.from_numpy(win).to(device)
        # Broadcast to [B, Hq, Sq, Skv] (no implicit broadcasting in the contract,
        # but torch SDPA accepts a broadcastable mask; expand to full rank to be safe).
        # .contiguous() so a later dtype cast / backend does not choke on the
        # zero-stride expanded view.
        return (
            t.view(1, 1, man["Sq"], man["Skv"])
            .expand(man["B"], man["Hq"], man["Sq"], man["Skv"])
            .contiguous()
        )
    return None


def _maybe_repeat_kv(k: Any, v: Any, Hq: int, Hkv: int) -> Tuple[Any, Any]:
    """Repeat K/V heads to Hq for the enable_gqa fallback path."""
    if Hkv == Hq:
        return k, v
    rep = Hq // Hkv
    k2 = k.repeat_interleave(rep, dim=1)
    v2 = v.repeat_interleave(rep, dim=1)
    return k2, v2


def _sdpa(
    q: Any,
    k: Any,
    v: Any,
    *,
    attn_mask: Optional[Any],
    is_causal: bool,
    scale: Optional[float],
    Hq: int,
    Hkv: int,
) -> Any:
    """Call F.scaled_dot_product_attention with enable_gqa, falling back to
    manual head repetition on older torch that lacks the kwarg."""
    import torch.nn.functional as F

    enable_gqa = Hkv != Hq
    try:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    except TypeError:
        # Older torch: no enable_gqa. Manually expand K/V heads to Hq.
        k2, v2 = _maybe_repeat_kv(k, v, Hq, Hkv)
        return F.scaled_dot_product_attention(
            q,
            k2,
            v2,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        )


def _save_f32(path: str, tensor: Any) -> None:
    """Save a tensor upcast to float32 as '<f4'."""
    import torch

    arr = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    np.save(path, arr.astype("<f4", copy=False))


def _mark_ok(status: dict, reference: str) -> None:
    """Record successful reference processing and clear stale failure metadata."""
    status["state"] = "ok"
    status["reference"] = reference
    status.pop("reason", None)
    status.pop("detail", None)


def _run_case_fp8(
    man: dict,
    q: Any,
    k: Any,
    v: Any,
    attn_mask: Optional[Any],
    *,
    requested_reference: str,
    is_causal: bool,
    scale: Optional[float],
    Hq: int,
    Hkv: int,
) -> dict:
    """Compute references for an fp8 case using the fp32-MATH fallback."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    files = man["files"]
    status = man.setdefault("status", {})

    q32, k32, v32 = q.float(), k.float(), v.float()
    q16, k16, v16 = q32.half(), k32.half(), v32.half()
    if attn_mask is not None:
        mask_hp = attn_mask.float()
        mask_lp = attn_mask.to(torch.float16)
    else:
        mask_hp = None
        mask_lp = None

    # fp32 MATH: both the effective reference and the high-precision diagnostic.
    with sdpa_kernel(SDPBackend.MATH):
        out_hp = _sdpa(
            q32,
            k32,
            v32,
            attn_mask=mask_hp,
            is_causal=is_causal,
            scale=scale,
            Hq=Hq,
            Hkv=Hkv,
        )
        torch.cuda.synchronize()
    _save_f32(files["reference_o"], out_hp)
    _save_f32(files["math_hp_o"], out_hp)

    # fp16 MATH: low-precision reference for the independent precision-gap budget.
    with sdpa_kernel(SDPBackend.MATH):
        out_lp = _sdpa(
            q16,
            k16,
            v16,
            attn_mask=mask_lp,
            is_causal=is_causal,
            scale=scale,
            Hq=Hq,
            Hkv=Hkv,
        )
        torch.cuda.synchronize()
    _save_f32(files["math_lp_o"], out_lp)

    _mark_ok(status, "pytorch-math")
    if requested_reference == "aotriton":
        status["requested_reference"] = "aotriton"
    else:
        status.pop("requested_reference", None)
    status.pop("reference_backend", None)
    return man


def _run_case(man: dict, reference: str) -> dict:
    """Compute and save references for one case; update its status."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    device = torch.device("cuda")
    dtype = man["dtype"]
    Hq, Hkv = man["Hq"], man["Hkv"]
    scale = man["scale"]  # None or float
    is_causal = bool(man["causal"])

    q = _load_qkv(man["files"]["q"], dtype, device)
    k = _load_qkv(man["files"]["k"], dtype, device)
    v = _load_qkv(man["files"]["v"], dtype, device)
    attn_mask = _build_attn_mask(man, device)  # fp32 base, or None

    if dtype in sdpa_cases.FP8_DTYPES:
        return _run_case_fp8(
            man,
            q,
            k,
            v,
            attn_mask,
            requested_reference=reference,
            is_causal=is_causal,
            scale=scale,
            Hq=Hq,
            Hkv=Hkv,
        )

    # Per-path mask precision: flash rejects any bias, so masked/windowed cases
    # fall through to the mem-efficient backend, which requires the bias in the
    # query dtype. math_lp uses the same dtype-cast bias so the budget reflects
    # the same quantization; math_hp uses fp32.
    if attn_mask is not None:
        mask_lp = attn_mask.to(q.dtype).contiguous()
        mask_hp = attn_mask.float()
    else:
        mask_lp = None
        mask_hp = None

    files = man["files"]
    status = man.setdefault("status", {})
    status.pop("requested_reference", None)
    status.pop("reference_backend", None)

    # --- math high-precision reference: upcast inputs to fp32. ---
    with sdpa_kernel(SDPBackend.MATH):
        q32, k32, v32 = q.float(), k.float(), v.float()
        out_hp = _sdpa(
            q32,
            k32,
            v32,
            attn_mask=mask_hp,
            is_causal=is_causal,
            scale=scale,
            Hq=Hq,
            Hkv=Hkv,
        )
        torch.cuda.synchronize()
    _save_f32(files["math_hp_o"], out_hp)

    # --- math low-precision: native dtype inputs, output upcast. ---
    with sdpa_kernel(SDPBackend.MATH):
        out_lp = _sdpa(
            q,
            k,
            v,
            attn_mask=mask_lp,
            is_causal=is_causal,
            scale=scale,
            Hq=Hq,
            Hkv=Hkv,
        )
        torch.cuda.synchronize()
    _save_f32(files["math_lp_o"], out_lp)

    if reference == "pytorch-math":
        _save_f32(files["reference_o"], out_hp)
        _mark_ok(status, "pytorch-math")
        return man

    # --- AOTriton path: flash, then efficient; never math. ---
    reference_backend: Optional[str] = None
    last_err: Optional[str] = None
    for backend, label in (
        (SDPBackend.FLASH_ATTENTION, "flash"),
        (SDPBackend.EFFICIENT_ATTENTION, "efficient"),
    ):
        try:
            with sdpa_kernel(backend):
                out = _sdpa(
                    q,
                    k,
                    v,
                    attn_mask=mask_lp,
                    is_causal=is_causal,
                    scale=scale,
                    Hq=Hq,
                    Hkv=Hkv,
                )
            torch.cuda.synchronize()
            _save_f32(files["reference_o"], out)
            reference_backend = label
            break
        except Exception as exc:  # backend cannot service this shape/config
            last_err = f"{label}: {type(exc).__name__}: {exc}"

    status["reference"] = "aotriton"
    if reference_backend is None:
        status["state"] = "skipped"
        status["reason"] = "aotriton-unsupported"
        status["detail"] = last_err
        return man

    _mark_ok(status, "aotriton")
    status["reference_backend"] = reference_backend
    return man


def run(run_dir: str, reference: str = "pytorch-math") -> int:
    """Process every case in ``run_dir``; rewrite each manifest with its status.

    Returns the number of cases that errored (not skipped); 0 on full success.
    """
    if reference not in ("pytorch-math", "aotriton"):
        raise ValueError(f"unsupported reference mode: {reference!r}")

    index = mf.read_index(run_dir)
    n_err = 0
    for name in index["cases"]:
        man = mf.read_manifest(run_dir, name)
        try:
            man = _run_case(man, reference)
        except Exception as exc:  # unexpected failure: record and continue
            man.setdefault("status", {})
            man["status"]["state"] = "error"
            man["status"]["reason"] = f"{type(exc).__name__}: {exc}"
            n_err += 1
        mf.write_manifest(run_dir, man)
    return n_err


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", required=True, help="run directory from gen_inputs"
    )
    parser.add_argument(
        "--reference",
        choices=("pytorch-math", "aotriton"),
        default="pytorch-math",
        help="reference mode (default: pytorch-math)",
    )
    args = parser.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    n_err = run(run_dir, args.reference)
    print(f"run_torch complete ({n_err} errored).")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
