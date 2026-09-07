# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Shared per-op device-time census for the model A/B samples.

Wraps the hot ``torch.nn.functional`` entry points in CUDA-event timing pairs for
one forward call and returns a ranked table. Kineto GPU capture is dead on this
ROCm build, so CUDA events are the reliable per-op device timer. This is measurement
only -- it does not route anything to hipDNN (that is what ``hipdnn_torch.install``
does); run it with the injection installed to time the routed ops, or uninstalled to
time native.
"""


# category label -> (F attribute, ...) -- conv + transformer ops the samples touch
_OPS = {
    "conv3d": ("conv3d",),
    "conv2d": ("conv2d",),
    "linear (GEMM)": ("linear",),
    "attention (SDPA)": ("scaled_dot_product_attention",),
    "group_norm": ("group_norm",),
    "layer_norm": ("layer_norm",),
    "rms_norm": ("rms_norm",),
    "gelu": ("gelu",),
    "silu": ("silu",),
}


def device_time_census(torch, run_forward) -> str:
    """Time each op category over a single ``run_forward()`` call via CUDA events."""
    F = torch.nn.functional
    events = []  # (category, start_event, end_event)

    def make(cat, real):
        def wrapper(*a, **k):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = real(*a, **k)
            e.record()
            events.append((cat, s, e))
            return out

        return wrapper

    reals = {}
    for cat, attrs in _OPS.items():
        for attr in attrs:
            real = getattr(F, attr, None)
            if real is not None:
                reals[(cat, attr)] = real

    with torch.no_grad():
        run_forward()  # warm
        torch.cuda.synchronize()
        for (cat, attr), real in reals.items():
            setattr(F, attr, make(cat, real))
        try:
            run_forward()
            torch.cuda.synchronize()
        finally:
            for (cat, attr), real in reals.items():
                setattr(F, attr, real)

    totals, calls = {}, {}
    for cat, s, e in events:
        totals[cat] = totals.get(cat, 0.0) + s.elapsed_time(e)
        calls[cat] = calls.get(cat, 0) + 1
    if not totals:
        return "  (no wrapped ops fired)"
    grand = sum(totals.values()) or 1.0
    lines = ["  category            calls   dev_ms   share"]
    for cat in sorted(totals, key=lambda k: -totals[k]):
        lines.append(
            f"  {cat:18s}  {calls[cat]:5d}  {totals[cat]:7.3f}  "
            f"{100 * totals[cat] / grand:5.1f}%"
        )
    lines.append(
        f"  {'(sum of wrapped)':18s}  {sum(calls.values()):5d}  "
        f"{grand:7.3f}  100.0%"
    )
    return "\n".join(lines)
