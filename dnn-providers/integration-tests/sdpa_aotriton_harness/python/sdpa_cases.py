"""Test-case definitions for the SDPA gpu_ref vs AOTriton numerical harness.

A :class:`Case` describes a single forward SDPA configuration: shapes, dtype,
masking mode, optional custom scale, and a deterministic seed. Cases are grouped
into tiers (``quick``, ``full``, ``irregular``) of increasing cost / coverage.

Scope (intersection of what the gpu_ref kernel and AOTriton-via-PyTorch both
support, forward-only):
  * plain MHA
  * causal (top-left aligned, square only: Sq == Skv)
  * additive float mask (full rank-4 [B, Hq, Sq, Skv])
  * sliding window (gpu_ref native left/right bounds; torch sees an equivalent
    additive -inf mask)
  * GQA / MQA (Hkv divides Hq; the same Hkv is used for both K and V)
  * custom scale

This module has no third-party dependencies so it always imports cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Case:
    """A single SDPA test configuration (everything in a manifest except files).

    Attributes mirror the per-case manifest schema consumed by the other
    scripts. ``mode`` is the high-level intent; ``left``/``right``/``top_left``/
    ``causal``/``has_mask`` are the concrete derived flags that drive both the
    C++ driver invocation and the torch reference.
    """

    name: str
    dtype: str  # "bf16" | "fp16"

    B: int
    Hq: int
    Hkv: int
    Sq: int
    Skv: int
    D: int

    scale: Optional[float]

    mode: str  # "plain" | "causal" | "window" | "mask"

    left: int  # sliding-window left bound (-1 disables)
    right: int  # sliding-window right bound (-1 disables)
    top_left: bool  # top-left alignment for window/causal

    causal: bool  # is_causal path (requires Sq == Skv)
    has_mask: bool  # additive float mask provided

    seed: int

    def validate(self) -> None:
        """Raise ``ValueError`` if the case violates a scope constraint."""
        if self.dtype not in ("bf16", "fp16"):
            raise ValueError(f"{self.name}: unsupported dtype {self.dtype!r}")
        if self.mode not in ("plain", "causal", "window", "mask"):
            raise ValueError(f"{self.name}: unsupported mode {self.mode!r}")
        if self.D > 256:
            raise ValueError(f"{self.name}: head dim {self.D} exceeds 256")
        if self.causal and self.Sq != self.Skv:
            raise ValueError(
                f"{self.name}: causal requires Sq == Skv (got {self.Sq} != {self.Skv})"
            )
        if self.Hq % self.Hkv != 0:
            raise ValueError(
                f"{self.name}: Hkv ({self.Hkv}) must divide Hq ({self.Hq})"
            )
        if self.mode == "causal" and not self.causal:
            raise ValueError(f"{self.name}: mode 'causal' must set causal=True")
        if self.mode == "mask" and not self.has_mask:
            raise ValueError(f"{self.name}: mode 'mask' must set has_mask=True")
        if self.mode == "window" and self.left < 0 and self.right < 0:
            raise ValueError(f"{self.name}: window mode needs a left or right bound")


def _name(
    dtype: str,
    B: int,
    Hq: int,
    Hkv: int,
    Sq: int,
    Skv: int,
    D: int,
    mode: str,
    *,
    scale: Optional[float] = None,
    left: int = -1,
    right: int = -1,
    top_left: bool = True,
) -> str:
    """Build a stable, filesystem-safe unique name encoding the config."""
    parts = [
        dtype,
        f"b{B}",
        f"hq{Hq}",
        f"hkv{Hkv}",
        f"sq{Sq}",
        f"skv{Skv}",
        f"d{D}",
        mode,
    ]
    if mode == "window":
        parts.append(f"l{left}r{right}")
        parts.append("tl" if top_left else "br")
    if scale is not None:
        # Encode the scale without a dot so the name stays path-safe.
        parts.append("scale" + str(scale).replace(".", "p").replace("-", "m"))
    return "_".join(parts)


# Deterministic seed assignment: a stable base plus the case ordinal. The
# ordinal is added by the tier builders so that re-ordering a list does not
# silently change every seed (we hash the name instead, see _seed).
def _seed(name: str, seed_base: int) -> int:
    """Derive a stable per-case seed from the case name and a base.

    Uses a simple deterministic string hash (FNV-1a, 32-bit) so the seed is
    reproducible across machines / Python versions (``hash()`` is salted).
    """
    h = 0x811C9DC5
    for ch in name:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return (seed_base + h) & 0x7FFFFFFF


def _make(
    dtype: str,
    B: int,
    Hq: int,
    Hkv: int,
    Sq: int,
    Skv: int,
    D: int,
    mode: str,
    seed_base: int,
    *,
    scale: Optional[float] = None,
    left: int = -1,
    right: int = -1,
    top_left: bool = True,
) -> Case:
    """Construct a validated :class:`Case`, deriving derived flags from ``mode``."""
    causal = mode == "causal"
    has_mask = mode == "mask"

    if mode == "causal":
        # top-left causal on a square matrix == torch is_causal=True.
        # gpu_ref expresses this as right=0, left=-1, top_left=True.
        eff_left, eff_right, eff_tl = -1, 0, True
    elif mode == "window":
        eff_left, eff_right, eff_tl = left, right, top_left
    else:  # plain, mask
        eff_left, eff_right, eff_tl = -1, -1, True

    name = _name(
        dtype,
        B,
        Hq,
        Hkv,
        Sq,
        Skv,
        D,
        mode,
        scale=scale,
        left=eff_left,
        right=eff_right,
        top_left=eff_tl,
    )
    case = Case(
        name=name,
        dtype=dtype,
        B=B,
        Hq=Hq,
        Hkv=Hkv,
        Sq=Sq,
        Skv=Skv,
        D=D,
        scale=scale,
        mode=mode,
        left=eff_left,
        right=eff_right,
        top_left=eff_tl,
        causal=causal,
        has_mask=has_mask,
        seed=_seed(name, seed_base),
    )
    case.validate()
    return case


def quick_cases(seed_base: int = 0) -> List[Case]:
    """Fast default tier: bf16 only, ~12-20 small cases covering all four modes."""
    cases: List[Case] = []

    # Square plain + causal across a couple of batch sizes and head dims.
    for B in (1, 4):
        for Sq, Skv in ((64, 64), (128, 128)):
            for D in (64, 128):
                cases.append(_make("bf16", B, 4, 4, Sq, Skv, D, "plain", seed_base))
                cases.append(_make("bf16", B, 4, 4, Sq, Skv, D, "causal", seed_base))

    # Non-square plain (NPOT seqlens) — exercises Sq != Skv path.
    cases.append(_make("bf16", 1, 4, 4, 143, 256, 64, "plain", seed_base))
    cases.append(_make("bf16", 4, 4, 4, 143, 256, 128, "plain", seed_base))

    # One sliding-window case (top-left, both bounds set).
    cases.append(
        _make(
            "bf16",
            1,
            4,
            4,
            128,
            128,
            64,
            "window",
            seed_base,
            left=16,
            right=8,
            top_left=True,
        )
    )

    # One additive-mask case.
    cases.append(_make("bf16", 1, 4, 4, 128, 128, 64, "mask", seed_base))

    return cases


def full_cases(seed_base: int = 0) -> List[Case]:
    """Comprehensive tier: bf16 + fp16, wide head dims, GQA, all modes, scale."""
    cases: List[Case] = []

    for dtype in ("bf16", "fp16"):
        # Head-dim sweep (plain + causal), small batch to keep the count sane.
        for D in (16, 32, 48, 64, 80, 96, 128, 256):
            cases.append(_make(dtype, 1, 4, 4, 128, 128, D, "plain", seed_base))
            cases.append(_make(dtype, 1, 4, 4, 128, 128, D, "causal", seed_base))

        # Sequence-length sweep, including an NPOT pair, plain + non-square.
        for Sq, Skv in ((256, 256), (512, 512), (143, 587)):
            cases.append(_make(dtype, 1, 4, 4, Sq, Skv, 64, "plain", seed_base))
        # Square NPOT causal.
        cases.append(_make(dtype, 1, 4, 4, 256, 256, 64, "causal", seed_base))

        # GQA / MQA pairs (Hkv divides Hq). Plain + causal.
        for Hq, Hkv in ((16, 8), (10, 2)):
            cases.append(_make(dtype, 1, Hq, Hkv, 128, 128, 64, "plain", seed_base))
            cases.append(_make(dtype, 1, Hq, Hkv, 128, 128, 64, "causal", seed_base))

        # Sliding-window: top-left and bottom-right variants; non-square too.
        cases.append(
            _make(
                dtype,
                1,
                4,
                4,
                128,
                128,
                64,
                "window",
                seed_base,
                left=32,
                right=0,
                top_left=True,
            )
        )
        cases.append(
            _make(
                dtype,
                1,
                4,
                4,
                143,
                256,
                64,
                "window",
                seed_base,
                left=16,
                right=16,
                top_left=False,
            )
        )

        # Additive-mask cases: square and non-square.
        cases.append(_make(dtype, 1, 4, 4, 128, 128, 64, "mask", seed_base))
        cases.append(_make(dtype, 1, 4, 4, 143, 256, 64, "mask", seed_base))

        # Explicit custom-scale variant.
        cases.append(
            _make(dtype, 1, 4, 4, 128, 128, 64, "plain", seed_base, scale=0.125)
        )

    return cases


def irregular_cases(seed_base: int = 0) -> List[Case]:
    """Opt-in slow tier: a small prime / NPOT sample (plain + causal).

    The reference kernel recomputes QK^T per output element, so these prime
    head-dims and odd sequence lengths are intentionally small in count.
    """
    cases: List[Case] = []
    for D in (73, 113):
        cases.append(_make("bf16", 1, 4, 4, 157, 257, D, "plain", seed_base))
        # Causal requires a square matrix; use a prime square length.
        cases.append(_make("bf16", 1, 4, 4, 157, 157, D, "causal", seed_base))
    return cases


_TIERS = {
    "quick": quick_cases,
    "full": full_cases,
    "irregular": irregular_cases,
}


def get_cases(tier: str, seed_base: int = 0) -> List[Case]:
    """Return the case list for ``tier``.

    Raises ``ValueError`` for an unknown tier name.
    """
    try:
        builder = _TIERS[tier]
    except KeyError as exc:
        valid = ", ".join(sorted(_TIERS))
        raise ValueError(f"unknown tier {tier!r}; valid tiers: {valid}") from exc
    cases = builder(seed_base)

    # Defensive: names must be unique within a tier (they key the manifests).
    seen: set[str] = set()
    for c in cases:
        if c.name in seen:
            raise ValueError(f"duplicate case name within tier {tier!r}: {c.name}")
        seen.add(c.name)
    return cases


def available_tiers() -> List[str]:
    """List the tier names usable with :func:`get_cases`."""
    return sorted(_TIERS)
