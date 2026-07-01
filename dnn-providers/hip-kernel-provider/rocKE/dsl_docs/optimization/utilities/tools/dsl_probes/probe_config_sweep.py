# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generic config sweep over a CK DSL spec template.

Replaces the family of one-off `(num_warps, tile_size, block_m_per_warp,
…)` sweep scripts that get rewritten every time a new lever lands.
Given:

- a build function ``build_fn(spec) -> KernelDef``;
- a base spec ``base_spec``;
- a list of override dicts (one variant per dict);
- a callable ``run_fn(kernel_artifact, label) -> float`` returning a per-
  iter latency in microseconds (the harness picks the timing tool you
  trust — see ``probe_targeted_bench.py`` for the canonical CUDA-event
  timer);

… the probe builds each variant, optionally verifies it compiles, and
reports the latency table sorted by best.

The probe **does not** drive a forward pass. It only owns the build
loop, the cache invalidation, and the result table. Wire it to your
favourite ``time_launches`` / ``torch.cuda.Event`` measurement function
so the perf number is in *your* methodology.

CLI demo (just builds-and-prints HSACO sizes for an attention sweep,
no GPU run required):

    python probe_config_sweep.py --demo attention_tiled_2d

Programmatic use:

    from probe_config_sweep import probe_config_sweep
    from rocke.instances.gfx950.attention_tiled_2d import (
        UnifiedAttention2DTiledSpec, build_unified_attention_2d_tiled,
    )

    base = UnifiedAttention2DTiledSpec(
        head_size=64, block_size=32, num_query_heads=64, num_kv_heads=8,
        dtype="bf16", use_sinks=True, sliding_window=0, has_softcap=False,
    )
    overrides = [
        dict(num_warps=1, tile_size=32),
        dict(num_warps=2, tile_size=64),
        dict(num_warps=4, tile_size=64),
        dict(num_warps=4, tile_size=64, waves_per_eu=3),
    ]
    def run_fn(artifact, label):
        # … your bench, returns latency in us …
        return 0.0
    probe_config_sweep(
        build_fn=build_unified_attention_2d_tiled,
        base_spec=base, overrides=overrides, run_fn=run_fn,
    )
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence


def _bootstrap_rocke() -> None:
    try:
        import rocke  # noqa: F401

        return
    except ImportError:
        pass
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "Python"
        if (candidate / "rocke" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return
        candidate = parent / "rocke" / "__init__.py"
        if candidate.exists():
            sys.path.insert(0, str(parent))
            return


_bootstrap_rocke()

from rocke.helpers.compile import compile_kernel  # noqa: E402


def _label_for(override: dict) -> str:
    """Make a short, sortable variant label from an override dict."""
    if not override:
        return "baseline"
    parts = []
    for k, v in override.items():
        if isinstance(v, bool):
            parts.append(f"{k}={int(v)}")
        else:
            parts.append(f"{k}={v}")
    return ".".join(parts)


def probe_config_sweep(
    *,
    build_fn: Callable[[Any], Any],
    base_spec: Any,
    overrides: Sequence[dict],
    run_fn: Callable[[Any, str], float] | None = None,
    only_build: bool = False,
    label_fn: Callable[[dict], str] | None = None,
    verify_fn: Callable[[Any, str], bool] | None = None,
) -> list[dict]:
    """Build (and optionally bench) each spec variant.

    Args:
        build_fn: callable ``spec -> KernelDef``.
        base_spec: the base dataclass; ``dataclasses.replace`` is used.
        overrides: iterable of override dicts; ``{}`` means baseline.
        run_fn: optional ``(KernelArtifact, label) -> latency_us``. If
            ``None`` (or ``only_build=True``), only build + HSACO size
            is reported.
        only_build: skip ``run_fn`` even if provided.
        label_fn: short-string label generator; defaults to ``_label_for``.
        verify_fn: optional ``(KernelArtifact, label) -> bool``. Variants
            that fail verification are reported but excluded from the
            "best" ranking.

    Returns the full list of result dicts (in input order).
    """
    label_fn = label_fn or _label_for
    rows: list[dict] = []
    for ov in overrides:
        label = label_fn(ov)
        try:
            spec = replace(base_spec, **ov) if ov else base_spec
        except (TypeError, ValueError) as e:
            print(f"  {label:<32} SPEC-FAIL: {type(e).__name__}: {e}")
            rows.append(
                {
                    "label": label,
                    "override": dict(ov),
                    "error": f"SPEC: {type(e).__name__}: {e}",
                }
            )
            continue
        row: dict = {"label": label, "override": dict(ov)}
        try:
            kdef = build_fn(spec)
            artifact = compile_kernel(kdef)
        except Exception as e:  # noqa: BLE001
            row["error"] = f"BUILD: {type(e).__name__}: {e}"
            rows.append(row)
            print(f"  {label:<32} BUILD-FAIL: {row['error']}")
            continue
        row["hsaco_bytes"] = len(artifact.hsaco)
        row["build_ms"] = artifact.timings.get("total", 0.0)
        if verify_fn is not None:
            try:
                row["verify_ok"] = bool(verify_fn(artifact, label))
            except Exception as e:  # noqa: BLE001
                row["verify_ok"] = False
                row["verify_error"] = f"{type(e).__name__}: {e}"
        if run_fn is not None and not only_build:
            try:
                row["latency_us"] = float(run_fn(artifact, label))
            except Exception as e:  # noqa: BLE001
                row["latency_us"] = None
                row["bench_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        line = (
            f"  {label:<32} build={row['build_ms']:6.1f}ms "
            f"hsaco={row['hsaco_bytes']:>7}B"
        )
        if row.get("verify_ok") is False:
            line += "  VERIFY=FAIL"
        if "latency_us" in row and row["latency_us"] is not None:
            line += f"  {row['latency_us']:7.1f}us"
        if "bench_error" in row:
            line += f"  BENCH-FAIL: {row['bench_error']}"
        print(line)

    benched = [
        r
        for r in rows
        if r.get("latency_us") is not None and r.get("verify_ok") is not False
    ]
    if benched:
        best = min(benched, key=lambda r: r["latency_us"])
        print(f"\nbest: {best['label']}  {best['latency_us']:.1f} us")
    return rows


# ---- Demo --------------------------------------------------------------


def _demo_attention_tiled_2d_build_only() -> None:
    from rocke.instances.gfx950.attention_tiled_2d import (
        UnifiedAttention2DTiledSpec,
        build_unified_attention_2d_tiled,
    )

    base = UnifiedAttention2DTiledSpec(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=True,
        sliding_window=0,
        has_softcap=False,
    )
    overrides = [
        {},  # baseline
        dict(num_warps=2, tile_size=64),
        dict(num_warps=4, tile_size=64),
        dict(num_warps=4, tile_size=64, waves_per_eu=3),
        dict(num_warps=4, tile_size=64, block_m_per_warp=32),
        # mfma_32x32 requires block_m_per_warp=32 — set both together.
        dict(num_warps=4, tile_size=64, block_m_per_warp=32, use_mfma_32x32=True),
        dict(
            num_warps=4,
            tile_size=64,
            block_m_per_warp=32,
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
        ),
        # Intentionally-invalid override to exercise the SPEC-FAIL path.
        dict(num_warps=4, tile_size=64, use_mfma_32x32=True),
    ]
    probe_config_sweep(
        build_fn=build_unified_attention_2d_tiled,
        base_spec=base,
        overrides=overrides,
        only_build=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--demo",
        choices=["attention_tiled_2d"],
        default="attention_tiled_2d",
        help="build-only smoke; wire run_fn programmatically for perf",
    )
    args = p.parse_args(argv)
    if args.demo == "attention_tiled_2d":
        _demo_attention_tiled_2d_build_only()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
