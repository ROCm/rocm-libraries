# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared multi-engine sweep lane for the prefill live benchmarks.

Single source for the ``--variants sweep`` lane used by both the gfx942 and
gfx950 ``benchmark_prefill2d_live.py`` scripts. The lane was copy-pasted into
both benches and the copies had already diverged (one hard-crashed in the
summary, the other silently marked its own timings as errors); hoisting it here
means the timing, per-path record shape, and summary key handling stay identical
across arches and the next fix lands once.
"""

from __future__ import annotations


def run_sweep(shape, data, sw, is_fp8, bench, *, arch, stream_handle, warmup, iters):
    """Time every engine the dispatcher registry offers for this problem.

    Builds one :class:`~dispatch.attention.AttentionRequest`, calls
    :func:`~dispatch.attention.attention_sweep_space` (the deduped spec of every
    *supported* candidate), groups the offered engines by their launched path,
    and times each distinct path via ``run_unified_attention_torch``.

    Returns a dict keyed by launched path. Each value is either a timed entry
    ``{"ms", "engines", "kernel", "out"}`` or -- if that one path raised --
    ``{"error", "engines"}``, so a failure on one path never discards the paths
    that already timed. An empty return means no candidate was eligible for the
    shape (the caller renders that as a visible marker, not a blank record).

    NOTE (framework phase): the registry decides the kernel *path* + candidate,
    not the CTA geometry (still owned by ``_tiled_spec_from_problem`` at launch).
    So distinct candidates that route to the same launched path collapse to one
    timed entry, which records the engine ``spec.name``s that mapped to it.
    """
    import torch
    from dispatch.attention import AttentionRequest, attention_sweep_space
    from kernels import run_unified_attention_torch
    from rocke.runtime import synchronize_and_release, time_launches

    dtype_str = "bf16" if shape.q_dtype == "torch.bfloat16" else "fp16"
    req = AttentionRequest(
        batch=shape.num_seqs,
        nhead_q=shape.num_query_heads,
        nhead_k=shape.num_kv_heads,
        seqlen_q=shape.max_seqlen_q,
        seqlen_k=shape.max_seqlen_k,
        hdim_q=shape.head_size,
        hdim_v=shape.head_size,
        arch=arch,
        dtype=dtype_str,
        sliding_window=sw,
        kv_block_size=shape.block_size,
        num_sms=bench.num_sms,
    )

    specs = attention_sweep_space(req)
    problem = bench._problem(shape, sw, is_fp8)

    # Group the offered engines by the launched path they resolve to.
    engines_by_path = {}
    for spec in specs:
        engines_by_path.setdefault(spec.path, []).append(spec.name)

    entries = {}
    for path, engine_names in engines_by_path.items():
        try:
            run_backend = "tiled" if path == "2d" else path
            kernel = _sweep_kernel_name(problem, run_backend)
            out = torch.empty_like(data["query"])

            def call_once(_backend=run_backend, _out=out):
                run_unified_attention_torch(
                    problem=problem,
                    q=data["query"],
                    k=data["key_cache"],
                    v=data["value_cache"],
                    out=_out,
                    cu_seqlens_q=data["cu_seqlens_q"],
                    seqused_k=data["kv_lens"],
                    softmax_scale=data["scale"],
                    block_table=data["block_tables"],
                    softcap=float(shape.softcap),
                    sinks=data["sinks"],
                    alibi_slopes=data["alibi_slopes"],
                    backend=_backend,
                    stream=stream_handle,
                )

            ms = time_launches(
                call_once, warmup=warmup, iters=iters, stream=stream_handle
            )
            synchronize_and_release(stream_handle)
            entries[path] = {
                "ms": ms,
                "engines": engine_names,
                "kernel": kernel,
                "out": out,
            }
        except Exception as exc:  # noqa: BLE001  # one bad path must not sink the rest
            entries[path] = {"error": repr(exc), "engines": engine_names}
    return entries


def _sweep_kernel_name(problem, run_backend):
    """Launched-kernel name for a swept path (mirrors ``_run_prod``'s
    ``instance_name`` so a sweep entry can be joined against a prod entry)."""
    if run_backend == "tiled":
        from kernels import supports_native_unified_attention_tiled
        from kernels.common.attention_unified import _tiled_spec_from_problem

        ok, _ = supports_native_unified_attention_tiled(problem)
        return _tiled_spec_from_problem(problem).kernel_name() if ok else "scalar"
    if run_backend == "3d":
        from kernels import supports_native_unified_attention_3d_tiled

        ok, _ = supports_native_unified_attention_3d_tiled(problem)
        return "3d" if ok else "scalar"
    return "scalar"


def record_sweep_entries(rec, sweep_entries, *, tri_out, tri_ms, tol, compare, best):
    """Fold sweep results into ``rec['variants']`` as ``sweep:<path>`` records.

    Returns the (possibly updated) ``best`` tuple. An empty sweep is recorded as
    a single visible ``sweep`` marker so "nothing was eligible" never renders
    identically to "the sweep ran cleanly". A per-path error is recorded as its
    own ``sweep:<path>`` entry rather than discarding the paths that succeeded.
    """
    if not sweep_entries:
        rec["variants"]["sweep"] = {"skipped": "no eligible engines for this shape"}
        return best
    for path, ent in sweep_entries.items():
        vname = f"sweep:{path}"
        if "error" in ent:
            rec["variants"][vname] = {
                "error": ent["error"],
                "engines": ent.get("engines", []),
            }
            continue
        s_err = compare(ent["out"], tri_out)
        s_ok = s_err <= tol
        s_ms = ent["ms"]
        s_spd = tri_ms / s_ms if s_ms > 0 else 0.0
        rec["variants"][vname] = {
            "ms": s_ms,
            "speedup": s_spd,
            "max_abs": s_err,
            "ok": s_ok,
            "engines": ent["engines"],
            "kernel": ent.get("kernel"),
        }
        # Track the winner by ms (lower = faster), the SAME unit the prod/single-
        # kernel lane uses for `best` -- so a mixed `--variants prod sweep` run
        # compares like with like. (Within a sweep-only run this is equivalent to
        # ranking by speedup, since speedup = tri_ms / s_ms with tri_ms fixed.)
        if s_ok and (best is None or s_ms < best[1]):
            best = (vname, s_ms)
    return best


def expand_variant_keys(args_variants, rec_variants):
    """Per-record: replace the literal ``sweep`` request with the concrete
    ``sweep:<path>`` keys actually emitted for this record (or the ``sweep``
    marker when the sweep was empty). Iterating the emitted keys is what keeps
    the summary from looking up a fixed ``sweep`` key the lane never writes."""
    out = []
    for v in args_variants:
        if v == "sweep":
            out.extend(
                sorted(
                    k for k in rec_variants if k == "sweep" or k.startswith("sweep:")
                )
            )
        else:
            out.append(v)
    return out


def all_variant_keys(args_variants, results):
    """Across all records: the ordered union of concrete variant keys, expanding
    ``sweep`` into every ``sweep:*`` key seen. Used by the per-variant geomean so
    each launched sweep path gets its own summary row."""
    keys = []
    seen = set()
    for v in args_variants:
        if v == "sweep":
            for r in results:
                for k in sorted(r["variants"]):
                    if (k == "sweep" or k.startswith("sweep:")) and k not in seen:
                        seen.add(k)
                        keys.append(k)
        elif v not in seen:
            seen.add(v)
            keys.append(v)
    return keys
