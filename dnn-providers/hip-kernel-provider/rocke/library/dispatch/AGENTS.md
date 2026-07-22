# Attention dispatcher — agent guide

This folder owns the **path-level dispatch** for unified attention: which kernel
family (2D tiled prefill vs 3D split-KV decode) and which specialized candidate
handles a given `AttentionRequest`.

## What the dispatcher decides (and what it does NOT)

**Decides:** kernel path (`"2d"` or `"3d"`), candidate name, algorithm tag,
and spec identity `(path, head_size, block_size)`.

**Does NOT decide:** CTA geometry (`num_warps`, `tile_size`), `num_segments`,
`waves_per_eu`, or any other performance knob. Those live in
`builders/common/attention_spec_builder.py` and
`kernels/common/attention_unified.py`. The parity identity with C++ is
`(path, head_size, block_size)` only — C++ reads `num_segments` as a parameter
passed from Python, it does not recompute it.

## Candidate registry — priority table

| priority | candidate | scope |
|---|---|---|
| 5 | `attention_gfx942_dense_pipe` | gfx942 fp16 2D prefill flash |
| 5 | `attention_d256_decode` | gfx942/gfx950 bf16 D256 3D decode |
| 10 | `attention_unified_2d` | generic 2D prefill fallback |
| 10 | `attention_unified_3d` | generic 3D decode fallback |

Lower priority number = higher precedence. Generic candidates (10) remain the
fallback for everything a specialized candidate (5) does not claim.

## How to add a new specialized candidate

Follow the `_make_d256_decode_candidate()` pattern in `attention.py`:

1. **Add a cohort predicate** in `kernels/common/attention_unified.py` —
   a pure function of `UnifiedAttentionProblem` that returns `True` for the
   target shape family. This is the single source of truth for membership;
   import it lazily inside the factory to keep `dispatch/` arch-neutral.

2. **Add a factory function** `_make_<name>_candidate()` in `attention.py`.
   The `support()` closure must check in order: request errors → arch/dtype
   gates → `_selector_matches` → `supports_native_unified_attention` →
   cohort predicate → path check (`select_path() == "2d"` or `"3d"`).

3. **Register** with `ATTENTION_REGISTRY.register(_make_<name>_candidate())`.

4. **Add CPU-only dispatch tests** in
   `tests/dispatch/attention/test_<name>_wiring.py`. Cover: registration,
   spec_id, algorithm, priority ordering, support gates (wrong arch/dtype/
   cohort/path), and routing for each target arch. Use `_PinnedArch` context
   manager to avoid GPU dependency.

5. **No C++ changes needed.** The dispatcher is Python-only.

## How to tune `num_segments` for a new cohort

See the worked example:
`benchmarks/gfx942/attention/decode/TUNING_D256.md`

## Testing (CPU-only, no GPU)

```bash
PYTHONPATH=library:platform/python python -m unittest discover \
    -s library/tests/dispatch/attention -p "test_*.py" -v
```

All dispatch tests complete in < 1 s.
