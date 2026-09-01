# Corpus Generation Gap Analysis: CK/rocKE vs. hipDNN UHD

**Date:** 2026-08-20  
**Context:** Defining benchmark corpus generation for hipDNN UHD training tooling  

---

## What CK/rocKE Have

### 1. Problem Space Generation

✅ **Stratified discrete sampling** (`generate_wide_coverage.py`):
- 13 hardcoded regimes (M=1, tiny/small/medium/large M, square, skinny/tall, deep/shallow K, primes, LLM-specific, non-power-of-2)
- Combinatorial sampling within bounded lists per regime
- Generates ~thousands of shapes (not millions) via `set()` deduplication

**Example (GEMM)**:
```python
# M=1 (single token) across various N, K
for n in [512, 1024, 1536, 2048, 3072, 4096, 4608, 7168, 8192, 11008, 14336, 28672]:
    for k in [256, 512, 1024, 1536, 2048, 2304, 4096, 7168, 8192]:
        shapes.add((1, n, k))

# Deep K (K >> M, N) -- compute-bound
for m in [16, 64, 256]:
    for n in [16, 64, 256]:
        for k in [4096, 8192, 16384, 32768]:
            shapes.add((m, n, k))

# Prime dimensions -- worst-case for tiling
primes = [17, 31, 37, 127, 251, 509, 1021, 2039, 4093]
for p in primes:
    shapes.add((p, p, p))
```

### 2. Kernel Space Enumeration

✅ **Explicit catalog enumeration**:
- Reads `gemm_universal_kernel_list.txt` from tile engine build
- Parses kernel name to extract config: `tile_m`, `tile_n`, `tile_k`, `warp_m`, `warp_n`, `warp_k`, pipeline, scheduler, epilogue, padding flags, persistent
- **Full enumeration**: Every kernel in the catalog is benchmarked

### 3. Benchmark Execution

✅ **Direct kernel timing**:
- One standalone executable per kernel: `benchmark_gemm_universal_fp8_rcr_<kernel_id>`
- Direct invocation (not through dispatcher/autotune)
- Parallel build via `ProcessPoolExecutor`, sequential timing
- Output: `{kernel_id, M, N, K, avg_time_ms, tflops, is_valid}`

### 4. Feature Engineering

✅ **Hardcoded feature derivation** (`feature_engine.py`):
- ~60 features for GEMM (device + problem + kernel + derived)
- Tile quantization: `ceil_div(M, tile_m)`, `tile_eff_m = (M % tile_m) / tile_m`
- Physics: arithmetic intensity, aspect ratios, utilization
- Categorical encoding: pipeline/scheduler/epilogue → integers

---

## What's Missing for hipDNN UHD

### 1. **Parameterized Corpus Generation** (not hardcoded)

❌ CK's regimes are **hardcoded in Python**, not driven by configuration:
- No `--tier smoke|standard|exhaustive` to control sampling density
- No per-operation customization (GEMM, SDPA, Conv all need different regimes)
- No way to override defaults without editing Python source

**What hipDNN needs**:
```bash
python -m uhd_gen generate-corpus \
    --operation sdpa_forward \
    --tier standard \
    --output corpus.json \
    [--config custom_ranges.yaml]  # Override defaults
```

**Requirements**:
- Per-operation corpus generators (`uhd_gen/corpus/gemm.py`, `sdpa.py`, `conv.py`, etc.)
- Tier-driven sampling density (smoke: 10-20, standard: 100-500, exhaustive: 1000+)
- Configuration file support for custom ranges

---

### 2. **Applicability Filtering** (explicit vs. implicit)

✅ **CK approach**: Benchmark **all** (problem, kernel) pairs, filter **implicitly**:
- Runs every kernel executable against every problem shape
- Kernel applicability logic is **inside the kernel** (returns error/empty if not applicable)
- Failed/timeout/invalid results are silently dropped via exception handling
- Only successful timing results are written to output

**What hipDNN UHD can do differently**:

**Option A: Follow CK** (filter implicitly during benchmarking):
```python
# Attempt to benchmark all pairs
for problem in corpus:
    for variant in all_variants:
        result = try_benchmark(problem, variant)
        if result.is_valid:
            write_to_csv(result)
        # else: silently skip (kernel not applicable to this problem)
```

**Option B: Filter explicitly** (query applicability before benchmarking):
```python
# Pre-filter via applicability API
for problem in corpus:
    graph = build_graph(problem)
    applicable_variants = [v for v in all_variants if engine.is_applicable(graph, v)]
    for variant in applicable_variants:
        result = benchmark(problem, variant)
        write_to_csv(result)
```

**Trade-offs**:
- **CK approach**: Simple (try everything, keep successes), but wastes time on inapplicable pairs
- **Explicit filtering**: More efficient (skip inapplicable pairs up front), but requires applicability API to accept variant configs, not just graphs

**Recommendation**: Start with CK approach (implicit filtering via benchmark success/failure), add explicit filtering later as optimization

---

### 3. **Config Validation Predicates** (kernel space filtering)

❌ CK reads **every** kernel from the catalog, no validation:
- Assumes all kernels in the catalog are valid
- For sparse catalogs with constraints (e.g., "tile_m × tile_k must fit LDS"), no way to express this

**What hipDNN needs** (two modes):

**Mode 1: Explicit variant enumeration** (KMD declares full list):
```json
{
  "variants": [
    {"tile_m": 128, "tile_n": 128, "tile_k": 8, "use_lds": true},
    {"tile_m": 256, "tile_n": 128, "tile_k": 8, "use_lds": true}
  ]
}
```

**Mode 2: Range + validation predicate** (KMD declares ranges + constraints):
```json
{
  "fields": [
    {"name": "tile_m", "type": "int64", "values": [64, 128, 256]},
    {"name": "tile_n", "type": "int64", "values": [64, 128, 256]},
    {"name": "tile_k", "type": "int64", "values": [8, 16, 32]},
    {"name": "use_lds", "type": "bool"}
  ],
  "has_config_validator": true
}
```

```python
# Tooling invokes provider predicate
cartesian = generate_cartesian_product(kmd.get_field_ranges())
valid_variants = [cfg for cfg in cartesian if engine.is_valid_config(cfg)]
```

**Requirements**:
- KMD schema extension to support explicit variants OR validation predicates
- Tooling to invoke validation before benchmarking
- Fallback to Cartesian sweep when no validator provided

---

### 4. **Autotune Integration** (not standalone executables)

❌ CK uses **standalone benchmark executables**, one per kernel:
- Each kernel built separately: `benchmark_gemm_universal_fp8_rcr_001`, `_002`, etc.
- Slow build time (even with parallel builds)
- Inflexible (hardcoded warmup/repeat, no autotune convergence)

**What hipDNN needs**:
```python
# Use hipDNN autotune API (RFC 0013)
for problem in applicable_corpus:
    graph = build_graph(problem)
    handle.add_engine_variants(engine_id, valid_variants)
    results = handle.autotune(graph, mode=EXHAUSTIVE, strategy=RUN_UNTIL_STABLE)
    # → [{engineId, knobSettings, minTimeMs, avgTimeMs, stddevMs, ...}]
```

**Requirements**:
- Integration with `hipdnnBackendAutotune()` API
- Graph construction per problem shape
- Variant enrollment (explicit list or sweep)
- Convergent timing (not fixed warmup/repeat)

---

### 5. **Roofline-Aware Stratification** (not just use-case regimes)

❌ CK regimes are **use-case based** (M=1 inference, training batches, LLM shapes):
- No explicit roofline stratification (compute-bound vs. memory-bound)
- Deep K / Shallow K regimes are proxies, but arithmetic intensity not explicit

**What hipDNN needs**:
```python
# Stratify by roofline position
def stratify_by_roofline(shapes, device_peak_flops, device_bandwidth):
    for shape in shapes:
        flops = compute_flops(shape)
        bytes = compute_bytes(shape)
        ai = flops / bytes  # arithmetic intensity
        
        if ai > (device_peak_flops / device_bandwidth):
            regime = "compute_bound"
        else:
            regime = "memory_bound"
        
        shape.metadata["roofline_regime"] = regime
```

**Requirements**:
- Roofline classification per shape (compute vs. memory-bound)
- Ensure corpus samples both regimes
- Validation: report % coverage of each regime

---

### 6. **Tile Alignment Stratification** (explicit, not implicit)

❌ CK primes are a **proxy** for misalignment, but not systematic:
- Prime dimensions (17, 31, 127, ...) force remainder tiles
- But no explicit "aligned vs. misaligned" stratification across typical tiles (64, 128, 256)

**What hipDNN needs**:
```python
# Explicit alignment stratification
common_tiles = [64, 128, 256]
for tile in common_tiles:
    # Well-aligned case
    shapes.add((tile * 16, tile * 16, tile * 8))  # M % tile == 0
    
    # Partial remainder case
    shapes.add((tile * 16 + tile // 2, tile * 16, tile * 8))  # M % tile != 0
    
    # Full remainder case
    shapes.add((tile * 16 + 7, tile * 16 + 13, tile * 8 + 3))  # All dims misaligned
```

**Requirements**:
- Generate shapes for aligned / partial-remainder / full-remainder cases
- Report tile alignment coverage per kernel variant

---

### 7. **Corpus Provenance & Reproducibility** (not just code)

❌ CK corpus is **in the Python code**, not a separate artifact:
- To reproduce a training run, you need the exact version of `generate_wide_coverage.py`
- No metadata about which corpus a model was trained on

**What hipDNN needs**:
```json
{
  "operation": "sdpa_forward",
  "tier": "standard",
  "generated_at": "2026-08-20T15:30:00Z",
  "generator_version": "uhd_gen 1.2.0",
  "count": 320,
  "regimes": {
    "prefill": 150,
    "decode": 120,
    "edge_cases": 50
  },
  "configs": [
    {"batch": 32, "heads": 32, "seqlen_q": 2048, "seqlen_k": 2048, "head_dim": 128, "is_causal": 1},
    ...
  ]
}
```

**Requirements**:
- Corpus saved as JSON with provenance metadata
- Training manifest references corpus file (or embeds it)
- Corpus versioning for reproducibility

---

### 8. **Multi-Operation Support** (not just GEMM)

❌ CK tooling is **GEMM-specific**:
- `generate_wide_coverage.py` hardcodes `(M, N, K)` dimensions
- No abstraction for other ops (SDPA, Conv, Normalization)

**What hipDNN needs**:
```python
# Operation-specific generators
class SdpaCorpusGenerator:
    def generate(self, tier: str) -> List[Dict]:
        # SDPA-specific regimes: prefill/decode, LLM configs
        ...

class ConvCorpusGenerator:
    def generate(self, tier: str) -> List[Dict]:
        # Conv-specific regimes: ResNet, VGG, EfficientNet, ...
        ...
```

**Requirements**:
- Per-operation corpus generator classes
- Shared base class with stratification helpers
- Registry to dispatch `--operation <name>` to correct generator

---

## Summary: What's Missing

| Capability | CK/rocKE | hipDNN UHD Needs |
|------------|----------|------------------|
| **Problem space sampling** | ✅ Hardcoded regimes | ❌ Parameterized (tier, config) |
| **Applicability filtering** | ❌ None | ❌ Provider predicate integration |
| **Config validation** | ❌ None | ❌ KMD validator support |
| **Benchmark substrate** | ✅ Standalone exes | ❌ Autotune API integration |
| **Roofline stratification** | ⚠️ Implicit (deep K) | ❌ Explicit AI classification |
| **Tile alignment** | ⚠️ Primes only | ❌ Systematic aligned/misaligned |
| **Corpus provenance** | ❌ Code-only | ❌ JSON with metadata |
| **Multi-operation** | ❌ GEMM-only | ❌ Per-op generators |

---

## Proposed Implementation Priority

1. **HIGH**: Parameterized corpus generation framework (tier-driven, per-operation)
2. **HIGH**: Applicability filtering integration
3. **HIGH**: Autotune API integration (replace standalone executables)
4. **MEDIUM**: Config validation predicates (KMD schema extension)
5. **MEDIUM**: Roofline-aware stratification
6. **MEDIUM**: Tile alignment stratification
7. **LOW**: Corpus provenance (JSON metadata)
8. **ONGOING**: Multi-operation support (add ops incrementally)

---

## Next Steps

1. Define per-operation corpus generator interface (`CorpusGenerator` base class)
2. Implement GEMM corpus generator (adopt CK's 13 regimes as defaults, add tier control)
3. Implement SDPA corpus generator (prefill/decode regimes)
4. Define applicability filtering workflow (graph construction + is_applicable call)
5. Define autotune integration workflow (variant enrollment + exhaustive timing)
