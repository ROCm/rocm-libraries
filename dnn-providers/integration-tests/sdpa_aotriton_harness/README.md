# SDPA gpu_ref vs AOTriton numerical harness

A forward-only numerical comparison harness that validates this branch's fp32
SDPA **gpu_ref** kernel against **AOTriton** (reached via PyTorch's
`torch.nn.functional.scaled_dot_product_attention` on ROCm).

For **fp8** inputs AOTriton is unavailable (torch SDPA rejects fp8 on every
backend), so fp8 cases are validated against torch's **fp32-MATH** reference
instead — see [fp8 datatypes](#fp8-datatypes).

## What it does and why

- **AOTriton** (the flash / mem-efficient backends behind PyTorch SDPA on ROCm)
  is the **oracle / reference of record**.
- The **gpu_ref** kernel is the **candidate under test**: it computes the whole
  attention in fp32 (QKᵀ, softmax, PV) regardless of the input dtype, and is
  validated against the AOTriton oracle.
- Inputs are generated **once** in Python, cast to the target dtype **once**,
  and the **exact same bit patterns** are fed to both sides (the C++ driver
  loads the same `.npy` files Python wrote), so neither side sees a divergent
  cast.

### Adaptive tolerance (independent precision-gap budget)

For each case, all math in float32:

```
err            = max(abs(gpuref_o - aotriton_o))   # candidate vs oracle (judged)
budget         = max(abs(math_hp_o - math_lp_o))    # independent fp32-vs-LP gap
atol_floor     = {bf16: 1e-2, fp16: 1e-3, fp8_*: 1e-3}
threshold      = max(atol_floor, fudge * budget)    # rtol = 0, fudge default 4.0
passed         = err <= threshold
# diagnostics (report-only):
gpuref_vs_fp32 = max(abs(gpuref_o  - math_hp_o))    # is the candidate a sound fp32 impl?
aotriton_vs_lp = max(abs(aotriton_o - math_lp_o))   # does the oracle behave like a standard LP impl?
```

`math_hp_o` is PyTorch's **MATH** backend with inputs upcast to fp32; `math_lp_o`
is the MATH backend on the native low-precision inputs. Their gap is the
**budget** — the inherent bf16/fp16 attention error, independent of both the
candidate and the oracle (using `|gpuref - math_lp|` would be circular now that
gpu_ref is the thing under test). The candidate must agree with the oracle to
within a small multiple of that budget, with an absolute floor per dtype.

The two diagnostics are report-only: `gpuref_vs_fp32` checks the candidate is a
sound fp32 implementation, and `aotriton_vs_lp` checks the oracle behaves like a
standard low-precision implementation.

## Prerequisites (gfx942 / MI300)

- A **ROCm PyTorch** with the AOTriton flash backend. `torch >= 2.8`
  (AOTriton 0.10b+) is recommended.

  ```bash
  pip install --pre torch --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/
  ```

  Alternatively, `projects/hipdnn/tools/dnn-benchmarking/setup.sh` sets up a
  ROCm-torch venv you can reuse.

- **numpy** (the only hard dependency for input generation and comparison).

This harness cannot run on non-gfx94X hardware: AOTriton only services those
GPUs through PyTorch SDPA.

## Building the C++ driver

The driver target (`sdpa_aotriton_ref_driver`) is **off by default**; enable it
with `-DBUILD_SDPA_AOTRITON_HARNESS=ON` at CMake configure time. It links the
in-tree `hipdnn_gpu_ref` library plus HIP/HIPRTC, so those dependencies must be
available.

### Option A — hipDNN superbuild (recommended)

A superbuild from the rocm-libraries root builds hipDNN and the integration
tests together via `add_subdirectory`, so every dependency (`hipdnn_gpu_ref`,
`hipdnn_test_sdk`, HIP, HIPRTC) is made available automatically — no install or
`find_package` step. Use any CMake preset that includes the
`hipdnn-integration-tests` component (the smallest is the preset of the same
name) and pass the harness option through on the configure line:

```bash
# From the rocm-libraries repo root:
cmake --preset hipdnn-integration-tests -DBUILD_SDPA_AOTRITON_HARNESS=ON
cmake --build build --target sdpa_aotriton_ref_driver
# -> build/bin/sdpa_aotriton_ref_driver
```

`BUILD_SDPA_AOTRITON_HARNESS` is a global CMake cache variable, so setting it on
the superbuild configure reaches the integration-tests subproject where the
option is defined. Any larger preset that also pulls in
`hipdnn-integration-tests` works too (e.g. `hip-kernel-provider`,
`hipdnn-providers`, `hipdnn-dev-all`) if you want the providers built as well.
See [`projects/hipdnn/docs/Superbuild.md`](../../../projects/hipdnn/docs/Superbuild.md)
for the full preset list and superbuild details.

### Option B — standalone integration-tests build

Build just the integration-tests subproject (requires the hipDNN SDK and HIP
dependencies to be installed / findable via `find_package`):

```bash
cmake -S dnn-providers/integration-tests -B build \
      -DBUILD_SDPA_AOTRITON_HARNESS=ON   # plus your usual hipDNN cmake args
ninja -C build sdpa_aotriton_ref_driver
# -> build/bin/sdpa_aotriton_ref_driver
```

## Running

```bash
python harness.py \
    --driver /path/to/build/bin/sdpa_aotriton_ref_driver \
    --tier quick
```

Tiers:

- `quick` (default) — bf16 small shapes across all four modes, plus a small fp8
  presence (plain + causal for each of the four fp8 formats) (~28 cases).
- `medium` — adds fp16, a head-dim sweep (16…256), longer / NPOT seqlens,
  GQA/MQA pairs, both window alignments, an explicit-scale case, and a per-format
  fp8 sweep (plain + causal at two head dims, window, mask).
- `large` — opt-in, **slow** (~10 min, 72 cases): larger tensors stressing
  AOTriton at scale — square seqlens up to 16384, head dims up to 256, bigger
  batch/head counts, GQA/MQA, long NPOT pairs, and large window/mask cases, in
  both bf16 and fp16. Every shape is AOTriton-serviceable (no SKIPs); batch and
  head counts are bounded where seqlen is long so the torch MATH oracle fits in
  memory.
- `irregular` — opt-in, **slow**: a small prime/NPOT sample. The gpu_ref kernel
  recomputes QKᵀ per output element, so large/odd shapes are expensive — keep
  this tier small.

Other flags: `--out DIR` (default: a timestamped dir under `./runs/`),
`--fudge FLOAT` (budget multiplier, default 4.0), `--seed-base INT`.

The pipeline can also be run stage-by-stage:

```bash
python gen_inputs.py --tier quick --out runs/manual
/path/to/sdpa_aotriton_ref_driver --q runs/manual/<case>/q.npy ...   # per case
python run_torch.py --run-dir runs/manual
python compare.py   --run-dir runs/manual --fudge 4.0
```

## Reading the results

`harness.py` prints a table and writes `results.json` into the run dir.

| Column      | Meaning |
|-------------|---------|
| `backend`   | which AOTriton backend serviced the case: `flash` or `efficient` (mem-efficient). |
| `err`       | max-abs difference between the gpu_ref candidate and the AOTriton oracle. |
| `budget`    | max-abs difference between torch MATH-HP and MATH-LP (independent fp32-vs-low-precision attention gap). |
| `threshold` | `max(atol_floor, fudge * budget)`. |
| `ratio`     | `err / threshold` (≤ 1.0 passes). |
| `g_vs_fp32` | gpu_ref candidate vs the independent fp32 MATH reference; a `!` flags `> 1e-2` (warn only). |
| `a_vs_lp`   | AOTriton oracle vs the low-precision MATH reference; a `!` flags `> 1e-2` (warn only). |
| `RESULT`    | `PASS` / `FAIL` / `SKIP` / `ERROR`. |

- **SKIP** means AOTriton cannot service that shape/config (both flash and
  mem-efficient backends raised). It is not a failure.
- **ERROR** means the C++ driver or the torch step failed for that case.
- The process exits non-zero if any case is `FAIL` or `ERROR` (skips are OK).

## fp8 datatypes

The harness also covers four fp8 input formats: `fp8_e4m3`, `fp8_e5m2` (OCP) and
`fp8_e4m3_fnuz`, `fp8_e5m2_fnuz` (FNUZ / MI300-native). They are stored on disk as
raw 8-bit patterns (`|u1`).

Because PyTorch's SDPA front-end hard-rejects fp8 on **every** backend
(`flash` / `efficient` / `math`) and cannot even run native-fp8 MATH, AOTriton
cannot be the oracle for fp8. Instead, fp8 cases use torch's **fp32-MATH** backend
as the oracle of record (fp8 inputs upcast losslessly to fp32), with the
low-precision budget leg computed in **fp16** (the smallest precision torch MATH
will run). In the results table the `backend` column shows `math-fp32` for these
cases.

Since the gpu_ref candidate and the oracle consume the *identical* fp8 bit
patterns, `err` measures the candidate's fp32 attention compute against torch's
fp32 compute (the fp8 quantization is shared, not part of `err`); the tolerance
floor is therefore `1e-3`, the same as fp16. The gpu_ref kernel decodes fp8 to
float in software inside the HIPRTC kernel, so this path is not gated on native
fp8 hardware support.

## Scope and limitations

- **Forward only**; dtypes **bf16**, **fp16**, and **fp8** (`e4m3` / `e5m2` and
  their `fnuz` variants; fp8 validated against the fp32-MATH oracle, not AOTriton).
- Feature scope is the intersection of what the gpu_ref and AOTriton-via-PyTorch
  both support: plain MHA, causal (top-left, **square only**), additive float
  mask (full rank-4, no broadcasting), sliding window (gpu_ref native bounds vs
  an equivalent additive `-inf` mask for torch), GQA/MQA (the same `Hkv` for K
  and V, via `enable_gqa`), and custom scale.
- **No LSE comparison against AOTriton**: torch SDPA does not return LSE. The
  driver can still emit LSE for the gpu_ref, but the harness does not request or
  compare it.
- **No dropout.**
- Masked and windowed cases generally route through the **mem-efficient**
  backend (still AOTriton), since flash does not accept an arbitrary additive
  mask.
- The reference kernel recomputes QKᵀ per output element, so large seqlens are
  slow — hence the tiered case sets.

## File layout

```
sdpa_aotriton_harness/
├── README.md            (this file)
└── python/
    ├── sdpa_cases.py    Case dataclass + tier definitions
    ├── manifest.py      manifest schema, file paths, window-mask synthesis (numpy only)
    ├── gen_inputs.py    generate inputs + manifests (torch for the dtype cast)
    ├── run_torch.py     AOTriton + math (HP/LP) references (torch, runs on GPU)
    ├── compare.py       adaptive-tolerance comparison (numpy only)
    └── harness.py       end-to-end orchestrator + summary table
```

### Per-case manifest schema

Each case directory holds a `manifest.json`:

```json
{
  "name": "bf16_b1_hq4_hkv4_sq128_skv128_d64_causal",
  "dtype": "bf16",
  "B": 1, "Hq": 4, "Hkv": 4, "Sq": 128, "Skv": 128, "D": 64,
  "scale": null,
  "mode": "causal",
  "left": -1, "right": 0, "top_left": true,
  "causal": true,
  "has_mask": false,
  "seed": 123456789,
  "files": {
    "q": ".../q.npy", "k": ".../k.npy", "v": ".../v.npy",
    "mask": null,
    "gpuref_o": ".../gpuref_o.npy", "gpuref_lse": ".../gpuref_lse.npy",
    "aotriton_o": ".../aotriton_o.npy",
    "math_hp_o": ".../math_hp_o.npy", "math_lp_o": ".../math_lp_o.npy"
  },
  "status": { "state": "ok", "backend_used": "flash" }
}
```

`status.state` is one of `pending` / `ok` / `skipped` / `error`; the C++ driver
sets nothing, `run_torch.py` sets `ok`/`skipped`, and the driver/torch steps set
`error` on failure.
