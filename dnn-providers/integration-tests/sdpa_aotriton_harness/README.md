# SDPA gpu_ref vs AOTriton numerical harness

A forward-only numerical comparison harness that validates this branch's fp32
SDPA **gpu_ref** kernel against **AOTriton** (reached via PyTorch's
`torch.nn.functional.scaled_dot_product_attention` on ROCm).

## What it does and why

- The **gpu_ref** kernel is treated as the **high-precision (HP) oracle**: it
  computes the whole attention in fp32 (QKᵀ, softmax, PV) regardless of the
  input dtype.
- **AOTriton** (the flash / mem-efficient backends behind PyTorch SDPA on ROCm)
  is the **low-precision (LP) result** under test.
- Inputs are generated **once** in Python, cast to the target dtype **once**,
  and the **exact same bit patterns** are fed to both sides (the C++ driver
  loads the same `.npy` files Python wrote), so neither side sees a divergent
  cast.

### Adaptive tolerance (mirrors AOTriton's own test methodology)

For each case, all math in float32:

```
err_aot    = max(abs(aotriton_o - gpuref_o))     # what we're judging
budget     = max(abs(gpuref_o   - math_lp_o))     # error from low precision alone
atol_floor = {bf16: 1e-2, fp16: 1e-3}
threshold  = max(atol_floor, fudge * budget)      # rtol = 0, fudge default 4.0
passed     = err_aot <= threshold
selfcheck  = max(abs(gpuref_o - math_hp_o))       # report-only oracle sanity
```

`math_lp_o` is PyTorch's **MATH** backend run on the native low-precision inputs
(measures the inherent quantization error); `math_hp_o` is the MATH backend with
inputs upcast to fp32 (an independent fp32 oracle used only as a sanity check on
the gpu_ref itself). AOTriton must agree with the gpu_ref to within a small
multiple of the quantization budget, with an absolute floor per dtype.

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

Configure the integration tests with the harness enabled and build the driver
target (it lands in the build `bin` directory):

```bash
cmake -S dnn-providers/integration-tests -B build \
      -DBUILD_SDPA_AOTRITON_HARNESS=ON   # plus your usual hipDNN cmake args
ninja -C build sdpa_aotriton_ref_driver
```

## Running

```bash
python harness.py \
    --driver /path/to/build/bin/sdpa_aotriton_ref_driver \
    --tier quick
```

Tiers:

- `quick` (default) — bf16 only, small shapes, all four modes (~12-20 cases).
- `full` — adds fp16, a head-dim sweep (16…256), longer / NPOT seqlens,
  GQA/MQA pairs, both window alignments, and an explicit-scale case.
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
| `err_aot`   | max-abs difference between AOTriton and the gpu_ref oracle. |
| `budget`    | max-abs difference between the gpu_ref and torch MATH-LP (quantization error). |
| `threshold` | `max(atol_floor, fudge * budget)`. |
| `ratio`     | `err_aot / threshold` (≤ 1.0 passes). |
| `selfcheck` | gpu_ref vs the independent fp32 MATH oracle; a `!` flags `> 1e-2` (warn only). |
| `RESULT`    | `PASS` / `FAIL` / `SKIP` / `ERROR`. |

- **SKIP** means AOTriton cannot service that shape/config (both flash and
  mem-efficient backends raised). It is not a failure.
- **ERROR** means the C++ driver or the torch step failed for that case.
- The process exits non-zero if any case is `FAIL` or `ERROR` (skips are OK).

## Scope and limitations

- **Forward only**; dtypes **bf16** and **fp16**.
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
