<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# CK DSL example-repeatability registry

The `rocke/examples/` tree (~40k LOC) is where CK DSL proves its kernels
build, launch, and verify on real hardware. Historically each driver was a
one-off `python3 -m ...` you ran by hand and eyeballed. `run_all.py` turns a
**curated subset** into a single-command, deterministic, golden-asserted
repeatability gate.

## Usage

GPU steps need passwordless sudo with `-E` on this box (the login user is not
in the GPU device group), exactly like the L6 numeric lane:

```bash
# list what's registered
python3 -m rocke.examples.run_all --list

# capture goldens (first time, or after an *intended* change)
sudo -n -E PYTHONPATH=Python TMPDIR=/tmp <venv>/python \
    -m rocke.examples.run_all --bless

# assert against goldens (CI / repeat); exits nonzero on any FAIL/DRIFT/ERROR
sudo -n -E PYTHONPATH=Python TMPDIR=/tmp <venv>/python \
    -m rocke.examples.run_all --check

# filter by name/family substring
... -m rocke.examples.run_all --check --only gemm,attention
```

Goldens live next to the driver in `examples/_goldens/<name>.json` so they
travel with the tree. Build artifacts and the run dashboard go to
`$TMPDIR` (default `/tmp`).

## How it works

For each registered example `run_all.py`:

1. Checks skip-gates (`needs_gpu`, `arch`) and SKIPs-with-reason if unmet --
   the run stays green, the skip is reported.
2. Runs `python -m <module> <fixed argv>` in a subprocess with deterministic
   env (`PYTHONHASHSEED=0`, fixed `TMPDIR`).
3. Distills stdout to a **DIGEST**: only the numeric-verdict lines
   (`max_abs_diff`, `bad=`, `max_abs=`, `margin=`, `-> PASS|FAIL`) are kept;
   volatile fields (hipcc/total timings, `ms`/`tflops`/`gbps` perf numbers,
   byte counts, device-name banners, `/tmp` paths) are dropped or scrubbed.
   The digest is `sha256` over those lines.
4. `--bless` writes the digest+lines as the golden; `--check` asserts the
   live digest equals the golden and reports a DRIFT (with a line-diff) if not.

A nonzero example exit code is a real **FAIL**, except `rc=2` (the rocke
convention for "validate rejected this spec on this arch") which is a SKIP.

## Determinism contract for a registered example

1. Accepts a fixed argv; no wall-clock / hostname / pid leaks into the
   verdict lines (timings are scrubbed, but keep them out of verdict lines).
2. Inputs are seeded -- rocke examples use `np.random.default_rng(0xC0FFEE)`
   or `torch.manual_seed(...)`; the manifest runner seeds from the manifest.
3. Returns `0` on success / nonzero on failure and prints its numeric verdict
   (`max_abs_diff` / `bad=` / `PASS|FAIL`).

## Registered examples

| name | family | module | argv | notes |
|------|--------|--------|------|-------|
| `elementwise_add_hip` | elementwise | `common.elementwise_verify_hip` | `--arch gfx950 --n 4096` | HIP-path binary add, numpy-seeded, exact verdict |
| `universal_gemm_hip_512` | gemm | `common.universal_gemm_verify_hip` | `--arch gfx950 --m 512 --n 512 --k 512` | HIP path -> `run_manifest --verify`; PerfJSON timings scrubbed, `max_abs_diff`/`bad_count` kept |
| `fmha_fwd_hip_mha` | attention | `common.fmha_fwd_verify_hip` | `--arch gfx950 --seqlen-q 64 --seqlen-k 64 --head-size 64 --heads 4 --batch 2` | unified tiled FMHA fwd, seeded numpy dense-attn reference |
| `distribution_reduce_demo` | reduce | `common.distribution_reduce_demo` | `--M 128 --N 4096 --block-size 256 --vec 8` | torch row-reduce, exact-match verdict |
| `numeric_differential_lane` | differential | `tests/instances/differential/numeric.py` | `--arch gfx950` | the full L6 numeric lane (GEMM/elementwise/norm/reduce/attention) as one repeatable example; per-config GREEN/XFAIL table digested |

All five are pinned to `gfx950` (the box's MI355X). On a different arch they
SKIP rather than fail.

## Registering another example

Add one `Example(...)` row to `REGISTRY` in `run_all.py`:

```python
Example(
    name="my_example",                       # unique; golden file name
    module="rocke.examples.gfx950.my_driver",
    argv=("--arch", "gfx950", "--size", "128"),  # fixed, deterministic
    family="gemm",
    arch="gfx950",        # SKIP if live device arch differs; None = any
    needs_gpu=True,       # SKIP if no GPU visible
    timeout=300,
    # optional: extra regexes whose matching lines are added to the digest,
    # for drivers whose verdict isn't covered by the default patterns
    digest_keep=(r"^my-verdict:",),
),
```

Then `--bless` once to capture the golden and commit
`examples/_goldens/my_example.json` alongside it.

### Tips for making a not-yet-repeatable driver registrable

- If the driver only prints perf (no correctness verdict), add a `--verify`
  / numeric-diff print, or add a `digest_keep` regex for whatever stable
  line it does emit.
- If it seeds from wall-clock, switch to a fixed seed
  (`np.random.default_rng(0xC0FFEE)` is the house default).
- If it writes to a fixed `/tmp` path that collides across users, namespace
  it (`run_all.py` namespaces its own dashboard by `uid`).
- If it needs data files not in the tree, gate it with `needs_gpu`/a custom
  skip or stage the data deterministically.
