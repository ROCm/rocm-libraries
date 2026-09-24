# GEKO `--tune` CLI contract

Source revision: `60c362c6ec96729a8cf307e99da199424ef133e1`.

## Bottom line

This command is incomplete:

```bash
geko --tune --arch gfx942 --backend ductile --devices=2 --n_slots 1 --up_thr 1.0
```

`--tune` requires exactly one workload source: `--inline`, `--list`, or
`--workload-log` ([`geko/cli.py:65-94`](geko/cli.py#L65-L94)). It also needs a
built hipBLASLt checkout. The in-tree `./bin/geko` launcher can auto-detect that
checkout; an installed `geko` command normally needs `--hipblaslt PATH` or
`GEKO_HIPBLASLT_PATH` ([`geko/cli.py:322-332`](geko/cli.py#L322-L332),
[`geko/paths.py:83-120`](geko/paths.py#L83-L120)).

`--devices=2` selects device ID 2; it does **not** request two devices. Multiple
devices are comma-separated, for example `--devices=0,1`
([`geko/cli.py:121-130`](geko/cli.py#L121-L130),
[`geko/utils.py:144-169`](geko/utils.py#L144-L169)). `--n_slots 1` then permits
one optimization job on each selected device.

## Three supported ways to specify GEMMs

### 1. One GEMM inline

```bash
./bin/geko --tune \
  --arch gfx942 --backend ductile --devices=2 --n_slots 1 \
  --inline 1024 4096 1 8192 B B S N T \
  --workdir geko_bf16_1024x4096x8192
```

The nine inline values are, in order:

```text
M N batch_count K DataType DestDataType ComputeDataType transA transB
```

The four dimensions must be integers; the resulting size is
`[M, N, batch_count, K]`. Transposes are `N`, `T`, or `C`, with `C` allowed only
for complex types ([`geko/cli.py:272-284`](geko/cli.py#L272-L284),
[`geko/schemas.py:262-310`](geko/schemas.py#L262-L310)). Sizes must ultimately be
four positive integers ([`geko/schemas.py:328-352`](geko/schemas.py#L328-L352)).

Inline dtype values use Tensile codes, which GEKO maps to hipBLASLt logical
types. The current canonical tokens are `B`, `H`, `S`, `D`, `F8`, `F8N`, `B8`,
`X`, `F4`, `C`, and `Z`; a mixed A/B type may be two one-character tokens such
as `BH` ([`geko/constants.py:16-29`](geko/constants.py#L16-L29),
[`geko/schemas.py:198-260`](geko/schemas.py#L198-L260)). The dtype triple must
round-trip consistently. Despite comments in the sample template, the current
CLI mapper rejects `I8`, `X1`, `F8B8`, and `B8F8`; those strings are absent from
the reverse mapping or exceed its one/two-character mixed-type rule.

### 2. Multiple sizes of one GEMM type via `--list`

`gfx942-bf16-nt.yaml`:

```yaml
TRANSA: N
TRANSB: T
DataType: B
DestDataType: B
ComputeDataType: S
ARCH: gfx942
SIZE_OPTION: 0
Sizes:
  - [1024, 4096, 1, 8192]
  - [2048, 4096, 1, 8192]
```

```bash
./bin/geko --tune \
  --arch gfx942 --backend ductile --devices=2 --n_slots 1 \
  --list gfx942-bf16-nt.yaml \
  --workdir geko_bf16_nt
```

For explicit lists, the required logical fields are `TRANSA`, `TRANSB`,
`DataType`, `DestDataType`, `ComputeDataType`, and `ARCH`; default
`SIZE_OPTION: 0` additionally requires a non-empty `Sizes` list
([`geko/config_generator/constants.py:323-340`](geko/config_generator/constants.py#L323-L340),
[`geko/config_generator/sizes.py:18-67`](geko/config_generator/sizes.py#L18-L67)).
`SIZE_OPTION: 1` instead generates an internal M/N/K grid and accepts optional
`GRID_DENSITY` ([`geko/config_generator/sizes.py:47-56`](geko/config_generator/sizes.py#L47-L56)).

Two current implementation details override the README wording:

- `geko --tune` requires `--arch` on the command line even if the list YAML has
  `ARCH` ([`geko/cli.py:286-287`](geko/cli.py#L286-L287)). The command-line value
  overwrites the YAML value ([`geko/config_generator/load_input_config.py:337-349`](geko/config_generator/load_input_config.py#L337-L349)).
- A normal `--list` YAML produces exactly one `GemmConfig`; the loader overwrites
  `GemmProblems` with a one-element list. Thus this form supports multiple sizes
  of one type, not multiple dtype/layout types. Use a workload log for multiple
  types ([`geko/config_generator/load_input_config.py:152-174`](geko/config_generator/load_input_config.py#L152-L174),
  [`geko/config_generator/load_input_config.py:354-363`](geko/config_generator/load_input_config.py#L354-L363)).

### 3. One or more types via a hipBLASLt workload log

```bash
HIPBLASLT_LOG_MASK=64 \
HIPBLASLT_LOG_FILE=hipblaslt-log-mask64.yaml \
python my_application.py

./bin/geko --tune \
  --arch gfx942 --backend ductile --devices=2 --n_slots 1 \
  --workload-log hipblaslt-log-mask64.yaml \
  --workdir geko_application_workload
```

The parser accepts YAML, or CSV when the filename ends in `.csv`
([`geko/bench/log.py:82-115`](geko/bench/log.py#L82-L115)). Every row must contain
these non-null fields:

```text
transA, transB, batch_count, M, N, K,
a_type, b_type, c_type, d_type, compute_type
```

That requirement is enforced by [`geko/constants.py:53-65`](geko/constants.py#L53-L65)
and [`geko/bench/log.py:117-126`](geko/bench/log.py#L117-L126). Lowercase `m`,
`n`, and `k` are normalized. `function`, `call_count`, and `scale_type` are
filled when absent; other recognized hipBLASLt fields are optional
([`geko/bench/log.py:117-145`](geko/bench/log.py#L117-L145)). A minimal YAML row is:

```yaml
- transA: N
  transB: T
  batch_count: 1
  M: 1024
  N: 4096
  K: 8192
  a_type: bf16_r
  b_type: bf16_r
  c_type: bf16_r
  d_type: bf16_r
  compute_type: c_f32_r
```

Unlike `--list`, a workload log can contain multiple dtype/layout groups; GEKO
groups them by transpose and logical A/B/C/compute types, while retaining each
`[M, N, batch_count, K]` size ([`geko/config_generator/load_input_config.py:116-143`](geko/config_generator/load_input_config.py#L116-L143)).

## Layout, bias, and activation semantics

For tuning identity, “layout” means only `transA` and `transB`. Inline/list
inputs have no leading-dimension or stride fields. Workload logs may contain
`lda`, `ldb`, `ldc`, `ldd`, and stride fields, but the configure summary reduces
each row to `GEMM_FIELDS`, and generated tuning sizes contain only
`[M, N, batch, K]` ([`geko/constants.py:39-51`](geko/constants.py#L39-L51),
[`geko/bench/log.py:441-497`](geko/bench/log.py#L441-L497)). Therefore GEKO does
not create separate tuned identities for different leading dimensions or
strides.

There is no CLI field for a particular bias or activation. A captured workload
may contain `bias_vector`, `bias_source`, `bias_type`, and `activation_type`, and
the parser recognizes them ([`geko/constants.py:75-122`](geko/constants.py#L75-L122)),
but configure collapses to the GEMM identity above, which omits all epilogue
fields. `--inline` and `--list` are first converted to synthetic rows containing
only the required GEMM fields ([`geko/cli.py:342-365`](geko/cli.py#L342-L365),
[`geko/schemas.py:312-369`](geko/schemas.py#L312-L369)).

Instead, the one-command tune path applies `EPILOGUES: True` by default and
generates broadly capable kernels with `ActivationType: hipblaslt_all`, bias,
and scale-alpha-vector support; f64 and complex GEMMs disable epilogues
([`geko/config_generator/constants.py:330-342`](geko/config_generator/constants.py#L330-L342),
[`geko/config_generator/config_sections_generator.py:47-51`](geko/config_generator/config_sections_generator.py#L47-L51),
[`geko/config_generator/config_sections_generator.py:71-103`](geko/config_generator/config_sections_generator.py#L71-L103)).
Bias type is derived from the GEMM dtype, not selected by a CLI argument
([`geko/config_generator/config_sections_generator.py:153-169`](geko/config_generator/config_sections_generator.py#L153-L169)).

Consequently, putting `EPILOGUES`, a specific bias, or a specific activation in
a `--list` file does not customize the one-command tune: the list is flattened
to a synthetic workload and `run_configure` builds a fresh tuning config from
only `ARCH`, backend, search space, and `GemmProblems`
([`geko/cli.py:49-56`](geko/cli.py#L49-L56),
[`geko/optim/optim.py:156-190`](geko/optim/optim.py#L156-L190)).

## Prerequisites

- Python 3.10+ and the GEKO runtime dependencies (`joblib`, PyYAML, pandas,
  NumPy, tqdm, pytest) ([`pyproject.toml:8-20`](pyproject.toml#L8-L20),
  [`requirements.txt:1-6`](requirements.txt#L1-L6)).
- A hipBLASLt checkout containing `tensilelite/` and an existing
  `build/release/`; all tune/search/bench workflows enforce those markers
  ([`geko/paths.py:35-53`](geko/paths.py#L35-L53),
  [`geko/cli.py:330-332`](geko/cli.py#L330-L332)).
- Usable AMD GPU device IDs. The configure path currently accepts IDs 0 through
  7 ([`geko/pipeline.py:388-401`](geko/pipeline.py#L388-L401)).
- A TensileLite client. GEKO builds/reuses it automatically; if a rebuild is
  needed, the Python `invoke` package must be installed
  ([`geko/utils.py:95-141`](geko/utils.py#L95-L141)). The optimizer then launches
  `tensilelite/Tensile/bin/Tensile` with that prebuilt client
  ([`geko/optim/optim.py:298-323`](geko/optim/optim.py#L298-L323)). Ductile itself
  is vendored under TensileLite and imported by its backend
  ([`../../tensilelite/Tensile/backends/ductile_backend.py:20-29`](../../tensilelite/Tensile/backends/ductile_backend.py#L20-L29)).

Live environment note (checked 2026-09-22): in container `vllm-rcm-a`, both
`/work/rocm-libraries/projects/hipblaslt` and
`/work/worktrees/users-alvasile-hypertensile/projects/hipblaslt` contain
`tensilelite/`, but neither contains `build/release/` or
`build/release/clients/hipblaslt-bench`. Both therefore fail GEKO's current
`require_built=True` gate and need a hipBLASLt clients build before tuning.

## Output and threshold behavior

Without `--workdir`, GEKO creates a unique `geko_YYYYmmdd_HHMMSS_microseconds`
directory in the current directory. With `--list` or `--inline`, it also writes
`synthetic_workload.yaml` there ([`geko/cli.py:38-46`](geko/cli.py#L38-L46),
[`geko/cli.py:334-365`](geko/cli.py#L334-L365)). Configure writes
`run_state.json`, `summary.csv`, `gemms.csv`, and tuning YAMLs under
`optimizations/` ([`geko/pipeline.py:418-453`](geko/pipeline.py#L418-L453)).

Optimization produces per-config build trees, merged YAMLs in `libs/`, benchmark
inputs, and `results/raw_results.csv` plus `results/metrics.json`. `full_libs/`
is always emitted after successful analysis; `results/final_results.csv` and
`final_libs/` exist only when at least one different tuned kernel is valid and
meets the uplift threshold ([`geko/optim/optim.py:447-505`](geko/optim/optim.py#L447-L505),
[`geko/pipeline.py:589-627`](geko/pipeline.py#L589-L627)).

Important current bug: although `--up_thr 1.0` is parsed, the `--tune` dispatch
does not pass it to `run_optimize`; only `--search` receives the CLI value
([`geko/cli.py:381-417`](geko/cli.py#L381-L417)). Therefore this command still
uses `run_optimize`'s default `up_thr=1.03`, requiring at least 3% uplift, plus a
different kernel and error below 0.03
([`geko/pipeline.py:456-464`](geko/pipeline.py#L456-L464),
[`geko/optim/optim.py:447-468`](geko/optim/optim.py#L447-L468)).

## Recommended concrete invocation

For one BF16 NT shape on GPU 2:

```bash
cd /path/to/rocm-libraries/projects/hipblaslt/utilities/geko
./bin/geko --tune \
  --hipblaslt /path/to/rocm-libraries/projects/hipblaslt \
  --arch gfx942 --backend ductile --search-space generic \
  --devices=2 --n_slots 1 \
  --inline 1024 4096 1 8192 B B S N T \
  --workdir geko_gfx942_bf16_nt_1024x4096x8192
```

Omit `--hipblaslt` only when using the in-tree launcher and its checkout is the
built tree you intend to tune against. Treat `--up_thr 1.0` as ineffective for
`--tune` until the dispatch forwards it.
