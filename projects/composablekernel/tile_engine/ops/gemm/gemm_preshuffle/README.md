# gfx1250 GEMM pipeline benchmark

`default_config_gfx1250.json` selects all five native pipelines through the
`gemm_preshuffle` benchmark bridge. The caller supplies ordinary A and B tensors;
the bridge packs B only for the two weight-preshuffle pipelines.

| Native pipeline | JSON `pipeline` | Scheduler | Epilogue | B upload |
| --- | --- | --- | --- | --- |
| PRESHUFFLE_V2 | `preshufflev2` | `default` | `default`, `cshuffle` | `shuffle_b_v0` |
| COMPUTE_TDM_V1 | `comp_tdm_v1` | `intrawave` | `tdm` | Ordinary B |
| COMPUTE_TDM_V2 | `comp_tdm_v2` | `intrawave` | `tdm` | Ordinary B |
| PRESHUFFLE_TDM | `preshuffle_tdm` | `default` | `default`, `cshuffle` | `shuffle_b_v0` |
| COMPUTE_ASYNC | `comp_async` | `intrawave` | `default`, `cshuffle` | Ordinary B |

The config targets fp16/bf16, `rcr`, a 128x128x64 block, 2x2x1 waves and a
16x16x32 warp tile. It enables M padding and uses complete N/K tiles. The two
weight-preshuffle pipelines cover both persistent and non-persistent launches;
the three compute pipelines expose only non-persistent execution:
**12 kernels per datatype, 24 across fp16 and bf16**.
Invalid pipeline/scheduler/epilogue combinations in the Cartesian sweep are
filtered before compilation. Persistent requests for compute pipelines are
rejected because these native pipelines do not expose `UsePersistentKernel`.
Compute TDM V2 requires exactly four waves. Compute TDM V1/V2 reject partial
N/K tiles and split-K at runtime: these native paths miscompute partial N/K
tiles on the validation target. Both the registry and legacy TE launch check
these restrictions, including for custom configurations with padding enabled. The
four additional pipelines are gated to gfx1250, matching fp16/bf16 inputs and
16x16x32 warp tiles; existing V2 configurations remain available on gfx9.

This is a bounded default configuration, not an exhaustive tuning search over
all native tile shapes. `permute_n` is false; the bridge does not expose the
permute-N packing mode. The `_preshuffle` kernel-name suffix identifies the
benchmark variant; the kernel's `Preshuffle` metadata determines whether B is
actually packed.

From `tile_engine/ops/gemm`, after building `dispatcher/build/libck_tile_dispatcher.a`:

```bash
python gemm_full_benchmark.py --variant gemm_preshuffle --arch gfx1250 \
    --dtype fp16 --layout rcr --verify --verify-tol 0.01 --csv gfx1250_fp16.csv
python gemm_full_benchmark.py --variant gemm_preshuffle --arch gfx1250 \
    --dtype bf16 --layout rcr --verify --verify-tol 0.01 --csv gfx1250_bf16.csv
```

Both commands select this JSON automatically. Pass a positional JSON to
choose another sweep, `--problems problems.json` for a custom M/N/K list, and
`--devices 1` to use one GPU. Build failures and failed measurements produce a
nonzero exit status, including when some kernels succeed. CSV output includes
latency, TFLOP/s and numerical verification for each kernel/problem pair.

The legacy TE CMake path selects the same default for a single gfx1250 target
and fp16/bf16. It can also be selected explicitly with
`-DGEMM_PRESHUFFLE_CONFIG_FILE=default_config_gfx1250.json`.

The GPU regression test covers the complete configuration, short/hot K loops,
odd/even loop counts, partial M tiles and changed B values. It also builds padded
compute TDM configurations and checks that partial N/K shapes are rejected:

```bash
python dispatcher/tests/test_preshuffle_gpu_correctness.py --gfx gfx1250
```
