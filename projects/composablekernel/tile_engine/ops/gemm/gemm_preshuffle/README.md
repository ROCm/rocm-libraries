# gfx1250 GEMM preshuffle pipelines

`default_config_gfx1250.json` selects all five gfx1250 pipelines through the
`gemm_preshuffle` benchmark bridge. The caller supplies ordinary A and B tensors;
the bridge packs B with `shuffle_b_v0` for every pipeline, since all five read
preshuffled B.

| Native pipeline | JSON `pipeline` | Scheduler | Epilogue |
| --- | --- | --- | --- |
| PRESHUFFLE_V2 | `preshufflev2` | `default` | `default`, `cshuffle` |
| PRESHUFFLE_TDM | `preshuffle_tdm` | `default` | `cshuffle` |
| COMPUTE_TDM_V1 | `comp_tdm` | `intrawave` | `tdm` |
| COMPUTE_TDM_V2 | `comp_tdm_v2` | `intrawave` | `tdm` |
| COMPUTE_ASYNC | `comp_async` | `intrawave` | `cshuffle` |

The config targets `rcr`, a 128x128 block with tile K 64 or 128, 1x4x1 waves
and a 16x16 warp tile (warp tile K 32 for 16-bit, 64 for 8-bit types). It
builds **10 kernels for fp16/bf16 and 8 for fp8/bf8** (`comp_async` has no
fp8/bf8 support yet). Invalid pipeline/scheduler/epilogue combinations in the
Cartesian sweep are filtered before compilation by one shared rule set
(`preshuffle_pipeline_reject_reason`, used by both the dispatcher and the
legacy TE):

- the four added pipelines are gfx1250 and `rcr` only, with no persistent
  kernel;
- the TDM pipelines (`preshuffle_tdm`, `comp_tdm`, `comp_tdm_v2`) require
  `pad_m=pad_n=pad_k=false`;
- `comp_tdm_v2` requires exactly four waves;
- `comp_tdm` and `comp_tdm_v2` also reject partial N/K tiles and split-K at
  launch time.

`preshufflev2` configurations for gfx9 are unchanged. `permute_n` is false; the
bridge does not expose the permute-N packing mode.

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

The legacy TE CMake path selects the same default for a single gfx1250 target.
It can also be selected explicitly with
`-DGEMM_PRESHUFFLE_CONFIG_FILE=default_config_gfx1250.json`.

The GPU regression test covers the complete configuration, short/hot K loops,
odd/even loop counts, partial M tiles and changed B values. It also checks that
compute TDM configurations reject partial N/K shapes:

```bash
python dispatcher/tests/test_preshuffle_gpu_correctness.py --gfx gfx1250
```
