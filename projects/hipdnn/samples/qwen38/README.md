# Qwen3.8-27B through hipDNN

`hipdnn_sample_qwen38` runs [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (the Hugging Face `qwen3_5` text path) end to end through hipDNN graphs, with greedy decoding. `run_qwen38.py` drives it:
- it tokenizes the prompt with the Qwen chat template, with thinking disabled;
- it runs the binary;
- it detokenizes the `TOKEN <id>` lines as they stream.

## What it demonstrates

Every op of the model is a hipDNN graph node:
- embedding gather;
- RMSNorm;
- matmul;
- causal conv1d;
- gated delta rule (GDN);
- RoPE;
- KV-cache store;
- causal GQA SDPA;
- pointwise;
- top-k.

The program itself only loads weights, builds graphs and moves buffers between them. The same model runs on two backends, which generate the same text.

### `--backend=cpu` (default)

- Weights are converted to fp32 on the host.
- A few multi-node graphs per layer run on the hipDNN CPU reference executor (`CpuReferenceGraphExecutor`, from `hipdnn_test_sdk`).
- Attention stores K and V in a paged cache through `paged_cache_store` and runs paged SDPA.

This backend needs no GPU or engine plugin.

### `--backend=gpu`

This is normal hipDNN execution:
- graphs are built with `graph->build(handle)` and run with `graph->execute(handle, ...)` on the engines of the loaded providers;
- weights stay on the device as bf16;
- activations are bf16, with FP32 compute;
- RMSNorm gammas, the GDN `A_log`/`dt_bias` and the GDN recurrent state are fp32.

How the graphs are built:
- **One graph per op.** The FlyDSL engines match single-node graphs.
- **One fusion.** The MLP gate GEMM and its SiLU are a two-node graph that hipBLASLt fuses into its epilogue. If no engine takes the fused graph, it is split into a matmul and a pointwise op.
- **Plan cache.** Plans are cached by shape and shared across layers and steps. Prefill and decode shapes are built once each. SDPA is the exception: its key length grows by one per decode step, so it builds one plan per step (about 1 ms).
- **Linear weights.** The nn.Linear `[out, in]` weights are the matmul B operand as a strided `[1, in, out]` view. hipBLASLt takes the transposed layout, so nothing is transposed at load.
- **CPU fallback.** If `build` finds no engine for a graph, that op runs on the CPU reference executor. Its operands are copied device→host, and its outputs host→device; weights are copied to the host once. The op's placement is logged.
- **Attention.** Attention keeps a dense per-layer KV buffer `[Smax, Hkv, D]` (BSHD).
  - The V projection writes straight into its cache rows, and the K RoPE writes its output into the K cache rows.
  - SDPA gets K/V views `[1, Hkv, kv_len, D]` (strides `{Smax·Hkv·D, D, Hkv·D, 1}`) covering exactly the valid length, with a bottom-right causal mask.
  - The FlyDSL CDNA SDPA engines decline paged KV and `seq_len` tensors, so this path doesn't use them.

On gfx942 every op runs on a GPU engine:

| Op | Engine |
|---|---|
| `matmul`, `matmul` + SiLU | `HIPBLASLT_ENGINE` (hipblaslt-provider) |
| `rmsnorm`, `sdpa`, `gather`, `rope`, `causal_conv1d`, `gdn`, `pointwise`, `topk` | `hipkernel:flydsl_*` (hip-kernel-provider) |

At startup every built graph is logged as `[graph] <op> <shape key> -> <engine or CPU reference>`. At exit a table lists the runs per op and engine, with the load, prefill and decode timings. `--profile` synchronizes around every op and adds per-op time.

## Requirements

- Linux.
- The `Qwen/Qwen3.8-27B` checkpoint (52 GB of bf16 safetensors), for example from `huggingface-cli download Qwen/Qwen3.8-27B`.
- Python 3.10 or later with the `tokenizers` package (`pip install tokenizers`).
- For `--backend=gpu`:
  - a GPU with about 52 GB of free memory for the bf16 weights, plus the KV cache and activations (measured on an MI300X);
  - the hip-kernel-provider plugin built with FlyDSL engines and the kernel ingestor (`-DHIPKERNELPROVIDER_ENABLE_FLYDSL=ON -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON`);
  - the hipblaslt-provider plugin.

  Without them, every op the providers don't cover falls back to the CPU reference executor.
- For `--backend=cpu`: enough host memory for the fp32 weights (about 108 GB) and many cores. It uses every core.

## Build

The sample builds with the other samples (Linux only). With the repository superbuild, enable the providers alongside the samples:

```bash
cmake -G Ninja -B build \
    -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hip-kernel-provider;hipblaslt-provider;hipdnn-samples" \
    -DHIPDNN_ENABLE_SDPA=ON -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPKERNELPROVIDER_ENABLE_FLYDSL=ON \
    -DCMAKE_BUILD_TYPE=Release
ninja -C build hipdnn_sample_qwen38 hipblaslt_plugin hip_kernel_provider
```

The build copies `run_qwen38.py` next to the binary in `build/bin`. The install puts both in `bin/`. The engine plugins load from `lib/hipdnn_plugins/engines` next to `libhipdnn_backend`.

## Run

```bash
python3 build/bin/run_qwen38.py "What is the capital of France? Answer in one sentence." --backend=gpu
```

`run_qwen38.py` options:
- `--backend cpu|gpu` (default `cpu`)
- `--profile`: time each op (GPU backend)
- `--max-new-tokens N` (default 24)
- `--layers N`: run only the first N decoder layers, for debugging
- `--snapshot DIR`: the model snapshot directory. By default, it is resolved from the Hugging Face cache (`HF_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`).
- `--binary PATH`: the `hipdnn_sample_qwen38` executable. By default, the one next to the script, else the one on `PATH`.

The binary itself takes `hipdnn_sample_qwen38 <snapshot> <ids,comma,separated> <max-new-tokens> [layers] [--backend=cpu|gpu] [--profile]`, and `--help`.

On a Slurm cluster, run it on a GPU node, for example:

```bash
srun --gres=gpu:1 -c 64 -t 00:30:00 \
    python3 build/bin/run_qwen38.py "What is the capital of France? Answer in one sentence." --backend=gpu
```

## Expected output

stdout carries the generated text:

```
The capital of France is Paris.
```

stderr carries the graph placements, then the summary table and timings.

Measured on gfx942 (MI300X), with the checkpoint in the page cache:

| Prompt | Prefill | Decode |
|---|---|---|
| "What is the capital of France? Answer in one sentence." (24 tokens) | 1.4–1.8 s | 22 ms/token |

- **Prefill** includes building the graphs (about 0.8 s).
- **Load** takes about 4.5 s with the checkpoint in the page cache. A cold load reads 52 GB from disk.
- **CPU reference** (`--backend=cpu`): about 13 s/token on 384 cores.

Both backends generate the same text.
