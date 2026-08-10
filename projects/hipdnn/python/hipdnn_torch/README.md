# hipdnn_torch — inject hipDNN into PyTorch

> [!CAUTION]
> **This is an experimental bring-up / E2E-testing aid, not a supported product.** It
> monkeypatches PyTorch internals and depends on a hipDNN engine plugin whose kernel
> family and shape constraints are still narrow. Expect gaps. The value today is
> **correctness parity + an honest census of what still falls back to native PyTorch**,
> which is the list that drives future kernel work.

`hipdnn_torch` routes a handful of `torch.nn.functional` calls — `F.linear`,
`F.rms_norm`, and `F.scaled_dot_product_attention` — onto a hipDNN engine, and falls
back to stock PyTorch (transparently, and logged) for anything the engine can't serve.
It exists so you can run a **real** model end-to-end on hipDNN kernels, measure them,
and get a ranked list of the ops/shapes hipDNN still needs.

```python
import hipdnn_torch

hipdnn_torch.enable_logging()   # optional: print each native fallback as it happens
hipdnn_torch.install()          # patch F.linear / F.rms_norm / F.scaled_dot_product_attention

model(inputs)                   # your unmodified model — matched calls route to hipDNN

print(hipdnn_torch.report())    # per-shape aot/native counts + why calls fell back
hipdnn_torch.uninstall()
```

No model code changes. `install()` patches the functional entry points, so every
`nn.Linear`, `nn.RMSNorm`, and `F.scaled_dot_product_attention` that resolves through
`torch.nn.functional` at call time is intercepted. Calls that don't meet the engine's
gate (wrong dtype, unsupported shape, masked/causal attention, …) run on native
PyTorch exactly as before — nothing breaks, it's just counted and logged.

## Contents

```
hipdnn_torch/
├── README.md                  # this file
├── hipdnn_torch/              # the reusable package (samples import this)
│   ├── __init__.py            # public API: install / uninstall / report / reset / enable_logging
│   ├── bootstrap.py           # one-time, env-parametrized backend+frontend+provider init
│   ├── base.py                # OpOverride: patch, graph cache, execute, census + fallback logging
│   ├── linear.py              # F.linear            -> hipDNN RCR matmul
│   ├── rmsnorm.py             # F.rms_norm          -> hipDNN 2-D RMSNorm
│   └── sdpa.py                # F.scaled_dot_product_attention -> hipDNN fused attention
└── samples/
    ├── minimal_block.py       # 5-minute "try it": a self-contained block, no external repo
    ├── microbench_ab.py       # per-op A/B (hipDNN vs native) over a shape sweep
    ├── sdpa_backends.py       # SDPA vs the real fused backends (AOTriton / fa-triton / hipDNN)
    └── ltx_video_ab.py        # ADVANCED: a real diffusion transformer (needs ComfyUI)
```

## Quick start

The neutral, self-contained sample builds a tiny transformer block out of exactly the
functionals this layer intercepts and A/Bs it against native PyTorch — no model
download, no external checkout. With the environment set up (next section):

```bash
HIPDNN_TORCH_PROVIDER_SO=<build>/lib/hipdnn_plugins/engines/libhip_kernel_provider.so \
    python samples/minimal_block.py
```

It runs one forward on plain PyTorch, installs `hipdnn_torch`, runs the same forward,
checks the two outputs match within a bf16 tolerance, and prints the report:

```
linear intercept census (shape -> aot / native):
  K=2048,N=6144,dtype=bf16          aot=    1  native=    0
  K=2048,N=2048,dtype=bf16          aot=    1  native=    0  biased=1
  ...
  TOTAL                             aot=    4  native=    0
rms_norm intercept census (shape -> aot / native):
  N=2048,dtype=bf16                 aot=    2  native=    0
  TOTAL                             aot=    2  native=    0
scaled_dot_product_attention intercept census (shape -> aot / native):
  Sq=128,Skv=128,dtype=bf16         aot=    1  native=    0
  TOTAL                             aot=    1  native=    0
```

When something falls back, the report ends with a ranked **fallback reasons** block —
that's the actionable "what hipDNN still needs" list. To see each fallback as it
happens, call `hipdnn_torch.enable_logging()` (or configure the `hipdnn_torch` logger
yourself — the package attaches only a `NullHandler` by default).

### Public API

| Call | Purpose |
|------|---------|
| `install(ops=("linear","rmsnorm","sdpa"))` | Patch the selected functionals. Triggers the one-time bootstrap. |
| `uninstall(ops=None)` | Restore the real functionals (default: all installed). |
| `reset(ops=None)` | Clear the census + fallback tally. |
| `report(ops=None) -> str` | Per-op census + ranked fallback reasons. |
| `census(ops=None) -> dict` | The raw counters, for programmatic use. |
| `enable_logging(level=logging.INFO)` | Print each native fallback to stderr. |
| `provider_ready() -> bool` | True if the environment bootstraps and a GPU is present. Never raises. |
| `overrides() -> dict` | The live override instances, for advanced inspection. |

Importing the package does **not** import torch or touch the GPU — that happens lazily
on the first `install()` (or `provider_ready()`), which is also where a misconfigured
environment surfaces a clear `BootstrapError` naming the variable to set.

## Environment setup

> [!NOTE]
> This is the part that is genuinely not a drop-in. The injection code is portable, but
> the runtime it plugs into is specific: a PyTorch build whose ROCm matches the hipDNN
> backend, plus a built engine plugin. Read this section before filing a "nothing
> routed" issue.

**1. A PyTorch whose ROCm build matches the hipDNN backend it loads (the version-skew
trap).** The frontend bindings bind to a `libhipdnn_backend.so`, and mixing a
system/SDK backend with the one PyTorch ships is the single most common hard-to-debug
failure. `bootstrap.py` avoids it by `dlopen`-ing **torch's own** bundled backend with
`RTLD_GLOBAL` before importing the frontend. That only works if your torch actually
bundles a compatible ROCm SDK. Use the ROCm torch build that pairs with your hipDNN.

**2. Import order is fixed and handled for you.** The bootstrap: imports torch → warms
the HIP/HSA stack (`torch.zeros(1, device="cuda")`) → `dlopen`s torch's backend
`RTLD_GLOBAL` → imports the frontend → points it at the provider `.so` → opens a
`Handle`. Because of this, **let `hipdnn_torch` bring torch up.** In practice: set any
env vars that must precede torch's CUDA init (see the SDPA backends note below) at the
very top of your script, then `import hipdnn_torch` and call `install()`.

**3. WSL only: the `librocdxg` shim.** On WSL2 the HIP runtime needs the WSL GPU shim
library discoverable on `LD_LIBRARY_PATH` before torch initializes CUDA, or device
enumeration fails. Add the directory containing it to `LD_LIBRARY_PATH` in your shell
before launching Python. (Native Linux and Windows don't need this.)

**4. Point the layer at the provider plugin.** The only **required** variable:

| Variable | Required | Meaning |
|----------|----------|---------|
| `HIPDNN_TORCH_PROVIDER_SO` | **yes** | Path to the built engine plugin, e.g. `<build>/lib/hipdnn_plugins/engines/libhip_kernel_provider.so`. |
| `HIPDNN_TORCH_ENGINE` | no | Engine name to pin (default `AOT_CATALOG_ENGINE`). |
| `HIPDNN_TORCH_FRONTEND_DIR` | no | Fallback path to a raw `frontend_bindings/build` dir, used only if the `hipdnn-frontend` wheel isn't importable. |
| `HIPDNN_TORCH_BACKEND_GLOB` | no | Override the glob used to find torch's bundled `libhipdnn_backend.so` (rarely needed). |

The frontend bindings themselves are found by importing the `hipdnn-frontend` wheel
first (build and install it per `../README.md`), then falling back to
`HIPDNN_TORCH_FRONTEND_DIR`. Any engine-specific configuration (catalog directories,
tuning caches, etc.) is left to the engine's own environment variables — `hipdnn_torch`
does not set or clear them.

**Check it's wired up** without running a model:

```bash
HIPDNN_TORCH_PROVIDER_SO=<...>/libhip_kernel_provider.so \
    python -c "import hipdnn_torch; print(hipdnn_torch.provider_ready())"
```

## Getting a provider with the operations this layer needs

> [!IMPORTANT]
> hipDNN is an **early-release** library. Default builds ship a deliberately limited set
> of providers and engines, and the more experimental ones are turned **off**.
> `hipdnn_torch` only routes a call to hipDNN when a loaded engine actually serves that
> operation and shape — otherwise it falls back to native PyTorch. So `provider_ready()`
> can return `True` and your model can run correctly while **nothing routes**, simply
> because the loaded provider doesn't yet cover `F.linear` / `F.rms_norm` /
> `F.scaled_dot_product_attention`. Getting calls to route takes a **custom build** that
> re-enables the pieces the default build leaves out.

There are two independent gates, both off by default:

**1. The provider itself is not in the default build.** The engine plugin this layer
loads comes from `hip-kernel-provider`, which is **not** part of the default
(`default:release`) or the "supported providers" presets — only the `hip-kernel-provider`,
`hipdnn-providers-all`, and `hipdnn-dev-all` presets include it. Build it explicitly from
the repository root:

```bash
cmake --preset hip-kernel-provider
cmake --build build
```

(equivalently, add `hip-kernel-provider` to `ROCM_LIBS_ENABLE_COMPONENTS`). This produces
the plugin under `build/.../hipdnn_plugins/engines/libhip_kernel_provider.so` — the path
you point `HIPDNN_TORCH_PROVIDER_SO` at.

**2. The engine that provides these ops is gated inside the provider.** Which engine
serves matmul / RMSNorm / SDPA — and whether it's compiled at all — is controlled by its
own CMake option, documented alongside that engine. `hipdnn_torch` selects the engine by
name through `HIPDNN_TORCH_ENGINE` (default `AOT_CATALOG_ENGINE`); set it to match the
engine your build actually registers.

> [!NOTE]
> The default engine name above, `AOT_CATALOG_ENGINE`, is **not yet on `develop`** — it
> currently lives in the draft PR
> [ROCm/rocm-libraries#10556](https://github.com/ROCm/rocm-libraries/pull/10556), which
> adds the matmul / RMSNorm / SDPA op coverage this layer was built against. That PR's own
> `README.md` is the authoritative source for its build option and setup (including which
> flags to pass and which are *not* needed). To exercise that coverage today, build
> `hip-kernel-provider` from that branch following its README, point
> `HIPDNN_TORCH_PROVIDER_SO` at the resulting plugin, and leave `HIPDNN_TORCH_ENGINE` at
> its default. That PR is also a good worked example of **how additional operation
> coverage is added to hipDNN** — the same pattern (a new engine / kernel family behind a
> build option) is how future ops will land and become routable here.

## Applicability & known limitations

**This only intercepts calls that go through `torch.nn.functional`.** That covers a lot
— every `nn.Linear`, `nn.RMSNorm`, most attention written against
`F.scaled_dot_product_attention` — but there are real ways a model bypasses it:

- **`torch.compile` / TorchInductor.** A compiled graph is lowered ahead of time and may
  never re-enter the Python `F.*` functions at run time. Install *before* compiling, and
  be aware that fused/traced regions can route around the patch entirely.
- **Custom fused attention.** FlashAttention, xformers, Triton kernels, AITER, and
  similar call their **own** ops (`torch.ops.*` / custom CUDA/HIP), not
  `F.scaled_dot_product_attention` — they are invisible to this layer. (See the next
  section for how to make a model use `F.SDPA` so the injection can see it.)
- **Direct `torch.ops.aten.*` or C++ module calls.** Anything that reaches the ATen
  dispatcher without going through the Python functional is not patched.
- **Quantized / low-bit paths.** Only f16/bf16 are gated in; quantized ops fall back.
- **Non-PyTorch frameworks** (JAX, TensorFlow, ONNX Runtime): out of scope entirely.

**Current engine coverage and shape constraints** (anything outside these gates falls
back to native and is counted with a reason):

| Op | Serves | Constraints |
|----|--------|-------------|
| `F.linear` | RCR matmul (`y = x @ Wᵀ + b`) | cuda, f16/bf16, weight 2-D, **M, N, K all multiples of 16**. Bias added natively after the matmul (no longer forces a fallback). |
| `F.rms_norm` | 2-D last-axis RMSNorm | cuda, f16/bf16, single-axis (last-dim) norm. Weightless norms (`weight=None`) are served via a synthesized ones-weight; `eps=None` resolves to `torch.finfo(dtype).eps` to match native exactly. |
| `F.scaled_dot_product_attention` | Dense, non-causal, unmasked fused attention | cuda, f16/bf16, rank-4 BHSD, `attn_mask=None`, `dropout_p=0`, not causal, no GQA, **B=1**, **H=32**, **D=64**, `S_q`/`S_kv` multiples of 16. |

> [!NOTE]
> The SDPA head count (32), head dim (64), and tile (16) are **baked into the shipped
> kernel family** and are named constants in `sdpa.py`. They must match the plugin you
> load. Driving these from catalog metadata (so a differently-baked family is picked up
> automatically) is a documented follow-up.

## Per-framework notes

The patch is global — it replaces the functional symbols once, and every framework that
ultimately calls them is covered. The only thing that differs between frameworks is
*which code paths actually reach `torch.nn.functional`*:

- **ComfyUI.** `comfy.ops` layers (and the model definitions built on them) resolve to
  `F.linear` / `F.scaled_dot_product_attention` at call time, so a single `install()`
  covers them. The advanced `ltx_video_ab.py` sample drives a real ComfyUI model this
  way. Some ComfyUI attention optimizations can be toggled to use non-SDPA backends —
  those bypass the injection (see above).
- **Hugging Face Transformers.** Load models with `attn_implementation="sdpa"` so
  attention routes through `F.scaled_dot_product_attention` (intercepted).
  `attn_implementation="flash_attention_2"` (or `"eager"`'s hand-rolled softmax)
  bypasses it. Linears and norms are intercepted regardless.
- **Diffusers.** Same story: the default attention processors call `F.SDPA` (visible);
  explicitly-selected FlashAttention/xformers processors do not.

The rule of thumb: if a framework has a knob to "use PyTorch SDPA" vs "use
FlashAttention/xformers/Triton," pick the PyTorch-SDPA option to let `hipdnn_torch` see
attention. Linear and RMSNorm rarely have such a knob and are almost always visible.

## Other PyTorch attention/op backends

Attention has several backends on ROCm; knowing which one PyTorch is actually using
matters, because benchmarking against the wrong baseline is misleading. On RDNA
(gfx1151) in particular, `F.scaled_dot_product_attention` **silently drops to the
unfused O(S²) math path** unless you opt into the experimental fused kernels — and the
math path makes any fused kernel look far better than it really is.

- **AOTriton (flash / memory-efficient).** PyTorch's built-in fused SDPA backends on
  ROCm. On RDNA they are gated behind `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, which
  **must be set before torch initializes CUDA**. Select a specific backend with
  `torch.nn.attention.sdpa_kernel([...])`. This is the *real* fused baseline to compare
  against.
- **ROCm flash-attention (Triton), a.k.a. "fa-triton".** The `flash_attn` package
  built from source with the Triton AMD backend; enabled at run time with
  `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`. Takes `[B, S, H, D]` layout (the sample
  permutes for you). Availability varies by pinned commit and GPU arch.
- **AITER / Composable Kernel (CK).** AMD's kernel libraries; some frameworks call into
  them directly for attention/GEMM. These call their own ops and bypass this layer.
- **xformers / SageAttention.** Third-party fused-attention libraries; likewise their
  own ops, invisible to the injection.
- **CDNA vs RDNA.** Much of the fused-attention tooling matured on CDNA (gfx942/950)
  first; RDNA (gfx1151) support is newer and often gated/experimental. Backend
  availability and performance differ substantially between the two — always confirm
  which backend actually ran on *your* GPU.

`samples/sdpa_backends.py` sets the two env vars for you, probes which backends are
usable on this machine, and benches hipDNN against math / AOTriton-flash /
AOTriton-efficient / fa-triton on the same shapes so you get an honest comparison.

## Samples

All samples require `HIPDNN_TORCH_PROVIDER_SO` and a provider-compatible torch; each
prints a clear message and exits if `provider_ready()` is false.

| Sample | What it does | Extra requirements |
|--------|--------------|--------------------|
| `minimal_block.py` | Self-contained block; parity + report. The "try it" path. | none |
| `microbench_ab.py` | Per-op A/B over a shape sweep; parity, timing, routed-or-fell-back. `--ops`, `--dtype`. | none |
| `sdpa_backends.py` | SDPA vs the real fused backends; backend probe + per-shape table. | optional `flash_attn` for fa-triton |
| `ltx_video_ab.py` | **Advanced.** A real LTX-Video diffusion transformer, native vs injected, with a per-op device-time census. | a ComfyUI checkout via `COMFYUI_PATH` |

Because the shipped gfx1151 kernels are correctness-first references, expect **output
parity to hold and the intended ops to route** — but do not expect a blanket
full-forward speedup yet. Kernel uplift is a separate, data-driven follow-on; the point
of this layer is to make that measurable on real models.

## Package follow-up

The `hipdnn_torch/` package is deliberately structured so it can become a
pip-installable wheel with a small lift (add a `pyproject.toml`, move to a `src/`
layout): importing it is torch-free and side-effect free, all environment discovery is
centralized in `bootstrap.py`, and the public API is a stable surface. That wheel, and
catalog-metadata-driven SDPA bakes, are the documented next steps — not part of this
initial drop.

## Going further

The functional monkeypatch this package ships is the shallowest of several ways to route
a framework's operations through hipDNN. Two design references in [`reference/`](reference/)
cover the fuller landscape — the deeper integration tiers, their tradeoffs, and how they
differ from what is implemented here. They are point-in-time design notes (roadmap and
provider maturity will age), written to guide both internal integration work and third
parties integrating with hipDNN:

- [`reference/pytorch-integration-techniques.md`](reference/pytorch-integration-techniques.md)
  — the general PyTorch path. Sets the functional monkeypatch (this package) and the
  ATen-dispatch override side by side (§3.1), then covers the deeper tiers above them: a
  C++ extension, TorchInductor fusion, and a native backend.
- [`reference/vllm-integration-techniques.md`](reference/vllm-integration-techniques.md)
  — vLLM specifically. vLLM bypasses `torch.nn.functional` for its hottest operations, so
  the monkeypatch here does not cover it; this charts the vLLM-specific path from a
  `ROCM_HIPDNN` attention-backend plugin to a first-class backend.
