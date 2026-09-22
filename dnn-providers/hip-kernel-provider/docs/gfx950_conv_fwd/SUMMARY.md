# rocKE convolution through hipDNN

**Purpose:** Run rocKE's GPU convolution kernels through the standard hipDNN
graph interface on gfx950 GPUs.

**What is implemented:** The `hipkernel:Gfx950ConvFwd` engine matches supported
convolution graphs and launches precompiled rocKE kernels. The implementation
includes native matching and launch code, descriptor-generation recipes,
packaging, correctness tests, and tools comparing hipDNN with direct rocKE
execution. It uses the existing runtime's automatic candidate selection and
persistent winner cache.

**How to use it:** Follow [SETUP.md](SETUP.md) to build on another development
machine or use an existing installation. Then, from the repository root, run:

```bash
python projects/hipdnn/tools/IngestorGenerator/tools/run_conv_demo.py smoke \
    --install-prefix "$CONV_INSTALL" --output-dir "$CONV_RESULTS"
```

Use `headline` instead of `smoke` for the 51-request comparison. The tools print
correctness results and GPU timings and save detailed JSON reports.

**Benefits:**

- Access rocKE kernels through hipDNN's graph and provider APIs.
- Deploy packaged kernels; the installed engine does not need Python.
- Compare tuning candidates and reuse their saved selection across processes.
- Check correctness and measure integration costs while experimenting.

**Current scope:** 61 requests and 71 variants for plain 2D forward convolution,
groups=1, channels-last FP16/BF16, FP32 accumulation, and symmetric padding.
New shapes require matching compiled entries; [the setup guide](SETUP.md#changing-shapes-or-kernels)
explains how to extend the catalog.
