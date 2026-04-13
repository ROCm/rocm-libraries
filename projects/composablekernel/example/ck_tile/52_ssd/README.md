# Mamba-2 SSD Forward — ck_tile Example

An implementation of the
**Mamba-2 SSD (State Space Decomposition)** forward pass on AMD GPUs.

> **Current status**
>
> - Data type: **fp32 only**.  bf16 / fp16 support is planned.
> - Features: `HAS_D=true`, `D_HAS_HDIM=true`, `HAS_Z=false`.
>   Output gating (`HAS_Z`) will be added in a future update.
>   Plan to use grouped gemm to replace batched gemm.

---

## Table of Contents

1. [Background — SSD Algorithm](#background--ssd-algorithm)
2. [Tensor Shapes and Parameters](#tensor-shapes-and-parameters)
3. [Template Flags (`HAS_D`, `D_HAS_HDIM`, `HAS_Z`)](#template-flags)
4. [Step-by-Step Algorithm](#step-by-step-algorithm)
5. [File Structure](#file-structure)
6. [Build](#build)
7. [Run](#run)
8. [GPU Targets](#gpu-targets)
9. [Implementation Notes](#implementation-notes)
10. [Roadmap](#roadmap)

---

## Background — SSD Algorithm

Mamba-2 decomposes a linear state-space model (SSM) into a form that can be
computed efficiently with matrix multiplications.  The key idea is to split the
input sequence of length `T = C * L` into **C chunks** of length **L** and
express the SSM recurrence as:

| Part | What it computes | How |
|------|-----------------|-----|
| **Intra-chunk** | Contributions *within* each chunk of L tokens | Two batched GEMMs (IntraBMM1, IntraBMM2) with a lower-triangular exponential-decay mask (**segsum**) |
| **Inter-chunk** | Contributions *across* chunks via hidden state propagation | Two batched GEMMs (InterBMM1, InterBMM2) with a sequential prefix scan over the C chunks |

The intra-chunk path uses a causal attention-like pattern (L x L matrix),
while the inter-chunk path propagates a compact (N x D) hidden state from one
chunk to the next.  Both paths are combined in the epilogue to produce the
final output Y, plus an optional skip connection through the D parameter.

---

## Tensor Shapes and Parameters

All tensors are stored in **row-major** order.

| Tensor   | Shape              | Description |
|----------|--------------------|-------------|
| `X`      | `[B, EH, D, C, L]`| Input activations |
| `DeltaA` | `[B, EH, C, L]`   | Log-space decay rates (cumsum of these gives the state decay) |
| `Delta`  | `[B, EH, C, L]`   | Scaling factors for input projection |
| `B_mat`  | `[B, G, N, C, L]` | Input-to-state projection matrix |
| `C_mat`  | `[B, G, N, C, L]` | State-to-output projection matrix |
| `D`      | `[EH, D]`         | Skip-connection (residual) parameter |
| `Y`      | `[B, EH, D, C, L]`| Output activations |
| `Fstate` | `[B, EH, D, N]`   | Final hidden state after the last chunk |

### Dimension key

| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| `B`    | Batch size | user-specified |
| `G`    | Number of groups (must be 1 currently) | 1 |
| `E`    | Expansion factor | user-specified |
| `H`    | Number of heads per group | user-specified |
| `EH`   | Total expanded heads = `E * H` | — |
| `C`    | Number of chunks | 8 |
| `L`    | Chunk length (sequence length per chunk) | 128 |
| `D`    | Head dimension | 64 |
| `N`    | State dimension | 128 |

---

## Template Flags

These flags control optional features in the SSD epilogue:

### `HAS_D` (default: `true`)

Whether to add the **skip connection** `D * X` to the output:

```
if HAS_D:
    Y[l, d] += D_param[d] * X[d, l]
```

This is analogous to a residual connection: the input is scaled per-dimension
and added directly to the output.  Setting `HAS_D=false` removes this term.

### `D_HAS_HDIM` (default: `true`)

Controls the shape of the D parameter:

- `true`: D has shape `[EH, D]` — each head-dimension pair has its own scale.
- `false`: D has shape `[EH, 1]` — a single scalar per head, broadcast across
  all D dimensions.

Only meaningful when `HAS_D=true`.

### `HAS_Z` (default: `false`, **not yet implemented**)

Whether to apply an **output gate** using a Z tensor:

```
if HAS_Z:
    Y[l, d] = Y[l, d] * Z[l, d] * sigmoid(Z[l, d])
```

Where `Z` has the same shape as `X`: `[B, EH, D, C, L]`.  This is the gated
activation from the Mamba-2 architecture.  When `HAS_Z=false` (current
behavior), the output Y is produced directly without gating.

---

## Step-by-Step Algorithm

For each `(batch, head)` pair, the forward pass runs these 10 steps:

### Step 1 — Cumsum

Compute the cumulative sum of `DeltaA` along the L dimension for each chunk:

```
cumsum[c, l] = sum_{i=0}^{l} DeltaA[c, i]
```

### Step 2 — IntraBMM1 (Batched GEMM)

Compute the intra-chunk "attention" matrix by contracting over the state
dimension N:

```
IntraBMM1[l1, l2] = sum_n C_mat[n, l1] * B_mat[n, l2]     (= C^T @ B)
```

Result shape: `[L, L]` per (head, chunk).

### Step 3 — SegSum + Pre-IntraBMM2 (fused custom kernel)

Build the lower-triangular exponential-decay mask from the cumsum and
element-wise multiply with Delta and IntraBMM1:

```
segsum[i, j] = exp(cumsum[i] - cumsum[j])   if j < i
             = 1                              if j == i
             = 0                              if j > i

PreIntra2[i, j] = segsum[i, j] * Delta[j] * IntraBMM1[i, j]
```

This mask enforces causality within the chunk: position `i` can only attend to
positions `j <= i`, with exponential decay.

### Step 4 — IntraBMM2 (Batched GEMM)

Multiply the masked attention matrix by the input to get intra-chunk output:

```
IntraBMM2[l, d] = sum_{l'} PreIntra2[l, l'] * X[d, l']     (= PreIntra2 @ X^T)
```

Result shape: `[L, D]` per (head, chunk).

### Step 5 — Pre-InterBMM1 + CumsumExp (custom kernel)

Prepare the inter-chunk input projection, scaled by the tail decay and Delta:

```
cumsum_exp[l]      = exp(cumsum[l])
cumsum_exp_last[l] = exp(last - cumsum[l])     where last = cumsum[L-1]

PreInter1[n, l] = cumsum_exp_last[l] * Delta[l] * B_mat[n, l]
```

### Step 6 — InterBMM1 (Batched GEMM)

Project the input into state space for inter-chunk propagation:

```
InterBMM1[n, d] = sum_l PreInter1[n, l] * X[d, l]     (= PreInter1 @ X^T)
```

Result shape: `[N, D]` per (head, chunk).

### Step 7 — State Propagation (sequential prefix scan)

Propagate the hidden state across chunks using a sequential recurrence:

```
State[0]  = 0
State[ci] = InterBMM1[ci-1] + exp(last[ci-1]) * State[ci-1]
```

This is inherently sequential over C (typically C=8, so the serial cost is
small).  Each `State[ci]` is an `[N, D]` matrix.

### Step 8 — InterBMM2 (Batched GEMM)

Project the propagated state back to output space:

```
InterBMM2[l, d] = sum_n C_mat[n, l] * State[n, d]     (= C^T @ State)
```

Result shape: `[L, D]` per (head, chunk).

### Step 9 — Epilogue (custom kernel)

Combine intra-chunk and inter-chunk contributions, plus the skip connection:

```
Y[d, l] = exp(cumsum[l]) * InterBMM2[l, d]     // inter-chunk, scaled by cumsum
         + IntraBMM2[l, d]                       // intra-chunk
         + D_param[d] * X[d, l]                  // skip connection (HAS_D)
```

### Step 10 — Final State (custom kernel)

Extract the hidden state after the last chunk for use as initial state in the
next forward pass (e.g., during autoregressive generation):

```
Fstate[d, n] = InterBMM1[C-1, n, d] + exp(last[C-1]) * State[C-1, n, d]
```

---

## File Structure

```
52_ssd/
├── CMakeLists.txt          CMake build (part of CK build tree)
├── ssd_problem.hpp         Problem definition: SsdHostArgs, SsdTileConfig
├── ssd_kernels.hpp         Custom HIP kernels (cumsum, segsum, state propagation, epilogue)
├── ssd_fwd.hpp             Host API: ssd_fwd() orchestrating 4 batched GEMMs + custom kernels
├── example_ssd_fwd.cpp     Example driver with CPU reference and verification
└── README.md               This file
```

### How it maps to ck_tile

| ck_tile concept | SSD usage |
|----------------|-----------|
| `BatchedGemmKernel` | 4 batched matrix multiplications (IntraBMM1/2, InterBMM1/2) |
| `TileGemmShape` | Tile sizes per GEMM variant (128x64x32 or 128x128x32) |
| `GemmPipelineAgBgCrMem` | Memory-bound pipeline for the moderate-sized GEMMs |
| `CShuffleEpilogue` | Standard shuffle-based epilogue for GEMM output |
| `DeviceMem` | Device memory allocation and HtoD/DtoH transfers |
| `launch_kernel` | Kernel launch wrapper with stream and occupancy support |
| Custom `__global__` kernels | Element-wise and scan operations that are not GEMMs |

---

## Build

**Requirements:** ROCm 6.0+, Composable Kernel source tree.

This example is built as part of the CK build tree (same as all other
`ck_tile` examples).

```bash
cd composable_kernel
mkdir -p build && cd build
bash ../script/cmake-ck-dev.sh ../ gfx950 -G Ninja
cmake --build . -- tile_example_ssd_fwd -j
```

---

## Run

```bash
# Default: B=2, E=2, H=2 with CPU verification
./bin/tile_example_ssd_fwd

# Custom sizes
./bin/tile_example_ssd_fwd -B=4 -E=2 -H=4

# Benchmark only (no CPU verification)
./bin/tile_example_ssd_fwd -B=4 -E=2 -H=4 -v=0 -warmup=10 -repeat=100
```

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `-B` | 2 | Batch size |
| `-G` | 1 | Groups (must be 1) |
| `-E` | 2 | Expansion factor |
| `-H` | 2 | Heads per group |
| `-v` | 1 | 0 = skip verification, 1 = verify against CPU reference |
| `-warmup` | 3 | GPU warmup iterations |
| `-repeat` | 10 | Benchmark iterations |

Fixed dimensions (not configurable): `C=8, L=128, D=64, N=128`.

---

## GPU Targets

| GPU | Architecture | Target flag |
|-----|-------------|-------------|
| MI210 / MI250 | CDNA2 | `gfx90a` |
| MI300X | CDNA3 | `gfx942` |
| MI350X | CDNA4 | `gfx950` |

---

## Implementation Notes

- B_mat/C_mat sub-matrices have non-unit stride (`C*L` between N-rows),
  so batched pack kernels copy them into contiguous buffers before each
  GEMM.  The GEMMs then run as a single batched launch with
  `batch_count = BEH * C`.  A production implementation would use a custom
  `tile_program` with strided tile windows to avoid the pack overhead.

- `SsdGemmConfig` tile sizes (M=128, N=64, K=32) are tuned for CDNA
  (MI-series) GPUs.  RDNA3 may benefit from different tile shapes.

- Only `G=1` (single group) is supported.

- The CPU reference in `example_ssd_fwd.cpp` computes everything in fp32 and
  `HAS_D=true`, `D_HAS_HDIM=true`, `HAS_Z=false`.

---

## Roadmap

- [ ] `HAS_Z` support (output gating: `Y = Y * Z * sigmoid(Z)`)
- [ ] bf16 / fp16 data types with fp32 accumulation
- [ ] Fused multi-chunk GEMM launch to reduce kernel launch overhead
- [ ] Support for `G > 1` (multiple groups)
