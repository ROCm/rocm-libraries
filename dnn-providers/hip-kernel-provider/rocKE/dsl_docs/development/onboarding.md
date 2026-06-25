# rocKE 3-Week Learning Program
## From Basics to Agentic Optimization

> **Which path are you on?** This program teaches you to **author and optimize
> kernels** — the right starting point for almost everyone. If instead you are
> going to **modify the engine internals** (anything under `rocke/core/`,
> `rocke/helpers/`, `rocke/instances/`, or the C++ engine in `Cpp/`), read
> [`engine_contributing.md`](./engine_contributing.md) first: the engine exists as
> two implementations kept byte-for-byte identical, and there are non-obvious
> rules ([`invariants.md`](./invariants.md)) you can break silently. Authors can
> ignore that; contributors cannot.

**Target Audience**: Engineers learning to write high-performance GPU kernels using rocKE
**Prerequisites**: Python programming, basic GPU concepts (threads, blocks, memory hierarchy)
**Hardware**: AMD GPU (gfx942/gfx950 recommended, gfx1151/gfx1201 for RDNA)
**AI Integration**: Each week includes AI-assisted learning checkpoints and exercises

> **Validation note**: Every code example, exercise, and command in this guide
> has been run and verified on an AMD Instinct MI355X (gfx950, ROCm 7.0). All
> custom kernels are bit-exact against their torch reference. Run commands from
> the `rocKE/` directory with `export PYTHONPATH=Python`.

---

## Program Overview

- **Week 1**: rocKE Foundations - Layers, primitives, basic kernels (vector add, GEMM), tiling experiments
- **Week 2**: Advanced Patterns - C-shuffle, epilogue fusion, attention, debugging correctness and performance
- **Week 3+**: Agentic Optimization - Using AI skills, hardware profiling, systematic tuning, building optimization knobs

---

# Week 1: rocKE Foundations

## Learning Objectives
By the end of Week 1, you will:
- Understand the rocKE compilation pipeline (Python → IR → LLVM → HSACO → HIP)
- Write simple kernels using core primitives
- Experiment with tiling parameters and observe performance impact
- Use AI assistants to explain unfamiliar concepts

---

## Day 1-2: Mental Model & Setup

### Morning: Environment Setup
**Hands-on Tasks:**
1. Set up your environment:
```bash
# Run all commands from the rocKE root:
cd <rocm-libraries-checkout>/dnn-providers/hip-kernel-provider/rocKE
export PYTHONPATH=Python
export PYTHONDONTWRITEBYTECODE=1

# Verify ROCm sees your GPU (look for your gfx target, e.g. gfx950)
rocminfo | grep -E "gfx|Marketing Name"

# Test basic import
python3 -c "from rocke import *; print('CK DSL ready!')"
```

> **GPU access tip**: If `rocminfo` prints "not member of video group" or torch
> reports `CUDA: False`, your shell is missing the GPU device groups. Confirm
> with `python3 -c "import torch; print(torch.cuda.is_available())"`. If the
> import itself fails with `No module named 'rocke'`, you are not in
> `rocKE/` or `PYTHONPATH=Python` is unset.

2. Run the validation suite to confirm everything works:
```bash
python tests/test_rocke.py
```
This runs ~245 static unit tests in a couple of seconds. A handful of
convolution-lowering tests may error in some checkouts (pre-existing, unrelated
to this course) — the GEMM, IR, and primitive tests you depend on in Week 1
all pass.

### Afternoon: Conceptual Foundation
**Reading (2-3 hours):**
- `dsl_docs/README.md` (sections 1-4 only)
- `dsl_docs/architecture/mental_model.md`
- `dsl_docs/architecture/authoring_model.md`

**AI-Assisted Learning Checkpoint:**
Use an AI assistant (Claude, GPT-4) to quiz yourself:
```
Prompt: "I just read about rocKE's mental model. Ask me 5 questions
to test my understanding of: (1) the compilation pipeline, (2) SSA IR,
(3) the difference between LLVM lowering and HIP lowering, (4) buffer
resource descriptors, and (5) why Value.__bool__ raises an error."
```

**Key Concepts to Internalize:**
- rocKE is a **Python-embedded DSL** that generates GPU kernels
- Compilation path: `Python spec → KernelDef IR → LLVM IR → HSACO → hipModule`
- The IR is **SSA-based** (Static Single Assignment) - values are immutable
- `IRBuilder` constructs the IR, not directly executes code
- Production path uses LLVM lowering, HIP lowering is for debugging only

---

## Day 3-4: First Kernel - Vector Add

### Morning: Simple Vector Add
**Study the example:**
```bash
# Read the 2D add demo - simpler than full vector add
cat Python/rocke/examples/common/distribution_2d_add_demo.py
```

**Exercise 1: Build and run the demo**
```bash
python Python/rocke/examples/common/distribution_2d_add_demo.py \
    --H 128 --W 256 --tile-m 32 --tile-n 64 --vec 8
```
Expected output (bit-exact):
```
2D distribution-driven add  H=128 W=256 tile=(32,64) vec=8  max_abs=0.000e+00
```

**Exercise 2: Create your own 1D vector add**
Create `my_first_kernel.py`:

> **Key API note**: rocKE has **no raw-pointer `gep`/`load`/`store`**. You
> address global memory with index-based primitives: `b.global_load_f32(ptr,
> idx)` reads element `idx` from a base pointer, and `b.global_store(ptr, idx,
> value)` writes it. Comparisons are `b.cmp_lt(a, b)` (and `cmp_le/gt/ge/eq/ne`),
> not a string-predicate `cmp(...)`. This kernel is verified bit-exact on gfx950.

```python
"""Simple 1D vector add: C = A + B"""
import torch
from rocke.core.arch import ArchTarget
from rocke.core.ir import F32, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.runtime.comgr import build_hsaco_from_llvm_ir
from rocke.runtime.hip_module import get_device_arch
from rocke.runtime.launcher import KernelLauncher, LaunchConfig, synchronize_and_release

BLOCK_SIZE = 256


def build_vector_add(vec_size: int = 4):
    """Build a simple vector add kernel.

    Each thread handles ``vec_size`` consecutive elements, addressed with the
    index-based ``global_load_f32`` / ``global_store`` primitives.
    """
    b = IRBuilder(f"vector_add_v{vec_size}")
    b.kernel.attrs["max_workgroup_size"] = BLOCK_SIZE

    # Kernel parameters
    A = b.param("A", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    B = b.param("B", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    N = b.param("N", I32)

    # Thread index -> first element this thread owns
    tid = b.thread_id_x()
    bid = b.block_id_x()
    global_idx = b.add(b.mul(bid, b.const_i32(BLOCK_SIZE)), tid)
    base_idx = b.mul(global_idx, b.const_i32(vec_size))

    for i in range(vec_size):
        idx = b.add(base_idx, b.const_i32(i))
        # Per-element bounds check keeps the tail safe for non-multiple sizes.
        in_bounds = b.cmp_lt(idx, N)
        with b.scf_if(in_bounds):
            a_val = b.global_load_f32(A, idx)
            b_val = b.global_load_f32(B, idx)
            c_val = b.fadd(a_val, b_val)
            b.global_store(C, idx, c_val)

    return b.kernel


# Test harness
if __name__ == "__main__":
    N = 1024 * 1024
    vec_size = 4

    kernel = build_vector_add(vec_size)
    arch = get_device_arch() or "gfx950"
    isa = ArchTarget.from_gfx(arch).isa_triple
    hsaco, _ = build_hsaco_from_llvm_ir(
        lower_kernel_to_llvm(kernel, arch=arch),
        isa=isa,
    )

    launcher = KernelLauncher(
        hsaco=hsaco,
        kernel_name=kernel.name,
        signature=[
            {"name": "A", "type": "ptr<f32, global>"},
            {"name": "B", "type": "ptr<f32, global>"},
            {"name": "C", "type": "ptr<f32, global>"},
            {"name": "N", "type": "i32"},
        ],
    )

    torch.manual_seed(0)
    A = torch.randn(N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, dtype=torch.float32, device="cuda")
    C = torch.zeros_like(A)

    grid_size = (N + BLOCK_SIZE * vec_size - 1) // (BLOCK_SIZE * vec_size)
    launcher(
        {"A": A, "B": B, "C": C, "N": N},
        config=LaunchConfig(grid=(grid_size, 1, 1), block=(BLOCK_SIZE, 1, 1)),
    )
    synchronize_and_release()

    ref = A + B
    diff = (C - ref).abs().max().item()
    print(f"Vector add N={N} vec_size={vec_size} max_diff={diff:.3e}")
    assert diff < 1e-5, f"Failed: diff={diff}"
    print("PASS")
```

### Afternoon: Understanding the IR
**Exercise 3: Inspect the generated code**

Modify your kernel to print IR and LLVM. Note `print_ir()` **returns** the
MLIR-style text as a string (it does not print on its own), so wrap it in
`print(...)`:
```python
from rocke.core.ir_print import print_ir

# After building the kernel:
print("=== SSA IR ===")
print(print_ir(kernel)[:2000])  # print_ir returns a str

print("\n=== LLVM IR ===")
llvm_ir = lower_kernel_to_llvm(kernel, arch=arch)
print(llvm_ir[:2000])  # First 2000 chars
```
In the SSA IR you'll see `arith.cmp`, `scf.if`, and `memref.global_load_typed`
ops; in the LLVM IR these become `icmp slt`, a `br i1` branch, and
`getelementptr`+`load`/`store` — a good way to confirm what your builder calls
actually lowered to.

**AI-Assisted Learning:**
```
Prompt: "Here's my rocKE kernel code [paste your code]. Explain:
1. What does IRBuilder.scf_if do differently than a Python if?
2. Why does rocKE use index-based global_load_f32(ptr, idx) instead of
   raw pointer arithmetic? (And why does Value.__bool__ raise?)
3. What happens during the lower_kernel_to_llvm() step?
4. How does the per-element bounds check keep the tail safe?"
```

---

## Day 5-6: Introduction to GEMM & Tiling

### Morning: Study GEMM Basics
**Reading:**
- `dsl_docs/instances/gemm.md` (first 50%)
- `dsl_docs/primitives/intrinsics_and_primitives.md`

**Run the shipped GEMM example:**
```bash
# Small GEMM to start. Note: flags are lowercase --m --n --k.
python -m rocke.examples.common.universal_gemm_verify \
    --arch gfx950 --m 1024 --n 1024 --k 1024
```
Expected tail (numbers vary by GPU):
```
verify max_abs_diff=0 bad=0/1048576
Perf: 0.0123 ms, ~174 TFlops, ~511 GB/s
[gfx950] GEMM 1024x1024x1024: max_abs_diff=0.000e+00 tol=0e+00 -> PASS
```
The per-arch wrapper `rocke.examples.gfx942.gemm_demo` forwards the same flags
and just pins `--arch gfx942` by default.

### Afternoon: Tiling Experiments
**Exercise 4: GEMM MFMA-atom sweep**

The shipped harness chooses block/warp tiling for you; the one tiling knob it
exposes on the CLI is `--warp-tile` — the **MFMA atom** (`MxNxK`), the matrix
shape one wave computes per instruction. Sweeping it shows how a single tiling
choice swings throughput. The legal fp16 atoms on gfx950 are `16x16x16`,
`16x16x32`, `32x32x8`, `32x32x16`.

Create `gemm_atom_sweep.py`:
```python
"""Sweep the GEMM MFMA atom (warp-tile) and report TFLOPS for each."""
import os
import re
import subprocess
import sys

PY = sys.executable
ATOMS = ["16x16x16", "16x16x32", "32x32x8", "32x32x16"]
M = N = K = 2048
ARCH = "gfx950"

results = []
for atom in ATOMS:
    print(f"\n{'='*60}\nTesting warp-tile (MFMA atom) = {atom}\n{'='*60}")
    proc = subprocess.run(
        [
            PY, "-m", "rocke.examples.common.universal_gemm_verify",
            "--arch", ARCH,
            "--m", str(M), "--n", str(N), "--k", str(K),
            "--warp-tile", atom,
        ],
        capture_output=True, text=True,
        env={"PYTHONPATH": "Python", **os.environ},
    )
    out = proc.stdout + proc.stderr
    perf = re.search(r"Perf:\s*[\d.]+\s*ms,\s*([\d.]+)\s*TFlops", out)
    passed = "-> PASS" in out
    tflops = float(perf.group(1)) if perf else 0.0
    results.append((atom, tflops, "PASS" if passed else "FAIL"))
    print(f"  {tflops:.1f} TFLOPS  {'PASS' if passed else 'FAIL'}")

print(f"\n{'='*60}\nSUMMARY ({M}x{N}x{K}, {ARCH})\n{'='*60}")
print(f"{'Atom (MxNxK)':<16}{'TFLOPS':<12}{'Status'}")
print("-" * 40)
for atom, tflops, status in sorted(results, key=lambda r: -r[1]):
    print(f"{atom:<16}{tflops:<12.1f}{status}")
```

Sample run on MI355X (gfx950) — note the >3x spread from atom choice alone:
```
Atom (MxNxK)    TFLOPS      Status
----------------------------------------
16x16x32        354.0       PASS
32x32x16        324.5       PASS
16x16x16        126.3       PASS
32x32x8         112.4       PASS
```

**Discussion Questions (AI-assisted):**
```
Prompt: "I swept the GEMM MFMA atom on gfx950 and got:
16x16x32 -> 354 TFLOPS, 32x32x16 -> 324, 16x16x16 -> 126, 32x32x8 -> 112.

Help me understand:
1. Why do the deeper-K atoms (x32, x16) so dramatically outperform x16/x8?
2. How does the MFMA atom's K depth relate to compute-to-issue ratio?
3. How does atom choice interact with register usage and occupancy?
4. When would a smaller atom ever win?"
```

---

## Day 7: Week 1 Review & Mini-Project

### Morning: Concepts Review
**Self-quiz (without AI first, then verify with AI):**
1. Draw the rocKE compilation pipeline from memory
2. Explain SSA IR in your own words
3. What's the difference between `b.scf_if()` and Python `if`?
4. What are the three main factors affected by tiling? (registers, LDS, parallelism)

### Afternoon: Mini-Project
**Build a fused kernel:** Element-wise add + ReLU

Start from your `vector_add.py` and fuse a ReLU onto the sum before storing.
The cleanest way is `b.fmax(x, zero)`; you can also do it with a compare +
select. Both are verified bit-exact on gfx950.

```python
"""Fused Add-ReLU: C = ReLU(A + B)"""
# Inside the per-element scf_if body, after computing the sum:
zero = b.const_f32(0.0)            # hoist this once before the loop
s = b.fadd(a_val, b_val)
r = b.fmax(s, zero)                # ReLU = max(x, 0)
# (equivalent: r = b.select(b.cmp_gt(s, zero), s, zero))
b.global_store(C, idx, r)

# Reference check:  ref = torch.relu(A + B)
```

**AI Code Review:**
```
Prompt: "Review my fused add-ReLU kernel [paste code].
Check for:
1. Correct bounds checking
2. Efficient memory access patterns
3. Proper use of SSA IR
4. Any potential optimizations"
```

**Week 1 Deliverable:** Working vector add and fused add-ReLU kernels with correctness verification

---

# Bridge: Writing an Instance Using Primitives

So far your kernels have been one-off scripts: a `build_*` function plus an
inline launch harness. Every shipped kernel in `rocke/instances/` follows a
more structured four-part pattern, and learning it now is what lets you read the
GEMM/attention instances in Week 2 and add your own knobs in Week 3.

An **instance** is the packaging convention that turns "a builder" into "a
reusable, dispatchable kernel":

| Part | Role | Example in the shipped tree |
|------|------|-----------------------------|
| **Spec dataclass** | the compile-time knobs that name one concrete kernel | `Reduce2DSpec`, `UniversalGemmSpec` |
| **Validity gate** | `is_valid_spec(spec) -> (ok, why)` rejects illegal configs *before* codegen | `reduce.is_valid_spec`, `GemmPipelinePolicy.validate` |
| **Builder** | `build_*(spec) -> KernelDef`, composing primitives + helpers | `build_reduce2d`, `build_universal_gemm` |
| **Signature + grid** | the launch contract: kernarg ABI + grid shape | `reduce2d_signature` / `reduce2d_grid` |

Study the canonical small-op instance first — it's the simplest end-to-end
example of all four parts:
```bash
sed -n '1,120p' Python/rocke/instances/common/reduce.py   # docstring + spec
cat dsl_docs/instances/small_ops.md          # the family doc
```

## Exercise: build a `row-sum` instance from scratch

Goal: `Y[m] = sum_n X[m, n]` for an `(M, N)` f32 matrix, one CTA per row. This
uses only the primitives from Week 1 (`thread_id_x`, `block_id_x`,
`global_load_f32`, `global_store`, `scf_if`, `cmp_eq`) plus **two helpers** you
haven't met yet:

- `make_lds_view(b, dtype=F32, shape=(BS,))` — allocates a workgroup-lifetime
  LDS (shared memory) buffer; `.base` is the pointer you pass to reducers.
- `block_lds_reduce(b, val, lds, tid, block_size=BS, combine="sum")` — folds one
  f32 partial per lane into a single value via an LDS tree, broadcast to every
  lane. Supported combiners: `sum`, `max`, `min`, `prod`.

The two-stage structure (per-thread partial → workgroup fold) is exactly what
`reduce.py` does; we just hand-roll stage 1 instead of using the distribution
machinery. Verified bit-exact-enough on gfx950 (`max_abs_diff ~2e-5`).

```python
"""A teaching instance: row-sum  Y[m] = sum_n X[m, n]."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
from rocke.core.arch import ArchTarget
from rocke.core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from rocke.helpers import compile_kernel
from rocke.helpers.reduction import block_lds_reduce
from rocke.helpers.spec import SignatureBuilder, ceil_div_grid
from rocke.helpers.tensor_view import make_lds_view
from rocke.runtime.hip_module import get_device_arch
from rocke.runtime.launcher import (
    KernelLauncher, LaunchConfig, synchronize_and_release,
)


# 1. Spec: the knobs that define one concrete kernel.
@dataclass(frozen=True)
class RowSumSpec:
    n: int                 # row length (compile-time tile of the N axis)
    block_size: int = 256
    name: str = "my_rowsum"

    def kernel_name(self) -> str:
        return f"{self.name}_N{self.n}_b{self.block_size}"


# 2. Validity gate: reject specs the builder can't honor.
def is_valid_spec(spec: RowSumSpec) -> Tuple[bool, str]:
    if spec.block_size not in (64, 128, 256, 512, 1024):
        return False, f"block_size {spec.block_size} not in supported set"
    if spec.n <= 0:
        return False, "n must be positive"
    # Keep this teaching kernel simple: require the row to tile evenly so every
    # thread owns the same number of columns and we need no tail handling.
    if spec.n % spec.block_size != 0:
        return False, f"n ({spec.n}) must be a multiple of block_size ({spec.block_size})"
    return True, "ok"


# 3. Builder: spec -> KernelDef, composing primitives + one helper.
def build_rowsum(spec: RowSumSpec) -> KernelDef:
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid rowsum spec: {why}")

    BS, N = spec.block_size, spec.n
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    _M = b.param("M", I32)   # ABI param (row count); grid encodes it
    _N = b.param("N", I32)   # ABI param (row length); N is the spec tile

    tid = b.thread_id_x()
    row = b.block_id_x()
    row_base = b.mul(row, b.const_i32(N))          # X is row-major (M, N)

    # Stage 1 (per thread): grid-stride sum over this thread's columns, in f32.
    # Unrolled at build time over N // BS chunks (the spec guarantees it
    # divides evenly), so no per-lane bounds check is needed.
    acc = b.const_f32(0.0)
    for chunk in range(N // BS):
        col = b.add(tid, b.const_i32(chunk * BS))
        idx = b.add(row_base, col)
        acc = b.fadd(acc, b.global_load_f32(X, idx))

    # Stage 2 (workgroup): fold the BS partials through an LDS tree.
    lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="rowsum_lds").base
    total = block_lds_reduce(b, acc, lds, tid, block_size=BS, combine="sum")

    # Lane 0 writes the row result.
    with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
        b.global_store(Y, row, total)

    return b.kernel


# 4. Signature + grid: the launch contract.
def rowsum_signature(spec: RowSumSpec):
    return (
        SignatureBuilder()
        .ptr("X", "f32").ptr("Y", "f32")
        .scalar("M", "i32").scalar("N", "i32")
        .build()
    )


def rowsum_grid(m: int) -> Tuple[int, int, int]:
    return ceil_div_grid((m, 1))   # one CTA per row


if __name__ == "__main__":
    arch = get_device_arch() or "gfx950"
    M, N = 128, 4096
    spec = RowSumSpec(n=N, block_size=256)

    art = compile_kernel(build_rowsum(spec), arch=arch)
    launcher = KernelLauncher(
        hsaco=art.hsaco, kernel_name=art.kernel_name,
        signature=rowsum_signature(spec),
    )

    torch.manual_seed(0)
    X = torch.randn(M, N, dtype=torch.float32, device="cuda")
    Y = torch.empty(M, dtype=torch.float32, device="cuda")
    launcher(
        {"X": X, "Y": Y, "M": M, "N": N},
        config=LaunchConfig(grid=rowsum_grid(M), block=(spec.block_size, 1, 1)),
    )
    synchronize_and_release()

    diff = (Y - X.sum(-1)).abs().max().item()
    print(f"row-sum {M}x{N}  max_abs_diff={diff:.3e}")
    assert diff < 1e-2
    print("PASS")
```

**What to notice (and how it maps to the real instances):**
- `compile_kernel(kernel, arch=...)` is the one-shot IR→LLVM→HSACO pipeline; its
  `art.hsaco` / `art.kernel_name` feed `KernelLauncher` directly. This is the
  same path `build_universal_gemm` uses.
- The **two trailing ABI params** `M`, `N` aren't read in the body — the row
  index comes from the grid and `N` is baked in as the spec tile — but they must
  appear so the kernarg signature matches what the launcher packs. (You saw the
  inverse of this footgun in Week 2's `gemm_multi_d` ABI note.)
- `block_lds_reduce` is the reusable primitive behind LayerNorm, RMSNorm, and
  attention's online-softmax fold — swapping `combine="max"` turns this into a
  row-max kernel for free.

**Extensions (try these):**
1. Add `op: Literal["sum","max","mean"]` to the spec; pick the identity element
   and `combine` per op (mirror `reduce._combine_scalar`). For `mean`, divide
   `total` by `N` with `b.fmul(total, b.rcp(b.const_f32(float(N))))`.
2. Relax the `N % BS == 0` gate by guarding the tail chunk. (Hint: a value
   assigned inside `scf_if` doesn't escape the region in SSA — instead, load
   `0.0` for out-of-range lanes via `b.select(b.cmp_lt(col, c_n), val, zero)`
   and always `fadd`.)
3. Compare your IR to the shipped kernel: `print_ir(build_rowsum(spec))` vs
   `print_ir(build_reduce2d(Reduce2DSpec(n_per_block=N, op="sum")))` and see what
   the distribution-driven version buys (warp-XOR butterfly vs the plain LDS tree).

---

# Week 2: Advanced Patterns & Debugging

## Learning Objectives
- Understand and implement C-shuffle for GEMM epilogues
- Write epilogue fusion patterns (bias, activation)
- Study attention kernel architecture
- Debug correctness issues systematically
- Profile and identify performance bottlenecks

---

## Day 8-9: C-Shuffle Pattern

### Morning: Understanding C-Shuffle
**Reading:**
- `dsl_docs/instances/gemm.md` (search for "cshuffle")
- Study the reference implementation:
```bash
# The cshuffle epilogue lives in the universal-GEMM builder. Grep for it:
grep -n "cshuffle" Python/rocke/instances/common/gemm_universal.py
```

**Key Concept:** C-shuffle optimizes the GEMM epilogue by:
1. Computing tile of C in thread-distributed (MFMA-shaped) registers
2. Staging C through **LDS** so threads can re-read it in a layout that gives
   coalesced global writes
3. Providing the hook where fused epilogue D-operands (bias/residual/gate) are
   applied in registers before the store (see Day 10-11)

In rocKE the epilogue is **not** a separate class — it's selected by the
`epilogue` string on `TraitSpec` (`"default"` = vectorised direct stores,
`"cshuffle"` = LDS-staged). You'll see the difference directly in the IR: the
cshuffle path emits noticeably more `smem_store`/`smem_load` ops.

**Visualization Exercise:**
```
Prompt: "Explain C-shuffle in GEMM with a concrete example.
Use a 64x64 output tile with 4 warps. Show:
1. Initial per-thread ownership of C elements
2. How cross-lane shuffles reorganize data
3. Final memory write pattern
4. Why this is faster than naive register→global writes"
```

### Afternoon: Implement C-Shuffle Study
**Exercise 5: Trace through C-shuffle**

The universal-GEMM spec is **nested** (`TileSpec` + `TraitSpec` + `DataSpec`
wrapped in `UniversalGemmSpec`) and carries no M/N/K — those are runtime launch
dims. `build_universal_gemm(spec, arch=...)` returns a single `KernelDef`. This
script builds both epilogues and counts their LDS ops so you can see cshuffle's
LDS staging:

```python
"""Study C-shuffle by comparing its IR to the default epilogue."""
from rocke.core.arch import ArchTarget
from rocke.core.ir_print import print_ir
from rocke.instances.common.gemm_universal import (
    DataSpec, TileSpec, TraitSpec, UniversalGemmSpec, build_universal_gemm,
)
from rocke.runtime.hip_module import get_device_arch


def make_spec(arch, epilogue):
    tile = TileSpec(
        tile_m=64, tile_n=64, tile_k=32,
        warp_m=2, warp_n=2, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
    )
    trait = TraitSpec(
        pipeline="mem", scheduler="intrawave", epilogue=epilogue,
        pad_m=True, pad_n=True, pad_k=True,
    )
    data = DataSpec(
        dtype_a="fp16", dtype_b="fp16", dtype_c="fp16",
        dtype_acc="fp32", layout="RCR",
    )
    return UniversalGemmSpec(
        name=f"ugemm_{arch}", tile=tile, trait=trait, data=data,
        wave_size=ArchTarget.from_gfx(arch).wave_size,
    )


arch = get_device_arch() or "gfx950"
for epilogue in ("default", "cshuffle"):
    spec = make_spec(arch, epilogue)
    ir = print_ir(build_universal_gemm(spec, arch=arch))
    n_smem = ir.count("smem_store") + ir.count("smem_load")
    print(f"epilogue={epilogue:<9} smem_ops={n_smem}")
```

Sample output on gfx950 (cshuffle stages C through LDS, so more smem ops):
```
epilogue=default   smem_ops=10
epilogue=cshuffle  smem_ops=28
```

**Analysis Task:**
1. Dump the full `print_ir(...)` for the cshuffle kernel and find the
   `smem_store` / `smem_load` ops that form the C staging
2. Locate the `barrier` between the LDS write and the re-read
3. Identify where the global stores happen *after* the shuffle
4. Compare the op counts to the `default` epilogue and explain the trade-off
   (extra LDS traffic + a barrier vs. better-coalesced global writes)

---

## Day 10-11: Epilogue Fusion

### Morning: Multi-D Epilogue Fusion (bias / residual / gate)

The real fused-epilogue instance is `gemm_multi_d`. It wraps a
`UniversalGemmSpec` (which **must** use `epilogue="cshuffle"` — that's where the
fused hook lives) and adds *D operands* that are combined into the accumulator
in registers before the store:

- `E = A*B + D0` — bias / residual add (`("D0", "add")`)
- `E = (A*B + D0) * D1` — bias + gate (`("D0","add"), ("D1","mul")`)

Each D operand is `(param_name, op)` with `op` in `{"add", "mul"}`.

**Study reference implementation:**
```bash
sed -n '1,40p' Python/rocke/instances/common/gemm_multi_d.py   # docstring
grep -n "GemmMultiDSpec\|d_operands" Python/rocke/instances/common/gemm_multi_d.py
```

> **Kernarg ABI gotcha** (called out in the source): the kernel param order is
> `A, B, C, M, N, K, [strides], D0, D1, ...` — D pointers come **last**, not the
> textbook `A, B, D0..., C`. Always build your launcher signature with
> `gemm_multi_d_signature(spec)` so the kernarg packing matches; hand-ordering
> it will read C from D0's slot and fault on the first store.

**Exercise 6: Fused bias GEMM (E = A*B + D0)** — verified bit-exact on gfx950:

```python
"""GEMM with fused bias via the multi-D epilogue:  E = A*B + D0."""
import torch
from rocke.core.arch import ArchTarget
from rocke.helpers import compile_kernel
from rocke.instances.common.gemm_universal import (
    DataSpec, TileSpec, TraitSpec, UniversalGemmSpec,
)
from rocke.instances.common.gemm_multi_d import (
    GemmMultiDSpec, build_gemm_multi_d,
    gemm_multi_d_signature, gemm_multi_d_grid,
)
from rocke.runtime.hip_module import get_device_arch
from rocke.runtime.launcher import (
    KernelLauncher, LaunchConfig, synchronize_and_release,
)

arch = get_device_arch() or "gfx950"
tile = TileSpec(
    tile_m=64, tile_n=64, tile_k=32,
    warp_m=2, warp_n=2, warp_k=1,
    warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
)
trait = TraitSpec(
    pipeline="mem", scheduler="intrawave", epilogue="cshuffle",  # required
    pad_m=True, pad_n=True, pad_k=True,
)
data = DataSpec(
    dtype_a="fp16", dtype_b="fp16", dtype_c="fp16",
    dtype_acc="fp32", layout="RCR",
)
base = UniversalGemmSpec(
    name=f"ugemm_{arch}", tile=tile, trait=trait, data=data,
    wave_size=ArchTarget.from_gfx(arch).wave_size,
)
spec = GemmMultiDSpec(base=base, d_operands=(("D0", "add"),))

art = compile_kernel(build_gemm_multi_d(spec, arch=arch), arch=arch)
launcher = KernelLauncher(
    hsaco=art.hsaco, kernel_name=art.kernel_name,
    signature=gemm_multi_d_signature(spec),  # canonical kernarg order
)

M = N = K = 512
torch.manual_seed(0)
# RCR layout: A (M,K) row-major, B (N,K) col-major, C/D0 (M,N) row-major.
A = torch.randint(-3, 4, (M, K), dtype=torch.float16, device="cuda")
B = torch.randint(-3, 4, (N, K), dtype=torch.float16, device="cuda")
D0 = torch.randint(-3, 4, (M, N), dtype=torch.float16, device="cuda")
C = torch.zeros(M, N, dtype=torch.float16, device="cuda")

launcher(
    {"A": A, "B": B, "C": C, "M": M, "N": N, "K": K, "D0": D0},
    config=LaunchConfig(grid=gemm_multi_d_grid(spec, M, N),
                        block=(spec.base.block_size, 1, 1)),
)
synchronize_and_release()

ref = (A.float() @ B.float().t() + D0.float()).half()
diff = (C.float() - ref.float()).abs().max().item()
print(f"GEMM+bias  {M}x{N}x{K}  max_abs_diff={diff:.3e}")
assert diff == 0.0
print("PASS")
```

### Afternoon: Bias + Gate, and where activation lives
**Exercise 7: Two D operands (bias + gate)**

Extend Exercise 6 to `E = (A*B + D0) * D1` by passing
`d_operands=(("D0", "add"), ("D1", "mul"))`, adding a `D1` tensor of shape
`(M, N)`, including `"D1"` in the launch dict, and updating the reference to
`((A@B.t() + D0) * D1)`.

> **Scope note**: `gemm_multi_d` supports `add`/`mul` D-operands only — it does
> **not** bake in an activation like ReLU or quantization. Pointwise
> activations are a separate elementwise pass; see the op menu in
> `examples/common/ck_tile_parity.py` (`relu`, `gelu`, `swish`, `sigmoid`, ...).
> Fusing an activation *into* the GEMM epilogue is a real extension project —
> revisit it in Week 3 once you can add your own knobs.

**Study the elementwise / quantization ops:**
```bash
# Activation + quant ops available as standalone elementwise kernels:
python Python/rocke/examples/common/ck_tile_parity.py --op elementwise
cat Python/rocke/instances/common/add_rmsnorm2d_rdquant.py   # fused quant ref
```

**AI-Assisted Design:**
```
Prompt: "In rocKE, gemm_multi_d fuses add/mul D-operands into the cshuffle
epilogue but has no activation hook. I want E = relu(A*B + D0) in one kernel.

Design the extension:
1. Where in the cshuffle epilogue would the relu go (register stage)?
2. What's the data flow: acc -> +D0 -> relu -> LDS stage -> store?
3. How would I add an 'activation' field to the spec?
4. How to verify against torch.relu(A @ B.t() + D0)?"
```

---

## Day 12-13: Attention Mechanisms

### Morning: Attention Architecture Study
**Reading:**
- `dsl_docs/instances/attention.md`
- Study example implementations:
```bash
ls Python/rocke/examples/gfx1151/attention/
cat Python/rocke/examples/gfx1151/attention/fmha_singlewave.py
```

**Key Attention Patterns:**
1. **Single-wave**: One warp processes entire sequence
2. **Multi-wave**: Parallel reduction across warps
3. **Pipelined**: Overlap compute with memory
4. **Tiled**: 2D/3D tiling for large sequences

**Conceptual Exercise:**
```
Prompt: "Explain Flash Attention's tiling strategy. For a
query length Q=2048, key length K=4096, and head dimension D=64:

1. How should I tile Q and K dimensions?
2. What's the memory footprint per tile?
3. When do I need online softmax vs standard softmax?
4. How does causal masking affect tiling?"
```

### Afternoon: Run Attention Examples
**Exercise 8: Attention parameter sweep**

The fwd verify harness uses these flags: `--seqlen-q`, `--seqlen-k`,
`--head-size`, `--heads`, `--kv-heads` (0 = MHA), `--batch`, `--causal`. It
checks correctness against a reference (it does not print TFLOPS).

```bash
# Single config (verifies correctness, prints max_abs_diff + PASS/FAIL)
python -m rocke.examples.common.fmha_fwd_verify_hip \
    --arch gfx950 --batch 2 --seqlen-q 1024 --seqlen-k 1024 \
    --heads 8 --head-size 64

# Vary sequence length
for s in 512 1024 2048; do
    echo "Testing seqlen=$s"
    python -m rocke.examples.common.fmha_fwd_verify_hip \
        --arch gfx950 --batch 2 --seqlen-q $s --seqlen-k $s \
        --heads 8 --head-size 64
done
```
Expected: each run prints `max_abs_diff` ~3e-5 (well under the 2e-2 tol) and
`-> PASS`. For *performance* scaling, use the dedicated attention parity /
benchmark harness in `examples/gfx950/attention/` (see `examples/index.md`),
which routes both backends through one HIP-event timer.

**Analysis:**
- How does the GROUP of GQA/MQA (`--kv-heads < --heads`) change the K/V read volume?
- What's the memory bottleneck as `--seqlen-k` grows?
- When does compute (the two matmuls) vs memory (K/V loads) dominate?

---

## Day 14: Debugging Correctness & Performance

### Morning: Correctness Debugging
**Systematic debugging workflow:**

1. **Isolate the bug:**
```python
# Add verification at each stage
print("Input A:", A[:10])  # Check inputs
print("Output C:", C[:10])  # Check outputs
print("Expected:", ref[:10])  # Check reference

# Verify intermediate values in kernel
# (use printf debugging or HIP debugging tools)
```

2. **Common correctness issues:**
- Index calculation errors → out-of-bounds access
- Race conditions → improper synchronization
- Numerical precision → mixed-precision handling
- Tail handling → incomplete work items

**Exercise 9: Debug a broken kernel**

```python
"""This kernel has a bug - find and fix it."""
def broken_matmul(M, N, K):
    b = IRBuilder("broken_matmul")
    # ... kernel code with subtle index bug
    # Hint: The bug is in the K-loop accumulation

    # Your task:
    # 1. Run and observe wrong results
    # 2. Add debug prints
    # 3. Find the bug
    # 4. Fix it
    # 5. Verify correctness
```

### Afternoon: Performance Debugging
**Profiling workflow:**

1. **High-level timing:** `time_launches` takes a **callable** (capture your
   launcher + args in a lambda) and keyword-only `warmup`/`iters`. It returns
   the **average per-call time in milliseconds** as a float (not a dict).
   Verified on gfx950.
```python
from rocke.runtime.launcher import time_launches

# Capture the launch in a zero-arg callable.
avg_ms = time_launches(
    lambda: launcher(args, config=config),
    warmup=10,
    iters=100,
)
print(f"Average time: {avg_ms:.5f} ms")

# Compute TFLOPS for a GEMM (flops = 2*M*N*K).
flops = 2 * M * N * K
tflops = flops / (avg_ms / 1e3) / 1e12
print(f"Performance: {tflops:.1f} TFLOPS")

# (For a memory-bound op like vector add, bytes/time is the useful metric:
#  gbps = (3 * N * 4) / (avg_ms / 1e3) / 1e9  # 2 reads + 1 write, f32)
```

2. **Identify bottlenecks:**
```bash
# Use rocprof for detailed profiling
rocprof --stats python my_kernel.py

# Check key metrics:
# - Wave occupancy
# - Memory bandwidth utilization
# - Instruction mix (VALU, MFMA, SALU)
# - LDS bank conflicts
```

**Exercise 10: Performance analysis**

Profile your GEMM from Week 1 and identify:
- Theoretical peak TFLOPS for your GPU
- Achieved TFLOPS
- Main bottleneck (memory or compute?)
- Occupancy percentage

**AI-Assisted Analysis:**
```
Prompt: "Here's my profiling output [paste rocprof stats].
My GPU is MI300X with peak 1.3 PFLOPS FP16.

Analyze:
1. What % of peak am I achieving?
2. Is this memory-bound or compute-bound?
3. What's limiting my occupancy?
4. What should I optimize first?"
```

**Week 2 Deliverable:**
- Working GEMM with C-shuffle epilogue (via `epilogue="cshuffle"`)
- Fused bias / bias+gate epilogue (via `gemm_multi_d`)
- Attention kernel understanding (run + verify existing examples)
- A timing measurement using `time_launches` (TFLOPS or GB/s)

---

# Week 3+: Agentic Optimization

## Learning Objectives
- Use AI skills for systematic optimization
- Apply hardware counter profiling
- Build parameterized optimization knobs
- Iterate on performance tuning
- Contribute optimizations back to the codebase

---

## Week 3 Structure: Iterative Optimization Cycle

Week 3 and beyond follows an **optimization sprint** model:
1. **Baseline** → Measure current performance
2. **Profile** → Identify bottleneck with hardware counters
3. **Hypothesize** → Form optimization theory
4. **Implement** → Add optimization knob
5. **Measure** → Verify improvement
6. **Iterate** → Repeat with next bottleneck

---

## Day 15-16: Setup Optimization Infrastructure

The optimization tooling lives under
`dsl_docs/optimization/`. Two things to anchor on:

1. **The runbook** — `optimization/optimization_runbook.md`. Read "The Loop
   (one-page summary)" first; it's the discipline the rest of Week 3 follows.
2. **The probes** — `optimization/utilities/tools/dsl_probes/`. These are real,
   runnable tools (verified on gfx950): `probe_occupancy.py`,
   `probe_isa_inspect.py`, `probe_config_sweep.py`, `probe_targeted_bench.py`,
   `probe_rocprof_single.py`.

### Morning: Resource & ISA inspection

Start by looking at what a kernel *costs* (registers, LDS, occupancy) and what
it *compiled to* (ISA mix) — before touching counters.

**Exercise 11: Occupancy + ISA probes** (both ship `--demo` builders so you can
run them immediately):

```bash
cd dsl_docs/optimization/utilities/tools/dsl_probes

# Resource/occupancy limiters for the implicit-GEMM conv demo.
python probe_occupancy.py --demo implicit_gemm --arch gfx950

# ISA instruction-mix + waitcnt patterns for the attention demo.
python probe_isa_inspect.py --demo attention_tiled_2d --mcpu gfx950
```

`probe_occupancy.py` output looks like this (real gfx950 run) — the `limit`
column tells you what's capping occupancy:
```
label                 vgpr  agpr  sgpr spill   lds B  waves/CU  wg/CU   limit
baseline_mem_sync       68    16    36     0   26624        24      6     LDS
async_dma               56    16    38     0   40960        16      4     LDS
```

**Key resources to understand (from the probe output):**
- `vgpr` / `agpr` / `sgpr` — register file pressure; high VGPR caps waves/CU
- `spill` — nonzero means register spilling (almost always bad)
- `lds B` — LDS bytes per workgroup; often the occupancy limiter (see `limit`)
- `waves/CU`, `wg/CU` — achieved occupancy estimate
- `limit` — which resource (VGPR/LDS/...) is the binding constraint

`probe_isa_inspect.py` reports the instruction histogram (`v_mfma`, `ds_read`,
`vmem_load/store`) and the **waitcnt patterns**, which reveal memory/LDS stalls.

### Afternoon: The runbook + AI skills

The `optimization/utilities/skills/` folder holds focused, code-adjacent
playbooks. Read the ones matching your kernel:

```bash
cd dsl_docs/optimization
sed -n '1,60p' optimization_runbook.md          # The Loop + lever catalog intro
cat utilities/skills/gemm-optimization-rocke.md  # GEMM-specific levers
cat utilities/skills/lds-optimization-rocke.md   # LDS layout / bank conflicts
cat utilities/skills/isa-inspection-rocke.md     # reading the ISA probe output
cat utilities/skills/prefetch-data-load-rocke.md # prefetch / async load levers
```

**AI-Assisted Optimization Setup** — feed the AI your *real* probe output, not
guesses:
```
Prompt: "I'm optimizing a GEMM on gfx950 (MI355X). Here is the occupancy probe:
[paste probe_occupancy.py table]
and the ISA mix:
[paste probe_isa_inspect.py histogram + waitcnt patterns]

Using the rocKE optimization runbook's loop:
1. What is the binding occupancy limiter, and is it the right thing to attack?
2. What do the waitcnt patterns say about memory vs LDS stalls?
3. Which single lever from the GEMM skill should I try first, and why?
4. How do I verify correctness + re-measure to confirm it moved?"
```

---

## Day 17-18: Optimization Iteration Loop

The universal-GEMM harness exposes three real, tunable levers, so you can
practice the runbook loop end-to-end without writing a new builder first:

| Lever        | CLI flag       | Values (gfx950 fp16)                  | What it changes |
|--------------|----------------|---------------------------------------|-----------------|
| MFMA atom    | `--warp-tile`  | `16x16x16/32`, `32x32x8/16`           | compute shape per wave |
| Pipeline     | `--pipeline`   | `mem`, `compv3`, `compv4`             | software pipelining / double-buffer |
| Epilogue     | `--epilogue`   | `default`, `cshuffle`                 | LDS-staged C write-out |

### Iteration 1: Establish a stable baseline

The harness already prints `Perf: <ms>, <TFLOPS>` and a `PASS/FAIL` correctness
gate. **Correctness first** (runbook rule: never report speed on a failing
kernel), then record the baseline number.
```bash
python -m rocke.examples.common.universal_gemm_verify \
    --arch gfx950 --m 2048 --n 2048 --k 2048 \
    --warp-tile 16x16x32 --pipeline mem --epilogue default
# -> Perf: ~0.048 ms, ~354 TFlops ... -> PASS   (your baseline)
```

### Iteration 2: Change one lever, re-verify, re-measure

**Exercise 12: Real knob sweep.** Each variant changes exactly one lever off the
baseline, then combines the winners — exactly the runbook loop. Create
`gemm_knob_sweep.py`:
```python
"""Sweep the real GEMM optimization knobs and rank by TFLOPS."""
import os, re, subprocess, sys

PY = sys.executable
M = N = K = 2048
ARCH = "gfx950"
BASE = {"warp_tile": "16x16x32", "pipeline": "mem", "epilogue": "default"}
VARIANTS = [
    ("baseline", {}),
    ("atom=32x32x16", {"warp_tile": "32x32x16"}),
    ("pipeline=compv3", {"pipeline": "compv3"}),
    ("pipeline=compv4", {"pipeline": "compv4"}),
    ("epilogue=cshuffle", {"epilogue": "cshuffle"}),
    ("compv4+cshuffle", {"pipeline": "compv4", "epilogue": "cshuffle"}),
]

def run(cfg):
    proc = subprocess.run(
        [PY, "-m", "rocke.examples.common.universal_gemm_verify",
         "--arch", ARCH, "--m", str(M), "--n", str(N), "--k", str(K),
         "--warp-tile", cfg["warp_tile"], "--pipeline", cfg["pipeline"],
         "--epilogue", cfg["epilogue"]],
        capture_output=True, text=True,
        env={"PYTHONPATH": "Python", **os.environ})
    out = proc.stdout + proc.stderr
    m = re.search(r"Perf:\s*[\d.]+\s*ms,\s*([\d.]+)\s*TFlops", out)
    return (float(m.group(1)) if m else 0.0), ("-> PASS" in out)

results = []
for label, override in VARIANTS:
    tflops, passed = run({**BASE, **override})
    results.append((label, tflops, "PASS" if passed else "FAIL"))
    print(f"{label:<20} {tflops:7.1f} TFLOPS  {'PASS' if passed else 'FAIL'}")

print("\nRANKED:")
for label, tflops, status in sorted(results, key=lambda r: -r[1]):
    print(f"{label:<20} {tflops:7.1f} TFLOPS  {status}")
```

Sample run on MI355X (gfx950) — all variants stay bit-exact (`PASS`):
```
RANKED:
pipeline=compv3        359.8 TFLOPS  PASS
compv4+cshuffle        356.0 TFLOPS  PASS
pipeline=compv4        354.2 TFLOPS  PASS
epilogue=cshuffle      354.2 TFLOPS  PASS
baseline               353.6 TFLOPS  PASS
atom=32x32x16          324.6 TFLOPS  PASS
```
Note these deltas are small and noisy at this shape — that's a *real* and
important lesson: at 2K square the kernel is already near a local optimum, so
re-run the winners a few times and watch the spread before declaring a victory.
Bigger swings show up at other shapes and on memory-bound problems.

### Iteration 3: Confirm *why* it moved (ISA / resource diff)

Don't trust a number you can't explain. Use the probes from Day 15 to see what
actually changed between two variants:
```bash
cd dsl_docs/optimization/utilities/tools/dsl_probes
# Occupancy/resource limiter for each variant (drive with a small custom
# script per the probe's --help "Programmatic use" block), or for the shipped
# demos directly:
python probe_occupancy.py --demo implicit_gemm --arch gfx950
```

**AI-Assisted Analysis** (feed it the real ranked table + probe output):
```
Prompt: "I swept GEMM knobs on gfx950 at 2048^3 and got:
[paste your RANKED table]
plus this occupancy probe for two variants:
[paste probe_occupancy.py rows]

1. Are these deltas inside the run-to-run noise? How many re-runs to be sure?
2. Why might compv3 edge out compv4 at this shape but not at larger K?
3. Which variant should I lock in, and what shape should I re-test at?
4. What single new lever (not on the CLI) would I add next, and why?"
```

---

## Day 19-20: Building Optimization Knobs

### Morning: From CLI lever to spec knob

The CLI flags map directly onto fields of the nested `UniversalGemmSpec` you met
in Week 2 (`TileSpec.warp_tile_*`, `TraitSpec.pipeline`, `TraitSpec.epilogue`).
"Building a knob" means parameterizing your *builder* over those fields and
adding a validity gate so the sweep skips illegal combos. The real validity gate
already exists — `GemmPipelinePolicy.validate(target, spec)` — use it:

```python
"""Parameterized GEMM builder driven by a small knob dataclass."""
from dataclasses import dataclass
from rocke.core.arch import ArchTarget
from rocke.helpers import compile_kernel
from rocke.instances import GemmPipelinePolicy
from rocke.instances.common.gemm_universal import (
    DataSpec, TileSpec, TraitSpec, UniversalGemmSpec, build_universal_gemm,
)


@dataclass(frozen=True)
class GemmKnobs:
    warp_tile: tuple = (16, 16, 32)   # MFMA atom (m, n, k)
    pipeline: str = "mem"             # mem | compv3 | compv4
    epilogue: str = "default"         # default | cshuffle
    warp_m: int = 2
    warp_n: int = 2


def make_spec(knobs: GemmKnobs, arch: str) -> UniversalGemmSpec:
    wtm, wtn, wtk = knobs.warp_tile
    tile = TileSpec(
        tile_m=knobs.warp_m * 2 * wtm, tile_n=knobs.warp_n * 2 * wtn,
        tile_k=max(32, wtk),
        warp_m=knobs.warp_m, warp_n=knobs.warp_n, warp_k=1,
        warp_tile_m=wtm, warp_tile_n=wtn, warp_tile_k=wtk,
    )
    trait = TraitSpec(
        pipeline=knobs.pipeline, scheduler="intrawave", epilogue=knobs.epilogue,
        pad_m=True, pad_n=True, pad_k=True,
    )
    data = DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16",
                    dtype_acc="fp32", layout="RCR")
    return UniversalGemmSpec(
        name=f"ugemm_{arch}", tile=tile, trait=trait, data=data,
        wave_size=ArchTarget.from_gfx(arch).wave_size,
    )


def build_if_valid(knobs: GemmKnobs, arch: str = "gfx950"):
    """Return a compiled artifact, or None if the spec is illegal for the arch."""
    spec = make_spec(knobs, arch)
    res = GemmPipelinePolicy().validate(ArchTarget.from_gfx(arch), spec)
    if not res.ok:
        return None  # skip illegal combos (e.g. bad atom/tile geometry)
    return compile_kernel(build_universal_gemm(spec, arch=arch), arch=arch)
```

**Exercise 13: Knob-space search with a validity gate**
```python
"""Enumerate the knob space, skip illegal combos, build the legal ones."""
import itertools

ARCH = "gfx950"
atoms = [(16, 16, 16), (16, 16, 32), (32, 32, 8), (32, 32, 16)]
pipelines = ["mem", "compv3", "compv4"]
epilogues = ["default", "cshuffle"]

legal, skipped = 0, 0
for atom, pipe, epi in itertools.product(atoms, pipelines, epilogues):
    knobs = GemmKnobs(warp_tile=atom, pipeline=pipe, epilogue=epi)
    art = build_if_valid(knobs, ARCH)
    if art is None:
        skipped += 1
        continue
    legal += 1
    print(f"OK  atom={atom} pipe={pipe:<7} epi={epi:<8} -> {art.kernel_name}")
print(f"\n{legal} legal / {skipped} skipped (gated out by GemmPipelinePolicy)")
```
This builds every legal combo (validity-gated) so you can hand the artifacts to
a timing loop. For actual benchmarking, reuse the subprocess `Perf:` parser from
Exercise 12, or drive `probe_targeted_bench.py` programmatically (see its
`--help`). Both are verified on gfx950.

### Afternoon: AI-Guided Tuning
**Use AI to narrow the search space** before you spend GPU time on it:

```
Prompt: "I'm tuning GEMM on gfx950. My legal knob space is:
- warp_tile (MFMA atom): 16x16x16, 16x16x32, 32x32x8, 32x32x16
- pipeline: mem, compv3, compv4
- epilogue: default, cshuffle
(GemmPipelinePolicy.validate gates out illegal tile/atom geometry.)

From Week 1 I know deeper-K atoms (x32/x16) far outperform x8 at 2048^3.

1. Which knobs likely interact (e.g. pipeline x epilogue) vs are independent?
2. Given the atom result, which 4-6 combos are worth timing first?
3. What second shape should I confirm the winner at to avoid overfitting 2048^3?
4. What's a real new knob NOT on this list I could add to the builder next?"
```

---

## Day 21: Week 3 Capstone Project

### Full Optimization Sprint
**Project: Optimize a real-world kernel end-to-end**

Choose one of:
1. **GEMM for specific shape** (e.g., LLaMA 70B shapes)
2. **Grouped convolution** (e.g., ResNet bottleneck)
3. **Attention** (e.g., GPT-style causal attention)

**Sprint checklist:**
- [ ] Correctness baseline (PASS) + a stable timing baseline (`time_launches`
      or the harness `Perf:` line)
- [ ] Resource/ISA inspection with the `dsl_probes/` tools
- [ ] Identify the binding bottleneck (occupancy limiter + waitcnt pattern)
- [ ] Sweep 3+ knobs one-at-a-time, re-verifying correctness each time
- [ ] Validity-gated knob-space search for the best combo
- [ ] Explain the winning delta with an ISA/resource diff
- [ ] Document learnings (consider adding to `empirical-case-studies.md`)

**AI Pair Programming:**
Use AI throughout the sprint for:
- Code review of each optimization
- Debugging performance regressions
- Suggesting next optimization to try
- Validating correctness of changes

**Example AI workflow:**
```
# After each change:
Prompt: "I just added [optimization X]. Here's the code [paste].
Before: 520 TFLOPS
After: 485 TFLOPS (regression!)

Debug:
1. What could cause this regression?
2. Check my code for issues
3. Suggest how to fix it
4. Should I keep or revert this change?"
```

---

## Week 4+: Advanced Topics & Specialization

After completing Week 3, choose your specialization path:

### Path A: Instance Development
**Focus:** Building production-quality kernel instances
- Study `instances/` implementations in depth
- Contribute new instances (e.g., new fusion patterns)
- Build dispatcher logic for kernel selection
- Create comprehensive test suites

**Reference:**
```bash
cat dsl_docs/instances/index.md          # how instances are organized
cat dsl_docs/development/extending.md     # adding a new instance
ls  Python/rocke/instances/common/                     # the shipped instance set
```

### Path B: Architecture Specialization
**Focus:** Deep expertise on specific AMD architectures
- Master one architecture (gfx942, gfx950, or gfx1151)
- Learn ISA-level optimization
- Build architecture-specific optimization rules
- Create optimal configurations for your arch

**Reference:**
```bash
# Deep-dive on your architecture (arch docs live under optimization/arch/)
cat dsl_docs/optimization/arch/gfx950.md
cat dsl_docs/optimization/arch/gfx942.md
cat dsl_docs/optimization/utilities/skills/isa-inspection-rocke.md
```

### Path C: Fusion & Custom Ops
**Focus:** Advanced fusion patterns
- Multi-stage fusion (e.g., attention + MLP)
- Custom quantization schemes
- Mixed-precision strategies
- Novel epilogue patterns

**Reference:**
```bash
cat dsl_docs/fusion/overview.md
# Real fused instances to study (epilogue fusion, fused norm+quant, fused conv):
cat Python/rocke/instances/common/gemm_multi_d.py        # GEMM epilogue fusion
cat Python/rocke/instances/common/add_rmsnorm2d_rdquant.py  # fused norm + quant
ls  Python/rocke/instances/common/ | grep -E "fmha|fused|rmsnorm|moe"
```

### Path D: Tooling & Infrastructure
**Focus:** Developer experience improvements
- Better profiling tools
- Automated tuning frameworks
- Visualization tools for optimization
- CI/CD for kernel testing

---

## Ongoing Learning Resources

### Daily Practice
1. **Read one case study per week** from
   `optimization/utilities/skills/empirical-case-studies.md`
2. **Profile one kernel per week** with the `dsl_probes/` tools (occupancy + ISA)
3. **Implement one optimization** from `optimization/optimization_runbook.md`
4. **Review PRs** in the CK repository

### AI-Assisted Learning Sessions
**Weekly AI Study Session (1 hour):**
```
Prompt: "This week I'm studying [topic]. Quiz me on:
1. [Key concept 1]
2. [Key concept 2]
3. [Key concept 3]

Then give me a hands-on exercise that tests my understanding."
```

**Monthly AI Code Review:**
```
Prompt: "Review all the kernels I wrote this month [paste code].
Provide:
1. Code quality feedback
2. Optimization opportunities I missed
3. Best practices I should adopt
4. Comparison to production CK kernels"
```

### Community Engagement
1. **Join CK development discussions** - ask questions, share learnings
2. **Contribute optimizations back** - PRs to rocm-libraries
3. **Share case studies** - document your optimization wins
4. **Mentor others** - help newcomers with Week 1-2 content

---

## Success Metrics

### Week 1 Success
- [ ] Built and ran vector add kernel
- [ ] Understood SSA IR and compilation pipeline
- [ ] Experimented with GEMM tiling
- [ ] Can explain rocKE architecture to someone else

### Week 2 Success
- [ ] Implemented C-shuffle epilogue
- [ ] Built fused epilogue (bias + activation)
- [ ] Ran and understood attention examples
- [ ] Can debug correctness and profile performance

### Week 3 Success
- [ ] Used the `dsl_probes/` tools (occupancy + ISA inspection)
- [ ] Swept the real GEMM knobs (`--warp-tile` / `--pipeline` / `--epilogue`)
- [ ] Built a validity-gated knob-space search (`GemmPipelinePolicy.validate`)
- [ ] Explained a measured delta with an ISA/resource diff (not just a number)

### Long-term Success (Month 3+)
- [ ] Contributing optimized kernels to production
- [ ] Mentoring other kernel developers
- [ ] Achieving >60% of theoretical peak on target kernels
- [ ] Building new optimization techniques

---

## Troubleshooting Guide

### Common Issues

**Week 1:**
- *"HSACO compilation fails"* → Check ROCm version, verify `rocminfo`
- *"Wrong results in vector add"* → Check index calculations, bounds checking
- *"Import errors"* → Verify `PYTHONPATH=Python`

**Week 2:**
- *"C-shuffle crashes"* → Check warp count, tile size alignment
- *"Fusion incorrect"* → Verify broadcast semantics, data types
- *"Attention OOM"* → Reduce tile sizes, check LDS usage

**Week 3:**
- *"probe can't find llvm-readelf/objdump"* → ensure ROCm `bin/` is on `PATH`
- *"Optimization regresses"* → re-measure (deltas can be in the noise), then
  diff occupancy/ISA before/after with the probes
- *"Knob sweep too slow"* → gate illegal combos with `GemmPipelinePolicy.validate`
  before building, and sweep one lever at a time

### Getting Help

1. **Check documentation first:**
   ```bash
   grep -r "your error" dsl_docs/
   ```

2. **Use AI for error diagnosis:**
   ```
   Prompt: "I got this error: [paste error]
   Context: [what you were trying to do]
   Help me understand and fix it."
   ```

3. **Ask in community** with full context:
   - What you're trying to do
   - Minimal reproducing code
   - Error message
   - What you've tried

---

## Appendix: Quick Reference

### Essential Commands
```bash
# All commands run from rocKE/ with PYTHONPATH=Python
export PYTHONPATH=Python

# Verify setup
python -c "from rocke import *; print('Ready')"

# Run the static unit suite
python tests/test_rocke.py

# Build + verify a GEMM (prints TFLOPS + PASS/FAIL)
python -m rocke.examples.common.universal_gemm_verify \
    --arch gfx950 --m 2048 --n 2048 --k 2048

# Inspect resources / ISA of a kernel (shipped demos)
cd dsl_docs/optimization/utilities/tools/dsl_probes
python probe_occupancy.py   --demo implicit_gemm     --arch gfx950
python probe_isa_inspect.py --demo attention_tiled_2d --mcpu gfx950

# Profile a script with rocprof (counter capture)
rocprof --stats python my_kernel.py
```

### Key File Locations
```
Python/rocke/
├── core/ir.py                              # SSA IR and IRBuilder
├── core/lower_llvm.py                      # production LLVM lowering
├── core/ir_print.py                        # print_ir() (returns MLIR-style text)
├── helpers/                                # high-level authoring helpers
├── instances/common/gemm_universal.py      # universal GEMM (DataSpec/TileSpec/TraitSpec)
├── instances/common/gemm_multi_d.py        # fused-epilogue GEMM (bias/gate)
├── instances/common/                       # conv, attention, norm, moe instances
├── runtime/launcher.py                     # KernelLauncher, time_launches
├── examples/common/                        # runnable verify harnesses
└── dsl_docs/                               # this documentation

dsl_docs/
├── optimization/
│   ├── optimization_runbook.md             # the loop + lever catalog
│   ├── arch/                               # per-arch guides (gfx942, gfx950)
│   └── utilities/
│       ├── skills/                         # focused playbooks (gemm/lds/isa/...)
│       └── tools/dsl_probes/               # probe_occupancy, probe_isa_inspect, ...
├── instances/                              # instance deep-dives
└── primitives/                             # primitive reference
```

### AI Prompt Templates

**Concept Explanation:**
```
Explain [concept] in rocKE for AMDGPU. Include:
1. What it is and why it matters
2. How it works at the hardware level
3. rocKE API for using it
4. Common pitfalls to avoid
```

**Code Review:**
```
Review this rocKE kernel [paste code].
Check for:
1. Correctness (bounds, synchronization, races)
2. Performance (memory access, occupancy, parallelism)
3. Best practices (SSA usage, helper patterns)
4. Comparison to CK production code
```

**Optimization Strategy:**
```
I'm optimizing [kernel] for [arch].
Current: [X] TFLOPS out of [Y] peak
Bottleneck: [profile results]

Design optimization strategy:
1. Top 3 optimizations to try in order
2. Expected improvement for each
3. Implementation approach
4. How to validate each step
```

**Debugging:**
```
My kernel produces wrong results [or: crashes, or: slow].
Code: [paste]
Expected: [paste]
Got: [paste]
Profile: [paste]

Help me:
1. Identify the root cause
2. Suggest fixes
3. Prevent this in the future
```

---

## Conclusion

This 3-week program takes you from rocKE basics to advanced optimization. The key to success:

1. **Week 1**: Build strong foundations - understand the IR and compilation model
2. **Week 2**: Master patterns - C-shuffle, fusion, attention are building blocks
3. **Week 3+**: Iterate relentlessly - profile, optimize, measure, repeat

**Remember:** Optimization is a skill learned through practice. Every kernel you optimize teaches you something new. Use AI as a learning partner and force multiplier, but always validate and understand the suggestions.

**Next steps:** Pick your Week 1 Day 1 exercises and start building! The best way to learn kernel optimization is to write kernels.

Good luck! 🚀
