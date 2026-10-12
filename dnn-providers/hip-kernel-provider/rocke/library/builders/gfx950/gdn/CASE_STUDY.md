# GDN/KDA decode on gfx950 — load-first case study: register counts and relative speed

Measured scope: the single-token GDN/KDA decode kernels of
`kernels/gfx950/gdn_decode.py` before and after the warp-tiled emitter issued
every global load first (ALGORITHM.md §4.3). Section 4 records static
register, wave-per-SIMD and scratch counts per compiler. Section 6 records
relative device-time speedup versus the merge base at three batch sizes, with
its correctness gate, method and one regression. No absolute latency appears.
ALGORITHM.md states the rule and the exception; this file keeps the
measurements, which change with the compiler. The register, wave and scratch
numbers are static counts read from the gfx950 code object, not hardware
counters.

## 1. Mechanism

The AMDGPU scheduler does not hoist a load above earlier math or across the
`k_lane == 0` output-store branch, so the emitter issues every global load
before the math. That keeps the raw state tile, `q`, `k`, each owned row's `v`
and, when fused, their conv taps and weights and the norm inputs live at once.
The register count rises on tiles where those inputs used to be loaded and
consumed one row at a time, and falls on tiles where the earlier order kept
intermediate values live longer.

The exception (ALGORITHM.md §4.3) is keyed on the wave count: with `fuse_conv`
and one wave, a lane owns many V rows, so that tile keeps the interleaved order
for each row's `v` conv inputs, each state chunk and the norm inputs. Multi-wave
`fuse_conv` tiles hoist everything. Their per-lane cap is `512 / ceil(NW / 4)`
VGPRs, so tiles with a small `WTK` (wide `q`/`k` slices with their taps and
weights) or many waves can cross it.

## 2. Spec set

Tiles are `(num_warps, warp_threads_k, blocks_per_v_dim)`. bf16 I/O, bf16 and
`f32` state, `DK = DV = 128` unless stated:

- **Dispatcher tiles (114 tile and state-dtype pairs):** GDN at `Hk/Hv = 16/32`,
  every tile in `CONFIGURED_TILES`; KDA at `32/32`, the three tiles in
  `_TUNED_TILES_KDA`; each with bf16 and `f32` state. Dispatch accepts only a
  16-bit state today, so the `f32` half is not reachable from a request yet.
- **Auto-route fallback (4):** GDN `(1,1,1)` at `DK = DV = 64` and `192`, where
  `DEFAULT_TILE` `(2,16,8)` is illegal (ALGORITHM.md §4.6).
- **Fused tiles (228):** GDN and KDA at `16/16`, `BPV = 1`, every legal
  `(NW, WTK)` with `NW` in 1–16 and `WTK` in 1–32, with `fuse_conv`,
  `fuse_out_norm` (`rn` below) or both (`cv+rn`). The registry never builds a
  fused spec; callers pick these tiles themselves.

The merge base is `153bf493555`, the interleaved order; the head is the
load-first emitter.

## 3. How the counts are read

The Python lowerer (`ROCKE_BACKEND=python`) emits the IR, comgr compiles it, and
`rocke.analysis.isa.analyze_hsaco` reads the VGPR count and private-segment
(scratch) bytes from the code object. Waves per SIMD are modeled, not measured:
`min(8, 512 // VGPRs rounded up to 8)`. A change in that number below means the
count crossed a granule boundary in that model; a case can cross more than one.

## 4. Results

### 4.1 ROCm 7.1.0 (`llvm20`, comgr SHA-256 `b3e6941ac89a1ba6…`)

Dispatcher tiles (modeled waves per SIMD, merge base → head):

- fewer waves: GDN `(1,4,1)` bf16 (2→1), GDN `(1,16,1)` bf16 (2→1), GDN `(1,16,4)` bf16 (5→4), GDN `(1,16,8)` bf16 (8→7);
- more waves: GDN `(4,16,1)` bf16 (4→5), GDN `(16,16,1)` bf16 (7→8), GDN `(1,8,8)` f32 (5→6), GDN `(2,8,4)` f32 (5→6), GDN `(2,16,8)` f32 (7→8), GDN `(4,8,2)` f32 (5→6), GDN `(4,16,4)` f32 (7→8), GDN `(8,8,1)` f32 (5→6), GDN `(8,16,2)` f32 (7→8), GDN `(16,8,1)` f32 (5→6), GDN `(16,16,1)` f32 (7→8);
- spill: grows: GDN `(1,1,1)` bf16 (448→508 B), GDN `(1,1,1)` f32 (664→712 B).

Auto-route fallback GDN `(1,1,1)`: `DK = 64` bf16 200→206 VGPRs, 0→0 B; `DK = 192` bf16 512→512 VGPRs, 3456→3140 B; `DK = 64` f32 235→235 VGPRs, 0→0 B; `DK = 192` f32 512→512 VGPRs, 3416→3656 B.

Fused tiles (228 cases; modeled waves per SIMD, merge base → head):

- fewer waves (21 cases): `(2,8,1)`: GDN bf16 rn (3→2); `(2,16,1)`: GDN bf16 cv (2→1), GDN bf16 cv+rn (2→1), GDN bf16 rn (3→2), GDN f32 cv (2→1), KDA bf16 cv (2→1), KDA bf16 cv+rn (2→1), KDA bf16 rn (3→2), KDA f32 cv (2→1); `(4,8,1)`: GDN bf16 rn (4→3), GDN f32 cv (2→1); `(4,16,1)`: GDN bf16 cv (3→2), GDN bf16 cv+rn (3→2), GDN f32 cv (3→2), KDA bf16 cv (3→2), KDA bf16 cv+rn (3→2), KDA f32 cv (3→2); `(8,16,1)`: GDN f32 cv (4→3), KDA f32 rn (5→4); `(16,16,1)`: GDN f32 rn (8→7), KDA bf16 rn (8→7);
- more waves (2 cases): `(8,16,1)`: GDN f32 rn (4→5); `(16,8,1)`: GDN bf16 rn (5→6);
- spill starts (10): `(2,4,1)`: GDN bf16 cv, GDN f32 cv+rn, KDA bf16 cv, KDA bf16 cv+rn; `(4,4,1)`: GDN bf16 cv, KDA bf16 cv, KDA f32 cv; `(8,8,1)`: GDN f32 cv; `(16,16,1)`: GDN bf16 cv, KDA bf16 cv;
- spill stops (2): `(1,4,1)`: GDN bf16 cv; `(16,16,1)`: GDN bf16 cv+rn;
- spill grows on 58 and shrinks on 17 cases that already spilled.

### 4.2 ROCm 7.13 (`llvm23`, comgr SHA-256 `dd9edec7b6e078a2…`)

Dispatcher tiles (modeled waves per SIMD, merge base → head):

- fewer waves: GDN `(1,16,2)` bf16 (4→3), GDN `(1,16,4)` bf16 (5→4), GDN `(2,16,1)` bf16 (4→3), GDN `(2,16,2)` bf16 (5→4), GDN `(4,8,1)` bf16 (4→3), GDN `(4,16,1)` bf16 (5→4), GDN `(8,8,1)` bf16 (5→4), KDA `(2,16,1)` bf16 (4→3);
- more waves: GDN `(1,16,16)` bf16 (7→8), GDN `(2,16,8)` bf16 (7→8), GDN `(1,8,16)` f32 (5→6), GDN `(2,8,8)` f32 (5→6), GDN `(2,16,2)` f32 (3→4), GDN `(4,8,4)` f32 (5→6), GDN `(8,8,2)` f32 (5→6);
- spill: grows: GDN `(1,1,1)` bf16 (364→372 B), GDN `(1,1,2)` bf16 (372→380 B), GDN `(1,1,1)` f32 (468→484 B).

Auto-route fallback GDN `(1,1,1)`: `DK = 64` bf16 267→267 VGPRs, 0→0 B; `DK = 192` bf16 512→512 VGPRs, 4376→4440 B; `DK = 64` f32 331→331 VGPRs, 0→0 B; `DK = 192` f32 512→512 VGPRs, 4264→4228 B.

Fused tiles (228 cases; modeled waves per SIMD, merge base → head):

- fewer waves (24 cases): `(1,16,1)`: KDA bf16 rn (2→1); `(2,8,1)`: GDN bf16 rn (3→2); `(2,16,1)`: GDN bf16 cv (2→1), GDN bf16 cv+rn (2→1), GDN bf16 rn (5→4), GDN f32 cv (2→1), GDN f32 cv+rn (2→1), KDA bf16 cv (2→1), KDA bf16 cv+rn (2→1), KDA bf16 rn (5→2), KDA f32 cv (2→1), KDA f32 cv+rn (2→1); `(4,8,1)`: GDN bf16 rn (4→3), GDN f32 cv (2→1), KDA bf16 cv (2→1); `(4,16,1)`: GDN bf16 cv (3→2), GDN bf16 cv+rn (3→2), KDA bf16 cv (3→2), KDA bf16 rn (4→3), KDA f32 rn (4→3); `(8,8,1)`: GDN bf16 rn (5→4), KDA bf16 rn (4→3); `(8,16,1)`: GDN bf16 cv (3→2); `(16,16,1)`: KDA bf16 rn (8→7);
- more waves (2 cases): `(2,4,1)`: KDA bf16 rn (1→2); `(4,8,1)`: KDA f32 rn (2→3);
- spill starts (10): `(2,4,1)`: KDA bf16 cv+rn, KDA f32 cv, KDA f32 cv+rn; `(4,4,1)`: GDN bf16 cv, GDN bf16 cv+rn, KDA bf16 cv, KDA bf16 cv+rn, KDA f32 cv; `(8,8,1)`: KDA bf16 cv; `(16,16,1)`: GDN bf16 cv;
- spill stops (3): `(1,2,1)`: GDN bf16 rn; `(1,4,1)`: GDN bf16 cv; `(4,4,1)`: GDN f32 cv;
- spill grows on 47 and shrinks on 28 cases that already spilled.

### 4.3 ROCm 10 pre-release (`llvm23`, comgr SHA-256 `f18da4b1298b9c6f…`)

Dispatcher tiles (modeled waves per SIMD, merge base → head):

- fewer waves: GDN `(1,16,2)` bf16 (4→3), GDN `(2,16,1)` bf16 (4→3), KDA `(2,16,1)` bf16 (4→3);
- more waves: GDN `(4,16,2)` bf16 (7→8), GDN `(1,8,16)` f32 (5→6), GDN `(2,8,8)` f32 (5→6), GDN `(4,8,4)` f32 (5→6), GDN `(8,8,2)` f32 (5→6);
- spill: shrinks: GDN `(1,1,1)` bf16 (432→364 B), GDN `(1,1,1)` f32 (468→348 B), GDN `(1,1,2)` f32 (60→52 B).

Auto-route fallback GDN `(1,1,1)`: `DK = 64` bf16 234→234 VGPRs, 0→0 B; `DK = 192` bf16 512→512 VGPRs, 2340→2492 B; `DK = 64` f32 256→256 VGPRs, 0→0 B; `DK = 192` f32 512→512 VGPRs, 2232→2132 B.

Fused tiles (228 cases; modeled waves per SIMD, merge base → head):

- fewer waves (11 cases): `(2,16,1)`: GDN bf16 cv (2→1), GDN f32 cv (2→1), KDA bf16 cv (2→1), KDA f32 cv (2→1); `(4,2,1)`: KDA bf16 rn (2→1), KDA f32 rn (2→1); `(4,16,1)`: GDN bf16 rn (5→4), GDN f32 cv (3→2), KDA bf16 cv (3→2), KDA f32 cv (3→2); `(8,16,1)`: GDN f32 rn (7→6);
- more waves (18 cases): `(1,8,1)`: GDN bf16 rn (1→2); `(1,16,1)`: GDN bf16 rn (1→2), GDN f32 rn (1→2), KDA bf16 rn (1→2), KDA f32 rn (1→3); `(2,4,1)`: GDN bf16 rn (1→2), GDN f32 rn (1→2), KDA bf16 rn (1→2); `(2,16,1)`: GDN bf16 rn (3→4), GDN f32 rn (2→4), KDA bf16 rn (3→4), KDA f32 cv+rn (1→2), KDA f32 rn (2→4); `(4,16,1)`: GDN f32 rn (4→5); `(8,16,1)`: GDN bf16 cv+rn (2→3); `(16,8,1)`: GDN bf16 rn (5→6); `(16,16,1)`: GDN f32 rn (7→8), KDA f32 rn (7→8);
- spill starts (5): `(4,4,1)`: GDN bf16 cv, GDN bf16 cv+rn, KDA bf16 cv, KDA bf16 cv+rn, KDA f32 cv;
- spill stops (3): `(1,2,1)`: GDN bf16 rn, KDA bf16 rn; `(2,1,1)`: KDA f32 rn;
- spill grows on 60 and shrinks on 36 cases that already spilled.

## 5. Decisions

- **Load-first is kept.** Every compiler moves some tiles up and others down; the
  direction for a given tile is not stable across compilers (for example GDN
  `(4,16,1)` bf16 gains a wave on ROCm 7.1, loses one on ROCm 7.13 and does not
  change on ROCm 10).
- **The exemption stays keyed on `NW == 1`.** The tiles that start to spill are
  multi-wave `fuse_conv` tiles, which dispatch never builds. Which ones spill
  differs per compiler, so a footprint-keyed exemption would encode one
  compiler's register allocator.
- **The scratch test holds the stable subset.**
  `test_lowers_and_compiles_without_scratch` lowers all 228 fused cases and
  compiles and holds at 0 B the 116 that ROCm 7.1, 7.13 and 10 all keep there:
  every mode on `(1,16,1)`, `(2,8,1)`, `(2,16,1)`, `(4,8,1)`, `(4,16,1)` and
  `(8,16,1)`, plus norm-only on every other legal tile except `(1,1,1)` and
  `(1,2,1)`.
- **Tile defaults were picked on the interleaved order.** Re-run `tune.py` before
  relying on their ranking (ALGORITHM.md §4.6).

## 6. Measured speed

Relative device-time speedup of the load-first emitter versus the merge base
(`153bf493555`, the interleaved order), on gfx950. Speedup is
`(base time / head time) − 1`; `+21.6%` means `1.216×` as fast. Each cell is the
median of 3 independent runs, and the Geomean column is the geometric mean of
a row's three cells.

| Decode path | State | Heads Hk/Hv | Batch 1 | Batch 16 | Batch 128 | Geomean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GDN plain | bf16 | 16/32 | +10.4% | +14.6% | +4.6% | +9.8% |
| GDN plain | bf16 | 4/8 | +11.6% | +3.2% | +11.8% | +8.8% |
| KDA plain | bf16 | 32/32 | +21.6% | +6.4% | +11.0% | +12.8% |
| KDA plain | bf16 | 8/8 | +21.8% | +6.2% | +15.7% | +14.4% |
| KDA plain | f32 | 32/32 | +7.6% | +8.6% | +5.0% | +7.1% |
| KDA fused conv + norm | f32 | 32/32 | +0.4% | **−1.7%** | +1.9% | +0.2% |

**Scope.** Single-token `DK = DV = 128` decode, bf16 I/O, batches 1, 16 and 128.
GDN uses the default `(2,16,8)` tile. KDA uses the dispatcher's `(4,16,4)`,
`(1,16,4)` and `(2,16,1)` tiles; the fused f32 case uses `(4,16,1)`. The f32-state
and fused arms were built as specs directly; a dispatch request can now ask for
both, and `auto` serves a fused request `(4,16,1)`. No f32 or fused tile was
tuned. GDN fused performance was not measured. The tile choices were made on the
old load order (§5).

**Method.** Both trees run in one process, with their arms interleaved. State,
conv-state and output buffers rotate through at least 1 GiB per cycle, so every
launch reads from HBM; every copy has its own cloned inputs. Timing replays a
HIP graph. The Python lowerer and `llvm23` build both trees. The runs used the
ROCm 7.13 container.

**Correctness gate.** Every timed arm is checked against the fp32 reference
before it is timed, with tolerance `1e-2` on the max-abs error of the output
and of the written state; for the fused-norm arm the output error is divided
by `max(1, |reference|)` (`out_error` in `gdn_decode.py`). All 54 measured
rows passed, and output, state and conv state were bitwise equal between the
two trees in every row.

**Honest loss.** The fused conv + norm f32 path is slower at batch 16: the three
independent runs give base/head ratios `0.9828`, `0.9834` and `0.9840`, so the
loss (−1.7%) is repeatable, while the gains at batches 1 and 128 are small
(+0.4% and +1.9%). The cause was not investigated. The plain paths improve in
every measured configuration, by at least +3.2%.

**Provenance.** The two-tree A/B driver that produced these numbers is not in
this tree, and absolute per-run times are not published here, so a reader
cannot replay this table from the repository. Section 7 reproduces the
register counts only.

## 7. Reproducing

Save the block below as `regs.py` anywhere outside the tree. Run it from
`rocke/library` once in a checkout of the merge base and once at the head, with
the same compiler, then join the two outputs. Each row is the kernel name,
VGPRs, modeled waves per SIMD and scratch bytes.

```python
import os
import tempfile
from itertools import product
from multiprocessing import Pool
from pathlib import Path

from dispatch.gdn.gfx950 import _TUNED_TILES_KDA, CONFIGURED_TILES
from kernels.gfx950.gdn_decode import GdnDecodeSpec, build_gdn_decode, is_valid_spec
from rocke.analysis.isa import analyze_hsaco
from rocke.helpers.compile import compile_kernel

FUSED = ((True, False), (False, True), (True, True))  # (fuse_conv, fuse_out_norm)


def specs():
    for st in ("bf16", "f32"):
        for nw, wtk, bpv in CONFIGURED_TILES:
            yield dict(gate_kind="gdn", num_k_heads=16, num_v_heads=32, state_dtype=st,
                       num_warps=nw, warp_threads_k=wtk, blocks_per_v_dim=bpv)
        for _, (nw, wtk, bpv), _ in _TUNED_TILES_KDA:
            yield dict(gate_kind="kda", num_k_heads=32, num_v_heads=32, state_dtype=st,
                       num_warps=nw, warp_threads_k=wtk, blocks_per_v_dim=bpv)
        for d in (64, 192):  # (2,16,8) is illegal here, so auto takes (1,1,1)
            yield dict(gate_kind="gdn", num_k_heads=16, num_v_heads=32, state_dtype=st,
                       head_k_dim=d, head_v_dim=d, num_warps=1, warp_threads_k=1,
                       blocks_per_v_dim=1)
        for gate, (cv, rn), nw, wtk in product(("gdn", "kda"), FUSED, (1, 2, 4, 8, 16),
                                               (1, 2, 4, 8, 16, 32)):
            yield dict(gate_kind=gate, num_k_heads=16, num_v_heads=16, state_dtype=st,
                       num_warps=nw, warp_threads_k=wtk, blocks_per_v_dim=1,
                       fuse_conv=cv, fuse_out_norm=rn)


def measure(kw):
    spec = GdnDecodeSpec(**kw)
    art = compile_kernel(build_gdn_decode(spec, arch="gfx950"), arch="gfx950",
                         capture_ir_text=False)
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as f:
        f.write(bytes(art.hsaco))
        f.flush()
        r = analyze_hsaco(Path(f.name)).resources
    waves = min(8, 512 // (-(-r.vgpr_count // 8) * 8))
    return spec.kernel_name(), r.vgpr_count, waves, r.scratch_bytes


if __name__ == "__main__":
    legal = [kw for kw in specs() if is_valid_spec(GdnDecodeSpec(**kw))[0]]
    with Pool(len(os.sched_getaffinity(0))) as pool:
        for row in pool.imap(measure, legal):
            print(*row, sep="\t", flush=True)
```

```bash
git worktree add /tmp/gdn-base 153bf493555
cd /tmp/gdn-base/dnn-providers/hip-kernel-provider/rocke/library
ROCKE_BACKEND=python PYTHONPATH=.:../platform/python python /path/to/regs.py | sort > /tmp/base.tsv
cd <head checkout>/dnn-providers/hip-kernel-provider/rocke/library
ROCKE_BACKEND=python PYTHONPATH=.:../platform/python python /path/to/regs.py | sort > /tmp/head.tsv
# name, base VGPRs, base waves, base scratch, head VGPRs, head waves, head scratch
join -t $'\t' /tmp/base.tsv /tmp/head.tsv
```

Select the compiler with `ROCKE_COMGR_LIB=<path to libamd_comgr.so.3>` and
`ROCKE_LLVM_FLAVOR` (`llvm20` for ROCm 7.1, `llvm23` for ROCm 7.13 and later),
and put that ROCm's `llvm/bin` first on `PATH` so `analyze_hsaco` uses a
`llvm-objdump` that knows the AMDGPU target. The pool uses every CPU the process
may run on; on a shared node, run one instance at a time.
