# CK Gentune Run Manifest — grouped conv2d fwd, XDL, gfx1250, NHWC, BF16

## Status

- **Mode**: SET UP + VALIDATED. Container created; **tuning BLOCKED by GPU health**.
- **Parser / cardinality / constraint preflight**: **PASS**
  (`preflight_check.py`: main naive=128 / constrained=32, 32 fully-resolved
  tuples, `B==A` coupling verified, seed tuple present, 24 BENCH == 24 VERIFY
  20-token shell-safe shapes; smoke constrained=1).
- **Container**: `ck-gentune-gfx1250` created and running (command below).
  In-container toolchain (AMD clang 23.0.0git) and `rocminfo` enumerate `gfx1250`
  fine; `hipGetDeviceCount` returns n=1 (gfx1250).
- **GPU execution**: **FAIL (2026-07-31 probe).** A minimal HIP test
  (`hipMalloc` + `hipMemcpy` H2D + trivial kernel launch + `hipDeviceSynchronize`)
  **SIGSEGVs (exit 139) before any output** — it dies at HIP init / first HSA
  queue creation. `rocm-smi` shows a phantom `GPU% 100%` with **no KFD PIDs**.
  This is the same wedged-accelerator signature as the 2026-07-29 run. (Host
  `rocminfo` also SIGSEGVs — a separate host-userspace/ROCm version mismatch.)
- **Smoke**: **ATTEMPTED 2026-07-31 → FAIL.** The instance compiled (4 files) and
  the conv problem set up correctly (`in {1,42,256,30,40}`, `wei {1,128,256,1,1}`,
  `out {1,42,128,30,40}` — shape mapping confirmed), but the benchmark
  **SIGSEGV'd (core dumped, exit 139)** at the GPU boundary → `0 performance
  figures gathered`, no winner. Log: `logs/smoke.log`. (Re-probe `/tmp/hipexec`
  also SIGSEGV'd exit 139.)
- **Full tuning**: **NOT LAUNCHED** — every one of the 24×32 candidates would
  crash identically (0 results, only wasted compiles), so it was not run.
  **Recovery: host/accelerator reboot by the system owner** (SMI/PCI-FLR reset is
  unsupported on this part), then re-probe (`docker exec ck-gentune-gfx1250
  /tmp/hipexec` must print `HIP_EXEC_OK`), then smoke, then the full run.

## Source and Environment

- Run ID: `grouped_conv_fwd_xdl_gfx1250_nhwc_20260731`
- Created UTC: `2026-07-31`
- Source commit: `ecf80c02360a66b2bfe97af5669666972487df66` (working tree: `.gitignore`
  modified; `gentune/runs/` untracked)
- Target arch: `gfx1250`, wavefront size 32
- Expected container (from the prior compile-confirmed run; user must provide a
  healthy one): `ck-gentune-gfx1250`,
  image `rocm/composable_kernel:ck_ub24.04_rocm7.13_develop`,
  mount `/home/ctsiaous/rocm-libraries:/workspace/rocm-libraries`,
  toolchain `/opt/rocm`, ROCm `7.13.x`,
  build dir `/workspace/rocm-libraries/projects/composablekernel/build`
  (`libutility.a` present on host).

> The prior run `grouped_conv_fwd_xdl_gfx1250_20260729_gpu0` hung the GPU (VM
> permission faults; accelerator lost from bus) and its container is no longer
> running. Confirm the GPU is healthy (`rocminfo` returns, `rocm-smi` shows 0%
> idle, no stuck KFD PIDs) before launching.

## Exact Workload

- Operation: `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle` (2D)
- Types: BF16 in/wei/CShuffle/out, FP32 accumulate, PassThrough elementwise
- Layout: **NHWC** — `GNHWC` input, `GKYXC` weight, `GNHWK` output
  (auto-selected by `num_dim_spatial == 2` in `run_convnd_fwd_example.inc`)
- GEMM spec: `MNKPadding`
- Objective: minimize the first float on the single `Perf:` line (ms latency)
- Shapes: **24 forward grouped-conv shapes** (MIOpen `convbfp16 ... -F 1`).
  Excluded per user: all `bnormbfp16` (batch norm), all `-F 2`/`-F 4` (backward),
  and 5 depthwise shapes (`-g` == channels → per-group C=K=1, a poor XDL fit).
- BENCH_ARGS format (2D): `verify init time 2  G N K C  Y X  Hi Wi  sH sW  dH dW
  lpH lpW  rpH rpW`. CK C/K are **per-group**; all 24 kept shapes are `-g 1`, so
  C=`-c`, K=`-k`. BENCH `verify=0`; VERIFY `verify=2` (GPU reference — fast
  enough for the budget on the large shapes; smoke uses `verify=1` CPU gold).

### MIOpen → CK shape map (all N=42, group=1)

| # | MIOpen (convbfp16 -F 1)                    | K   | C   | Y×X  | Hi×Wi   | stride |
|---|--------------------------------------------|-----|-----|------|---------|--------|
| 1 | c128 H120 W160 k10  1x1                     | 10  | 128 | 1×1  | 120×160 | 1      |
| 2 | c128 H120 W160 k128 3x3 p1                  | 128 | 128 | 3×3  | 120×160 | 1      |
| 3 | c128 H30 W40 k128 1x1                       | 128 | 128 | 1×1  | 30×40   | 1      |
| 4 | c128 H30 W40 k128 3x3 p1                    | 128 | 128 | 3×3  | 30×40   | 1      |
| 5 | c128 H30 W40 k512 1x1                       | 512 | 128 | 1×1  | 30×40   | 1      |
| 6 | c16 H1 W1 k256 1x1                          | 256 | 16  | 1×1  | 1×1     | 1      |
| 7 | c192 H120 W160 k48 1x1                      | 48  | 192 | 1×1  | 120×160 | 1      |
| 8 | c192 H60 W80 k64 1x1                        | 64  | 192 | 1×1  | 60×80   | 1      |
| 9 | c24 H240 W320 k128 1x1                      | 128 | 24  | 1×1  | 240×320 | 1      |
|10 | c24 H240 W320 k24 3x3 p1                    | 24  | 24  | 3×3  | 240×320 | 1      |
|11 | c24 H240 W320 k96 3x3 p1 s2                 | 96  | 24  | 3×3  | 240×320 | 2      |
|12 | c256 H1 W1 k16 1x1                          | 16  | 256 | 1×1  | 1×1     | 1      |
|13 | c256 H30 W40 k128 1x1                       | 128 | 256 | 1×1  | 30×40   | 1      |
|14 | c256 H60 W80 k64 1x1                        | 64  | 256 | 1×1  | 60×80   | 1      |
|15 | c32 H1 W1 k512 1x1                          | 512 | 32  | 1×1  | 1×1     | 1      |
|16 | c3 H480 W640 k24 3x3 p1 s2 (odd C)          | 24  | 3   | 3×3  | 480×640 | 2      |
|17 | c48 H120 W160 k128 1x1                      | 128 | 48  | 1×1  | 120×160 | 1      |
|18 | c48 H120 W160 k192 1x1                      | 192 | 48  | 1×1  | 120×160 | 1      |
|19 | c48 H120 W160 k192 3x3 p1                   | 192 | 48  | 3×3  | 120×160 | 1      |
|20 | c512 H1 W1 k32 1x1                          | 32  | 512 | 1×1  | 1×1     | 1      |
|21 | c512 H30 W40 k128 1x1                       | 128 | 512 | 1×1  | 30×40   | 1      |
|22 | c64 H60 W80 k128 1x1                        | 128 | 64  | 1×1  | 60×80   | 1      |
|23 | c64 H60 W80 k256 1x1                        | 256 | 64  | 1×1  | 60×80   | 1      |
|24 | c96 H120 W160 k48 1x1                       | 48  | 96  | 1×1  | 120×160 | 1      |

## Search Space (constraint-based, bounded)

- Fixed 16×16 skeleton (checked-in gfx1250 instance / compile-confirmed smoke):
  `NumGemmKPrefetch=1, BlockSize=256, MPerBlock=64, NPerBlock=64, KPerBlock=32,
  AK1=BK1=8, MPerXDL=NPerXDL=16, MXdlPerWave=NXdlPerWave=2,
  A/B cluster S<4,64,1>, arrange/access S<1,0,2>, srcVecDim=2, dstScalarPerVec=8,
  LdsExtra=1, CShuffleM/NXdl=1, CDE cluster S<1,32,1,4>`.
  On NHWGC+bf16+Default/Filter1x1* on wave32, `Wave32Force16MNPerXDL` forces the
  MFMA to 16×16, matching this skeleton.
- Tuned individual params + constraints + suffix assembly:
  - `CK_CONVSPEC ∈ {Default, Filter1x1Stride1Pad0}` — 18/24 shapes are 1×1
    stride-1 pad-0 and get the fast specialization; on non-1×1 shapes the 1×1
    kernel reports unsupported and is skipped.
  - `CK_A_SRC_VEC, CK_B_SRC_VEC ∈ {1,2,4,8}` with `CONSTRAINT CK_B_SRC_VEC ==
    CK_A_SRC_VEC` (both load along channel C, so coupled).
  - `CK_CDE_VEC ∈ {1,2,4,8}` (output write along K).
- **Constrained cardinality = 2 × 4 × 1 × 4 = 32** candidates/shape. Finite,
  parser-enumerable (< 10000). With `A=B=CDE=4` the suffix equals checked-in
  instance #17; `=8` equals the compile-confirmed smoke tuple.
- Randomness: Gentune has no random-seed option. Resume: optimizer state is
  in-memory only; logs/HPP snapshots are evidence, not a checkpoint.

## Budget / stop policy (fits well under 4 h)

- 24 shapes × ≤32 candidates ≈ ≤768 compile+bench worst case (exhaustive; the
  optimizer brute-forces after 50% coverage), plus GPU verify only on new bests.
- Compile threads 16, status `-p 60`, verbosity 1 (2 only for smoke).
- External wall guard: `timeout --signal=INT --kill-after=5m 4h`.
- Stop on finite exhaustion, unsafe disk, GPU/process health failure, or wall
  expiry. Failures are skipped (compile fail / unsupported / verify fail).

## Commands

Container (already created on 2026-07-31; recreate after a host reboot):
```bash
docker rm -f ck-gentune-gfx1250 2>/dev/null
docker run -d --name ck-gentune-gfx1250 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size=16G \
  -v /home/ctsiaous/rocm-libraries:/workspace/rocm-libraries \
  -w /workspace/rocm-libraries/projects/composablekernel/gentune \
  rocm/composable_kernel:ck_ub24.04_rocm7.13_develop \
  sleep infinity
```

GPU health re-probe (must print `HIP_EXEC_OK` before launching anything):
```bash
docker exec ck-gentune-gfx1250 bash -lc 'timeout 60 /tmp/hipexec; echo exit=$?'
```

Preflight (GPU-free; host or container):
```bash
cd /home/ctsiaous/rocm-libraries/projects/composablekernel/gentune
python3 runs/grouped_conv_fwd_xdl_gfx1250_nhwc_20260731/preflight_check.py
```

Smoke (inside a healthy gfx1250 container; -v 2 keeps build output + failed
instances; establishes the seed's baseline latency for shape #13):
```bash
docker exec -e HIP_VISIBLE_DEVICES=0 ck-gentune-gfx1250 bash -lc \
  'cd /workspace/rocm-libraries/projects/composablekernel/gentune && \
   python3 -u tuner_main.py \
     -g runs/grouped_conv_fwd_xdl_gfx1250_nhwc_20260731/ \
     -i generation/grouped_conv_fwd_xdl_nhwc_smoke.gentune \
     -v 2 -t 1 -p 30 2>&1 | \
   tee runs/grouped_conv_fwd_xdl_gfx1250_nhwc_20260731/logs/smoke.log'
```

Full tuning (only after smoke PASS):
```bash
docker exec -e HIP_VISIBLE_DEVICES=0 ck-gentune-gfx1250 bash -lc \
  'set -o pipefail; cd /workspace/rocm-libraries/projects/composablekernel/gentune && \
   timeout --signal=INT --kill-after=5m 4h \
   python3 -u tuner_main.py \
     -g runs/grouped_conv_fwd_xdl_gfx1250_nhwc_20260731/ \
     -i generation/grouped_conv_fwd_xdl_nhwc.gentune \
     -v 1 -t 16 -p 60 2>&1 | \
   tee runs/grouped_conv_fwd_xdl_gfx1250_nhwc_20260731/logs/tune.log'
```

## Known limitations / future extensions

- **Single tile skeleton (64×64).** Only vectorization + 1×1 specialization are
  tuned, because 64×64/16×16 is the only *gfx1250-compile-confirmed* tile for
  this exact device-op class. Adding larger tiles (the checked-in 32×32 NHWGC
  instances that run as 16×16 on wave32) is a worthwhile next step but must be
  gated by a smoke test — it is not statically verifiable.
- `OddC` specialization was left out to stay ≤40 and because `Default` +
  `MNKPadding` handles the one odd-C shape (#16, C=3) via `CK_A_SRC_VEC=1`.
- The example throws (aborts) on unsupported args instead of printing the clean
  "does not support this problem" sentinel; Gentune records that as
  `ERROR_EXECUTION_FAILED` and skips it — safe, just noisier logs.

## Artifacts

- Keep: `generation/*.gentune`, `templates/grouped_conv_fwd_xdl_nhwc.txt`,
  `preflight_check.py`, this manifest, `logs/*.log`, winner report.
- Clean up after review: `test_instances/`, `test_instances_smoke/` (generated
  sources, objects, executables, failed builds).
