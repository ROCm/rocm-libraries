# Troubleshooting the engine and build

This is the failure-mode catalog for the **engine and build** — the things that
look like bugs but usually aren't. For debugging a *running kernel* (wrong
output, hangs, faults, slowness), see the "Debugging Patterns" section of
[`testing.md`](./testing.md). For the rules behind several of these, see
[`invariants.md`](./invariants.md).

> **First rule of this codebase:** when something that should work doesn't, suspect
> a **stale or mis-flagged build product before you suspect the source**. Almost
> every phantom failure here has been an old archive, a missing build flag, or a
> stale synced copy — used silently. Rebuild fresh, then debug.

---

## Build / link failures

### "undefined symbol: rocke_… " when loading the provider or pybind module
The consumer linked against an archive that predates your change, or against the
wrong archive. **Fix:** rebuild the engine fresh and point the consumer at *that*
archive. Use `tools/check_byte_identity.py` (it builds the engine fresh and links the
provider against it by construction). See [`invariants.md`](./invariants.md#7-build-on-local-disk-never-trust-a-stale-archive).

### "it doesn't compile" — but only in one checkout / on the cluster
You are almost certainly looking at a **stale synced copy**, not current source.
Re-sync (content-verified, not a timestamp/rsync filter) and rebuild. A clean
build from current source in `/tmp` is the source of truth.

### The provider build can't find hipDNN headers / `_deps` / `SDPA`
A by-hand build dropped a required flag. The full set lives in
[`../../../BUILD.md`](../../../BUILD.md) ("the flags that get
forgotten") — `-D__HIP_PLATFORM_AMD__`, the generated `build/` include set
including `_deps`, the SDPA enable, and an explicit fresh `ROCKE_LIB`. Prefer
`tools/build.sh`, which bakes them all.

### Sanitizer build fails with a `-Wformat-truncation` "null format string"
This is a known interaction between sanitizer instrumentation and the format
checker on the variadic emit wrappers — it is a false positive, not real UB. The
sanitizer build path already suppresses it (`-Wno-format-truncation` in the
sanitizer block only; the normal build keeps the check). If you re-introduce it,
suppress it in the sanitizer block, not globally.

### Building on NFS takes forever
Expected — NFS makes the compiler and `comgr` pathologically slow. Build into a
local dir (`/tmp`). This is a documented requirement, not a machine problem.

---

## Differential-gate failures

### `--mode ll` is red on a family
Your change altered emitted LLVM-IR bytes on one engine but not the other. This is
a **real contract break** — fix it before merging. Narrow it:
```bash
run_diff --mode ll --only <family>
```
Then look at whether the two engines diverged in the lowering (most likely if you
edited `lower_llvm`) or in the builder.

### `--mode ll` went red right after a lint or format pass
A `ruff check --fix` (or an editor auto-fix) almost certainly **deleted a
side-effecting builder call**: `F841` treats `c8 = b.const_i32(8)` as an unused
variable and removes it, even though the call emits an op into the kernel. Restore
the assignment (add `# noqa: F841` if the handle is intentionally unused) and
re-run. **Never run `ruff check --fix` on `rocke/core`, `helpers`, or
`instances`** — see [`invariants.md`](./invariants.md) rule 9.

### `--mode ll` is green but `--mode ir` shows `STRUCT_DRIFT`
The lowerer absorbed an upstream builder divergence (it renumbers SSA), so the
final `.ll` matched by luck. The IR still diverged. Re-run with `--canonical`:
```bash
run_diff --mode ir --canonical --only <family>
```
- `CANON_EQUAL` → incidental SSA-id renumbering; harmless.
- `STRUCT_DRIFT` → real. The usual cause is **argument-evaluation order**: a
  builder emitted operands as nested call arguments, which the C++ compiler
  evaluates right-to-left while Python goes left-to-right. Hoist the operands to
  temporaries in source order. See
  [`invariants.md`](./invariants.md#2-the-ir-builder-must-emit-operands-left-to-right).

### `RANGE_DRIFT` on `fmha_appendkv` or `gfx1151_wmma_gemm_iu8_dequant`
Pre-existing and benign: the C emitter enumerates a slightly wider config range
than the Python reference; the in-range bytes are identical. Not a failure.

### `--check-golden` fails after a change I *meant* to make
If you intentionally changed what a kernel emits, re-bless the golden **from a
green state** and let your reviewer read the diff:
```bash
run_diff --record-golden
```
Never re-bless to silence a red gate you don't understand — that discards the one
signal that protects the contract.

### The gate says "archive not found"
You didn't build it (or built it elsewhere). Build into the default location, or
pass `--archive`:
```bash
cmake -S Cpp -B /tmp/rocke_verify -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/rocke_verify -j
# default archive path is $TMPDIR/rocke_verify/librocke_core.a
```
`tools/check_byte_identity.py` does the build + gate together.

---

## Environment / GPU

### `torch.cuda.is_available()` is False even though there's a GPU
On a box where your user isn't in the GPU device group, the device nodes
(`/dev/kfd`, `renderD*`) aren't accessible from a plain shell, so torch reports no
device. Compilation/lowering (`comgr`) does **not** need the GPU and works
normally; only kernel *launch* does. Run launch steps through the privileged path
your environment provides (e.g. `sudo -n -E <venv>/bin/python …`), and write
artifacts somewhere you own (e.g. `/tmp`), not into a root-owned repo path.

### `No module named 'rocke'` / `Cpp`
`PYTHONPATH` isn't pointing at `dnn-providers/hip-kernel-provider/rocKE/Python`. Set it
(`PYTHONPATH=<…>/dnn-providers/hip-kernel-provider/rocKE/Python`) and run from there.

### A probe can't find `llvm-readelf` / `objdump`
The ROCm `bin/` directory isn't on `PATH`. Add it.

---

## Multi-architecture (cluster) runs

### A cluster job lands on a node with an empty home / missing inputs
The synced repo and the virtualenv live on shared storage that only exists on
specific nodes; a job that lands elsewhere mounts an empty home. Constrain the
job to the nodes where your inputs actually are (the submit script does this with
a node constraint) — see the engine CI directory.

### A whole architecture is "missing" from results
Some arches are scarce and queue longer; some share a generic container image
rather than a dedicated one. The submit script encodes per-arch resources, images,
and walltimes — a longer walltime lets a scarce-GPU job wait in queue instead of
being rejected. Check `squeue` before concluding a run failed.

---

## When in doubt

1. Rebuild fresh from current source into `/tmp`.
2. Run `tools/check_byte_identity.py`.
3. Only then debug the source.

Most of the time, steps 1–2 resolve it.
