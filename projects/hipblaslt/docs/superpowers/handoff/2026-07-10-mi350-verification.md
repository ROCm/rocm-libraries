# MI350 (gfx950) Verification Handoff — hipBLASLt Python Interface

**Audience:** a Claude instance (or engineer) that has checked out branch
`users/talumbau/python-interface` on a **gfx950 / MI350** host, tasked with
verifying the device paths that could NOT be tested on the gfx942 dev host.

## Context

The `hipblaslt` Python package (under `python/`) was implemented and tested on a
gfx942 / MI300 host. Everything runs there EXCEPT three device-GEMM correctness
paths that require gfx950 / MI350. Those were written full best-effort,
cross-referenced against the C++ sources, and marked `@pytest.mark.mi350`. They
skip on gfx942 (via `NOT_SUPPORTED` → `pytest.skip`) and must be confirmed here.

Read first: `docs/superpowers/specs/2026-07-10-python-interface-design.md`
(design) and `docs/superpowers/plans/2026-07-10-python-interface.md` (plan).

## Step 1: Build and sanity-check

```bash
# Activate the dev environment (same conda env used during implementation).
conda activate pydev313

cd projects/hipblaslt
invoke build -ca gfx950 --python          # or your usual arch flags + --python
cd python
python -c "import hipblaslt; print(hipblaslt._core.hip_available())"   # -> True
python -m pytest tests/ -m "not mi350" -v  # full non-deferred suite must pass
```

If `pydev313` is not present on this machine, create it:
```bash
conda create -n pydev313 python=3.13 numpy pandas -y
conda activate pydev313
python -m pip install "scikit-build-core>=0.10" "nanobind>=2.0" "ml_dtypes>=0.5.0" pytest
```

If the non-deferred suite does not pass on MI350, STOP — that is a regression
unrelated to the deferred work; investigate before touching the mi350 tests.

## Step 2: Run the deferred suite

```bash
python -m pytest tests/ -m mi350 -v
```

The three deferred tests:

| Test | File | What it verifies |
|------|------|------------------|
| `test_ocp_fp8_gemm` | `tests/test_fp8_gemm.py` | OCP E4M3 fp8 GEMM correctness vs numpy f32 reference |
| `test_mx_gemm_matches_reference` | `tests/test_mx.py` | MX GEMM, `A/B_SCALE_MODE=VEC32_UE8M0`, canonical UE8M0 scales |
| `test_mx_gemm_preswizzle_mode1001` | `tests/test_mx.py` | MX GEMM, `BLK32_UE8M0_32_8_EXT` (1001), PRE-SWIZZLED scales |

Expected on MI350: they no longer skip. Each either PASSES or reveals a real
issue (see below).

Note: `test_mx_gemm_preswizzle_mode1001` also guards against
`c.ScaleMode.BLK32_UE8M0_32_8_EXT` being absent from the SDK — if the installed
hipBLASLt SDK on this machine still does not expose that enum value, the test
will skip with "BLK32_UE8M0_32_8_EXT not available in this SDK version". If it
skips for that reason, confirm the SDK version and file a follow-up task.

## Step 3: What is most likely wrong (search for `VERIFY-ON-MI350`)

Grep the tree for the flag: `grep -rn "VERIFY-ON-MI350" python/`. The high-risk
spots, in priority order:

1. **The swizzle permutation** — `python/hipblaslt/mx.py::swizzle_scales`. Only
   the roundtrip (swizzle→unswizzle) was verified on gfx942; whether the FORWARD
   layout matches what the gfx950 subtile kernel reads is unverified. Ground
   truth: `tensilelite/client/src/DataInitialization.cpp` `generateMXInput`
   (~lines 1977–2016), `preSwizzleTile = {tileMN, tileK, subTileK} = {32, 8, 4}`.
   If `test_mx_gemm_preswizzle_mode1001` fails but the canonical MX test passes,
   the permutation is the culprit — re-derive it from the C++ and update
   `swizzle_scales` (and its inverse). The roundtrip test guards the inverse.

2. **Scale-tensor layout / transpose for `VEC32_UE8M0`** — in
   `test_mx_gemm_matches_reference` the `dsa`/`dsb` scale tensors are uploaded
   transposed to match the element tensors; the exact orientation the kernel
   expects is unverified. If numbers are wrong but structurally close, try the
   non-transposed scale layout and adjust `MatrixLayout`/stride accordingly.

3. **fp8 element layout for OCP GEMM** — `test_ocp_fp8_gemm` reuses the f32 GEMM
   transpose convention. Confirm the column-major handling holds for 8-bit
   elements (leading dimensions are in elements, not bytes).

4. **Tolerances** — the deferred tests use `rtol/atol = 0.1–0.15`. If a test
   passes only with a much looser tolerance, that itself is a finding: note the
   achievable tolerance rather than loosening silently.

## Known encoding divergences (from Task 16, gfx942)

These were discovered during the fp8 encoding cross-check between ml_dtypes and
hipBLASLt. They are documented here as context — not bugs to fix. They do NOT
appear in the linspace(-8.0, 8.0, 257) sweep used by the cross-check tests.

1. **FNUZ out-of-range clamping**: For inputs outside the representable range
   (e.g., 448.0 > e4m3_fnuz max of 240.0), hipBLASLt clamps to the representable
   maximum, while ml_dtypes converts to the NaN sentinel (0x80). Only surfaces for
   inputs outside [-240, 240].

2. **e5m2 NaN payload bit difference**: hipBLASLt encodes NaN as 0x7F; ml_dtypes
   encodes it as 0x7E. Both decode to float NaN — semantically identical, only the
   bit pattern differs. Does not occur in the linspace sweep (no NaN inputs).

Neither divergence affects the deferred GEMM tests (which operate within normal
representable ranges), but they are relevant context if you extend the cross-check
suite on MI350.

## Step 4: Record results

For each deferred test, record in the PR / a follow-up commit: PASS, or the
specific divergence (which elements, what tolerance, suspected cause). If a path
is genuinely unsupported even on this MI350 build/ROCm version, convert the test
to `xfail(reason=...)` with the ROCm version, rather than deleting it.

## Ground-truth references

- Scale-mode enum + "Not supported yet" notes: `library/include/hipblaslt/hipblaslt.h` (`hipblasLtMatmulMatrixScale_t`).
- Swizzle layout: `tensilelite/client/src/DataInitialization.cpp` (`generateMXInput`, ~1977–2016; arch gate at ~964; canonical-reference invariant at ~2056).
- fp8 element types + host converters: `library/include/hipblaslt/hipblaslt_float8.h`.
