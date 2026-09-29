# Target-agnostic SDPA recipe integration

Keep one hipDNN engine and the existing bounded BF16 causal SDPA contract
(B=1, Hq=Hkv=4, D=128; S=512/768/1024). Initially support gfx942 and gfx950
through separate recipes in the same installed bundle.

1. Bind the recipe ABI by semantic argument name and type, using launch-plan
   offsets. Support graph-derived batch and sequence scalars as well as scale
   and Q/K/V/O pointers. Validate the ABI at preparation, rejecting unknown fields.
2. Feed HKP's selected GPU_TARGETS / AMDGPU_TARGETS into the producer. Require
   explicit targets and LLVM flavor for standalone production. Select each
   target's builder through a small registry and fail unsupported requests.
3. Roll each target independently under common logical recipe keys. Generate
   the descriptor target list from the same selection, retain per-target LLVM
   references and provenance, and verify geometry against builder helpers.
4. Remove the architecture restriction from graph matching. Use descriptor and
   bundle admission for target support; preserve the existing normalized lookup
   and device-specific COMGR target path.
5. Parameterize native/frontend validation by the actual or expected target.
   Check missing-target rejection, single/dual-target packaging, Python/native
   LLVM identity, COMGR compilation, and fresh Slurm numerical runs on both GPUs.

Acceptance: the same installed provider and dual-target bundle pass native and
frontend numerical tests on gfx942 and gfx950, without rebuilding between GPUs.
Compile-only checks and historical GPU runs are not numerical acceptance.

Implementation order: ABI correction first, then producer/packaging/runtime
target selection and gfx942 enablement. Keep the changes narrowly scoped to this
integration; broader SDPA shapes, persistent caches, and additional devices are
separate work.
