# hipCCL docs (placeholder)

This directory is the intended home for hipCCL-level documentation: an overview
page linking out to each component, and (eventually) a unified Sphinx/rocm-docs-core
build the way [NVIDIA/cccl's `docs/`](https://github.com/NVIDIA/cccl/tree/main/docs)
aggregates docs for CUB, Thrust, and libcudacxx into one site.

**Not implemented yet.** For now, each component keeps its own docs where it
already had them:

- `../rocprim/docs/`
- `../hipcub/docs/`
- `../rocthrust/docs/`
- `../libhipcxx/docs/`

`rocm-libraries`' existing docs-preview infrastructure
(`.github/workflows/docs-pr-preview*.yml`, `.github/docs-config.json`) already
builds and previews `rocprim`/`hipcub`/`rocthrust` docs independently via their
Read the Docs slugs (`advanced-micro-devices-{rocprim,hipcub,rocthrust}`); that
continues to work unchanged for the `projects/rocprim`, `projects/hipcub`,
`projects/rocthrust` copies outside `hipccl/`, since those directories were left
in place (see `../../../docs/hipccl-repository-split-proposal.md`).

TODO for a future pass:
- Decide whether hipCCL wants a single combined docs site (CCCL-style) or four
  independently-published docs sets that just happen to share a landing page.
- If combined: design the aggregation (likely a top-level `index.rst`/`index.md`
  plus per-component subdirectories, similar to CCCL's structure).
