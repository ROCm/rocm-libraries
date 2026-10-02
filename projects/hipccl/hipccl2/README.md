# hipccl2 (compatibility snapshot - not a build root)

This directory is intentionally just `rocprim/`, `hipcub/`, and `rocthrust/` as
independent, unmodified project folders - the exact same commit as
`rocm-libraries`' `projects/rocprim`, `projects/hipcub`, `projects/rocthrust`
(and the same commit as their counterparts under `../hipccl3/`). There is
deliberately **no root-level `CMakeLists.txt`, version file, or any other file
binding these three together** - each is built/tested/used exactly as it is
today under `projects/rocprim`, `projects/hipcub`, `projects/rocthrust`.

This exists purely for backward-compatibility during the transition described
in [`../../../docs/hipccl-repository-split-proposal.md`](../../../docs/hipccl-repository-split-proposal.md).
The forward-looking, unified project lives in [`../hipccl3/`](../hipccl3/).
Eventually only one of `hipccl2`/`hipccl3` will remain.
