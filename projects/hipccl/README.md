# hipCCL

Working area for consolidating rocPRIM, hipCUB, rocThrust, and libhipcxx into a
single hipCCL project, mirroring the role [NVIDIA/cccl](https://github.com/NVIDIA/cccl)
plays for CUB, Thrust, and libcudacxx. See
[`../../docs/hipccl-repository-split-proposal.md`](../../docs/hipccl-repository-split-proposal.md)
for the full extraction plan.

This directory temporarily has two parallel copies while that transition is
in progress:

- **[`hipccl2/`](hipccl2/)** - a loose, unbound compatibility snapshot of
  rocPRIM/hipCUB/rocThrust exactly as they exist in `rocm-libraries` today.
  Nothing at its root binds the three together.
- **[`hipccl3/`](hipccl3/)** - the forward-looking unified hipCCL project:
  common root-level CMake, docs, and CI scaffolding across rocPRIM, hipCUB,
  rocThrust, and libhipcxx, with a single version and a single install layout
  (`<prefix>/include/hipccl/<component>`).

Both currently pin rocPRIM/hipCUB/rocThrust to the same commit (tip of
`rocm-libraries`' `develop` as of this writing); `hipccl3` additionally
includes libhipcxx, pinned to `ROCm/libhipcxx@5ac455d737937ba2dfd1a4e85ad13f19a775f692`
(`amd-develop`). `hipccl3`'s rocPRIM/hipCUB/rocThrust copies are expected to
be merged forward to align with upstream CCCL 3.0 at a later time; libhipcxx
was brought in as a starting point for that same effort.

The original `projects/rocprim`, `projects/hipcub`, `projects/rocthrust`
directories are untouched and remain fully functional - this is additive, not
a cutover. Removing them (and repointing everything that currently depends on
those paths) is a separate, later step; see the proposal doc's Phase 6 for
what that involves.
