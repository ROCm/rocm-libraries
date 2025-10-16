| author | created | modified | status |
|--------|---------|----------|--------|
| Sambhav Jain, Aaron St. George and Mahesh Ravishankar | <TBD> | <TBD> | draft |


# Add IREE as a kernel provider for hipDNN

## Overview

[IREE](https://github.com/iree-org/iree/) is an open source ML
compiler stack built using MLIR that is intended to support the
compilation and execution of ML models. While IREE is setup to be
multi-targeting, over the past couple of years a lot of effort has
gone into improving the codegeneration for AMDGPUs, specifically
Instinct class GPUs. While a lot of the IREE compiler stack is meant
to optimize execution of full-scale ML models, one key component of
the work is to have efficient kernel code generation for MI300+
cards. [Fusilli](https://github.com/nod-ai/shark-ai/tree/main/sharkfuser)
is a C++ graph API and JIT fronend that leverages the kernel
codegeneration capabilities of IREE and packages it to be useable as a
kernel provider within hipDNN. This allows use of IREE for targeted
portions of the program, even for training use cases. The advantages
of using IREE this way are

1) IREE has been built from the ground-up as a fusion compiler. The
   kinds of fusions that libraries like hipDNN are expected to provide
   are supported out-of-the box in IREE.

2) Using a compiler as a kernel provider through Fusilli's JIT
   interface helps pick IREE generated kernels without having to ship
   pre-built kernels with hipDNN - saving both build time and space.

This RFC is to propose adding a path to using IREE as a kernel
provider to hipDNN within TheRock. There are three components to
reason about

1. The hipDNN backend plugin to Fusilli. Currently it lives
   [here](https://github.com/nod-ai/shark-ai/tree/main/fusilli-plugin)

2. Fusilli. Currently lives
   [here](https://github.com/nod-ai/shark-ai/tree/main/sharkfuser)

3. IREE, which lives in its own Github Org and is a Linux Foundation
   Project. It currently lives [here](https://github.com/iree-org/iree)

## Workplan

### Immediate next steps

The immediate workplan is to move just the hipDNN backend plugin to
Fusilli (so component 1 above) into TheRock. The plugin will be built
conditionally (not on by default) and will pull in Fusilli and IREE as
external dependencies. Nested dependencies of Fusilli are (TODO:
Adjust/clarify the following).

1. IREE runtime sources are built into Fusilli

2. IREE compile using the command-line interface, i.e. the
   `iree-compile` binary needs to be available (typically through
   pip-install of the IREE compiler package)

### Medium term/Long term requirements.

While the initial integration will just focus on pulling in the hipDNN
plugin to IREE into monorepo, long term the expectation is that
Fusilli and IREE are brought in through official release mechanisms
that allow TheRock to seamlessly pull them in (through the usual
versioning mechanisms). Some question that need to be answered for
those are

1. Where does Fusilli live? Fusilli is a C++ API around IREE and such
   is tightlt coupled with IREE. A natural home for Fusilli is within
   the same repo/github organization as IREE itself.

2. The expectation is that Fusilli will stop using `iree-compile` as a
   binary, but rather use the C-API of the IREE compiler to JIT
   compile the (fused) kernel computation. This would require
   significant changes to current IREE workflow. Apart from resolving
   where the IREE project lives, i.e. if it should move into the
   monorepo as well, another challenge to solve there is which LLVM
   version should IREE use. IREE currently tracks top-of-main of LLVM
   pretty closely. This would need to change to use either the LLVM
   version within monorepo or a release version of LLVM/MLIR.
