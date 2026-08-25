---
id: project-hipdnn
title: "hipDNN (convolution path)"
type: project
repo: ROCm/rocm-libraries
tree: projects/hipdnn
tags: [hipdnn, convolution]
operator_families: [convolution]
architecture_families: [cdna, rdna]
related: [family-convolution, project-miopen, kernel-conv-implicit-gemm]
kernel_types: [convolution]
---

# hipDNN convolution

There is no `projects/hipconv` directory on `develop`. Convolution in this
monorepo’s graph API lives under hipDNN:
`projects/hipdnn/backend/...Convolution{Fwd,Bwd,Wrw}OperationDescriptor.*`
plus flatbuffer attributes.

Use hipDNN as the **API/graph** source; MIOpen and rocke/CK Tile as the
**kernel** sources. When someone says “hipconv,” start here and at
`family-convolution`.
