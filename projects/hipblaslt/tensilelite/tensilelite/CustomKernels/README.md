<!--
Copyright (C) 2025 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Custom kernel validation

In the context of TensileCreateLibrary, Custom kernel meta data is taken from the logic
file where the custom kernel is referenced. The metadata from the logic file can be
overwritten by specifying the parameters of interest in the `custom.config` section of
the kernel assembly file. The majority of the override parameters are validated at build
time when building a custom kernel with exception of the nine element MatrixInstruction.
While this validation is no longer applied during the build process it can be applied by
using the **TensileLogic** script as follows:

```
Tensile/bin/TensileLogic --check-only-custom-kernels <path to logic file>
```

where `<path to logic file>` is a path to a logic file containing a solution where
`CustomKernelName` references the custom kernel requiring validation.
