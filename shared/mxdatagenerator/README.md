# Temporary mxDataGenerator compatibility project

The mxDataGenerator implementation has been removed. Its live CPU numerical
functionality and physical layout transforms now live in ROCHostNumerics.

This directory contains no numerical code or installed headers. It exists only
because the TheRock revision currently pinned by rocm-libraries still
configures and stages `shared/mxdatagenerator` as a support subproject.

`MXDATAGENERATOR_BUILD_TESTING=ON` builds the temporary
`mxDataGeneratorTests` compatibility-target smoke executable expected by older
Jenkins jobs. It links only the empty `roc::mxDataGenerator` interface target
and does not restore any removed numerical implementation.

Do not add new consumers or functionality here.

Delete this directory after:

- TheRock removes the `mxDataGenerator` support subproject and build
  dependencies; and
- rocm-libraries advances its TheRock pin to include that change.
