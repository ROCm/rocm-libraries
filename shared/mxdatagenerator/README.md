# Temporary mxDataGenerator compatibility project

The mxDataGenerator implementation has been removed. Its live CPU numerical
functionality and physical layout transforms now live in ROCHostValidation.

This directory contains no numerical code or installed headers. It exists only
because the TheRock revision currently pinned by rocm-libraries still
configures and stages `shared/mxdatagenerator` as a support subproject.

Do not add new consumers or functionality here.

Delete this directory after:

- TheRock removes the `mxDataGenerator` support subproject and build
  dependencies; and
- rocm-libraries advances its TheRock pin to include that change.
