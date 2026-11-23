# hipCCL (hipccl2 layout - legacy compatibility snapshot)

This directory is the legacy hipCCL layout: a frozen snapshot of `rocprim/`,
`hipcub/`, and `rocthrust/` from before the CCCL 3.x alignment work
(rocPRIM, hipCUB, and rocThrust at 4.7.0; hipCUB tracks CCCL 2.8.2). Their
library and algorithm code is not kept in sync with upstream. The unified,
actively developed layout is [`../hipccl3/`](../hipccl3/).

A root-level [`CMakeLists.txt`](CMakeLists.txt) builds, versions, and packages
the three components together, mirroring `hipccl3`:

* one umbrella version (`HIPCCL_VERSION`, 2.8.0) for the `hipccl` package;
  each component keeps its own version,
* headers installed under `<prefix>/include/hipccl/<component>`, with the
  optional `HIPCCL_INSTALL_LEGACY_HEADER_SYMLINKS` compatibility paths,
* one `hipccl` package and one `find_package(hipccl)` entry point,
* the same `HIPCCL_BUILD_ROCPRIM`, `HIPCCL_BUILD_HIPCUB`,
  `HIPCCL_BUILD_ROCTHRUST`, and `HIPCCL_BUILD_ALL` options.

Unlike `hipccl3`, there is no libhipcxx and no `HIPCCL_BUILD_LIBHIPCXX` option.

The component copies carry the same build-system changes as `hipccl3` that are
needed to nest them in one build (for example, standalone packaging is
disabled and `generate_resource_spec` targets are renamed per component). See
[`../docs/migration-notes.md`](../docs/migration-notes.md).

To build it, use the layout selector in [`../`](../) with
`-DHIPCCL_BUILD_LEGACY=ON`, or run `cmake` from this directory. See
[`../docs/install/build.rst`](../docs/install/build.rst).

hipCCL2 is temporary and will be removed in a future release; see
[`CHANGELOG.md`](CHANGELOG.md).
