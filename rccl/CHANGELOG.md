# Changelog for RCCL

Full documentation for RCCL is available at [https://rccl.readthedocs.io](https://rccl.readthedocs.io)

## Unreleased - RCCL 2.22.3 for ROCm 6.5.0

### Added

* Added new GPU target `gfx950`
* Added support for `unroll=1` in device-code generation to improve performance

### Known issue

* Using more than 64 channels can cause a segmentation fault when multiple different collectives are used in the same `ncclGroup()` call

## Unreleased - RCCL 2.22.3 for ROCm 6.4.0

### Added

* `RCCL_SOCKET_REUSEADDR` and `RCCL_SOCKET_LINGER` environment parameters
* Setting `NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=VERBS` will generate traces for fifo and data ibv_post_sends
* Added `--log-trace` flag to enable traces through the install.sh script (e.g. `./install.sh --log-trace`)

### Changed

* Compatibility with NCCL 2.22.3
* Added support for the rail-optimized tree algorithm for the MI300 series. This feature requires the use of all eight GPUs within
  each node. It limits NIC traffic to use only GPUs of the same index across nodes and should not impact performance
  on non-rail-optimized network topologies. The original method of building trees can be enabled by setting the
  environment variable `RCCL_DISABLE_RAIL_TREES=1`.
* Additional debug information about how the trees are built can be logged to the GRAPH logging subsys by setting
  `RCCL_OUTPUT_TREES=1`.
* Added documentation about the NPS4 and CPX partition modes performance benefits on the MI300X.

## RCCL 2.21.5 for ROCm 6.3.1

### Added

### Changed

* Enhanced user documentation

### Resolved issues

* Corrected user help strings in `install.sh`

## RCCL 2.21.5 for ROCm 6.3.0

### Added

* MSCCL++ integration for AllReduce and AllGather on gfx942
* Performance collection to rccl_replayer
* Tuner Plugin example for MI300
* Tuning table for large number of nodes
* Support for amdclang++
* Allow NIC ID remapping using `NCCL_RINGS_REMAP` environment variable

### Changed

* Compatibility with NCCL 2.21.5
* Increased channel count for MI300X multi-node
* Enabled MSCCL for single-process multi-threaded contexts
* Enabled gfx12
* Enabled CPX mode for MI300X
* Enabled tracing with rocprof
* Improved version reporting
* Enabled GDRDMA for Linux kernel 6.4.0+

### Resolved issues

* Fixed model matching with PXN enable

## RCCL 2.20.5 for ROCm 6.2.1
### Fixed
- GDR support flag now set with DMABUF
### Known issues
- On systems running Linux kernel 6.8.0, such as Ubuntu 24.04, Direct Memory Access (DMA) transfers between the GPU and NIC are disabled and impacts multi-node RCCL performance.
  - This issue was reproduced with RCCL 2.20.5 (ROCm 6.2.0 and 6.2.1) on systems with Broadcom Thor-2 NICs and affects other systems with RoCE networks using Linux 6.8.0 or newer.
  - Older RCCL versions are also impacted.
  - This issue will be addressed in a future ROCm release.

## RCCL 2.20.5 for ROCm 6.2.0
### Changed
- Compatibility with NCCL 2.20.5
- Compatibility with NCCL 2.19.4
- Performance tuning for some collective operations on MI300
- Enabled NVTX code in RCCL
- Replaced rccl_bfloat16 with hip_bfloat16
- NPKit updates:
  - Removed warm-up iteration removal by default, need to opt in now
  - Doubled the size of buffers to accommodate for more channels
- Modified rings to be rail-optimized topology friendly
- Replaced ROCmSoftwarePlatform links with ROCm links
### Added
- Support for fp8 and rccl_bfloat8
- Support for using HIP contiguous memory
- Implemented ROC-TX for host-side profiling
- Enabled static build
- Added new rome model
- Added fp16 and fp8 cases to unit tests
- New unit test for main kernel stack size
- New -n option for topo_expl to override # of nodes
- Improved debug messages of memory allocations
### Fixed
- Bug when configuring RCCL for only LL128 protocol
- Scratch memory allocation after API change for MSCCL

## RCCL 2.18.6 for ROCm 6.1.0
### Changed
- Compatibility with NCCL 2.18.6

## RCCL 2.18.3 for ROCm 6.0.0
### Changed
- Compatibility with NCCL 2.18.3

## RCCL 2.17.1-1 for ROCm 5.7.0
### Changed
- Compatibility with NCCL 2.17.1-1
- Performance tuning for some collective operations
### Added
- Minor improvements to MSCCL codepath
- NCCL_NCHANNELS_PER_PEER support
- Improved compilation performance
- Support for gfx94x
### Fixed
- Potential race-condition during ncclSocketClose()

## RCCL 2.16.2 for ROCm 5.6.0
### Changed
- Compatibility with NCCL 2.16.2
### Fixed
- Remove workaround and use indirect function call

## RCCL 2.15.5 for ROCm 5.5.0
### Changed
- Compatibility with NCCL 2.15.5
- Unit test executable renamed to rccl-UnitTests
### Added
- HW-topology aware binary tree implementation
- Experimental support for MSCCL
- New unit tests for hipGraph support
- NPKit integration
### Fixed
- rocm-smi ID conversion
- Support for HIP_VISIBLE_DEVICES for unit tests
- Support for p2p transfers to non (HIP) visible devices
### Removed
- Removed TransferBench from tools.  Exists in standalone repo: https://github.com/ROCm/TransferBench

## RCCL-2.13.4 for ROCm 5.4.0
### Changed
- Compatibility with NCCL 2.13.4
- Improvements to RCCL when running with hipGraphs
- RCCL_ENABLE_HIPGRAPH environment variable is no longer necessary to enable hipGraph support
- Minor latency improvements
### Fixed
- Resolved potential memory access error due to asynchronous memset

## RCCL-2.12.10 for ROCm 5.3.0
### Changed
- Improvements to LL128 algorithms
### Added
- Adding initial hipGraph support via opt-in environment variable RCCL_ENABLE_HIPGRAPH
- Integrating with NPKit (https://github.com/microsoft/NPKit) profiling code

## RCCL-2.12.10 for ROCm 5.2.3
### Added
- Compatibility with NCCL 2.12.10
- Packages for test and benchmark executables on all supported OSes using CPack.
- Adding custom signal handler - opt-in with RCCL_ENABLE_SIGNALHANDLER=1
  - Additional details provided if Binary File Descriptor library (BFD) is pre-installed
- Adding support for reusing ports in NET/IB channels
  - Opt-in with NCCL_IB_SOCK_CLIENT_PORT_REUSE=1 and NCCL_IB_SOCK_SERVER_PORT_REUSE=1
  - When "Call to bind failed : Address already in use" error happens in large-scale AlltoAll
    (e.g., >=64 MI200 nodes), users are suggested to opt-in either one or both of the options
    to resolve the massive port usage issue
  - Avoid using NCCL_IB_SOCK_SERVER_PORT_REUSE when NCCL_NCHANNELS_PER_NET_PEER is tuned >1
### Removed
- Removed experimental clique-based kernels

## RCCL-2.11.4 for ROCm 5.2.0
### Changed
- Unit testing framework rework
- Minor bug fixes
### Known issues
- Managed memory is not currently supported for clique-based kernels

## RCCL-2.11.4 for ROCm 5.1.0
### Added
- Compatibility with NCCL 2.11.4
### Known issues
- Managed memory is not currently supported for clique-based kernels

## RCCL-2.10.3 for ROCm 5.0.0
### Added
- Compatibility with NCCL 2.10.3
### Known issues
- Managed memory is not currently supported for clique-based kernels

## RCCL-2.9.9 for ROCm 4.5.0
### Changed
- Packaging split into a runtime package called rccl and a development package called rccl-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.
### Added
- Compatibility with NCCL 2.9.9
### Known issues
- Managed memory is not currently supported for clique-based kernels

## [RCCL-2.8.4 for ROCm 4.3.0]
### Added
- Ability to select the number of channels to use for clique-based all reduce (RCCL_CLIQUE_ALLREDUCE_NCHANNELS).  This can be adjusted to tune for performance when computation kernels are being executed in parallel.
### Optimizations
- Additional tuning for clique-based kernel AllReduce performance (still requires opt in with RCCL_ENABLE_CLIQUE=1)
- Modification of default values for number of channels / byte limits for clique-based all reduce based on device architecture
### Changed
- Replaced RCCL_FORCE_ENABLE_CLIQUE to RCCL_CLIQUE_IGNORE_TOPO
- Clique-based kernels can now be enabled on topologies where all active GPUs are XGMI-connected
- Topologies not normally supported by clique-based kernels require RCCL_CLIQUE_IGNORE_TOPO=1
### Fixed
- Install script '-r' flag invoked alone no longer incorrectly deletes any existing builds.
### Known issues
- Managed memory is not currently supported for clique-based kernels

## [RCCL-2.8.4 for ROCm 4.2.0]
### Added
- Compatibility with NCCL 2.8.4

### Optimizations
- Additional tuning for clique-based kernels
- Enabling GPU direct RDMA read from GPU
- Fixing potential memory leak issue when re-creating multiple communicators within same process
- Improved topology detection
### Known issues
- None

## [RCCL-2.7.8 for ROCm 4.1.0]
### Added
- Experimental support for clique-based kernels (opt in with RCCL_ENABLE_CLIQUE=1)
- Clique-based kernels may offer better performance for smaller input sizes
- Clique-based kernels are currently only enabled for AllReduce under a certain byte limit (controlled via RCCL_CLIQUE_ALLREDUCE_BYTE_LIMIT)
### Optimizations
- Performance improvements for Rome-based systems
### Known issues
- Clique-based kernels are currently experimental and have not been fully tested on all topologies.  By default, clique-based kernels are disabled if the detected topology is not supported (override with RCCL_FORCE_ENABLE_CLIQUE)
- Clique-based kernels may hang if there are differences between environment variables set across ranks.
- Clique-based kernels may fail if the input / output device pointers are not the base device pointers returned by hipMalloc.


## [RCCL-2.7.8 for ROCm 3.9.0]
### Added
- Adding support for alltoallv RCCL kernel
### Optimizations
- Modifications to topology based on XGMI links
### Known issues
- None

## [RCCL-2.7.6 for ROCm 3.8.0]
### Added
- Support for static library builds
### Known issues
- None

## [RCCL-2.7.6 for ROCm 3.7.0]
### Added
- Updated to RCCL API version of 2.7.6
- Added gather, scatter and all-to-all collectives

## [RCCL-2.7.0 for ROCm 3.6.0]
### Added
- Updated to RCCL API version of 2.6.4

## [RCCL-2.7.0 for ROCm 3.5.0]
### Added
- Compatibility with NCCL 2.6
- Network interface improvements with API v3
### Optimizations
- Fixing issues and built time improvements for hip-clang
- Network topology detection
- Improved CPU type detection
- Infiniband adaptive routing support
### Changed
- Switched to hip-clang as default compiler
### Deprecated
- Deprecated hcc build
