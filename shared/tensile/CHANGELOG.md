# Changelog for Tensile

## (Unreleased) Tensile 4.44.0

### Added

- Added support for gfx950
- Added code object compression via bundling
- Added support for non-default HIP SDK installations on Windows
- Added master solution library documentation
- Added compiler version dependent assembler and architecture capabilities
- Added documentation from GitHub Wiki to ROCm docs
- Added support for gfx1150

### Changed

- Loosened check for CLI compiler choices
- Introduced 4-tuple targets for bundler invocations
- Introduced PATHEXT extensions on Windows when searching for toolchain components
- Enabled passing fully qualified paths to toolchain components
- Enabled environment variable overrides when searching for a ROCm stack
- Improved default toolchain configuration
- Ignored f824 flake errors

### Removed

- Removed support for the gfx940 and gfx941 targets
- Removed unused tuning files
- Removed disabled tests

### Resolved issues

- Fixed configure time path not being invoked at build
- Fixed find_package for msgpack to work with versions 5 and 6
- Fixed rhel9 testing
- Fixed gfx908 builds
- Fixed "argument list too long" error
- Fixed version typo in 6.3 changelog
- Fixed improper use of aliases as nested namespace specifiers

## Tensile 4.43.0 for ROCm 6.4.0

### Added

- Nightly builds with performance statistics
- Cache asm capabilities for reuse
- venv for Tensile create on Linux
- Flag to keep build_tmp when running Tensile
- Generalized profiling scripts
- GFX1151 support
- Single-threaded support in TensileCreateLibrary
- Logic to remove temporary build artifacts

### Changed

- Updated Tensile documents (API reference, README.md, and comments)
- Disabled asm-cache for tests
- Used hipcc.bat as a compiler on Windows instead of the Perl script
- Improved clarity of CHANGELOG.md
- Enabled external CI
- Improved Tensile documentation
- Refactored kernel source and header creation
- Refactored writeKernels in TensileCreateLibrary
- Suppressed developer warnings (simplifying the Tensile output)
- Used an explicit cast when invoking min is called
- Used cache abbreviations to compute kernel names

### Removed

- OCL backend
- Unsupported tests
- Deep copy in TensileCreateLibrary

### Optimized

- Linearized asm register search to reduce build time

### Resolved issues

- Fixed Stream-K dynamic grid model
- Fixed logic related to caching asm capabilities
- Fixed accvgpr overflow
- Fixed test failures in SLES containers when running TensileTests
- Fixed a regression that prevents TensileCreateLibrary from completing when fallback logic is not available

## Tensile 4.42.0 for ROCm 6.3.0

### Added

- Contributor and developer guide
- Testing and documentation for `MasterSolutionLibrary.ArchitectureIndexMap` and `remapSolutionIndicesStartingFrom`
- gfx12 support
- Functions for writing master file
- `tPrint` and reconcile printing options
- Python unit test coverage report
- Factor embed library logic into function and test
- `clang++` as cxx-compiler option for Windows
- Logic to cope with different compilers
- `toFile` function to include `generateManifest` and moved to utilities
- Profiling CI job
- Support for `amdclang` and use defaults
- Architecture management functions in `TensileCreateLibrary`
- `TensileCreateLibrary` CLI reference docs
- New documentation for sphinx prototype and build out skeleton
- Prediction model for optimal number of Stream-K tiles to run
- Two-tile algorithm with Stream-K after DP
- Atomic two-tile Stream-K and clean-up tuning parameters
- Using glob to find logic files in `TensileCreateLibrary`
- Function to confirm supported compiler rather than raw logic

### Changed

- Improved rocBLAS build output by allowing warning suppression, ignoring developer warnings, displaying progress bar and quiet printing
- Reordered extensions for Windows in `which` function
- Updated `amdclang++` and `asm` directories
- Updated duplicate marking tests with mocks
- Restored print ordering
- Print option
- Bumped rocm-docs-core from 1.2.0 to 1.5.0 in `/docs/sphinx`
- Refactored kernel duplicate matching
- Refactored `generateLogicDataAndSolutions`
- Restricted XCC mapping to gfx942
- Refactored argument parsing in `TensileCreateLibrary`
- Disabled failing rhel9 tests
- Changed line length to 100 characters for formatting
- Changed YAML operations to use C `libyaml` backend
- Improved warning text
- Updated clang support for Windows
- Updated `supportedCompiler` function
- Clang support on Windows to require use of conditional choices and defaults
- Refactored sanity check in `TensileCreateLibrary`
- Moved client config logic from `TensileCreateLibrary` main into `createClientConfig`
- Updated `verifyManifest` in `TensileCreateLibrary`
- Updated RTD configs
- Cleaned up CMake to avoid redundant work during client builds
- Updated Stream-K debug settings

### Removed

- Deprecated flag from CI profiling job
- Diagnostic print
- Globals from `prepAsm`
- Deprecated `package-library` option
- Duplicate `which` function and minor cleanup

### Optimized

To optimize the performance of Stream-K kernels:

- Introduced analytical grid size prediction model
- Remapped XCC-based workgroup

### Resolved issues

- Fixed stream-K XCC configs for gfx942
- Updated WMMA capability command for ISA 10+
- Fixed progress bar character encoding error on Windows
- Fixed solution redundancy removal
- Fixed tuning imports for `pyyaml`
- Fixed printing of ASM capabilities for ROCm versions prior to 6.3
- Fixed code objects by filtering kernels with build errors and unprocessed kernels
- Fixed fully qualified `std::get` in contraction solutions
- Fixed `add -v flag` and change system invocation
- Used conditional imports for new dependencies to fix yaml `CSafe` load and dump import and rich terminal print import
- Fixed comments on `scalarStaticDivideAndRemainder`

## Tensile 4.41.0 for ROCm 6.2.0
### Additions
- new tuning script to summarize rocBLAS log file
- new environment variable to test fixed grid size with Stream-K kernels
- new Stream-K dynamic mode to run large problems at slightly reduced CU count if it improves work division and power
- add reject conditions for SourceKernel + PrefetchGlobalRead/LoopDoWhile
- add reject condition for PreloadKernelArguments (disable PreloadKernelArguments if not supported (instead of rejecting kernel generation))
- support NT flag for global load and store for gfx94x
- new Kernarg preloading feature (DelayRemainingArgument: initiate the load of the remaining (non-preloaded) arguments, updated AsmCaps, AsmRegisterPool to track registers for arguments and preload)
- add option for rotating buffers timing with cache eviction
- add predicate for arithmetic intensity
- add DirectToVgpr + packing for f8/f16 + TLU cases
- enable negative values for ExtraLatencyForLR to reduce interval of local read and wait for DTV
- add test cases for DirectToVgpr + packing
- add batch support for Stream-K kernels and new test cases
- new tuning scripts to analyze rocblas-bench results and remove tuned sizes from liblogic
- enable VgprForLocalReadPacking + PrefetchLocalRead=1 (removed the reject condition for VFLRP + PLR=1, added test cases for VFLRP + PLR=1)
- support VectorWidthB (new parameter VectorWidthB)
- support VectorWidth + non SourceSwap
- add test cases for VectorWidthB, VectorWidth + non SourceSwap
- add code owners file
- new environment variables to dynamically adjust number of CUs used in Stream-K
- add new parameters to specify global load width for A and B separately (GlobalLoadVectorWidthA, B (effective with GlobalReadVectorWidth=-1))
- add xf32 option to rocblas-bench input creator

### Optimizations
- initialization optimizations (reordered init code for PreloadKernelArguments opt, used s_mov_b64 for 64 bit address copy, used v_mov_b64/ds_read_b64 for C register initialization, added undefine AddressC/D with PreloadKernelArguments, optimized waitcnt for prefetch global read with DirectToVgpr, refactored waitcnt code for DTV and moved all asm related code to KernelWriterAssembly.py)
- optimize temp vgpr allocation for ClusterLocalRead (added if condition to allocate temp vgpr only for 8bit datatype)
- reverse MFMA order in inner loop for odd outer iteration
- optimize waitcnt lgkmcnt for 1LDSBuffer + PGR>1 (removed redundant waitcnt lgkmcnt after 1LDSBuffer sync)
- enhance maximum value of DepthU to 1024 (used globalParameters MaxDepthU to define maximum value of DepthU)

### Changes
- update rocBLAS-bench-input-create script (added number of iteration based on performance, rotating buffer flag)
- limit build threads based on CPUs/RAM available on system (for tests)
- update required workspace size for Stream-K, skip kernel initialization when possible
- use fallback libraries for archs without optimized logic
- use hipMemcpyAsync for validation (replace hipMemcpy with hipMemcpyAsync + hipStreamSynchronize in ReferenceValidator)
- remove OCL tests
- disable HostLibraryTests
- reduce extended test time by removing extra parameters in the test config files
- disable InitAccVgprOpt for Stream-K
- skip sgemm 64bit offset tests for gfx94x
- skip DTV, DTL, LSU+MFMA tests for gfx908
- increase extended test timeout to 720 min
- update xfail test (1sum tests only failing on gfx90a)
- update lib logic convertor script
- test limiting CI threads for only gfx11
- WGM related kernargs are removed if they are not needed (WGM=-1,0,1)
- cleanup on unused old code, mostly related to old client
- change GSUA to SingleBuffer if GlobalSplitU=1 + MultipleBuffer, instead of rejecting it
- update efficiency script for new architecture and xf32 datatype
- re-enable negative values for WorkGroupMapping (asm kernel only)
- disable HW monitor for aquvavanjaram941
- pre-apply offsets for strided batch kernels
- update tensile build with 16 threads

### Fixes
- fix WorkspaceCheck implementation when used in rocBLAS
- ignore asm cap check for kernel arg preload for rocm6.0 and older
- fix Stream-K partials cache behavior
- fix MasterSolutionLibrary indexing for multiple architecture build
- fix memory allocation fail with FlushMemorySize + StridedBatched/Batched cases (multiply batch count size when calculating array size)
- fix BufferLoad=False with Stream-K
- fix mismatch issue with GlobalReadCoalesceGroup
- fix rocblas build fail on gfx11 (used state["ISA"] for reject conditions instead of globalParameters["CurrentISA"])
- fix for LdsPad auto (fixed incorrect value assignment for autoAdjusted, set LdsBlockSizePerPadA or B = 0 if stride is not power of 2)
- fix inacurate vgpr allocation for ClusterLocalRead
- fix mismatch issue with LdsBlockSizePerPad + MT1(or 0) not power of 2
- fix mismatch issue with InitAccOpt + InnerUnroll (use const 0 for src1 of MFMA only if index of innerUnrll (iui) is 0)
- fix HostLibraryTests on gfx942 and gfx941
- fix LLVM crash issue
- fix for newer windows vcpkg msgpack and vcpkg version package name
- fix an error with DisableKernelPieces + 32bit ShadowLimit

## Tensile 4.40.0 for ROCm 6.1.0
### Additions
- new DisableKernelPieces values to invalidate local read, local write, and global read
- stream-K kernel generation, including two-tile stream-k algorithm by setting StreamK=3
- feature to allow testing stream-k grid multipliers
- debug output to check occupancy for Stream-K
- reject condition for FractionalLoad + DepthU!=power of 2
- new TENSILE_DB debugging value to dump the common kernel parameters
- predicate for APU libs
- new parameter (ClusterLocalRead) to turn on/off wider local read opt for TileMajorLDS
- new parameter (ExtraLatencyForLR) to add extra interval between local read and wait
- new logic to check LDS size with auto LdsPad(=1) and change LdsPad to 0 if LDS overflows
- initialization type and general batched options to the rocblas-bench input creator script

### Optimizations
- enabled MFMA + LocalSplitU=4 for MT16x16
- enabled (DirectToVgpr + MI4x4) and supported skinny MacroTile
- optimized postGSU kernel: separate postGSU kernels for different GSU values, loop unroll for GSU loop, wider global load depending on array size, and parallel reduction depending on array size
- auto LdsPad calculation for TileMajorLds + MI16x16
- auto LdsPad calculation for UnrollMajorLds + MI16x16 + VectorWidth

### Changes
- cleared hipErrorNotFound error since it is an expected part of the search
- modified hipcc search path for Linux
- changed PCI ID from 32bit to 64bit for ROCm SMI HW monitor
- changed LdsBlockSizePerPad to LdsBlockSizePerPadA, B to specify LBSPP separately
- changed the default value of LdsPadA, B, LdsBlockSizePerPadA, B from 0 to -1
- updated test cases according to parameter changes for LdsPad, LBSPP and ClusterLocalRead
- Replaced std::regex with fnmatch()/PathMatchSpec as a workaround to std::regex stack overflow known bug

### Fixes
- hipcc compile append flag parallel-jobs=4
- race condition in Stream-K that appeared with large grids and small sizes
- mismatch issue with LdsPad + LdsBlockSizePerPad!=0 and TailLoop
- mismatch issue with LdsPad + LdsBlockSizePerPad!=0 and SplitLds
- incorrect reject condition check for DirectToLds + LdsBlockSizePerPad=-1 case
- small fix for LdsPad optimization (LdsElement calculation)

## Tensile 4.39.0 for ROCm 6.0
### Added
- Added aquavanjaram support: gfx940/gfx941/gfx942, fp8/bf8 datatype, xf32 datatype, and stochastic rounding for various datatypes
- Added/updated tuning scripts
- Added DirectToLds support for larger data types with 32bit global load (old parameter DirectToLds is replaced with DirectToLdsA and DirectToLdsB), and the corresponding test cases
- Added the average of frequency, power consumption, and temperature information for the winner kernels to the CSV file
- Added asmcap check for MFMA + const src
- Added support for wider local read + pack with v_perm (with VgprForLocalReadPacking=True)
- Added a new parameter to increase miLatencyLeft

### Optimizations
- Enabled InitAccVgprOpt for MatrixInstruction cases
- Implemented local read related parameter calculations with DirectToVgpr
- Adjusted miIssueLatency for gfx940
- Enabled dedicated vgpr allocation for local read + pack
- Optimized code initialization
- Optimized sgpr allocation
- Supported DGEMM TLUB + RLVW=2 for odd N (edge shift change)
- Enabled miLatency optimization for (gfx940/gfx941 + MFMA) for specific data types, and fixed instruction scheduling

### Changed
- Removed old code for DTL + (bpe * GlobalReadVectorWidth > 4)
- Changed/updated failed CI tests for gfx11xx, InitAccVgprOpt, and DTLds
- Removed unused CustomKernels and ReplacementKernels
- Added a reject condition for DTVB + TransposeLDS=False (not supported so far)
- Removed unused code for DirectToLds
- Updated test cases for DTV + TransposeLDS=False
- Moved parameter MinKForGSU from globalparameter to BenchmarkCommonParameter to support smaller K
- Changed how to calculate latencyForLR for miLatency
- Set minimum value of latencyForLRCount for 1LDSBuffer to avoid getting rejected by overflowedResources=5 (related to miLatency)
- Refactored allowLRVWBforTLUandMI and renamed it as VectorWidthB
- Supported multi-gpu for different architectures in lazy library loading
- Enabled dtree library for batch > 1
- Added problem scale feature for dtree selection
- Enabled ROCm SMI for gfx940/941.
- Modified non-lazy load build to skip experimental logic

### Fixed
- Fixed predicate ordering for fp16alt impl round near zero mode to unbreak distance modes
- Fixed boundary check for mirror dims and re-enable disabled mirror dims test cases
- Fixed merge error affecting i8 with wmma
- Fixed mismatch issue with DTLds + TSGR + TailLoop
- Fixed a bug with InitAccVgprOpt + GSU>1 and a mismatch issue with PGR=0
- Fixed override for unloaded solutions when lazy loading
- Fixed build some errors (adding missing headers)
- Fixed boost link for a clean build on ubuntu22
- Fixed bug in forcestoresc1 arch selection
- Fixed compiler directive for gfx941 and gfx942
- Fixed formatting for DecisionTree_test.cpp

## Tensile 4.38.0 for ROCm 5.7
### Added
- Added support for FP16 Alt Round Near Zero Mode (this feature allows the generation of alternate kernels with intermediate rounding instead of truncation)
- Added user-driven solution selection feature
### Optimizations
- Enabled LocalSplitU with MFMA for I8 data type
- Optimized K mask code in mfmaIter
- Enabled TailLoop code in NoLoadLoop to prefetch global/local read
- Enabled DirectToVgpr in TailLoop for NN, TN, and TT matrix orientations
- Optimized DirectToLds test cases to reduce the test duration
### Changed
- Removed DGEMM NT custom kernels and related test cases
- Changed noTailLoop logic to apply noTailLoop only for NT
- Changed the range of AssertFree0ElementMultiple and Free1
- Unified aStr, bStr generation code in mfmaIter
### Fixed
- Fixed LocalSplitU mismatch issue for SGEMM
- Fixed BufferStore=0 and Ldc != Ldd case
- Fixed mismatch issue with TailLoop + MatrixInstB > 1

## Tensile 4.37.0 for ROCm 5.6
### Added
- Added user driven tuning API
- Added decision tree fallback feature
- Added SingleBuffer + AtomicAdd option for GlobalSplitU
- DirectToVgpr support for fp16 and Int8 with TN orientation
- Added new test cases for various functions
- Added SingleBuffer algorithm for ZGEMM/CGEMM
- Added joblib for parallel map calls
- Added support for MFMA + LocalSplitU + DirectToVgprA+B
- Added asmcap check for MIArchVgpr
- Added support for MFMA + LocalSplitU
- Added frequency, power, and temperature data to the output
### Optimizations
- Improved the performance of GlobalSplitU with SingleBuffer algorithm
- Reduced the running time of the extended and pre_checkin tests
- Optimized the Tailloop section of the assembly kernel
- Optimized complex GEMM (fixed vgpr allocation, unified CGEMM and ZGEMM code in MulMIoutAlphaToArch)
- Improved the performance of the second kernel of MultipleBuffer algorithm
### Changed
- Updated custom kernels with 64-bit offsets
- Adapted 64-bit offset arguments for assembly kernels
- Improved temporary register re-use to reduce max sgpr usage
- Removed some restrictions on VectorWidth and DirectToVgpr
- Updated the dependency requirements for Tensile
- Changed the range of AssertSummationElementMultiple
- Modified the error messages for more clarity
- Changed DivideAndReminder to vectorStaticRemainder in case quotient is not used
- Removed dummy vgpr for vectorStaticRemainder
- Removed tmpVgpr parameter from vectorStaticRemainder/Divide/DivideAndReminder
- Removed qReg parameter from vectorStaticRemainder
### Fixed
- Fixed tmp sgpr allocation to avoid over-writing values (alpha)
- 64-bit offset parameters for post kernels
- Fixed gfx908 CI test failures
- Fixed offset calculation to prevent overflow for large offsets
- Fixed issues when BufferLoad and BufferStore are equal to zero
- Fixed StoreCInUnroll + DirectToVgpr + no useInitAccVgprOpt mismatch
- Fixed DirectToVgpr + LocalSplitU + FractionalLoad mismatch
- Fixed the memory access error related to StaggerU + large stride
- Fixed ZGEMM 4x4 MatrixInst mismatch
- Fixed DGEMM 4x4 MatrixInst mismatch
- Fixed ASEM + GSU + NoTailLoop opt mismatch
- Fixed AssertSummationElementMultiple + GlobalSplitU issues
- Fixed ASEM + GSU + TailLoop inner unroll

## Tensile 4.36.0 for ROCm 5.5.0
### Added
- Add functions for user-driven tuning
- Add GFX11 support: HostLibraryTests yamls, rearragne FP32(C)/FP64(C) instruction order, archCaps for instruction renaming condition, adjust vgpr bank for A/B/C for optimize, separate vscnt and vmcnt, dual mac
- Add binary search for Grid-Based algorithm
- Add reject condition for (StoreCInUnroll + BufferStore=0) and (DirectToVgpr + ScheduleIterAlg<3 + PrefetchGlobalRead==2)
- Add support for (DirectToLds + hgemm + NN/NT/TT) and (DirectToLds + hgemm + GlobalLoadVectorWidth < 4)
- Add support for (DirectToLds + hgemm(TLU=True only) or sgemm + NumLoadsCoalesced > 1)
- Add GSU SingleBuffer algorithm for HSS/BSS
- Add gfx900:xnack-, gfx1032, gfx1034, gfx1035
- Enable gfx1031 support
### Optimizations
- Use AssertSizeLessThan for BufferStoreOffsetLimitCheck if it is smaller than MT1
- Improve InitAccVgprOpt
### Changed
- Use global_atomic for GSU instead of flat and global_store for debug code
- Replace flat_load/store with global_load/store
- Use global_load/store for BufferLoad/Store=0 and enable scheduling
- LocalSplitU support for HGEMM+HPA when MFMA disabled
- Update Code Object Version
- Type cast local memory to COMPUTE_DATA_TYPE in LDS to avoid precision loss
- Update asm cap cache arguments
- Unify SplitGlobalRead into ThreadSeparateGlobalRead and remove SplitGlobalRead
- Change checks, error messages, assembly syntax, and coverage for DirectToLds
- Remove unused cmake file
- Clean up the LLVM dependency code
- Update ThreadSeparateGlobalRead test cases for PrefetchGlobalRead=2
- Update sgemm/hgemm test cases for DirectToLds and ThreadSepareteGlobalRead
### Fixed
- Add build-id to header of compiled source kernels
- Fix solution index collisions
- Fix h beta vectorwidth4 correctness issue for WMMA
- Fix an error with BufferStore=0
- Fix mismatch issue with (StoreCInUnroll + PrefetchGlobalRead=2)
- Fix MoveMIoutToArch bug
- Fix flat load correctness issue on I8 and flat store correctness issue
- Fix mismatch issue with BufferLoad=0 + TailLoop for large array sizes
- Fix code generation error with BufferStore=0 and StoreCInUnrollPostLoop
- Fix issues with DirectToVgpr + ScheduleIterAlg<3
- Fix mismatch issue with DGEMM TT + LocalReadVectorWidth=2
- Fix mismatch issue with PrefetchGlobalRead=2
- Fix mismatch issue with DirectToVgpr + PrefetchGlobalRead=2 + small tile size
- Fix an error with PersistentKernel=0 + PrefetchAcrossPersistent=1 + PrefetchAcrossPersistentMode=1
- Fix mismatch issue with DirectToVgpr + DirectToLds + only 1 iteration in unroll loop case
- Remove duplicate GSU kernels: for GSU = 1, GSUAlgorithm SingleBuffer and MultipleBuffer kernels are identical
- Fix for failing CI tests due to CpuThreads=0
- Fix mismatch issue with DirectToLds + PrefetchGlobalRead=2
- Remove the reject condition for ThreadSeparateGlobalRead and DirectToLds (HGEMM, SGEMM only)
- Modify reject condition for minimum lanes of ThreadSeparateGlobalRead (SGEMM or larger data type only)

## Tensile 4.35.0 for ROCm 5.4.0
### Added
- Async DMA support for Transpose Data Layout (ThreadSeparateGlobalReadA/B)
- Option to output library logic in dictionary format
- No solution found error message for benchmarking client
- Exact K check for StoreCInUnrollExact
- Support for CGEMM + MIArchVgpr
- client-path parameter for using prebuilt client
- CleanUpBuildFiles global parameter
- Debug flag for printing library logic index of winning solution
- NumWarmups global parameter for benchmarking
- Windows support for benchmarking client
- DirectToVgpr support for CGEMM
- TensileLibLogicToYaml for creating tuning configs from library logic solutions
### Optimizations
- Put beta code and store separately if StoreCInUnroll = x4 store
- Improved performance for StoreCInUnroll + b128 store
### Changed
- Re-enable HardwareMonitor for gfx90a
- Decision trees use MLFeatures instead of Properties
### Fixed
- Reject DirectToVgpr + MatrixInstBM/BN > 1
- Fix benchmark timings when using warmups and/or validation
- Fix mismatch issue with DirectToVgprB + VectorWidth > 1
- Fix mismatch issue with DirectToLds + NumLoadsCoalesced > 1 + TailLoop
- Fix incorrect reject condition for DirectToVgpr
- Fix reject condition for DirectToVgpr + MIWaveTile < VectorWidth
- Fix incorrect instruction generation with StoreCInUnroll

## Tensile 4.34.0 for ROCm 5.3.0
### Added
- Lazy loading of solution libraries and code object files
- Support for dictionary style logic files
- Support for decision tree based logic files using dictionary format
- DecisionTreeLibrary for solution selection
- DirectToLDS support for HGEMM
- DirectToVgpr support for SGEMM
- Grid based distance metric for solution selection
- Support for gfx11xx
- Support for DirectToVgprA/B + TLU=False
- ForkParameters Groups as a way of specifying solution parameters
- Support for a new Tensile yaml config format
- TensileClientConfig for generating Tensile client config files
- Options for TensileCreateLibrary to build client and create client config file
### Optimizations
- Solution generation is now cached and is not repeated if solution parameters are unchanged
### Changed
- Default MACInstruction to FMA
### Fixed
- Accept StaggerUStride=0 as valid
- Reject invalid data types for UnrollLoopEfficiencyEnable
- Fix invalid code generation issues related to DirectToVgpr
- Return hipErrorNotFound if no modules are loaded
- Fix performance drop for NN ZGEMM with 96x64 macro tile
- Fix memory violation for general batched kernels when alpha/beta/K = 0

## Tensile 4.33.0 for ROCm 5.2.0
### Added
- TensileUpdateLibrary for updating old library logic files
- Support for TensileRetuneLibrary to use sizes from separate file
- ZGEMM DirectToVgpr/DirectToLds/StoreCInUnroll/MIArchVgpr support
- Tests for denorm correctness
- Option to write different architectures to different TensileLibrary files
### Optimizations
- Optimize MessagePackLoadLibraryFile by switching to fread
- DGEMM tail loop optimization for PrefetchAcrossPersistentMode=1/DirectToVgpr
### Changed
- Alpha/beta datatype remains as F32 for HPA HGEMM
- Force assembly kernels to not flush denorms
- Use hipDeviceAttributePhysicalMultiProcessorCount as multiProcessorCount
### Fixed
- Fix segmentation fault when run i8 datatype with TENSILE_DB=0x80

## Tensile 4.32.0 for ROCm 5.1.0
### Added
- Better control of parallelism to control memory usage
- Support for multiprocessing on Windows for TensileCreateLibrary
- New JSD metric and metric selection functionality
- Initial changes to support two-tier solution selection
### Optimizations
- Optimized runtime of TensileCreateLibraries by reducing max RAM usage
- StoreCInUnroll additional optimizations plus adaptive K support
- DGEMM NN optimizations with PrefetchGlobalRead(PGR)=2 support
### Changed
- Update Googletest to 1.11.0
### Removed
- Remove no longer supported benchmarking steps

## Tensile 4.31.0 for ROCm 5.0.0
### Added
- DirectToLds support (x2/x4)
- DirectToVgpr support for DGEMM
- Parameter to control number of files kernels are merged into to better parallelize kernel compilation
- FP16 alternate implementation for HPA HGEMM on aldebaran
### Optimizations
- Add DGEMM NN custom kernel for HPL on aldebaran
### Changed
- Update tensile_client executable to std=c++14
### Removed
- Remove unused old Tensile client code
### Fixed
- Fix hipErrorInvalidHandle during benchmarks
- Fix addrVgpr for atomic GSU
- Fix for Python 3.8: add case for Constant nodeType
- Fix architecture mapping for gfx1011 and gfx1012
- Fix PrintSolutionRejectionReason verbiage in KernelWriter.py
- Fix vgpr alignment problem when enabling flat buffer load

## Tensile 4.30.0 for ROCm 4.5.0
### Added
- Custom Kernel mechanism for adding custom assembly kernels to Tensile
- New assertions for problems sizes, alpha/beta values, and C equals D
- Support setting VectorWidth in M dimension in MFMA SourceSwap configuration
### Fixed
- Fix merge.py keeping duplicate solutions
- Fix ScheduleIterAlg 2,3 cases for aldebaran

## Tensile 4.28.0 for ROCm 4.3.0
### Added
- TensileRetuneLibrary for updating existing library logic files
- Support GFX1030
- Support NHWC

### Fixed
- TensileCreateLibrary crash with relative output and --merge-files

### Changed
- Change cmake_minimum_required to VERSION 3.13

## Tensile 4.27.0 for ROCm 4.2.0
### Added
- Benchmarking and library support for CU efficiency vs. overall speed
- support general batch GEMM
- Support offset for each input/output buffer in Tensile
- support support ldc != ldd for all GEMM kernel

### Optimizations
- Refactor ConvolutionVsContraction

### Fixed
- Fixed MasterSolutionLibrary having duplicated hardware rows
- channel stride is incorrect when converting conv problem into tensor contraction problem

## Tensile 4.26.0 for ROCm 4.1.0
### Added
- Make messagepack python dependency optional
- TensileCreateLibraryFiles: auto create target for build time lib generation
- Tensile cluster tuning tool
- Framework for filtering solutions
- Workflow for manually editing Kernels
- Tuning client design doc
- MatrixInstruction for general int8
- Tensile integration test for TensileCreateLibrary
- Trig float and random narrow init patterns for new client
- Summation dimension mirroring (contributed by timlathy & Slimakanzer)
- ROCm 4.1 TargetID support in Tensile; source kernels force xnack=OFF
- Tensile/Utilities/merge.py revamp for merging logic yaml files
  - now merge.py requires python3
  - add `-v` verbosity levels (up to 2)
  - add `--notrim` to retain leading dimensions in sizes
- New BoundsCheck design: Access guard page will trigger memory fault
- Solution fitness metric
- Auto-tuning documentation and build script improvements
- Support for High Precision Accumulate FP16/BF16 In FP32 Out
- CHANGELOG.md

### Optimizations
- Refine PersistentKernel: support PKn1, EPS, optimize LW-vmcnt and sMagicDiv2

### Fixed
- targets to clang-offload-bundler updated to use hipv4 prefix when appropriate
- Fix bugs of tail-loop branch label, and LR addr restore
- locateExe in Tensile/Common.py looks in defaultPath first
- Honor $ENV{ROCM_PATH} to support relocatable ROCm location
