# Changelog for hipTensor

Full documentation for hipTensor is available at [rocm.docs.amd.com/projects/hiptensor](https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html).

## hipTensor 2.6.0 for ROCm 10.3

### Added
* Added support for the following new GPU targets:
  * gfx11-7: gfx1170, gfx1171.
* Added a backend interface between the solution layer and the kernels. `hiptensorCreatePlan` selects and prepares the backend of every operation, and executions run on that backend. The Composable Kernel backend serves every plan by default. The rocKE backend is built with `-DHIPTENSOR_ENABLE_ROCKE_BACKEND=ON`. The environment variables `HIPTENSOR_BACKEND` (`auto`, `ck`, `rocke`), `HIPTENSOR_BACKEND_FALLBACK` (`1`, `0`), and `HIPTENSOR_BACKEND_AB` (`<primary>,<shadow>`, debug comparison of two backends) select the backend; unknown values make `hiptensorCreatePlan` and `hiptensorEstimateWorkspaceSize` return `HIPTENSOR_STATUS_INVALID_VALUE`.
* Added rocKE kernels to the rocKE backend for contractions (f16, bf16, f32, f64, complex f32, complex f64), permutations, elementwise binary and trinary operations, and reductions. The kernels take their problem sizes and strides at run time and are compiled once, at build time, into a kernel pack per GPU target (`share/hiptensor/rocke/<target>/rocke.htpk`, built from `HIPTENSOR_ROCKE_SOURCE_DIR` or copied from `HIPTENSOR_ROCKE_PACK_DIR`). A C++ host tool of the build writes the packs with rocKE's C++ kernel builders and the COMGR library, so building them needs no Python. The library loads the pack and its code objects without Python. `HIPTENSOR_ROCKE_PACK_PATH` overrides where the library looks for packs.
* Added a JIT and a kernel cache to the rocKE backend (`-DHIPTENSOR_ENABLE_ROCKE_JIT=ON`, the default with the rocKE backend; links COMGR). Plans created with `HIPTENSOR_JIT_MODE_DEFAULT` replace their elementwise launches (permutations, elementwise operations, and the packing and epilogue launches of contractions) by kernels compiled for the problem at plan creation, with constant sizes and strides and vector accesses where the layout allows them. The library builds these kernels with rocKE's C++ kernel builders, which it links, and compiles them with COMGR; neither the build nor the library runs Python, and the elementwise JIT needs no data file besides the kernel pack. `HIPTENSOR_JIT_MODE_NONE` never compiles. Compiled kernels go to a process-wide, content-addressed kernel cache that `hiptensorWriteKernelCacheToFile` and `hiptensorReadKernelCacheFromFile` write and read per device class.
* Added kernel selection to the rocKE backend at plan creation: capability checks of every candidate kernel program without launching anything, then rocKE's tuned route table, then a selection model per problem class shipped in the kernel pack (gradient-boosted trees evaluated in C++ on integer features; `HIPTENSOR_ROCKE_SELECT_MODEL_DIR` at build time), then rocKE's rules. `HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK` selects the k-th candidate. `HIPTENSOR_ALGO_DEFAULT_PATIENT` times the first candidates on the device and stores the winner in the plan cache. `hiptensorEstimateWorkspaceSize` and `HIPTENSOR_PLAN_REQUIRED_WORKSPACE` report the workspace of the selected kernel.
* The rocKE backend stores the kernel every plan selects in the handle's plan cache (`HIPTENSOR_CACHE_MODE_PEDANTIC`), not only the winner of `HIPTENSOR_ALGO_DEFAULT_PATIENT`.
* Added contractions read and written in place to the rocKE backend: the ring GETT (rocKE's tensor-contraction body with run-time index maps, f16 and bf16 data on gfx950) and the direct GETT (every data type, compute type, operator and layout, on every target) read A, B and C and write D through their own strides in one launch with no workspace. Contractions planned with a workspace limit of zero, which the rocKE backend declined, are served, and so are contractions with a 16-bit compute type on gfx90a and gfx942. The selection models of gfx950 and gfx942 rank these routes with the others; where no model scores them, rocKE's rules put the ring GETT first when it applies (after rocKE's tuned route table; operands whose fastest free run is not a multiple of 8 elements are packed first when the workspace limit allows), and the direct GETT on gfx90a and gfx942 for 16-bit and f32 compute classes. The ring GETT and the tuned plans run the direct GETT when alpha or beta is zero, so that a NaN or an infinity of a tensor whose scale is zero no longer reaches D, and when a buffer is not aligned to their vector accesses. Permutations and elementwise operations whose inputs are fastest along another mode than the output run on a tiled kernel, and those with an input contiguous along the output's fastest mode on vector forms of the generic kernel, without the JIT. The operand packs and epilogues of contractions run on the tiled kernel too when they transpose their tensor or have more modes than the generic kernel's index slots (one launch instead of several), with the generic launches as their fallback for buffers or workspaces not aligned to 16 bytes. Permutations and elementwise operations of small planes compute several batch indices per workgroup.
* Added the contraction routes of rocKE's AOT route table to the rocKE backend's kernel pack (f16 and bf16 data on gfx950). For each problem of the table, the pack carries the launch program rocKE's contraction planner selects: a GETT on one of rocKE's GEMM bodies that reads the operands in place, optionally after packing one operand into the workspace, or a GETT on K-contiguous copies of both operands made in the workspace. The selection puts these routes ahead of the ring GETT, together with rocKE's tuned route table. They run the direct GETT when alpha or beta is zero and when a buffer or the workspace isn't aligned to their vector accesses, and they need the workspace of their operand copies (`hiptensorEstimateWorkspaceSize` reports it). The table's kernels have the problem's sizes and strides built in, so they serve only contractions whose extents, strides, mode order, and data types are exactly those of a table problem; any other contraction (another extent, padded strides, another mode order, or another GPU target) runs on the ring GETT or the other routes as before, or, with `HIPTENSOR_JIT_MODE_DEFAULT`, on the contraction JIT's route.
* Added every unary operator to the rocKE backend: `HIPTENSOR_OP_SIN`, `HIPTENSOR_OP_COS`, `HIPTENSOR_OP_TAN`, `HIPTENSOR_OP_SINH`, `HIPTENSOR_OP_COSH`, `HIPTENSOR_OP_ASIN`, `HIPTENSOR_OP_ACOS`, `HIPTENSOR_OP_ATAN`, `HIPTENSOR_OP_ASINH`, `HIPTENSOR_OP_ACOSH`, `HIPTENSOR_OP_ATANH`, `HIPTENSOR_OP_CEIL`, and `HIPTENSOR_OP_FLOOR` on the operands of permutations, elementwise operations, reductions, and contractions of f16, bf16, f32, and f64 data, which it declined before. They're evaluated in f64 (with an f32 compute type the result is rounded once to f32), with the errors listed in the programmer's guide. The rocKE backend's `HIPTENSOR_OP_SIGMOID` now keeps the small results of large negative arguments (it returned 0 below about -88.7 in f32 and -709.8 in f64). `HIPTENSOR_OP_CONJ` of a real tensor is the identity, as on the Composable Kernel backend; operators other than identity and conjugate on complex tensors are still declined.
* Added native f32 contractions to the rocKE backend: with `HIPTENSOR_COMPUTE_DESC_32F` and `HIPTENSOR_COMPUTE_DESC_C32F`, contractions of f32 and complex f32 data run on an f32 strided GEMM with f32 products and f32 accumulation (gfx942 and gfx950; the f64 route on gfx90a). The environment variable `HIPTENSOR_ROCKE_F32_RECIPE` selects another recipe: `x9`, `x6` (f32 accuracy class), `x4`, `x3` (about 16 significant bits per operand), which split the operands into bf16 parts on gfx950, or `f64`. Every recipe also lists the f64 route, so no recipe selects a kernel less accurate than its own. The programmer's guide gives the accuracy class and error bound of each recipe.
* Added trinary contractions, contractions of mixed data types, and tensors of 2 GiB and more to the rocKE backend. A trinary contraction runs as two contractions with the intermediate tensor, in the output's data type, in the workspace, and the backend selects the kernels of both steps. A, B, C, and D of a contraction can each be f16, bf16, f32, or f64 (or each complex f32 or complex f64): with a 16-bit compute type every operand is rounded once to that type, with `HIPTENSOR_COMPUTE_DESC_64F` the contraction runs in f64, and the result is rounded once to D's type. Tensors past the kernels' 32-bit offsets run on kernels with 64-bit addresses or as several launches over pieces of the tensors.
* Added rocKE's tuned reduction kernels to the rocKE backend's kernel pack. The kernels are rocKE's native strided reductions (vector loads, several threads per output, unrolled loads, multi-workgroup splits with f32 or f64 partial results) with the problem's sizes and strides as kernel arguments: a reduction runs in its own layout, with no permutation. The pack carries, per GPU target, a palette of these kernels and the launches of the reductions in rocKE's reduction route table (gfx950), which the selection puts first for those problems; for any other reduction, the selection puts first the palette kernel with the knobs of the nearest reduction measured on gfx950 (the pack carries those measurements; on gfx90a and gfx942 the same knobs run with the split scaled to the device's compute units, which on the benchmark reductions ran faster than the generic kernels on gfx942 and on gfx90a on all but one f64 reduction) and offers the next nearest as candidates (`HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK`, `HIPTENSOR_ALGO_DEFAULT_PATIENT`). When the partial results of a split need more workspace than the plan's limit, the split is lowered to the largest power of two that fits (no split for a limit of zero), so plans created with `HIPTENSOR_WORKSPACE_MIN` or a zero workspace keep these kernels. `HIPTENSOR_ROCKE_DISABLE_TUNED` runs reductions on the generic reduction kernels only. They keep the reduction's semantics: `HIPTENSOR_OP_MAX` and `HIPTENSOR_OP_MIN` propagate NaN, `alpha` or `beta` of zero doesn't read A or C, `opA` (other than the elementary functions) and `opC` are applied, partial results stay in the compute type and D is rounded once. A reduction whose A, C, D or workspace isn't aligned to the kernel's vector accesses runs the generic reduction kernel instead. Reductions with an elementary-function operator, with more than eight reduced or kept modes after merging, or with element counts beyond 32-bit indices keep the generic reduction kernels.
* Added a flat elementwise kernel to the rocKE backend's kernel pack (gfx942 and gfx950): permutations and elementwise binary and trinary operations of f16 or bf16 tensors that are all one contiguous run of elements in the same order run without index arithmetic, with two 16-byte vectors of D per thread, streaming loads and stores, and branch-free loads when no scale is zero. Results are those of the generic kernel; buffers not aligned to its vectors run the generic kernel. `HIPTENSOR_ROCKE_DISABLE_TUNED` turns it off.
* Added a contraction JIT to the rocKE backend (`-DHIPTENSOR_ENABLE_ROCKE_JIT=ON`, gfx950). A plan created with `HIPTENSOR_JIT_MODE_DEFAULT` for an f16 or bf16 contraction (one data type, `HIPTENSOR_COMPUTE_DESC_32F`, no operators, C laid out as D) that rocKE's tuned and AOT route tables don't hold runs rocKE's tensor-contraction planner at plan creation: the C++ twin of rocKE's product planner, with its GEMM dispatcher and selection models, chooses between a GETT on one of rocKE's GEMM bodies that reads the operands in place (optionally after packing one or both into the workspace) and a GETT on K-contiguous copies of both operands. The library builds the chosen kernels with rocKE's C++ kernel builders, with the problem's sizes and strides built in, compiles them with COMGR, and keeps them in the kernel cache (`hiptensorWriteKernelCacheToFile`, `hiptensorReadKernelCacheFromFile`, `HIPTENSOR_KERNEL_CACHE_DIR`), so a later plan of the same problem, also one whose descriptors list the modes under other labels or in another order, compiles nothing. The gfx950 selection model of the class ranks this route with the ring GETT, the GEMM routes, the f64 route and the direct GETT, and the plan compiles the route's kernels only when it selects (or times) the route; without a model that scores it, the route goes first. The route runs the direct GETT when alpha or beta is zero, when a buffer or the workspace isn't aligned to its 16-byte accesses, and on another device than the plan's. When its kernels can't be compiled (`HIPTENSOR_JIT_MAX_MS` spent, a compiler error), the plan takes the next candidate; past `HIPTENSOR_JIT_MAX_MS` the route still takes its kernels from the kernel cache and from `HIPTENSOR_KERNEL_CACHE_DIR` files (the budget is checked before each compile and doesn't stop one in progress). `HIPTENSOR_ROCKE_GETT_TABLES=off` (for checks) lets the planner also plan the problems of rocKE's route tables; a test checks that, for every problem of the tables described with other mode labels, descriptor orders and extent-1 modes, it chooses exactly the tables' launches. The planner's selection models are data files that the build installs next to the gfx950 kernel pack (`share/hiptensor/rocke/gfx950/gemm_models`, checked against their SHA-256 digests when they're loaded); without them, contractions keep their other routes. Plans created with `HIPTENSOR_JIT_MODE_NONE` don't change. `HIPTENSOR_ROCKE_DISABLE_TUNED` and `HIPTENSOR_ROCKE_JIT=0` turn the route off. Contractions that the planner doesn't serve (other data types, batch modes, K extents that the GEMM bodies' K tiles don't divide, layouts without a 4-byte run) keep their routes.
* Extended the rocKE backend's JIT to every elementwise launch of its kernel pack: the operand packing and epilogue launches of mixed-type contractions, launches with the operators `HIPTENSOR_OP_SIN` to `HIPTENSOR_OP_FLOOR` whose inputs aren't all contiguous along the output's fastest mode (constant geometry without vector accesses, with `HIPTENSOR_ROCKE_JIT=all`), and launches with 64-bit offsets and those operators. They kept their kernel-pack kernels before. The JIT specializes the launches of the generic elementwise kernel and its vector classes only: the flat elementwise, GETT, tuned reduction, and generic reduction launches keep their kernel-pack kernels.

### Changed
* Changed the rocKE backend's kernel selection for contractions so that its selection models rank every route: the contraction JIT's route (rocKE's tensor-contraction planner), the ring GETT, the direct GETT, the GEMM routes, the native f32 route and the f64 route. The gfx950 and gfx942 models of the contraction classes (f16, bf16 and f32 with a 16-bit compute type; f32 and complex f32 with the default `native` recipe, new classes `contraction.s.native` and `contraction.c.native`; f64; complex f64) were retrained on cold, interleaved timings of every candidate of model-guided problem sweeps. Before, the JIT route and the ring GETT went first by rule, the direct GETT followed the models' picks, and the `native` recipe had no model. A model now names the routes it scores and a margin by which its pick must beat the first candidate of rocKE's order (model format v3: also the digest of the contraction JIT planner's GEMM selection models when the model reads that planner's predicted time; the pack build fails on a mismatch, and the library doesn't use such a model with other planner models). Feature schema 3 adds the JIT route's planner route, predicted time and bytes moved, and the plan's mode (whether the plan specializes its elementwise launches with the JIT): the gfx950 models are trained on timings of both plan modes and on contractions without contracted modes. Routes a model does not score, and models of the earlier formats, keep the rules. The new CPU test `rocke_select_model_test` walks every shipped model on the exporter's check vectors (`<class>.check` next to each model). Contractions of rocKE's tuned and AOT route tables still take their table's launches. `HIPTENSOR_ROCKE_SELECT_MODELS` reads the models from a directory instead of the kernel pack (training and evaluation), and `HIPTENSOR_ROCKE_SELECT_DUMP` also records the candidates of plans that are not measured.
* Changed the plan cache to keep its entries per backend, and the plan cache file to format version 2, which records the backends of the library. `hiptensorHandleReadPlanCacheFromFile` returns `HIPTENSOR_STATUS_INVALID_VALUE` for files of the earlier format and for files written by a library with other backends.
* Changed `hiptensorWriteKernelCacheToFile` and `hiptensorReadKernelCacheFromFile` to return `HIPTENSOR_STATUS_NOT_INITIALIZED` for a null `handle` and `HIPTENSOR_STATUS_INVALID_VALUE` for a null `fileName`. With no backend that keeps a kernel cache, both still succeed without touching the file.
* Changed the elementwise operations and reductions to follow cuTENSOR for zero scaling factors: an input whose scaling factor is zero is not read and its operator is not applied, so NaN or Inf in that input, or produced by its operator (for example `HIPTENSOR_OP_LOG` of zero), no longer reaches the output, and the input may be `NULL`. This applies to `alpha` in `hiptensorPermute`, `alpha` and `gamma` in `hiptensorElementwiseBinaryExecute`, `alpha`, `beta`, and `gamma` in `hiptensorElementwiseTrinaryExecute`, and `alpha` and `beta` in `hiptensorReduce`, where `beta == 0` also no longer reads the previous contents of D.
* Changed `hiptensorCreatePlan` to return `HIPTENSOR_STATUS_NOT_SUPPORTED` for permutations, elementwise operations, and reductions that no kernel can run (data types, output rank, or reduced modes without a kernel), and for reductions whose C and D differ in descriptor or modes, whose C carries a mode A doesn't, or with a mode repeated within a tensor, and `HIPTENSOR_STATUS_INVALID_VALUE` for reductions whose C has a mode with a different extent than in A. These problems were previously accepted and failed during execution, mostly with `HIPTENSOR_STATUS_INTERNAL_ERROR`.
* Changed `hiptensorEstimateWorkspaceSize` to compute the workspace from the kernels that can run the contraction (the largest of them, or the smallest for `HIPTENSOR_WORKSPACE_MIN`), plus the intermediate tensor of a trinary contraction, instead of only from kernels recorded in the plan preference by an earlier plan. It accepts a `NULL` plan preference. `HIPTENSOR_PLAN_REQUIRED_WORKSPACE` now returns the workspace the plan's kernel needs, at most the `workspaceSizeLimit` passed to `hiptensorCreatePlan`, instead of that limit, and `hiptensorPlanGetAttribute` returns `HIPTENSOR_STATUS_INVALID_VALUE` when `sizeInBytes` isn't the attribute's size or `buf` is `NULL`.
* Changed `hiptensorCreatePermutation`, `hiptensorCreateElementwiseBinary`, `hiptensorCreateElementwiseTrinary`, and `hiptensorCreateReduction` to return `HIPTENSOR_STATUS_INVALID_VALUE` when the operator of an operand isn't a unary operator, which was applied as `HIPTENSOR_OP_IDENTITY` before, and `hiptensorCreateReduction` to return it for a reduction operator other than `HIPTENSOR_OP_ADD`, `HIPTENSOR_OP_MUL`, `HIPTENSOR_OP_MAX`, or `HIPTENSOR_OP_MIN`.
* Changed `hiptensorReduce` for reductions whose output keeps every mode of A to compute `alpha * opA(A) + beta * opC(C)` with a kernel of its own instead of an elementwise binary operation, which adds rank-1 support.

### Resolved issues
* Fixed `hiptensorCreateContraction` and `hiptensorCreateContractionTrinary` accepting modes the kernels can't address, which wrote past the library's internal mode tables (more than six M, N, or K modes) or read modes that weren't there (an output mode carried by neither input, no M or no N mode). They now return `HIPTENSOR_STATUS_NOT_SUPPORTED` for more than six modes in a group, for a mode carried by one tensor only, for a contraction without M or N modes, and for a mode repeated within a tensor, and `HIPTENSOR_STATUS_INVALID_VALUE` for a mode with different extents in two tensors. For trinary contractions this applies to both steps.
* Fixed `hiptensorCreateTensorDescriptor` reading a `NULL` `lens` array when `numModes` is greater than zero and accepting non-positive extents, which now return `HIPTENSOR_STATUS_INVALID_VALUE`, and negative strides, which now return `HIPTENSOR_STATUS_NOT_SUPPORTED`. A `NULL` `desc` returns `HIPTENSOR_STATUS_NOT_INITIALIZED`.
* Fixed `hiptensorReduce` copying C to D with a blocking copy on the null stream, which didn't wait for work on a non-blocking `stream` that was still writing C, and copying C's packed size, which for a padded or strided D wrote the wrong elements and the memory between them. C is now copied on `stream`, element by element when D's elements aren't contiguous.
* Fixed `hiptensorReduce` changing the plan's reduction operator for reductions whose output keeps every mode of A, and failing with `HIPTENSOR_STATUS_INTERNAL_ERROR` for such reductions of rank 1.

## hipTensor 2.5.0 for ROCm 10.2

### Added
* Added fp64 support to hiptensorPermute for ranks 2-6.
* Added broadcast support to the elementwise operations. An input tensor of `hiptensorCreatePermutation`, `hiptensorCreateElementwiseBinary`, and `hiptensorCreateElementwiseTrinary` may now carry a subset of the output tensor's modes, in which case it's broadcast along the modes it lacks. For example, `A` with modes `{n}` can be combined with `B`, `C`, and `D` with modes `{m,n}`.
* Added `gfx1250-strict` as a supported offload target and classified its device pass as `gfx1250`.

### Changed
* Changed `hiptensorCreatePermutation`, `hiptensorCreateElementwiseBinary`, and `hiptensorCreateElementwiseTrinary` to return `HIPTENSOR_STATUS_INVALID_VALUE` when a mode shared by an input tensor and the output tensor has a different extent in each, and `HIPTENSOR_STATUS_NOT_SUPPORTED` when a mode is repeated within one tensor. An input tensor that carries a mode the output tensor doesn't carry is still rejected with `HIPTENSOR_STATUS_NOT_SUPPORTED`, since that would have to be reduced away rather than broadcast.
* Changed `libhiptensor.so` to export symbols with hidden visibility by default, so only the documented public C API and the internal `hiptensor::Logger` and `hiptensor::HiptensorOptions` singletons, which the hipTensor tests share with the library, are exposed. Other internal C++ classes and Composable Kernel template instantiations that were previously visible are now local to the library. Code that relied on those undocumented internal symbols will no longer find them.
* Changed the plan cache file written by `hiptensorHandleWritePlanCacheToFile` to a versioned format that records the hipTensor version and the device's architecture and compute unit count. `hiptensorHandleReadPlanCacheFromFile` now returns `HIPTENSOR_STATUS_INVALID_VALUE` for a file of another hipTensor version, including every file written by earlier versions, or of another device; `HIPTENSOR_STATUS_IO_ERROR` for a damaged file; and `HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE`, with the number of cache lines the file needs in `numCachelinesRead`, when the file holds more lines than the plan cache. The plan cache is left unchanged in all these cases, and reading a file no longer changes the size of the plan cache, so resize the plan cache with `hiptensorHandleResizePlanCache` before reading a file with more lines. `hiptensorHandleWritePlanCacheToFile` and `hiptensorHandleReadPlanCacheFromFile` now return `HIPTENSOR_STATUS_NOT_INITIALIZED` for a null `handle` and `HIPTENSOR_STATUS_INVALID_VALUE` for a null `fileName` (or `numCachelinesRead`), also when the plan cache is disabled. With the plan cache disabled, `hiptensorHandleReadPlanCacheFromFile` sets `numCachelinesRead` to 0.
* Changed `hiptensorCreatePlan` to return `HIPTENSOR_STATUS_NOT_SUPPORTED` for any tensor whose elements span 2^31 bytes or more (for complex types, per real or imaginary part), measured from its lengths and strides, and, for `HIPTENSOR_R_64F` permutations and elementwise binary and trinary operations with an input laid out differently from the output, 2^29 bytes or more. The limits are deliberately conservative, so they also refuse some sizes that previously gave correct results, for example some 16-bit elementwise operations above 2 GiB and reductions of exactly 2^31 bytes.

### Resolved issues
* Fixed `hiptensorPermute` and the element-wise binary/trinary execute paths ignoring user-supplied output tensor strides, which caused the output to always be written contiguously regardless of the strides set on the output descriptor.
* Fixed elementwise operations reading the wrong elements when two input tensors ordered the same modes differently. Every tensor's strides are now permuted into the output tensor's mode order before reaching the kernel.
* Fixed scalars passed with a BF16 compute descriptor being read as zero, which made BF16 element-wise binary operations and BF16 reductions return zeros instead of the scaled result.
* Fixed `hiptensorElementwiseTrinaryExecute` applying `opAB` and `opABC` in swapped positions. It now computes `opABC(opAB(alpha * A, beta * B), gamma * C)` as documented. Results change whenever `opAB` and `opABC` differ.
* Fixed `hiptensorCreateElementwiseBinary` and `hiptensorCreateElementwiseTrinary` accepting an operator other than `HIPTENSOR_OP_ADD`, `HIPTENSOR_OP_MUL`, `HIPTENSOR_OP_MAX`, or `HIPTENSOR_OP_MIN` for `opAC`, `opAB`, or `opABC`, which left the output undefined. They now return `HIPTENSOR_STATUS_INVALID_VALUE`.
* Fixed elementwise kernel selection reading past the end of its tuning table for extents above the largest tuned size (rank 2 above 33,554,432, rank 3 above 524,288, and rank 4 above 131,072). Such extents now use the largest tuned size.
* Fixed `hiptensorPermute` returning an unscaled result when a plan whose first execution used `alpha == 1` was executed again with a different `alpha` under `HIPTENSOR_CACHE_MODE_NONE`.
* Fixed `hiptensorContract` and `hiptensorContractTrinary` returning wrong results for packed column-major tensors, including descriptors created with `strides = NULL`, whose modes weren't stored as all output modes followed by all contracted modes in the same order in every tensor (for example A with modes `{m0,k0,m1,k1}`), and for packed column-major tensors combined with row-major or padded tensors. The column-major stride optimization is now applied only when it addresses the same elements as the tensors' own strides.
* Fixed contractions reading C (and, in trinary contractions, D) with the output tensor's strides, which gave wrong results whenever it was laid out differently from the output. It is now read through its own modes and strides. `hiptensorCreateContraction` and `hiptensorCreateContractionTrinary` now return `HIPTENSOR_STATUS_NOT_SUPPORTED` when this tensor doesn't carry the same modes as the output, or, for complex compute types, when its layout differs from the output's, and `HIPTENSOR_STATUS_INVALID_VALUE` when a mode has a different extent than in the output.
* Fixed `hiptensorCreatePlan` with `HIPTENSOR_ALGO_DEFAULT` or `HIPTENSOR_ALGO_DEFAULT_PATIENT` aborting with an illegal memory access for tensors with padded (non-packed) strides. The scratch buffers used to time candidate kernels are now sized from the strides.
* Fixed loading hipTensor ending the host process on a GPU whose architecture it doesn't support and on a host without a visible GPU. The library now loads on any host, and `hiptensorCreate` returns `HIPTENSOR_STATUS_ARCH_MISMATCH` for an unsupported architecture or `HIPTENSOR_STATUS_HIP_ERROR` when the device can't be queried.
* Fixed HIP errors inside the library, such as running out of device memory, ending the host process or escaping the C API as a C++ exception. The API call now returns `HIPTENSOR_STATUS_ALLOC_FAILED` for an allocation failure, `HIPTENSOR_STATUS_HIP_ERROR` for any other failed HIP call, and `HIPTENSOR_STATUS_INTERNAL_ERROR` for any other error raised by the kernel backend. `hiptensorCreatePlan` now returns `HIPTENSOR_STATUS_HIP_ERROR` rather than `HIPTENSOR_STATUS_ALLOC_FAILED` when a HIP call other than an out-of-memory allocation fails while it times candidate kernels.
* Fixed `hiptensorOperationDescriptorSetAttribute` and `hiptensorOperationDescriptorGetAttribute` copying `sizeInBytes` bytes without checking them, which overwrote neighboring fields of the operation descriptor (for example when `HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT` or `HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT` was set with one `uint32_t` per output mode), and writing through a null pointer for `HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE`. A null buffer or a size that doesn't match the attribute now returns `HIPTENSOR_STATUS_INVALID_VALUE`. `PADDING_LEFT` and `PADDING_RIGHT` take one `uint32_t` per mode of the output tensor, and `PADDING_VALUE` takes one element of the output tensor's type. No kernel pads its output, so non-zero padding returns `HIPTENSOR_STATUS_NOT_SUPPORTED`.
* Fixed `hiptensorCreatePlan` dereferencing a null `pref`, which is documented as allowed. A null preference now selects the defaults of `hiptensorCreatePlanPreference` with `HIPTENSOR_ALGO_DEFAULT` and `HIPTENSOR_JIT_MODE_NONE`. A null `handle`, `plan`, or `desc` returns `HIPTENSOR_STATUS_NOT_INITIALIZED`.
* Fixed `hiptensorCreatePlan` leaving an allocated plan in `*plan` when it failed after creating it, for example when no kernel could run a contraction, which leaked unless the caller destroyed the plan. It now sets `*plan` to `NULL` whenever it doesn't return `HIPTENSOR_STATUS_SUCCESS`.
* Fixed `hiptensorCreatePermutation`, `hiptensorCreateElementwiseBinary`, `hiptensorCreateElementwiseTrinary`, `hiptensorCreateReduction`, `hiptensorCreateContraction`, and `hiptensorCreateContractionTrinary` reading an undefined scalar type for `HIPTENSOR_COMPUTE_DESC_NONE`, the integer compute descriptors, and unknown values. They now return `HIPTENSOR_STATUS_INVALID_VALUE`. The contraction functions previously returned `HIPTENSOR_STATUS_ARCH_MISMATCH` for these.
* Fixed `hiptensorContract` reading released kernel arguments to log the required workspace size after returning `HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE`.
* Fixed permutations, elementwise binary and trinary operations, and reductions returning wrong results without an error, and contractions failing during plan creation or execution or ending the process (complex types), when a tensor's elements span 2^31 bytes or more. `hiptensorCreatePlan` now returns `HIPTENSOR_STATUS_NOT_SUPPORTED` for such a tensor, measured from its lengths and strides (for complex types, per real or imaginary part), and for the intermediate `A * B` tensor of a trinary contraction. For `HIPTENSOR_R_64F` permutations and elementwise binary and trinary operations with an input laid out differently from the output, including a broadcast input, which wrote zeros for elementwise tensors of 1 GiB, the limit is 2^29 bytes.
* Fixed the plan cache growing without bound. Adding a line to a full plan cache now evicts the least recently used line, and updating a line that is already cached no longer evicts another one.
* Fixed heap corruption after `hiptensorHandleWritePlanCacheToFile`, which crashed later executions on the same handle or the destruction of the handle.
* Fixed plan cache lookups matching a different problem, which then reused that problem's kernel. The key now also covers the device, the output tensor of trinary contractions, the operators, and the rank of each tensor.
* Fixed `hiptensorCreatePlan` with `HIPTENSOR_ALGO_ACTOR_CRITIC` selecting a contraction kernel that can't run the problem, after which `hiptensorContract` and `hiptensorContractTrinary` failed with `HIPTENSOR_STATUS_INTERNAL_ERROR`, and failing with `HIPTENSOR_STATUS_EXECUTION_FAILED` when its table names no kernel for the problem or the kernel it names isn't available. The plan now falls back to timing the kernels that can run the problem, and returns `HIPTENSOR_STATUS_NOT_SUPPORTED` when none can.
* Fixed `hiptensorCreatePlan` for trinary contractions selecting, with any algorithm, a kernel that can run the second step (`(A * B) * C`) but not the first (`A * B`), after which `hiptensorContractTrinary` failed in the first step. Only kernels that can run both steps are now selected.
* Fixed `hiptensorContractTrinary` with a D tensor returning NaN wherever the workspace held NaN before the call. The intermediate `A * B` tensor in the workspace was computed by a kernel that also read its previous contents, scaled by zero. The intermediate tensor is now cleared first.
* Fixed `hiptensorContractTrinary` returning `HIPTENSOR_STATUS_INTERNAL_ERROR` from the second execution of a plan created with `HIPTENSOR_ALGO_ACTOR_CRITIC` under `HIPTENSOR_CACHE_MODE_PEDANTIC`. The plan's kernel was replaced by a plan cache lookup among candidates that the plan doesn't keep, which left the plan without a kernel.
* Fixed the hipTensor test binaries exiting with an "Unknown command line argument" error when passed GoogleTest flags such as `--gtest_filter`.

## hipTensor 2.4.0 for ROCm 10.1

### Added
* Added `ffm-quick` and `ffm-full` test categories for emulation tests.
* Added a Windows version resource so `hiptensor.dll` exposes file metadata (file description, version, product name, and copyright) in File Explorer properties.

### Changed
* Reorganized test configurations per tier and category.

### Optimized
* Reduced test execution duration.

### Resolved issues
* Enabled `-frtti` on Windows to fix RTTI-related build failures.
* Fixed batched contractions reporting success while producing incorrect results. `hiptensorCreateContraction` and `hiptensorCreateContractionTrinary` now return `HIPTENSOR_STATUS_NOT_SUPPORTED` when a mode is shared by both inputs and the output of a contraction.
* Changed `hiptensorCreatePermutation`, `hiptensorCreateElementwiseBinary`, and `hiptensorCreateElementwiseTrinary` to return `HIPTENSOR_STATUS_NOT_SUPPORTED` when an input tensor doesn't carry the same modes as the output tensor. These configurations previously produced a valid descriptor and plan, then failed with `HIPTENSOR_STATUS_INTERNAL_ERROR` at execution.
* Removed the internal compiler flags `-amdgpu-early-inline-all=true` and `-amdgpu-function-calls=false` from the build, which caused excessive compile-time memory usage (OOM) with newer ROCm/LLVM toolchains (JIRA: LCOMPILER-2589).

## hipTensor 2.3.0 for ROCm 7.14

### Added
* Added Windows support.
* Added contraction support with FP16 and BF16 data and compute types for gfx11 and gfx12 targets.
* Added support for the following new GPU targets:
  * gfx11: gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1152, gfx1153.
  * gfx12: gfx1200, gfx1201, gfx1250.
* Added unary element-wise operators to contraction, including the new `BilinearUnary` class, dedicated instances, samples, and tests.
* Added Dockerfiles (prebuilt and full build) and documentation to streamline hipTensor build environment setup.
* Added the `CREATE_TEST_APP_LOCAL_DEPLOY` CMake option to stage required ROCm DLLs on Windows, and updated the Windows build documentation accordingly.
* Added YAML-driven CTest test filter standardization, applying `quick`/`standard`/`comprehensive`/`full` tier labels to the installed test tree so tests can be run by tier with `ctest -L <tier>`.
* Added support for trinary contractions.
* Added hipTensor to the TheRock build system, enabling source builds, artifact distribution, and CI testing on both Linux and Windows.
* Added native Linux packages with HPC SDK metapackages: `amdrocm-hiptensor` (runtime), `amdrocm-hiptensor-devel` (headers and CMake config), and `amdrocm-hiptensor-test` (CTest binaries).

### Changed
* Replaced numeric UID-based actor-critic kernel lookup with platform-stable string-based kernel name comparison to enable cross-platform compatibility.
* Adopted FNV-1a string hashing in place of `std::hash` to ensure plan cache files are portable across platforms.
* Switched to ROCm-provided CMake install functions (`rocm_export_targets`) for consistency with other ROCm libraries.
* Adapted hipTensor to CK namespace changes for `host_tensor` functions.
* Cleaned up `rtest` script formatting and removed invalid run commands from `rtest.xml`.

### Removed
* Removed the legacy `.jenkins` folder since CI migration to rocJenkins.

### Optimized
* Improved column-major contraction performance by applying CK-style stride reordering for column-major inputs.
* Achieved 2x–3x speedup in contraction TFLOPS/s by using switch-case dispatch in `HiptensorUnaryOp` instead of static table lookup.
* Re-selected `HIPTENSOR_ALGO_ACTOR_CRITIC` winning kernels for all contraction type/rank/layout combinations to reflect current Composable Kernel tile parameters.

### Resolved issues
* Fixed use-after-free bug where `hiptensorCreatePlan` held dangling pointers to user-provided descriptors; the plan now deep-copies all descriptors.
* Fixed incorrect BF16 results in contraction with unary ops caused by silent `bhalf_t`-to-float integer promotion in cross-type overloads.

## hipTensor 2.2.0 for ROCm 7.2.0

### Added

* Added software-managed plan cache support.
* Added `hiptensorHandleWritePlanCacheToFile` to write the plan cache of a hipTensor handle to a file.
* Added `hiptensorHandleReadPlanCacheFromFile` to read a plan cache from a file into a hipTensor handle.
* Added `simple_contraction_plan_cache` to demonstrate plan cache usages.
* Added `plan_cache_test` to test the plan cache across various tensor ranks.
* Added C API headers to enable compatibility with C programs.
* Added a CMake function to allow projects to query architecture support.
* Added an option to configure the memory layout for tests and benchmarks.

### Changed

* hipTensor has been moved into the new rocm-libraries "monorepo" repository (https://github.com/ROCm/rocm-libraries). This repository consolidates a number of separate ROCm libraries and shared components.
  * The repository migration requires a few changes to the CMake configuration of hipTensor.
* Updated C++ standard from C++17 to C++20.
* Include files `hiptensor/hiptensor.hpp` and `hiptensor/hiptensor_types.hpp` are now deprecated. Use `hiptensor/hiptensor.h` and `hiptensor/hiptensor_types.h` instead.
* Converted include guards from #ifndef/#define/#endif to #pragma once.

### Resolved issues

* Removed large tensor sizes causing problem in benchmarks.

## hipTensor 2.1.0 for ROCm 7.1.0

### Added

* Added large tensor lengths in benchmark YAML files.
* Added a new "-l" option to tests for redirecting logs to a file.

### Changed

* Replaced `permutation` with `elementwise` or `elementwise_permute` across folder names, file names, function names, and variable names.

### Resolved issues

* Fixed an issue where `hiptensor-test` files are not removed after uninstallation.
* Fixed memory leaks in the test code and cleared Valgrind leak reports.

## hipTensor 2.0.0 for ROCm 7.0.0

### Added

* Added element-wise binary operation support.
* Added element-wise trinary operation support.
* Added support for new GPU target gfx950.
* Added dynamic unary and binary operator support for element-wise operations and permutation.
* Added a CMake check for `f8` datatype availability.
* Added `hiptensorDestroyOperationDescriptor` to free all resources related to the provided descriptor.
* Added `hiptensorOperationDescriptorSetAttribute` to set attribute of a `hiptensorOperationDescriptor_t` object.
* Added `hiptensorOperationDescriptorGetAttribute` to retrieve an attribute of the provided `hiptensorOperationDescriptor_t` object.
* Added `hiptensorCreatePlanPreference` to allocate the `hiptensorPlanPreference_t` and enabled users to limit the applicable kernels for a given plan or operation.
* Added `hiptensorDestroyPlanPreference` to free all resources related to the provided preference.
* Added `hiptensorPlanPreferenceSetAttribute` to set attribute of a `hiptensorPlanPreference_t` object.
* Added `hiptensorPlanGetAttribute` to retrieve information about an already-created plan.
* Added `hiptensorEstimateWorkspaceSize` to determine the required workspaceSize for the given operation.
* Added `hiptensorCreatePlan` to allocate a `hiptensorPlan_t` object, select an appropriate kernel for a given operation and prepare a plan that encodes the execution.
* Added `hiptensorDestroyPlan` to free all resources related to the provided plan.

### Changed

* Removed architecture support for gfx940 and gfx941.
* Generalized opaque buffer now for any descriptor.
* Replaced `hipDataType` with `hiptensorDataType_t` for all supported types, for example, `HIP_R_32F` to `HIPTENSOR_R_32F`.
* Replaced `hiptensorComputeType_t` with `hiptensorComputeDescriptor_t` for all supported types.
* Replaced `hiptensorInitTensorDescriptor` with `hiptensorCreateTensorDescriptor`.
* Changed handle type and API usage from `*handle` to `handle`.
* Replaced `hiptensorContractionDescriptor_t` with `hipTensorOperationDescriptor_t`.
* Replaced `hiptensorInitContractionDescriptor` with `hiptensorCreateContraction`.
* Replaced `hiptensorContractionFind_t` with `hiptensorPlanPreference_t`.
* Replaced `hiptensorInitContractionFind` with `hiptensorCreatePlanPreference`.
* Replaced `hiptensorContractionGetWorkspaceSize` with `hiptensorEstimateWorkspaceSize`.
* Replaced `HIPTENSOR_WORKSPACE_RECOMMENDED` with `HIPTENSOR_WORKSPACE_DEFAULT`.
* Replaced `hiptensorContractionPlan_t` with `hiptensorPlan_t`.
* Replaced `hiptensorInitContractionPlan` with `hiptensorCreatePlan`.
* Replaced `hiptensorContraction` with `hiptensorContract`.
* Replaced `hiptensorPermutation` with `hiptensorPermute`.
* Replaced `hiptensorReduction` with `hiptensorReduce`.
* Replaced `hiptensorElementwiseBinary` with `hiptensorElementwiseBinaryExecute`.
* Replaced `hiptensorElementwiseTrinary` with `hiptensorElementwiseTrinaryExecute`.
* Removed function `hiptensorReductionGetWorkspaceSize`.

## hipTensor 1.5.0 for ROCm 6.4.0

### Added

* Added benchmarking suites for contraction, permutation, and reduction. YAML files are categorized into bench and validation folders for organization
* Added emulation test suites for contraction, permutation, and reduction
* Support has been added for changing the default data layout using the `HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR` environment variable

### Changed

* Used `GPU_TARGETS` instead of `AMDGPU_TARGETS` in `cmakelists.txt`
* Binary sizes can be reduced on supported compilers by using the `--offload-compress` compiler flag

### Optimized

* Optimized the hyper-parameter selection algorithm for permutation

### Resolved issues

* For CMake bug workaround, set `CMAKE_NO_BUILTIN_CHRPATH` when `BUILD_OFFLOAD_COMPRESS` is unset

## hipTensor 1.4.0 for ROCm 6.3.0

### Added

* Added support for tensor reduction, including APIs, CPU reference, unit tests, and documentation

### Changed

* ASAN builds only support xnack+ targets.
* ASAN builds use `-mcmodel=large` to accommodate library sizes greater than 2GB.
* Updated the permute backend to accommodate changes to element-wise operations.
* Updated the actor-critic implementation.

### Optimized

* Split kernel instances to improve build times

### Resolved issues

* Fixed a bug in randomized tensor input data generation.
* Fixed the default strides calculation to be in column major order.
* Fixed a small memory leak by properly destroying HIP event objects in tests.
* Default strides calculations now follow column-major convention.
* Various documentation formatting updates and fixes.

## hipTensor 1.3.0 for ROCm 6.2.0

### Additions

* Added support for tensor permutation of ranks of 2, 3, 4, 5 and 6
* Added tests for tensor permutation of ranks of 2, 3, 4, 5 and 6
* Added support for tensor contraction of M6N6K6: M, N, K up to rank 6
* Added tests for tensor contraction of M6N6K6: M, N, K up to rank 6
* Added new test YAML parsing to support sequential parameters ordering

### Changes

* Documentation updates for installation, programmer's guide and API reference
* Prefer amd-llvm-devel package before system LLVM library
* Preferred compilers changed to CC=amdclang CXX=amdclang++
* Updated actor-critic selection for new contraction kernel additions

### Fixes

* Fixed LLVM parsing crash
* Fixed memory consumption issue in complex kernels
* Work-around implemented for compiler crash during debug build
* Allow random modes ordering for tensor contractions

## hipTensor 1.2.0 for ROCm 6.1.0

### Additions

* API support for permutation of rank 4 tensors: f16 and f32
* New datatype support in contractions of rank 4: f16, bf16, complex f32, complex f64
* Added scale and bilinear contraction samples and tests for new supported data types
* Added permutation samples and tests for f16, f32 types

### Fixes

* Fixed bug in contraction calculation with data type f32

## hipTensor 1.1.0 for ROCm 6.0.0

### Additions

* Architecture support for gfx940, gfx941, and gfx942
* Client tests configuration parameters now support YAML file input format

### Changes

* Doxygen now treats warnings as errors

### Fixes

* Client tests output redirections now behave accordingly
* Removed dependency static library deployment
* Security issues for documentation
* Compile issues in debug mode
* Corrected soft link for ROCm deployment

## hipTensor 1.0.0 for ROCm 5.7.0

### Additions

* Initial prototype enablement of hipTensor library that supports tensor operations
* Kernel selection support for Default and Actor-Critic algorithms
* API support for:
  * Definition and contraction of rank 4 tensors
  * Contextual logging and output redirection
  * Kernel selection caching
* Data type support for f32 and f64
* Architecture support for gfx908 and gfx90a
