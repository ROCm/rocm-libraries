# Changelog for rocALUTION

Full documentation for rocALUTION is available at [https://rocm.docs.amd.com/projects/rocALUTION/en/latest/](https://rocm.docs.amd.com/projects/rocALUTION/en/latest/).

## (Unreleased) rocALUTION

### Added
* Added the generic `amdgcnspirv` SPIR-V target to the default list of GPU architectures.
* Added `MoveToBackend()` to vectors and matrices. It moves an object to the backend of another object and, unlike `CloneBackend()`, keeps the parallel manager of the object.

### Changed
* `LocalMatrix::DiagonalMatrixMultL()` now requires a vector with one entry per row of the matrix, and `LocalMatrix::DiagonalMatrixMultR()` one with one entry per column. Before, both accepted either size.
* `LocalMatrix::ExtractColumnVector()`, `LocalMatrix::ExtractRowVector()`, `LocalMatrix::ReplaceColumnVector()` and `LocalMatrix::ReplaceRowVector()` now require the index of the row or column to be in range. Before, an index out of range could access memory out of bounds or create entries outside the matrix, and did nothing for a matrix without entries.
* `LocalVector::Asum()` and `GlobalVector::Asum()` of complex vectors on the host now return the sum of the moduli `|z|` with a zero imaginary part. Before, they returned the sums of the absolute real and imaginary parts as a complex number. This changes the L1 residual norm of complex solves on the host.
* The `cg-rsamg_mpi`, `cg-saamg_mpi` and `cg-uaamg_mpi` samples now print the total runtime, as the non-MPI samples do.
* ROCmCMakeBuildTools 0.11.0 or later is now required. By default, CMake configuration fails when it isn't found instead of downloading rocm-cmake from GitHub. Configure with `-DALLOW_FETCH_DEPS=ON` to download rocm-cmake 0.14.0 from ROCm 7.2.4 instead. The deprecated `ROCM` package fallback was removed.
* GoogleTest must be installed before building the tests. Alternatively, set `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` to a googletest source tree, or configure with `-DALLOW_FETCH_DEPS=ON` to download GoogleTest 1.15.2. Both build GoogleTest statically and don't install it.
* CMake 3.21 or later is now required to build rocALUTION (previously 3.5).
* `LocalMatrix::ConvertTo()`, `LocalMatrix::ConvertToBCSR()` and the `GlobalMatrix` functions of the same names now convert a BCSR matrix to the requested block dimension instead of keeping its own. A matrix whose sizes are not a multiple of it falls back to CSR, also a matrix without entries, which was padded to whole blocks before. With a block dimension of 1 or less, such as the default of `ConvertTo()`, a BCSR matrix still keeps its own. `LocalMatrix::ReadFileRSIO()` and `GlobalMatrix::ReadFileRSIO()` with `maintain_initial_format` now also convert a BCSR file to the block dimension the matrix had before the read.
* Converting a matrix that is not BCSR to BCSR now requires a block dimension greater than 1, also on the accelerator and for a matrix without entries, which got block dimension 2 before. So `ConvertTo(BCSR)` without a block dimension, and `SetPrecondMatrixFormat()` of the AI, MultiColored and MultiElimination preconditioners or `BaseAMG::SetOperatorFormat()` with `BCSR` and a block dimension of 1 or less, now end the process for such a matrix.
* `LocalMatrix::ConnectivityOrder()` now returns the new position of each row, the inverse of the permutation it returned before. Pass it to `Permute()` directly. Code that inverted it before `Permute()` must drop the inversion, and code that applied it with `PermuteBackward()` and undid it with `Permute()` must swap the two calls.
* `LocalMatrix::CMK()` now returns the Cuthill-McKee ordering and `LocalMatrix::RCMK()` its reverse, so their permutations differ from those of earlier releases.
* After `Build()` of an AMG solver, `SetHostLevels()` with a smaller number of host levels than before now needs a `MoveToAccelerator()` call to move the other levels back to the accelerator, as without `Build()`. Before, `Build()` with the finest operator on the accelerator left all levels there, so the smaller number applied without it. `Build()` now keeps the number of host levels, clamped to the new hierarchy, also if a later `Build()` has more levels again.
* `PairwiseAMG` now builds different aggregates and hierarchies, as `beta` now sets the strength threshold and the orderings visit the nodes in order, so the number of levels and of iterations can change.
* `AS::Build()` and `RAS::Build()` now assert that there are no more blocks than rows. Before, such a split solved no rows, or with an overlap read before the first row.
* `MultiColoredGS` and `MultiColoredSGS` now apply SOR and SSOR with the relaxation parameter given to `SetRelaxation()`, so relaxation values tuned for earlier versions need to be tuned again. Before, `MultiColoredGS` applied SOR with 1 / omega, and `MultiColoredSGS` with decomposition applied SSOR with 1 / omega up to a scalar factor.
* Without decomposition, `MultiColoredSGS::SetRelaxation()` must now be called before `Build()`, and asserts otherwise. Before, the relaxation was ignored in this mode.
* If the upper eigenvalue bound is positive, `Chebyshev::Set()` now replaces a real lower bound that is not positive, such as the lower bound of `LocalMatrix::Gershgorin()`, which is at most 0, by the upper bound divided by 30 and prints a warning. Likewise, a real upper bound of 0 with a negative lower bound is replaced by the lower bound divided by 30. With such bounds, the Chebyshev iteration does not converge. The bounds are replaced also if they are given in reverse order. If a bound has a nonzero imaginary part, neither bound is replaced.
* `Chebyshev::Set()` now terminates the program if a bound is not finite or if the segment between the bounds still contains 0, such as for the bounds `0` and `0`. Previously, the solver wrote Inf or NaN into the solution or did not converge.
* On Linux, the HIP library is now built with CMake's HIP language support instead of the `FindHIP` module and hipcc. The HIP compiler is `amdclang++` from `ROCM_PATH`, unless `CMAKE_HIP_COMPILER` or `HIPCXX` selects another one. The host sources can still be built with any C++ compiler. `GPU_TARGETS` sets the GPU architectures of the HIP sources.

### Removed
* Removed the `USE_HIPCXX` option. CMake's HIP language support is now always used on Linux.
* Removed the GoogleTest superbuild in `deps` and its `BUILD_GTEST` option, which `install.sh -d` invoked without enabling it.

### Resolved issues
* Fixed the HIP backend ignoring the errors returned by rocPRIM calls, such as too small temporary storage, which could lead to allocations of undefined size and to undefined scan, reduction, sort and nonzero count results.
* Fixed the HIP backend ignoring the errors returned by HIP runtime calls for allocations, frees, memsets, copies, synchronizations and device queries, and not checking several kernel launches for errors.
* Fixed the ILU0 factorization and the triangular solve analysis of CSR and BCSR matrices on the accelerator ignoring errors from the rocSPARSE buffer size queries.
* Fixed `info_rocalution()` printing uninitialized memory when the rocBLAS or rocSPARSE version query fails.
* Fixed `LocalMatrix::ReadFileRSIO()` not checking the element size query when the index or value type of the file differs from the type of the matrix, and leaking temporary buffers when a read fails.
* Fixed `LocalMatrix::WriteFileRSIO()` reporting success when closing the file fails, for example on a full disk, leaving an incomplete file.
* Fixed `LocalMatrix::ExtractDiagonal()` and `LocalMatrix::ExtractInverseDiagonal()` aborting on CSR matrices with more rows than columns.
* Fixed `LocalMatrix::DiagonalMatrixMultL()` rejecting a vector with one entry per row on non-square CSR matrices, and reading out of bounds on CSR matrices with more rows than columns.
* Fixed `init_rocalution()` reporting an accelerator after `stop_rocalution()` when the accelerator was disabled with `disable_accelerator_rocalution()`.
* Fixed a HIP stream leak in `init_rocalution()` in multi-node builds when creating the HIP streams fails.
* Fixed the `Chebyshev` solver using a recurrence that is not the Chebyshev iteration and depends on the scale of the operator. It converged more slowly than the Chebyshev iteration and, depending on the scale of the operator, could diverge even with exact eigenvalue bounds, for example with FSAI preconditioning.
* Fixed the format warning of `GlobalMatrix::AMGUnsmoothedAggregation()` naming `AMGSmoothedAggregation()`.
* Fixed `LocalVector::Sync()` and `LocalMatrix::Sync()` deleting both the host and the accelerator copy of an object after `MoveToHostAsync()`, which lost its data and crashed on the next access.
* Fixed `LocalMatrix::MoveToAcceleratorAsync()` and `LocalMatrix::MoveToHostAsync()` leaking a full copy of the matrix for each additional call before `Sync()`. Additional calls before `Sync()` now do nothing, so changes made to the matrix between the first call and `Sync()` are not transferred.
* Fixed `LocalVector::Amax()` on the accelerator returning the index and value of the entry after the largest one, and reading past the end of the vector when the largest entry is the last one. This also affected the Linf residual norm and `GetAmaxResidualIndex()` of the iterative solvers on the accelerator.
* Fixed wrong results of the iterative triangular solves on the accelerator when both `LocalMatrix::ItLAnalyse()` and `LocalMatrix::ItUAnalyse()` are called on the same matrix, as the SGS preconditioner does with `TriSolverAlg_Iterative`.
* Fixed `ItILU0::GetConvergenceHistory()` on the accelerator returning an unwritten history and an undefined iteration count when rocSPARSE rejects the history query. The history is now empty in this case. Also fixed a memory leak when building ItILU0 with a convergence history in single precision.
* Fixed `ItILU0Algorithm::SyncSplit` on the accelerator stopping after the first sweep and returning zero factors, so the preconditioner is the identity, when `ItILU0Option::StoppingCriteria` or `ItILU0Option::ConvergenceHistory` is set without a norm option. `ItILU0Algorithm::AsyncSplit` failed with an illegal memory access in that case. The residual norm is now computed whenever the stopping criteria or the convergence history is requested, or when a history buffer is passed to `LocalMatrix::ItILU0Factorize()`. With only `ItILU0Option::ComputeNrmCorrection` set, the SyncSplit stopping test now also takes the residual into account.
* Fixed `init_rocalution()` making every host OpenMP parallel region, including those of the application, run on a single thread when rocALUTION is built with OpenMP 5.0 or newer, which is the default with `amdclang++`.
* Fixed `stop_rocalution()` aborting when nested parallelism was enabled before `init_rocalution()`, for example with `OMP_MAX_ACTIVE_LEVELS`, `OMP_NESTED` or a nested `OMP_NUM_THREADS` list, in builds with OpenMP 5.0 or newer. In builds with older OpenMP versions, such as with GCC, it restored a different maximum number of active parallel levels.
* Fixed a data race in the host `LocalVector::Amax()` with more than one OpenMP thread, which could return the index of a later entry when the largest magnitude occurs more than once. It now returns the first one, as the sequential code and the HIP backend do.
* Fixed a data race in the host Laplace 2D stencil `Apply()` that made it run slower with several OpenMP threads than with one.
* Fixed GS, SGS, ILU, ItILU0 and the LU solver aborting or leaking the triangular solve analysis when they are moved to the backend they are already on, for example by calling `MoveToAccelerator()` on a preconditioner that its solver has already moved.
* Fixed ILUT and IC failing to solve after they are moved to another backend. IC now moves its inverse diagonal, and the IC factorization on the accelerator now computes it.
* Fixed a HIP error that ended the process when an MCSR matrix that was never allocated or was cleared is copied to, from or on the accelerator. This also affected `LocalMatrix::MatrixMult()` into an MCSR matrix and reading a matrix file into an MCSR matrix on the accelerator.
* Fixed an assertion or a HIP error, such as out of memory, when a BCSR matrix that was never allocated, was cleared, or handed over its arrays with `LocalMatrix::LeaveDataPtrBCSR()` is copied, because the block counts were left uninitialized or kept. After `LeaveDataPtrBCSR()`, the matrix now also keeps its block dimension. This also affected `LocalMatrix::MatrixMult()` into a BCSR matrix and `LocalMatrix::ReadFileCSR()` into a BCSR matrix on the accelerator.
* Fixed BCSR matrices ignoring the requested block dimension. `LocalMatrix::AllocateBCSR()` used block dimension 2 for a matrix that was not BCSR yet and kept the block dimension of one that was; converting a matrix without entries to BCSR used block dimension 2; and the `LocalMatrix::MatrixAdd()` fallback converted back to BCSR with block dimension 1 and ended the process, which also affected `LocalMatrix::ILUpFactorize()` without levels.
* Fixed `LocalMatrix::ReplaceColumnVector()` on CSR matrices dropping the entry that follows an inserted one, which could corrupt memory. All other formats except DENSE fall back to the host CSR code, so they dropped the entry on both backends. On the accelerator, every call on a CSR matrix with entries also read the new number of entries from uninitialized memory, which could end the process with an illegal memory access.
* Fixed `LocalMatrix::ExtractColumnVector()`, `LocalMatrix::ExtractRowVector()`, `LocalMatrix::ReplaceColumnVector()` and `LocalMatrix::ReplaceRowVector()` doing nothing for a matrix without entries. The extractions now return zeros, and the replacements insert the nonzero values.
* Fixed `LocalMatrix::ExtractRowVector()` on DENSE host matrices filling only as many entries of the vector as the matrix has rows, and reading past the end of the matrix and writing past the end of the vector when the matrix has more rows than columns.
* Fixed the strength test of the pairwise aggregation (`PairwiseAMG`), which used a wrong threshold. The initial pass took every negative coupling as strong in rows without a positive coupling, and in rows with one measured the couplings against the largest positive coupling instead. The further passes took every negative coupling, and small positive ones, as strong, except in rows with a positive ghost coupling, which then set the threshold. A coupling is now strong if it is below `-beta` times the largest magnitude of the negative couplings of the row, ghost couplings included.
* Fixed the CMK, RCMK, MIS and multicoloring orderings of the pairwise aggregation (`PairwiseAMG::SetOrdering()`) visiting the nodes in the wrong order.
* Fixed `LocalMatrix::InitialPairwiseAggregation()` and `LocalMatrix::FurtherPairwiseAggregation()` dereferencing an empty permutation for an ordering other than 0 to 5. They now stop with an error.
* Fixed the strength of connection of `RugeStuebenAMG`, with the `Greedy` and the `PMIS` coarsening, for rows with a negative diagonal. Those rows took the couplings below the strength threshold as strong instead of those above it; strength now does not change when a row with a nonzero diagonal is scaled by -1. This also fixes `LocalMatrix::RSCoarsening()` writing past the end of an internal array for a row with a negative diagonal and no positive coupling, or for a row whose couplings are so small that the strength threshold rounds to zero.
* Fixed the PMIS strength of connection on the accelerator using an undefined diagonal sign for a row without diagonal. Such a row now counts as one with a positive diagonal, as on the host.
* Fixed `PairwiseAMG` and `LocalMatrix::InitialPairwiseAggregation()` reading the diagonal of a row without diagonal at an undefined index, which could crash. A missing diagonal now counts as zero.
* Fixed `LocalMatrix::MaximalIndependentSet()` returning an invalid permutation, with positions out of range, for matrices with a nonsymmetric sparsity pattern. This affected the MIS ordering of the pairwise aggregation (`PairwiseAMG::SetOrdering(4)`), which read and wrote out of bounds and skipped rows, and the `MultiElimination` preconditioner. The set is now independent in the pattern of `A + A^T`. The pairwise aggregation now also stops with an error instead of using an ordering that is not a permutation.
* Fixed `LocalMatrix::ConnectivityOrder()` returning the inverse permutation. It now returns the new position of each row, which `Permute()` takes.
* Fixed `LocalMatrix::CMK()` and `LocalMatrix::RCMK()` not numbering the neighbours of each node by increasing degree, and numbering every node after the first connected component in index order. The first component now also starts from node 0 when row 0 stores no diagonal.
* Fixed `PairwiseAMG::Build()` repeating the pairwise aggregation pass without end when a pass forms no pair before the coarsening factor is reached, for example on a level with nothing left to pair or, with a `GlobalMatrix`, when on every process the ghost couplings make the local couplings weak. Each repetition doubled the memory of the aggregation, until the build ran out of memory or failed an assert, also when a pass formed only a few pairs. A level now ends at the first further pass that pairs fewer than half of the nodes, keeping the coarsening reached so far, and the hierarchy ends at a level that does not coarsen.
* Fixed `PairwiseAMG::Build()` failing an assert on a level whose rows are all strongly diagonally dominant, such as a diagonal matrix, where the pairwise aggregation forms no aggregate. The hierarchy now ends before that level, which on the finest level is the "Could not build initial AMG level" error. With a `GlobalMatrix`, the hierarchy ends there on every process when a single process forms no aggregate.
* Fixed `LocalMatrix::Apply()` and `LocalMatrix::MatrixMult()` of complex DENSE matrices on the accelerator ending the program.
* Fixed `ApplyAdd()` of HYB and DENSE matrices on the accelerator overwriting the output instead of adding to it. This also affected preconditioners that build their blocks in these formats, such as the multicolored preconditioners and `MultiElimination`.
* Fixed `Permute()` and `PermuteBackward()` of COO matrices leaving the entries unsorted, which made a later conversion assert on the host and give wrong results on the accelerator. This also affected `LocalMatrix::PermuteBackward()` of the other formats (it goes through COO), `ReBuildNumeric()` of the multicolored ILU without decomposition, the multicolored preconditioners, `MultiElimination` and `DiagJacobiSaddlePointPrecond` with a COO operator, and `BlockPreconditioner` with a COO operator and a permutation.
* Fixed `LocalVector::Asum()`, `GlobalVector::Asum()` and `LocalVector::Amax()` of complex vectors on the accelerator ending the program, which also ended complex solves on the accelerator with the L1 residual norm, and complex `LocalMatrix` solves on the accelerator with the Linf residual norm.
* Fixed `LocalVector::Asum()` and `GlobalVector::Asum()` of complex vectors on the host returning the sums of the absolute real and imaginary parts as a complex number, whose real plus imaginary part is the BLAS scasum. On both backends they now return the sum of the moduli `|z|` with a zero imaginary part, consistent with `LocalVector::Amax()`, which returns the index and the modulus of the entry with the largest modulus. The L1 residual norm of complex solves on the host is now this sum, which is at least as large as before, so these solves can stop after a different number of iterations, and QMRCGStab and IDR(s), which also use the norm in their iteration, compute different iterates.
* Fixed `SGS` scaling by the inverse of the diagonal instead of the diagonal between its two triangular solves after `Build()`.
* Fixed `Jacobi::ResetOperator()`, `GS::ResetOperator()` and `SGS::ResetOperator()` ignoring the given operator and rebuilding from the previous one.
* Fixed `FSAI` and `LocalMatrix::FSAI()` for a row of the pattern with a single entry, which kept its pattern value unless that row of the matrix started at the diagonal. The entry is now computed from the diagonal entry of A in its column, which is what the general formula gives for a single entry. Such a row is a valid FSAI pattern only when its entry is on the diagonal.
* Fixed `AIChebyshev` building a wrong approximate inverse: the series started from the matrix instead of the identity, the shifted matrix was scaled only on its diagonal, and all coefficients were halved.
* Fixed `AS` and `RAS` leaving the last rows in no block when the number of rows is not a multiple of the number of blocks, `AS` weighting the overlaps of a block between two others wrongly, and not clipping to the matrix a block whose overlap reaches before the first or past the last row. `AS` now weights each row by 1 / (number of blocks that hold it).
* Fixed `MultiColoredGS` and `MultiColoredSGS` applying a wrong relaxation for `SetRelaxation()` values other than 1, and `MultiColoredSGS` without decomposition ignoring the relaxation.
* Fixed `MultiColoredSGS::ReBuildNumeric()` crashing without decomposition, where it deleted the blocks that only the decomposition allocates. Without decomposition, it now also analyses the new triangular factors for the solves, and with decomposition it frees the permuted copy of the matrix as `Build()` does.
* Fixed `LocalMatrix::SymbolicPower()` computing the pattern of A^(p+1) instead of A^p for p above 8. FSAI with a matrix power above 8 and ILU(p) without level control for p of 8 or more got larger patterns than requested, and multi-colored ILU was colored from the larger pattern for q above 8, which `MultiColoredILU::Set(p)` uses for p of 8 or more.
* Fixed `LocalMatrix::LLSolve()` on the host taking the last entry of each row as the diagonal of factors that store zeros above the diagonal, such as IC with a BCSR operator on the host. Without the inverse diagonal this gave NaN. With it, which the IC preconditioner passes, the results depended on the previous contents of the output vector. Without the inverse diagonal, `LLSolve()` now ends the program for a row that does not store its diagonal.
* Fixed the `ExtPI` interpolation of `RugeStuebenAMG`, with and without `SetInterpolationFF1Limit(true)`, for a strong coupling to an F point that has no couplings to distribute it through. The coupling was dropped, or gave infinite or NaN weights for negative diagonals and in the ghost part of a distributed matrix. It is now lumped into the diagonal, as in hypre, so the interpolation reproduces constants for zero row sums. A row whose diagonal is zero, for example after this lumping, is now left without interpolation instead of getting infinite or NaN weights.
* Fixed the scaled coarse-grid correction of the multigrid cycles using the right-hand side instead of the residual on the coarse levels. With four or more levels, this changes the convergence of `MultiGrid`, which scales by default, and of the AMG solvers with `BaseMultiGrid::SetScaling(true)`.
* Fixed `MultiGrid` crashing in `Clear()` and in its destructor after `Build()`, because it deleted transfer maps that only the AMG solvers allocate.
* Fixed the W-cycle returning the V-cycle result at the cost of a W-cycle. Its second coarse cycle now continues from the first instead of restarting from zero, and the coarsest level is solved once per visit.
* Fixed the K-cycle returning NaN when a CG step on a coarse level breaks down: for a zero coarse residual, a zero first step, or a first step that solves the level exactly.
* Fixed `UAAMG` failing an assert at a level without aggregates. Like `SAAMG`, it now stops coarsening at that level.
* Fixed `UAAMG::ReBuildNumeric()` not scaling the coarse operators by the over-interpolation factor.
* Fixed `ReBuildNumeric()` of the AMG solvers after `SetHostLevels()`. It failed a backend assert, and for a `GlobalMatrix` it gave a coarse operator a wrong parallel manager, or for `PairwiseAMG` none, which broke the solve. The coarse operators now end up on the backend of their level and keep their parallel manager.
* Fixed the AMG solvers after `SetHostLevels()` and another `Build()`, also by an outer solver that uses the AMG as preconditioner. The new hierarchy kept the number of host levels, but had all its levels on the backend of the finest operator. With the finest operator on the accelerator, the solve and `ReBuildNumeric()` then failed a backend assert, or could crash in a release build. `Build()` now moves the coarsest levels of the new hierarchy and its coarse grid solver to the host, at most all levels but the finest.
* Fixed the preconditioned BiCGStab solver returning a wrong solution after a breakdown in the second half of an iteration.
* Fixed the BiCGStab solver not detecting an infinite omega for complex types.
* Fixed the QMRCGStab solver starting a second solve from the search direction of the previous solve, returning NaN when the first half step solves the system exactly, and reporting one iteration more than it performed.
* Fixed the MixedPrecisionDC solver counting a residual check as an iteration, so that a maximum of one iteration made no correction, and `ReBuildNumeric()` keeping the values of the old matrix in its low precision copy, also after `ResetOperator()`. The reset operator must have the same sparsity pattern.
* Fixed the BiCGStab(l) solver iterating when the initial residual is already within the tolerance.
* Fixed `FixedPoint::SolveZeroSol()` returning the previous solution instead of zero when it does not iterate. This also fixes the multigrid cycles without pre-smoothing, where the coarse levels started from their previous correction.
* Fixed the minimum iteration count being ignored at the restarts of GMRES and FGMRES, in the BiCG part of BiCGStab(l) and in the inner steps of IDR. At these checks, `GetCurrentResidual()` now returns the residual just checked, and reaching the maximum iteration count takes precedence over the divergence tolerance, as at the other checks.
* Fixed GMRES and FGMRES returning NaN when a cycle solves the system exactly below the minimum iteration count. They now stop with the exact solution and status 0, as no tolerance is reached below the minimum.
* Fixed the BiCGStab(l) solver returning NaN when it iterates far past convergence, below the minimum iteration count or with zero tolerances, until its residuals underflow, and at a breakdown with omega = 0. It now stops with the last solution and status 0.
* Fixed the residual history file dropping the last residual and, after a second solve, holding the residuals of the first solve.
* Fixed `ReBuildNumeric()` of the IDR solver clearing its shadow space and of the Chebyshev iteration un-building it, so that the next solve failed. `ReBuildNumeric()` of the Chebyshev iteration keeps its eigenvalue bounds; to change them, call `Chebyshev::Set()`, which needs no new `Build()`.
* Fixed the IDR solver ending the program at a breakdown. It now stops and returns the last iterate and its residual. The breakdown checks also catch an M(k,k) or omega of -inf, and infinite values of complex types. When a step solves the system exactly below the minimum iteration count, IDR now stops with the exact solution and status 0 instead of a breakdown.
* Fixed the LU solver ending the program for a DENSE operator with the default triangular solve.
* Fixed the host conversion of non-square matrices to DIA writing past the end of the values when the matrix has more rows than columns, for example in `LocalMatrix::ExtractSubMatrices()` and in the multicolored preconditioners. When the matrix has more columns than rows, `LocalMatrix::LeaveDataPtrDIA()` aborted on the result. The host now sizes the DIA values and decides whether the conversion succeeds as the accelerator does, so some non-square matrices that the host converted to DIA before now stay in CSR, with the fallback warning.
* Fixed the host `LocalMatrix::Apply()` and `LocalMatrix::ApplyAdd()` of non-square DIA matrices ignoring the entries in the columns at or past the number of rows, and reading past the end of the input vector when the matrix has more rows than columns. They also skipped diagonals in the last rows, for square matrices too, when the offsets set with `LocalMatrix::SetDataPtrDIA()` were not in ascending order.
* Fixed the conversion to DIA of matrices with more than 2^31 - 1 DIA entries, which overflowed the number of entries or the index into the values. These matrices now stay in CSR, with the fallback warning.
* Fixed `LocalMatrix::RSExtPIInterpolation()`, `LocalMatrix::AMGSmoothedAggregation()` and `LocalMatrix::CompressAdd()` on the accelerator, and the `GlobalMatrix` functions that use them, synchronizing with a block-wide barrier that not every wavefront of the block reaches. Under GPU AddressSanitizer, this was a heap-use-after-free.
* Fixed `LocalMatrix::Check()` accepting CSR column indices equal to the number of columns, values of minus infinity and complex values with an infinite part, rejecting complex matrices with a stored zero, and reading past the column indices of a matrix whose last row has no entries.
* Fixed `LocalVector::Check()` rejecting integer and complex vectors with a zero entry, and accepting complex vectors with an infinite entry.
* Fixed `LocalVector::SetRandomUniform()` on the host setting every entry of an integer vector to the lower bound and of a bool vector to the upper bound. Integer and bool vectors now get values uniformly distributed from the lower to the upper bound.
* Fixed `LocalMatrix::ItLUSolve()` and `LocalMatrix::ItLLSolve()` on the host limiting the second triangular solve to the iterations of the first one, and `LocalMatrix::ItLLSolve()` stopping the transposed solve before it reaches the tolerance.
* Fixed `LocalMatrix::CreateFromMap()` doing nothing when the matrix has no entries yet, and losing entries and leaving values at zero for maps with unassigned nodes, which also affected the restriction of `PairwiseAMG`.
* Fixed `LocalMatrix::Transpose()` of a matrix without entries not swapping the sizes, and leaving the output matrix unchanged when it transposes into another matrix. `LocalMatrix::Transpose()` into a matrix that is not in CSR format no longer aborts, which made FSAI abort when it was built again with a preconditioner matrix format other than CSR. The output matrix gets the format of the transposed matrix, or CSR if the conversion to that format fails.
* Fixed `LocalStencil::ApplyAdd()` overwriting the output instead of adding to it, and the Laplace stencil on the host ignoring the scalar.

## (Unreleased) rocALUTION 4.1.1

### Added
* Added support for the `gfx1250-strict` architecture.
* Added the matrix-matrix based interpolation operators `MMExtPI` and `MMExtPE` to the `InterpolationType` enumeration, selectable through `RugeStuebenAMG::SetInterpolationType()`.
* Added `RugeStuebenAMG::SetInterpolationTruncationFactor()` and `RugeStuebenAMG::SetInterpolationMaxElmts()` to truncate the interpolation operator, dropping entries that are small relative to the largest entry of their row and capping the number of entries per row.
* Added `BaseAMG::SetMaxLevels()` to limit the number of levels of an AMG hierarchy. Combined with passing another AMG solver as the coarse grid solver, this allows hybrid hierarchies such as unsmoothed aggregation on the finest levels followed by Ruge-Stueben AMG on the coarser levels.
* Added the `cg-uaamg-rsamg` sample demonstrating a hybrid unsmoothed aggregation / Ruge-Stueben AMG preconditioner.

### Changed
* All device memory in the HIP backend, including temporary rocPRIM buffers, is now allocated and released through the rocALUTION allocation layer instead of direct `hipMalloc` and `hipFree` calls.

### Resolved issues
* Fixed wrong smoothed aggregation prolongation operators from the deprecated `LocalMatrix::AMGSmoothedAggregation()` on wave32 devices for rows with 64 or more entries.
* Fixed the host fallback of the Ruge-Stueben AMG extended+i interpolation operator when a row exceeds the LDS capacity.
* Fixed `LocalMatrix::ItLUSolve()` and `LocalMatrix::ItLLSolve()` on the accelerator limiting the iterations of the second triangular solve to the number of iterations performed by the first one.
* Fixed a heap corruption in `GlobalMatrix::CoarsenOperator()`, used by the pairwise AMG with MPI, when a process sends its coarse boundary to more than one neighboring process.
* Fixed `LocalMatrix::Sort()` on the accelerator ignoring errors when creating the identity permutation for CSR and COO matrices.
* Fixed wrong coarse operators of the AMG preconditioners with MPI on wave32 (RDNA) hardware, caused by `LocalMatrix::CompressAdd()` on the accelerator.
* Fixed a crash in `LocalMatrix::CompressAdd()` on the accelerator when a row of the interior part exceeds the supported number of entries and the computation falls back to the host.
* Fixed an out-of-bounds access and wrong coarse ghost columns in `GlobalMatrix::CoarsenOperator()`, used by the pairwise AMG with MPI, when boundary points are not aggregated.
* Fixed the TNS preconditioner keeping its approximate inverse on the host when moved to the accelerator.
* Fixed HIP work in the distributed CSR receive and in `CompressAdd` running on the null stream instead of the current stream.
* Fixed the Smoothed-Aggregation AMG preconditioner not falling back to the host when the prolongation fill failed on the accelerator or in a non-CSR format.
* Fixed the FCG and QMRCGStab solvers returning NaN when the initial residual is already below the absolute tolerance, for example a zero right-hand side on the coarsest AMG level.
* Fixed `init_rocalution()` reporting the HIP backend as available when creating the rocBLAS or rocSPARSE handle failed on a device selected with `set_device_rocalution()`, which caused `info_rocalution()` to crash. rocALUTION now falls back to the host backend in this case.
* Fixed a crash or out-of-bounds access in the SA-AMG, UA-AMG and Ruge-Stueben AMG setup with MPI when a process sets only the local part of a `GlobalMatrix` with `SetLocalDataPtrCSR()` or `SetLocalDataPtrCOO()` and its neighbors reference its columns. Such a matrix now has an empty ghost part with one row per local row.

## rocALUTION 4.1.0 for ROCm 7.2.0

### Added
* Added `--clients-only` option to the `install.sh` and `rmake.py` scripts to allow building only the clients while using an already installed version of rocALUTION.

### Resolved issues
* Fixed a GPU memory access fault in the Smoothed-Aggregation AMG preconditioner on wave32 (RDNA) hardware.

## rocALUTION 4.0.1 for ROCm 7.1.0

### Added
* Added support for gfx950.

### Changed
* Updated the default build standard to C++17 when compiling rocALUTION from source (previously C++14).

### Optimized
* Improved and expanded user documentation.

### Resolved issues
* Fixed a bug in the GPU hashing algorithm that occurred when not compiling with -O2/-O3.
* Fixed an issue with the SPAI preconditioner when using complex numbers.

## rocALUTION 3.2.3 for ROCm 6.4.1

### Added
* The `-a` option has been added to the `rmake.py` build script. This option provides a way to select specific architectures when building on Windows.

### Resolved issues
* Fixed an issue where the `HIP_PATH` environment variable was being ignored when compiling on Windows.

## rocALUTION 3.2.2 for ROCm 6.4.0

### Changed
* Improved documentation

## rocALUTION 3.2.1 for ROCm 6.3.0

### Added

* Support for gfx1200, gfx1201, and gfx1151.

### Changed

* Changed the default compiler from `hipcc` to `amdclang` in the installation script and cmake files.
* Changed the address sanitizer build targets. Now only `gfx908:xnack+`, `gfx90a:xnack+`, `gfx940:xnack+`, `gfx941:xnack+`, and `gfx942:xnack+` are built with `BUILD_ADDRESS_SANITIZER=ON`.

### Resolved issues

* Fix hang in `RS-AMG` for Navi on some specific matrix sparsity patterns.
* Fix wrong results in `Apply` on multi-GPU setups.

## rocALUTION 3.2.0 for ROCm 6.2.0

### Additions
* New file I/O based on rocsparse I/O format
* `GetConvergenceHistory` for ItILU0 preconditioner

### Deprecations
* `LocalMatrix::ReadFileCSR`
* `LocalMatrix::WriteFileCSR`
* `GlobalMatrix::ReadFileCSR`
* `GlobalMatrix::WriteFileCSR`

## rocALUTION 3.1.1 for ROCm 6.1.0

### Additions

* `TripleMatrixProduct` functionality for `GlobalMatrix`
* Multi-Node/GPU support for `UA-AMG`, `SA-AMG` and `RS-AMG`
* Iterative ILU0 preconditioner `ItILU0`
* Iterative triangular solve, selectable via `SolverDecr` class

### Deprecations

* `LocalMatrix::AMGConnect`
* `LocalMatrix::AMGAggregate`
* `LocalMatrix::AMGPMISAggregate`
* `LocalMatrix::AMGSmoothedAggregation`
* `LocalMatrix::AMGAggregation`
* `PairwiseAMG`

### Known Issues
* `PairwiseAMG` does currently not support matrix sizes that exceed int32 range
* `PairwiseAMG` might fail building the hierarchy on certain input matrices

## rocALUTION 3.0.3 for ROCm 6.0.0

### Additions

* Support for 64bit integer vectors
* Inclusive and exclusive sum functionality for vector classes
* Transpose functionality for `GlobalMatrix` and `LocalMatrix`
* `TripleMatrixProduct` functionality for `LocalMatrix`
* `Sort()` function for `LocalVector` class
* Multiple stream support to the HIP backend

### Optimizations

* `GlobalMatrix::Apply()` now uses multiple streams to better hide communication

### Changes

* Matrix dimensions and number of non-zeros are now stored using 64-bit integers
* Improved the ILUT preconditioner

### Deprecations

* `LocalVector::GetIndexValues(ValueType*)`
* `LocalVector::SetIndexValues(const ValueType*)`
* `LocalMatrix::RSDirectInterpolation(const LocalVector&, const LocalVector&, LocalMatrix*, LocalMatrix*)`
* `LocalMatrix::RSExtPIInterpolation(const LocalVector&, const LocalVector&, bool, float, LocalMatrix*, LocalMatrix*)`
* `LocalMatrix::RugeStueben()`
* `LocalMatrix::AMGSmoothedAggregation(ValueType, const LocalVector&, const LocalVector&, LocalMatrix*, LocalMatrix*, int)`
* `LocalMatrix::AMGAggregation(const LocalVector&, LocalMatrix*, LocalMatrix*)`

### Fixes

* Unit tests no longer ignore BCSR block dimension
* Fixed documentation typos
* Bug in multi-coloring for non-symmetric matrix patterns

## rocALUTION 2.1.11 for ROCm 5.7.0

### Additions

* Support for gfx940, gfx941, and gfx942

### Fixes

* OpenMP runtime issue with Windows toolchain

## rocALUTION 2.1.9 for ROCm 5.6.0

### Fixes

* Synchronization issues in level 1 routines

## rocALUTION 2.1.8 for ROCm 5.5.0

### Additions

* Build support for Navi32

### Fixes

* Typo in MPI backend
* Bug with the backend when HIP support is disabled
* Bug in SAAMG hierarchy building on the HIP backend
* Improved SAAMG hierarchy build performance on the HIP backend

### Deprecations

* `LocalVector::GetIndexValues(ValueType*)`: use
  `LocalVector::GetIndexValues(const LocalVector&, LocalVector*)` instead
* `LocalVector::SetIndexValues(const ValueType*)`: use
  `LocalVector::SetIndexValues(const LocalVector&, const LocalVector&)` instead
* `LocalMatrix::RSDirectInterpolation(const LocalVector&, const LocalVector&, LocalMatrix*, LocalMatrix*)`:
  use `LocalMatrix::RSDirectInterpolation(const LocalVector&, const LocalVector&, LocalMatrix*)`
  instead
* `LocalMatrix::RSExtPIInterpolation(const LocalVector&, const LocalVector&, bool, float, LocalMatrix*, LocalMatrix*)`:
  use `LocalMatrix::RSExtPIInterpolation(const LocalVector&, const LocalVector&, bool, LocalMatrix*)`
  instead
* `LocalMatrix::RugeStueben()`
* `LocalMatrix::AMGSmoothedAggregation(ValueType, const LocalVector&, const LocalVector&, LocalMatrix*, LocalMatrix*, int)`:
  use `LocalMatrix::AMGAggregation(ValueType, const LocalVector&, const LocalVector&, LocalMatrix*, int)`
  instead
* `LocalMatrix::AMGAggregation(const LocalVector&, LocalMatrix*, LocalMatrix*)`: use
  `LocalMatrix::AMGAggregation(const LocalVector&, LocalMatrix*)` instead

## rocALUTION 2.1.3 for ROCm 5.4.0

### Additions

* Build support for Navi31 and Navi33
* Support for non-squared global matrices

### Fixes

* Memory leak in MatrixMult on HIP backend
* Global structures can now be used with a single process

### Changes

* Switched GTest death test style to 'threadsafe'
* Removed the native compiler option that was used during default library compilation

### Deprecations

* `GlobalVector::GetGhostSize()`
* `ParallelManager::GetGlobalSize(), ParallelManager::GetLocalSize()`, `ParallelManager::SetGlobalSize()`,
  and `ParallelManager::SetLocalSize()`
* `Vector::GetGhostSize()`
* `Multigrid::SetOperatorFormat(unsigned int)`: use `Multigrid::SetOperatorFormat(unsigned int, int)`
  instead
* `RugeStuebenAMG::SetCouplingStrength(ValueType)`: use `SetStrengthThreshold(float)` instead

## rocALUTION 2.1.0 for ROCm 5.3.0

### Additions

* Benchmarking tool
* Ext+I Interpolation with sparsify strategies added for RS-AMG

### Optimizations

* ParallelManager

## rocALUTION 2.0.3 for ROCm 5.2.0

### Additions

* New packages for test and benchmark executables on all supported operating systems using CPack

## rocALUTION 2.0.2 for ROCm 5.1.0

### Additions

* Added out-of-place matrix transpose functionality
* Added LocalVector<bool>

## rocALUTION 2.0.1 for ROCm 5.0.0

### Changes

* Changed to C++ 14 Standard
* Added sanitizer option
* Improved documentation

### Deprecations

* `GlobalPairwiseAMG` class: use `PairwiseAMG` instead

## rocALUTION 1.13.2 for ROCm 4.5.0

### Additions

* AddressSanitizer build option
* Enabled beta support for Windows 10

### Changes

* Packaging has been split into a runtime package (`rocalution`) and a development package
  (`rocalution-devel`):
  The development package depends on the runtime package. When installing the runtime package,
  the package manager will suggest the installation of the development package to aid users
  transitioning from the previous version's combined package. This suggestion by package manager is
  for all supported operating systems (except CentOS 7) to aid in the transition. The `suggestion`
  feature in the runtime package is introduced as a deprecated feature and will be removed in a future
  ROCm release.

### Deprecations

* `GlobalPairwiseAMG`: use `PairwiseAMG` instead

### Optimizations

* Improved (A)MG smoothing and convergence rate
* Improved IDR Gram-Schmidt process
* Optimized (A)MG solving phase

## rocALUTION 1.12.1 for ROCm 4.3.0

### Additions

* Support for gfx90a target
* Support for gfx1030 target

### Optimizations

* Install script

## rocALUTION 1.11.5 for ROCm 4.0.0

### Additions

* Changelog
* Block compressed sparse row (BCSRR) format support

### Changes

* Update to the Debian package name
* CMake file adjustments

### Fixes

* NaN issues

## rocALUTION 1.10 for ROCm 3.9

### Additions

* rocRAND support for GPU sampling of random data

## rocALUTION 1.9.3 for ROCm 3.8

### Additions

* `csr2dense` and `dense2csr` to HIP backend

## rocALUTION 1.9.1 for ROCm 3.5

### Additions

* Static build
* BCSR matrix format for SpMV

### Fixes

* Bug in conversion from CSR to HYB format
