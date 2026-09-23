/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_BASE_MATRIX_HPP_
#define ROCALUTION_BASE_MATRIX_HPP_

#include "backend_manager.hpp"
#include "matrix_formats.hpp"
#include "rocalution/utils/types.hpp"

namespace rocalution
{
    enum _itilu0_alg : unsigned int;
    typedef _itilu0_alg ItILU0Algorithm;

    template <typename ValueType>
    class BaseVector;
    template <typename ValueType>
    class HostVector;
    template <typename ValueType>
    class HIPAcceleratorVector;

    template <typename ValueType>
    class HostMatrixCSR;
    template <typename ValueType>
    class HostMatrixCOO;
    template <typename ValueType>
    class HostMatrixDIA;
    template <typename ValueType>
    class HostMatrixELL;
    template <typename ValueType>
    class HostMatrixHYB;
    template <typename ValueType>
    class HostMatrixDENSE;
    template <typename ValueType>
    class HostMatrixMCSR;
    template <typename ValueType>
    class HostMatrixBCSR;

    template <typename ValueType>
    class HIPAcceleratorMatrixCSR;
    template <typename ValueType>
    class HIPAcceleratorMatrixMCSR;
    template <typename ValueType>
    class HIPAcceleratorMatrixBCSR;
    template <typename ValueType>
    class HIPAcceleratorMatrixCOO;
    template <typename ValueType>
    class HIPAcceleratorMatrixDIA;
    template <typename ValueType>
    class HIPAcceleratorMatrixELL;
    template <typename ValueType>
    class HIPAcceleratorMatrixHYB;
    template <typename ValueType>
    class HIPAcceleratorMatrixDENSE;

    /// Base class for all host/accelerator matrices
    template <typename ValueType>
    class BaseMatrix
    {
    public:
        BaseMatrix();
        virtual ~BaseMatrix();

        /** \brief Return the number of rows in the matrix */
        int GetM(void) const;
        /** \brief Return the number of columns in the matrix */
        int GetN(void) const;
        /** \brief Return the non-zeros of the matrix */
        int64_t GetNnz(void) const;
        /** \brief Shows simple info about the object */
        virtual void Info(void) const = 0;
        /** \brief Return the matrix format id (see matrix_formats.hpp) */
        virtual unsigned int GetMatFormat(void) const = 0;
        /** \brief Return the block dimension of the matrix */
        virtual int GetMatBlockDimension(void) const;
        /** \brief Copy the backend descriptor information */
        virtual void set_backend(const Rocalution_Backend_Descriptor& local_backend);
        /** \brief Perform a sanity check of the matrix */
        virtual bool Check(void) const;

        /** \brief Allocate CSR Matrix */
        virtual void AllocateCSR(int64_t nnz, int nrow, int ncol);
        /** \brief Allocate BCSR Matrix */
        virtual void AllocateBCSR(int64_t nnzb, int nrowb, int ncolb, int blockdim);
        /** \brief Allocate MCSR Matrix */
        virtual void AllocateMCSR(int64_t nnz, int nrow, int ncol);
        /** \brief Allocate COO Matrix */
        virtual void AllocateCOO(int64_t nnz, int nrow, int ncol);
        /** \brief Allocate DIA Matrix */
        virtual void AllocateDIA(int64_t nnz, int nrow, int ncol, int ndiag);
        /** \brief Allocate ELL Matrix */
        virtual void AllocateELL(int64_t nnz, int nrow, int ncol, int max_row);
        /** \brief Allocate HYB Matrix */
        virtual void
            AllocateHYB(int64_t ell_nnz, int64_t coo_nnz, int ell_max_row, int nrow, int ncol);
        /** \brief Allocate DENSE Matrix */
        virtual void AllocateDENSE(int nrow, int ncol);

        /** \brief Initialize a COO matrix on the Host with externally allocated data */
        virtual void
            SetDataPtrCOO(int** row, int** col, ValueType** val, int64_t nnz, int nrow, int ncol);
        /** \brief Leave a COO matrix to Host pointers */
        virtual void LeaveDataPtrCOO(int** row, int** col, ValueType** val);

        /** \brief Initialize a CSR matrix on the Host with externally allocated data */
        virtual void SetDataPtrCSR(
            PtrType** row_offset, int** col, ValueType** val, int64_t nnz, int nrow, int ncol);
        /** \brief Leave a CSR matrix to Host pointers */
        virtual void LeaveDataPtrCSR(PtrType** row_offset, int** col, ValueType** val);

        /** \brief Initialize a BCSR matrix on the Host with externally allocated data */
        virtual void SetDataPtrBCSR(int**       row_offset,
                                    int**       col,
                                    ValueType** val,
                                    int64_t     nnzb,
                                    int         nrowb,
                                    int         ncolb,
                                    int         blockdim);
        /** \brief Leave a BCSR matrix to Host pointers */
        virtual void LeaveDataPtrBCSR(int** row_offset, int** col, ValueType** val, int& blockdim);

        /** \brief Initialize a MCSR matrix on the Host with externally allocated data */
        virtual void SetDataPtrMCSR(
            int** row_offset, int** col, ValueType** val, int64_t nnz, int nrow, int ncol);
        /** \brief Leave a MCSR matrix to Host pointers */
        virtual void LeaveDataPtrMCSR(int** row_offset, int** col, ValueType** val);

        /** \brief Initialize an ELL matrix on the Host with externally allocated data */
        virtual void
            SetDataPtrELL(int** col, ValueType** val, int64_t nnz, int nrow, int ncol, int max_row);
        /** \brief Leave an ELL matrix to Host pointers */
        virtual void LeaveDataPtrELL(int** col, ValueType** val, int& max_row);

        /** \brief Initialize a DIA matrix on the Host with externally allocated data */
        virtual void SetDataPtrDIA(
            int** offset, ValueType** val, int64_t nnz, int nrow, int ncol, int num_diag);
        /** \brief Leave a DIA matrix to Host pointers */
        virtual void LeaveDataPtrDIA(int** offset, ValueType** val, int& num_diag);

        /** \brief Initialize a DENSE matrix on the Host with externally allocated data */
        virtual void SetDataPtrDENSE(ValueType** val, int nrow, int ncol);
        /** \brief Leave a DENSE matrix to Host pointers */
        virtual void LeaveDataPtrDENSE(ValueType** val);

        /** \brief Clear (free) the matrix */
        virtual void Clear(void) = 0;

        /** \brief Set all the values to zero */
        virtual bool Zeros(void);

        /** \brief Scale all values */
        virtual bool Scale(ValueType alpha);
        /** \brief Scale the diagonal entries of the matrix with alpha */
        virtual bool ScaleDiagonal(ValueType alpha);
        /** \brief Scale the off-diagonal entries of the matrix with alpha */
        virtual bool ScaleOffDiagonal(ValueType alpha);
        /** \brief Add alpha to all values */
        virtual bool AddScalar(ValueType alpha);
        /** \brief Add alpha to the diagonal entries of the matrix */
        virtual bool AddScalarDiagonal(ValueType alpha);
        /** \brief Add alpha to the off-diagonal entries of the matrix */
        virtual bool AddScalarOffDiagonal(ValueType alpha);

        /** \brief Extrat a sub-matrix with row/col_offset and row/col_size */
        virtual bool ExtractSubMatrix(int                    row_offset,
                                      int                    col_offset,
                                      int                    row_size,
                                      int                    col_size,
                                      BaseMatrix<ValueType>* mat) const;

        /** \brief Extract the diagonal values of the matrix into a LocalVector */
        virtual bool ExtractDiagonal(BaseVector<ValueType>* vec_diag) const;
        /** \brief Extract the inverse (reciprocal) diagonal values of the matrix into a LocalVector */
        virtual bool ExtractInverseDiagonal(BaseVector<ValueType>* vec_inv_diag) const;
        /** \brief Extract the upper triangular matrix */
        virtual bool ExtractU(BaseMatrix<ValueType>* U) const;
        /** \brief Extract the upper triangular matrix including diagonal */
        virtual bool ExtractUDiagonal(BaseMatrix<ValueType>* U) const;
        /** \brief Extract the lower triangular matrix */
        virtual bool ExtractL(BaseMatrix<ValueType>* L) const;
        /** \brief Extract the lower triangular matrix including diagonal */
        virtual bool ExtractLDiagonal(BaseMatrix<ValueType>* L) const;

        /** \brief Perform (forward) permutation of the matrix */
        virtual bool Permute(const BaseVector<int>& permutation);

        /** \brief Perform (backward) permutation of the matrix */
        virtual bool PermuteBackward(const BaseVector<int>& permutation);

        /** \brief Create permutation vector for CMK reordering of the matrix */
        virtual bool CMK(BaseVector<int>* permutation) const;
        /** \brief Create permutation vector for reverse CMK reordering of the matrix */
        virtual bool RCMK(BaseVector<int>* permutation) const;
        /** \brief Create permutation vector for connectivity reordering of the matrix (increasing nnz per row) */
        virtual bool ConnectivityOrder(BaseVector<int>* permutation) const;

        /** \brief Perform multi-coloring decomposition of the matrix; Returns number of
        * colors, the corresponding sizes (the array is allocated in the function)
        * and the permutation
        */
        virtual bool
            MultiColoring(int& num_colors, int** size_colors, BaseVector<int>* permutation) const;

        /** \brief Perform maximal independent set decomposition of the matrix; Returns the
        * size of the maximal independent set and the corresponding permutation */
        virtual bool MaximalIndependentSet(int& size, BaseVector<int>* permutation) const;

        /** \brief Return a permutation for saddle-point problems (zero diagonal entries),
        * where all zero diagonal elements are mapped to the last block;
        * the return size is the size of the first block */
        virtual bool ZeroBlockPermutation(int& size, BaseVector<int>* permutation) const;

        /** \brief Convert the matrix from another matrix (with different structure) */
        virtual bool ConvertFrom(const BaseMatrix<ValueType>& mat) = 0;

        /** \brief Copy from another matrix */
        virtual void CopyFrom(const BaseMatrix<ValueType>& mat) = 0;

        /** \brief Copy to another matrix */
        virtual void CopyTo(BaseMatrix<ValueType>* mat) const = 0;

        /** \brief Async copy from another matrix */
        virtual void CopyFromAsync(const BaseMatrix<ValueType>& mat);

        /** \brief Copy to another matrix */
        virtual void CopyToAsync(BaseMatrix<ValueType>* mat) const;

        /** \brief Copy from CSR array (the matrix has to be allocated) */
        virtual void CopyFromCSR(const PtrType* row_offsets, const int* col, const ValueType* val);

        /** \brief Copy to CSR array (the arrays have to be allocated) */
        virtual void CopyToCSR(PtrType* row_offsets, int* col, ValueType* val) const;

        /** \brief Copy from COO array (the matrix has to be allocated) */
        virtual void CopyFromCOO(const int* row, const int* col, const ValueType* val);

        /** \brief Copy to COO array (the arrays have to be allocated) */
        virtual void CopyToCOO(int* row, int* col, ValueType* val) const;

        /** \brief Allocates and copies a host CSR matrix */
        virtual void CopyFromHostCSR(const PtrType*   row_offset,
                                     const int*       col,
                                     const ValueType* val,
                                     int64_t          nnz,
                                     int              nrow,
                                     int              ncol);

        /** \brief Create a restriction matrix operator based on an int vector map */
        virtual bool CreateFromMap(const BaseVector<int>& map, int n, int m);
        /** \brief Create a restriction and prolongation matrix operator based on an int vector map */
        virtual bool
            CreateFromMap(const BaseVector<int>& map, int n, int m, BaseMatrix<ValueType>* pro);

        /** \brief Read matrix from MTX (Matrix Market Format) file */
        virtual bool ReadFileMTX(const std::string& filename);
        /** \brief Write matrix to MTX (Matrix Market Format) file */
        virtual bool WriteFileMTX(const std::string& filename) const;

        /** \brief Read matrix from CSR (ROCALUTION binary format) file */
        virtual bool ReadFileCSR(const std::string& filename);
        /** \brief Write matrix to CSR (ROCALUTION binary format) file */
        virtual bool WriteFileCSR(const std::string& filename) const;

        /** \brief Read matrix from rocsparse I/O file */
        virtual bool ReadFileRSIO(const std::string& filename);
        /** \brief Write matrix to rocsparse I/O file */
        virtual bool WriteFileRSIO(const std::string& filename) const;

        /** \brief Perform symbolic computation (structure only) of |this|^p */
        virtual bool SymbolicPower(int p);

        /** \brief Perform symbolic matrix-matrix multiplication (i.e. determine the structure),
        * this = this*src */
        virtual bool SymbolicMatMatMult(const BaseMatrix<ValueType>& src);
        /** \brief Multiply two matrices, this = A * B */
        virtual bool MatMatMult(const BaseMatrix<ValueType>& A, const BaseMatrix<ValueType>& B);
        /** \brief Perform symbolic matrix-matrix multiplication (i.e. determine the structure),
        * this = A*B */
        virtual bool SymbolicMatMatMult(const BaseMatrix<ValueType>& A,
                                        const BaseMatrix<ValueType>& B);
        /** \brief Perform numerical matrix-matrix multiplication (i.e. value computation),
        * this = A*B */
        virtual bool NumericMatMatMult(const BaseMatrix<ValueType>& A,
                                       const BaseMatrix<ValueType>& B);
        /** \brief Multiply the matrix with diagonal matrix (stored in LocalVector),
        * this=this*diag (right multiplication) */
        virtual bool DiagonalMatrixMultR(const BaseVector<ValueType>& diag);
        /** \brief Multiply the matrix with diagonal matrix (stored in LocalVector),
        * this=diag*this (left multiplication) */
        virtual bool DiagonalMatrixMultL(const BaseVector<ValueType>& diag);
        /** \brief Perform matrix addition, this = alpha*this + beta*mat;
        * if structure==false the structure of the matrix is not changed,
        * if structure==true new data structure is computed */
        virtual bool MatrixAdd(const BaseMatrix<ValueType>& mat,
                               ValueType                    alpha,
                               ValueType                    beta,
                               bool                         structure);

        /** \brief Perform ILU(0) factorization */
        virtual bool ILU0Factorize(void);
        /** \brief Perform LU factorization */
        virtual bool LUFactorize(void);
        /** \brief Perform ILU(t,m) factorization based on threshold and maximum
        * number of elements per row */
        virtual bool ILUTFactorize(double t, int maxrow);
        /** \brief Perform ILU(p) factorization based on power (see power(q)-pattern method, D. Lukarski
        * "Parallel Sparse Linear Algebra for Multi-core and Many-core Platforms - Parallel Solvers
        * and
        * Preconditioners", PhD Thesis, 2012, KIT) */
        virtual bool ILUpFactorizeNumeric(int p, const BaseMatrix<ValueType>& mat);
        /// Perform Iterative ILU0 factorization
        virtual bool ItILU0Factorize(ItILU0Algorithm alg,
                                     int             option,
                                     int             max_iter,
                                     double          tolerance,
                                     int*            niter,
                                     double*         history);

        /** \brief Perform IC(0) factorization */
        virtual bool ICFactorize(BaseVector<ValueType>* inv_diag);

        /** \brief Analyse the structure (level-scheduling) */
        virtual void LUAnalyse(void);
        /** \brief Delete the analysed data (see LUAnalyse) */
        virtual void LUAnalyseClear(void);
        /** \brief Solve LU out = in; if level-scheduling algorithm is provided then the graph
        * traversing is performed in parallel */
        virtual bool LUSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

        /** \brief Analyse the structure (level-scheduling) */
        virtual void LLAnalyse(void);
        /** \brief Delete the analysed data (see LLAnalyse) */
        virtual void LLAnalyseClear(void);
        /** \brief Solve LL^T out = in; if level-scheduling algorithm is provided then the graph */
        // traversing is performed in parallel
        virtual bool LLSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
        virtual bool LLSolve(const BaseVector<ValueType>& in,
                             const BaseVector<ValueType>& inv_diag,
                             BaseVector<ValueType>*       out) const;

        /** \brief Analyse the structure (level-scheduling) L-part
        * diag_unit == true the diag is 1;
        * diag_unit == false the diag is 0; */
        virtual void LAnalyse(bool diag_unit = false);
        /** \brief Delete the analysed data (see LAnalyse) L-party */
        virtual void LAnalyseClear(void);
        /** \brief Solve L out = in; if level-scheduling algorithm is provided then the
        * graph traversing is performed in parallel */
        virtual bool LSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

        /** \brief Analyse the structure (level-scheduling) U-part;
        * diag_unit == true the diag is 1;
        * diag_unit == false the diag is 0; */
        virtual void UAnalyse(bool diag_unit = false);
        /** \brief Delete the analysed data (see UAnalyse) U-party */
        virtual void UAnalyseClear(void);
        /** \brief Solve U out = in; if level-scheduling algorithm is provided then the
        * graph traversing is performed in parallel */
        virtual bool USolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

        /// Analyse the structure for Iterative solve
        virtual void ItLUAnalyse(void);
        /// Delete the analysed data (see ItLUAnalyse)
        virtual void ItLUAnalyseClear(void);
        /// Solve LU out = in iteratively using the Jacobi method.
        virtual bool ItLUSolve(int                          max_iter,
                               double                       tolerance,
                               bool                         use_tol,
                               const BaseVector<ValueType>& in,
                               BaseVector<ValueType>*       out) const;

        /// Analyse the structure (level-scheduling)
        virtual void ItLLAnalyse(void);
        /// Delete the analysed data (see ItLLAnalyse)
        virtual void ItLLAnalyseClear(void);
        /// Solve LL^T out = in iteratively using the Jacobi method.
        virtual bool ItLLSolve(int                          max_iter,
                               double                       tolerance,
                               bool                         use_tol,
                               const BaseVector<ValueType>& in,
                               BaseVector<ValueType>*       out) const;
        virtual bool ItLLSolve(int                          max_iter,
                               double                       tolerance,
                               bool                         use_tol,
                               const BaseVector<ValueType>& in,
                               const BaseVector<ValueType>& inv_diag,
                               BaseVector<ValueType>*       out) const;

        /// Analyse the structure (level-scheduling) L-part
        /// diag_unit == true the diag is 1;
        /// diag_unit == false the diag is 0;
        virtual void ItLAnalyse(bool diag_unit = false);
        /// Delete the analysed data (see ItLAnalyse) L-party
        virtual void ItLAnalyseClear(void);
        /// Solve L out = in iteratively using the Jacobi method.
        virtual bool ItLSolve(int                          max_iter,
                              double                       tolerance,
                              bool                         use_tol,
                              const BaseVector<ValueType>& in,
                              BaseVector<ValueType>*       out) const;

        /// Analyse the structure (level-scheduling) U-part;
        /// diag_unit == true the diag is 1;
        /// diag_unit == false the diag is 0;
        virtual void ItUAnalyse(bool diag_unit = false);
        /// Delete the analysed data (see ItUAnalyse) U-party
        virtual void ItUAnalyseClear(void);
        /// Solve U out = in iteratively using the Jacobi method.
        virtual bool ItUSolve(int                          max_iter,
                              double                       tolerance,
                              bool                         use_tol,
                              const BaseVector<ValueType>& in,
                              BaseVector<ValueType>*       out) const;

        /** \brief Compute Householder vector */
        virtual bool Householder(int idx, ValueType& beta, BaseVector<ValueType>* vec) const;
        /** \brief QR Decomposition */
        virtual bool QRDecompose(void);
        /** \brief Solve QR out = in */
        virtual bool QRSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

        /** \brief Invert this */
        virtual bool Invert(void);

        /** \brief Compute the spectrum approximation with Gershgorin circles theorem */
        virtual bool Gershgorin(ValueType& lambda_min, ValueType& lambda_max) const;

        /** \brief Apply the matrix to vector, out = this*in; */
        virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const = 0;
        /** \brief Apply and add the matrix to vector, out = out + scalar*this*in; */
        virtual void ApplyAdd(const BaseVector<ValueType>& in,
                              ValueType                    scalar,
                              BaseVector<ValueType>*       out) const = 0;

        /** \brief Delete all entries abs(a_ij) <= drop_off;
        * the diagonal elements are never deleted */
        virtual bool Compress(double drop_off);

        /** \brief Transpose the matrix */
        virtual bool Transpose(void);
        /** \brief Transpose the matrix */
        virtual bool Transpose(BaseMatrix<ValueType>* T) const;

        /** \brief Sort the matrix indices */
        virtual bool Sort(void);

        /** \brief Return key for row, col and val */
        virtual bool Key(long int& row_key, long int& col_key, long int& val_key) const;

        /** \brief Replace a column vector of a matrix */
        virtual bool ReplaceColumnVector(int idx, const BaseVector<ValueType>& vec);

        /** \brief Replace a column vector of a matrix */
        virtual bool ReplaceRowVector(int idx, const BaseVector<ValueType>& vec);

        /** \brief Extract values from a column of a matrix to a vector */
        virtual bool ExtractColumnVector(int idx, BaseVector<ValueType>* vec) const;

        /** \brief Extract values from a row of a matrix to a vector */
        virtual bool ExtractRowVector(int idx, BaseVector<ValueType>* vec) const;

        /** \brief Extract number of boundary row nnz */
        virtual bool ExtractBoundaryRowNnz(BaseVector<PtrType>*         row_nnz,
                                           const BaseVector<int>&       boundary_index,
                                           const BaseMatrix<ValueType>& gst) const;

        /** \brief Extract boundary rows */
        virtual bool ExtractBoundaryRows(const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                         BaseVector<int64_t>*         bnd_csr_col_ind,
                                         BaseVector<ValueType>*       bnd_csr_val,
                                         int64_t                      global_column_offset,
                                         const BaseVector<int>&       boundary_index,
                                         const BaseVector<int64_t>&   ghost_mapping,
                                         const BaseMatrix<ValueType>& gst) const;

        /** \brief Extract global column indices */
        virtual bool ExtractGlobalColumnIndices(int                        ncol,
                                                int64_t                    global_offset,
                                                const BaseVector<int64_t>& l2g,
                                                BaseVector<int64_t>*       global_col) const;

        /** \brief Extract non zeros of matrix extension */
        virtual bool ExtractExtRowNnz(int offset, BaseVector<PtrType>* row_nnz) const;
        virtual bool MergeToLocal(const BaseMatrix<ValueType>& mat_int,
                                  const BaseMatrix<ValueType>& mat_gst,
                                  const BaseMatrix<ValueType>& mat_ext,
                                  const BaseVector<int>&       vec_ext);

        virtual bool AMGConnect(ValueType eps, BaseVector<int>* connections) const;
        virtual bool AMGAggregate(const BaseVector<int>& connections,
                                  BaseVector<int>*       aggregates) const;
        virtual bool AMGPMISAggregate(const BaseVector<int>& connections,
                                      BaseVector<int>*       aggregates) const;
        virtual bool AMGSmoothedAggregation(ValueType              relax,
                                            const BaseVector<int>& aggregates,
                                            const BaseVector<int>& connections,
                                            BaseMatrix<ValueType>* prolong,
                                            int                    lumping_strat) const;
        virtual bool AMGAggregation(const BaseVector<int>& aggregates,
                                    BaseMatrix<ValueType>* prolong) const;

        virtual bool AMGGreedyAggregate(const BaseVector<bool>& connections,
                                        BaseVector<int64_t>*    aggregates,
                                        BaseVector<int64_t>*    aggregate_root_nodes) const;

        /// Parallel maximal independent set aggregation
        virtual bool AMGBoundaryNnz(const BaseVector<int>&       boundary,
                                    const BaseVector<bool>&      connections,
                                    const BaseMatrix<ValueType>& ghost,
                                    BaseVector<PtrType>*         row_nnz) const;
        virtual bool AMGExtractBoundary(int64_t                      global_column_begin,
                                        const BaseVector<int>&       boundary,
                                        const BaseVector<int64_t>&   l2g,
                                        const BaseVector<bool>&      connections,
                                        const BaseMatrix<ValueType>& ghost,
                                        const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                        BaseVector<int64_t>*         bnd_csr_col_ind) const;
        virtual bool AMGComputeStrongConnections(ValueType                    eps,
                                                 const BaseVector<ValueType>& diag,
                                                 const BaseVector<int64_t>&   l2g,
                                                 BaseVector<bool>*            connections,
                                                 const BaseMatrix<ValueType>& ghost) const;

        virtual bool AMGPMISInitializeState(int64_t                      global_column_begin,
                                            const BaseVector<bool>&      connections,
                                            BaseVector<int>*             state,
                                            BaseVector<int>*             hash,
                                            const BaseMatrix<ValueType>& ghost) const;
        virtual bool AMGExtractBoundaryState(const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                             const BaseVector<bool>&      connections,
                                             const BaseVector<int>&       max_state,
                                             const BaseVector<int>&       hash,
                                             BaseVector<int>*             bnd_max_state,
                                             BaseVector<int>*             bnd_hash,
                                             int64_t                      global_column_offset,
                                             const BaseVector<int>&       boundary_index,
                                             const BaseMatrix<ValueType>& gst) const;
        virtual bool AMGPMISFindMaxNeighbourNode(int64_t                      global_column_begin,
                                                 int64_t                      global_column_end,
                                                 bool&                        undecided,
                                                 const BaseVector<bool>&      connections,
                                                 const BaseVector<int>&       state,
                                                 const BaseVector<int>&       hash,
                                                 const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                                 const BaseVector<int64_t>&   bnd_csr_col_ind,
                                                 const BaseVector<int>&       bnd_state,
                                                 const BaseVector<int>&       bnd_hash,
                                                 BaseVector<int>*             max_state,
                                                 BaseVector<int64_t>*         aggregates,
                                                 const BaseMatrix<ValueType>& ghost) const;
        virtual bool
                     AMGPMISAddUnassignedNodesToAggregations(int64_t                    global_column_begin,
                                                             const BaseVector<bool>&    connections,
                                                             const BaseVector<int>&     state,
                                                             const BaseVector<int64_t>& l2g,
                                                             BaseVector<int>*           max_state,
                                                             BaseVector<int64_t>*       aggregates,
                                                             BaseVector<int64_t>*       aggregate_root_nodes,
                                                             const BaseMatrix<ValueType>& ghost) const;
        virtual bool AMGPMISInitializeAggregateGlobalIndices(
            int64_t                    global_column_begin,
            const BaseVector<int64_t>& aggregates,
            BaseVector<int64_t>*       aggregate_root_nodes) const;
        /// Smoothed aggregation
        virtual bool AMGSmoothedAggregation(ValueType                  relax,
                                            const BaseVector<bool>&    connections,
                                            const BaseVector<int64_t>& aggregates,
                                            BaseMatrix<ValueType>*     prolong,
                                            int                        lumping_strat) const;
        /// Unsmoothed aggregation
        virtual bool AMGUnsmoothedAggregation(const BaseVector<int64_t>& aggregates,
                                              BaseMatrix<ValueType>*     prolong) const;

        virtual bool
            AMGSmoothedAggregationProlongNnz(int64_t                      global_column_begin,
                                             int64_t                      global_column_end,
                                             const BaseVector<bool>&      connections,
                                             const BaseVector<int64_t>&   aggregates,
                                             const BaseVector<int64_t>&   aggregate_root_nodes,
                                             const BaseMatrix<ValueType>& ghost,
                                             BaseVector<int>*             f2c,
                                             BaseMatrix<ValueType>*       prolong_int,
                                             BaseMatrix<ValueType>*       prolong_gst) const;

        virtual bool
            AMGSmoothedAggregationProlongFill(int64_t                      global_column_begin,
                                              int64_t                      global_column_end,
                                              int                          lumping_strat,
                                              ValueType                    relax,
                                              const BaseVector<bool>&      connections,
                                              const BaseVector<int64_t>&   aggregates,
                                              const BaseVector<int64_t>&   aggregate_root_nodes,
                                              const BaseVector<int64_t>&   l2g,
                                              const BaseVector<int>&       f2c,
                                              const BaseMatrix<ValueType>& ghost,
                                              BaseMatrix<ValueType>*       prolong_int,
                                              BaseMatrix<ValueType>*       prolong_gst,
                                              BaseVector<int64_t>*         global_ghost_col) const;

        virtual bool
            AMGUnsmoothedAggregationProlongNnz(int64_t                      global_column_begin,
                                               int64_t                      global_column_end,
                                               const BaseVector<int64_t>&   aggregates,
                                               const BaseVector<int64_t>&   aggregate_root_nodes,
                                               const BaseMatrix<ValueType>& ghost,
                                               BaseVector<int>*             f2c,
                                               BaseMatrix<ValueType>*       prolong_int,
                                               BaseMatrix<ValueType>*       prolong_gst) const;

        virtual bool
            AMGUnsmoothedAggregationProlongFill(int64_t                      global_column_begin,
                                                int64_t                      global_column_end,
                                                const BaseVector<int64_t>&   aggregates,
                                                const BaseVector<int64_t>&   aggregate_root_nodes,
                                                const BaseVector<int>&       f2c,
                                                const BaseMatrix<ValueType>& ghost,
                                                BaseMatrix<ValueType>*       prolong_int,
                                                BaseMatrix<ValueType>*       prolong_gst,
                                                BaseVector<int64_t>* global_ghost_col) const;

        /** \brief Ruge Stueben coarsening */
        virtual bool RSCoarsening(float eps, BaseVector<int>* CFmap, BaseVector<bool>* S) const;

        /** \brief Parallel maximal independent set coarsening */
        virtual bool RSPMISStrongInfluences(float                        eps,
                                            BaseVector<bool>*            S,
                                            BaseVector<float>*           omega,
                                            int64_t                      global_row_offset,
                                            const BaseMatrix<ValueType>& ghost) const;
        virtual bool RSPMISUnassignedToCoarse(BaseVector<int>*         CFmap,
                                              BaseVector<bool>*        marked,
                                              const BaseVector<float>& omega) const;
        virtual bool RSPMISCorrectCoarse(BaseVector<int>*             CFmap,
                                         const BaseVector<bool>&      S,
                                         const BaseVector<bool>&      marked,
                                         const BaseVector<float>&     omega,
                                         const BaseMatrix<ValueType>& ghost) const;
        virtual bool RSPMISCoarseEdgesToFine(BaseVector<int>*             CFmap,
                                             const BaseVector<bool>&      S,
                                             const BaseMatrix<ValueType>& ghost) const;
        virtual bool RSPMISCheckUndecided(bool& undecided, const BaseVector<int>& CFmap) const;

        /** \brief Ruge Stueben Direct Interpolation */
        virtual bool RSDirectProlongNnz(const BaseVector<int>&       CFmap,
                                        const BaseVector<bool>&      S,
                                        const BaseMatrix<ValueType>& ghost,
                                        BaseVector<ValueType>*       Amin,
                                        BaseVector<ValueType>*       Amax,
                                        BaseVector<int>*             f2c,
                                        BaseMatrix<ValueType>*       prolong_int,
                                        BaseMatrix<ValueType>*       prolong_gst) const;
        virtual bool RSDirectProlongFill(const BaseVector<int64_t>&   l2g,
                                         const BaseVector<int>&       f2c,
                                         const BaseVector<int>&       CFmap,
                                         const BaseVector<bool>&      S,
                                         const BaseMatrix<ValueType>& ghost,
                                         const BaseVector<ValueType>& Amin,
                                         const BaseVector<ValueType>& Amax,
                                         BaseMatrix<ValueType>*       prolong_int,
                                         BaseMatrix<ValueType>*       prolong_gst,
                                         BaseVector<int64_t>*         global_ghost_col) const;

        /** \brief Ruge Stueben Ext+i Interpolation */
        virtual bool RSExtPIBoundaryNnz(const BaseVector<int>&       boundary,
                                        const BaseVector<int>&       CFmap,
                                        const BaseVector<bool>&      S,
                                        const BaseMatrix<ValueType>& ghost,
                                        BaseVector<PtrType>*         row_nnz) const;
        virtual bool RSExtPIExtractBoundary(int64_t                      global_column_begin,
                                            const BaseVector<int>&       boundary,
                                            const BaseVector<int64_t>&   l2g,
                                            const BaseVector<int>&       CFmap,
                                            const BaseVector<bool>&      S,
                                            const BaseMatrix<ValueType>& ghost,
                                            const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                            BaseVector<int64_t>*         bnd_csr_col_ind) const;
        virtual bool RSExtPIProlongNnz(int64_t                      global_column_begin,
                                       int64_t                      global_column_end,
                                       bool                         FF1,
                                       const BaseVector<int64_t>&   l2g,
                                       const BaseVector<int>&       CFmap,
                                       const BaseVector<bool>&      S,
                                       const BaseMatrix<ValueType>& ghost,
                                       const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                       const BaseVector<int64_t>&   bnd_csr_col_ind,
                                       BaseVector<int>*             f2c,
                                       BaseMatrix<ValueType>*       prolong_int,
                                       BaseMatrix<ValueType>*       prolong_gst) const;
        virtual bool RSExtPIProlongFill(int64_t                      global_column_begin,
                                        int64_t                      global_column_end,
                                        bool                         FF1,
                                        const BaseVector<int64_t>&   l2g,
                                        const BaseVector<int>&       f2c,
                                        const BaseVector<int>&       CFmap,
                                        const BaseVector<bool>&      S,
                                        const BaseMatrix<ValueType>& ghost,
                                        const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                        const BaseVector<int64_t>&   bnd_csr_col_ind,
                                        const BaseVector<PtrType>&   ext_csr_row_ptr,
                                        const BaseVector<int64_t>&   ext_csr_col_ind,
                                        const BaseVector<ValueType>& ext_csr_val,
                                        BaseMatrix<ValueType>*       prolong_int,
                                        BaseMatrix<ValueType>*       prolong_gst,
                                        BaseVector<int64_t>*         global_ghost_col) const;

        /** \brief Factorized Sparse Approximate Inverse assembly for given system
        * matrix power pattern or external sparsity pattern */
        virtual bool FSAI(int power, const BaseMatrix<ValueType>* pattern);

        /** \brief SParse Approximate Inverse assembly for given system matrix pattern */
        virtual bool SPAI(void);

        /** \brief Initial Pairwise Aggregation scheme */
        virtual bool InitialPairwiseAggregation(ValueType        beta,
                                                int&             nc,
                                                BaseVector<int>* G,
                                                int&             Gsize,
                                                int**            rG,
                                                int&             rGsize,
                                                int              ordering) const;
        /** \brief Initial Pairwise Aggregation scheme for split matrices */
        virtual bool InitialPairwiseAggregation(const BaseMatrix<ValueType>& mat,
                                                ValueType                    beta,
                                                int&                         nc,
                                                BaseVector<int>*             G,
                                                int&                         Gsize,
                                                int**                        rG,
                                                int&                         rGsize,
                                                int                          ordering) const;
        /** \brief Further Pairwise Aggregation scheme */
        virtual bool FurtherPairwiseAggregation(ValueType        beta,
                                                int&             nc,
                                                BaseVector<int>* G,
                                                int&             Gsize,
                                                int**            rG,
                                                int&             rGsize,
                                                int              ordering) const;
        /** \brief Further Pairwise Aggregation scheme for split matrices */
        virtual bool FurtherPairwiseAggregation(const BaseMatrix<ValueType>& mat,
                                                ValueType                    beta,
                                                int&                         nc,
                                                BaseVector<int>*             G,
                                                int&                         Gsize,
                                                int**                        rG,
                                                int&                         rGsize,
                                                int                          ordering) const;
        /** \brief Build coarse operator for pairwise aggregation scheme */
        virtual bool CoarsenOperator(BaseMatrix<ValueType>* Ac,
                                     int                    nrow,
                                     int                    ncol,
                                     const BaseVector<int>& G,
                                     int                    Gsize,
                                     const int*             rG,
                                     int                    rGsize) const;

        virtual bool CombineAndRenumber(int                        ncol,
                                        int64_t                    ext_nnz,
                                        int64_t                    col_begin,
                                        int64_t                    col_end,
                                        const BaseVector<int64_t>& l2g,
                                        const BaseVector<int64_t>& ext,
                                        BaseVector<int>*           merged,
                                        BaseVector<int64_t>*       mapping,
                                        BaseVector<int>*           local_col) const;

        virtual bool SplitInteriorGhost(BaseMatrix<ValueType>* interior,
                                        BaseMatrix<ValueType>* ghost) const;

        /** \brief Create ghost columns from global ids */
        virtual bool CopyGhostFromGlobalReceive(const BaseVector<int>&       boundary,
                                                const BaseVector<PtrType>&   recv_csr_row_ptr,
                                                const BaseVector<int64_t>&   recv_csr_col_ind,
                                                const BaseVector<ValueType>& recv_csr_val,
                                                BaseVector<int64_t>*         global_col);

        virtual bool CopyFromGlobalReceive(int                          nrow,
                                           int64_t                      global_column_begin,
                                           int64_t                      global_column_end,
                                           const BaseVector<int>&       boundary,
                                           const BaseVector<PtrType>&   recv_csr_row_ptr,
                                           const BaseVector<int64_t>&   recv_csr_col_ind,
                                           const BaseVector<ValueType>& recv_csr_val,
                                           BaseMatrix<ValueType>*       ghost,
                                           BaseVector<int64_t>*         global_col);

        /** \brief Renumber global indices to local indices */
        virtual bool RenumberGlobalToLocal(const BaseVector<int64_t>& column_indices);
        virtual bool CompressAdd(const BaseVector<int64_t>&   l2g,
                                 const BaseVector<int64_t>&   global_ghost_col,
                                 const BaseMatrix<ValueType>& ext,
                                 BaseVector<int64_t>*         global_col);

    protected:
        /** \brief Number of rows */
        int nrow_;
        /** \brief Number of columns */
        int ncol_;
        /** \brief Number of non-zero elements */
        int64_t nnz_;

        /** \brief Backend descriptor (local copy) */
        Rocalution_Backend_Descriptor local_backend_;

        friend class BaseVector<ValueType>;
        friend class HostVector<ValueType>;
        friend class AcceleratorVector<ValueType>;
        friend class HIPAcceleratorVector<ValueType>;
    };

    template <typename ValueType>
    class HostMatrix : public BaseMatrix<ValueType>
    {
    public:
        HostMatrix();
        virtual ~HostMatrix();
    };

    template <typename ValueType>
    class AcceleratorMatrix : public BaseMatrix<ValueType>
    {
    public:
        AcceleratorMatrix();
        virtual ~AcceleratorMatrix();

        /** \brief Copy (accelerator matrix) from host matrix */
        virtual void CopyFromHost(const HostMatrix<ValueType>& src) = 0;

        /** \brief Async copy (accelerator matrix) from host matrix */
        virtual void CopyFromHostAsync(const HostMatrix<ValueType>& src);

        /** \brief Copy (accelerator matrix) to host matrix */
        virtual void CopyToHost(HostMatrix<ValueType>* dst) const = 0;

        /** \brief Async opy (accelerator matrix) to host matrix */
        virtual void CopyToHostAsync(HostMatrix<ValueType>* dst) const;
    };

    template <typename ValueType>
    class HIPAcceleratorMatrix : public AcceleratorMatrix<ValueType>
    {
    public:
        HIPAcceleratorMatrix();
        virtual ~HIPAcceleratorMatrix();
    };

} // namespace rocalution

#endif // ROCALUTION_BASE_MATRIX_HPP_
