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

#ifndef ROCALUTION_GLOBAL_MATRIX_HPP_
#define ROCALUTION_GLOBAL_MATRIX_HPP_

#include "local_vector.hpp"
#include "operator.hpp"
#include "parallel_manager.hpp"
#include "rocalution/utils/types.hpp"

namespace rocalution
{

    template <typename ValueType>
    class GlobalVector;
    template <typename ValueType>
    class LocalMatrix;

    /** \ingroup op_vec_module
  * \class GlobalMatrix
  * \brief GlobalMatrix class
  * \details
  * A GlobalMatrix is called global, because it can stay on a single or on multiple nodes
  * in a network. For this type of communication, MPI is used.
  *
  * \tparam ValueType - can be int, float, double, std::complex<float> and
  *                     std::complex<double>
  *
  * A number of matrix formats are supported. These are CSR, BCSR, MCSR, COO, DIA, ELL, HYB, and DENSE.
  * \note For CSR type matrices, the column indices must be sorted in increasing order. For COO matrices, the row
  * indices must be sorted in increasing order. The function \p Check can be used to check whether a matrix
  * contains valid data. For CSR and COO matrices, the function \p Sort can be used to sort the row or column
  * indices respectively.
  */
    template <typename ValueType>
    class GlobalMatrix : public Operator<ValueType>
    {
    public:
        GlobalMatrix();
        /** \brief Initialize a global matrix with a parallel manager */
        explicit GlobalMatrix(const ParallelManager& pm);
        virtual ~GlobalMatrix();

        /** \brief Return the number of rows in the global matrix. */
        virtual int64_t GetM(void) const;
        /** \brief Return the number of columns in the global matrix. */
        virtual int64_t GetN(void) const;
        /** \brief Return the number of non-zeros in the global matrix. */
        virtual int64_t GetNnz(void) const;
        /** \brief Return the number of rows in the interior matrix. */
        virtual int64_t GetLocalM(void) const;
        /** \brief Return the number of columns in the interior matrix. */
        virtual int64_t GetLocalN(void) const;
        /** \brief Return the number of non-zeros in the interior matrix. */
        virtual int64_t GetLocalNnz(void) const;
        /** \brief Return the number of rows in the ghost matrix. */
        virtual int64_t GetGhostM(void) const;
        /** \brief Return the number of columns in the ghost matrix. */
        virtual int64_t GetGhostN(void) const;
        /** \brief Return the number of non-zeros in the ghost matrix. */
        virtual int64_t GetGhostNnz(void) const;

        /** \brief Return the global matrix format id (see matrix_formats.hpp) */
        unsigned int GetFormat(void) const;

        /** \private */
        const LocalMatrix<ValueType>& GetInterior() const;
        /** \private */
        const LocalMatrix<ValueType>& GetGhost() const;

        /** \brief Move all data (i.e. move the part of the global matrix stored on this rank) to the accelerator */
        virtual void MoveToAccelerator(void);
        /** \brief Move all data (i.e. move the part of the global matrix stored on this rank) to the host */
        virtual void MoveToHost(void);
        /** \brief Shows simple info about the matrix. */
        virtual void Info(void) const;

        /** \brief Perform a sanity check of the matrix
        * \details
        * Checks, if the matrix contains valid data, i.e. if the values are not infinity
        * and not NaN (not a number) and if the structure of the matrix is correct (e.g.
        * indices cannot be negative, CSR and COO matrices have to be sorted, etc.).
        *
        * \retval true if the matrix is ok (empty matrix is also ok).
        * \retval false if there is something wrong with the structure or values.
        */
        virtual bool Check(void) const;

        /** \brief Allocate CSR Matrix */
        void AllocateCSR(const std::string& name, int64_t local_nnz, int64_t ghost_nnz);
        /** \brief Allocate COO Matrix */
        void AllocateCOO(const std::string& name, int64_t local_nnz, int64_t ghost_nnz);

        /** \brief Clear (free) the matrix */
        virtual void Clear(void);

        /** \brief Set the parallel manager of a global matrix */
        void SetParallelManager(const ParallelManager& pm);

        /** \brief Initialize a CSR matrix on the host with externally allocated data */
        void SetDataPtrCSR(PtrType**   local_row_offset,
                           int**       local_col,
                           ValueType** local_val,
                           PtrType**   ghost_row_offset,
                           int**       ghost_col,
                           ValueType** ghost_val,
                           std::string name,
                           int64_t     local_nnz,
                           int64_t     ghost_nnz);
        /** \brief Initialize a COO matrix on the host with externally allocated data */
        void SetDataPtrCOO(int**       local_row,
                           int**       local_col,
                           ValueType** local_val,
                           int**       ghost_row,
                           int**       ghost_col,
                           ValueType** ghost_val,
                           std::string name,
                           int64_t     local_nnz,
                           int64_t     ghost_nnz);

        /** \brief Initialize a CSR matrix on the host with externally allocated local data */
        void SetLocalDataPtrCSR(
            PtrType** row_offset, int** col, ValueType** val, std::string name, int64_t nnz);
        /** \brief Initialize a COO matrix on the host with externally allocated local data */
        void SetLocalDataPtrCOO(
            int** row, int** col, ValueType** val, std::string name, int64_t nnz);

        /** \brief Initialize a CSR matrix on the host with externally allocated ghost data */
        void SetGhostDataPtrCSR(
            PtrType** row_offset, int** col, ValueType** val, std::string name, int64_t nnz);
        /** \brief Initialize a COO matrix on the host with externally allocated ghost data */
        void SetGhostDataPtrCOO(
            int** row, int** col, ValueType** val, std::string name, int64_t nnz);

        /** \brief Leave a CSR matrix to host pointers */
        void LeaveDataPtrCSR(PtrType**   local_row_offset,
                             int**       local_col,
                             ValueType** local_val,
                             PtrType**   ghost_row_offset,
                             int**       ghost_col,
                             ValueType** ghost_val);
        /** \brief Leave a COO matrix to host pointers */
        void LeaveDataPtrCOO(int**       local_row,
                             int**       local_col,
                             ValueType** local_val,
                             int**       ghost_row,
                             int**       ghost_col,
                             ValueType** ghost_val);
        /** \brief Leave a local CSR matrix to host pointers */
        void LeaveLocalDataPtrCSR(PtrType** row_offset, int** col, ValueType** val);
        /** \brief Leave a local COO matrix to host pointers */
        void LeaveLocalDataPtrCOO(int** row, int** col, ValueType** val);
        /** \brief Leave a CSR ghost matrix to host pointers */
        void LeaveGhostDataPtrCSR(PtrType** row_offset, int** col, ValueType** val);
        /** \brief Leave a COO ghost matrix to host pointers */
        void LeaveGhostDataPtrCOO(int** row, int** col, ValueType** val);

        /** \brief Clone the entire matrix (values,structure+backend descr) from another
        * GlobalMatrix
        */
        void CloneFrom(const GlobalMatrix<ValueType>& src);
        /** \brief Copy matrix (values and structure) from another GlobalMatrix */
        void CopyFrom(const GlobalMatrix<ValueType>& src);

        /** \brief Convert the matrix to CSR structure */
        void ConvertToCSR(void);
        /** \brief Convert the matrix to MCSR structure */
        void ConvertToMCSR(void);
        /** \brief Convert the matrix to BCSR structure */
        void ConvertToBCSR(int blockdim);
        /** \brief Convert the matrix to COO structure */
        void ConvertToCOO(void);
        /** \brief Convert the matrix to ELL structure */
        void ConvertToELL(void);
        /** \brief Convert the matrix to DIA structure */
        void ConvertToDIA(void);
        /** \brief Convert the matrix to HYB structure */
        void ConvertToHYB(void);
        /** \brief Convert the matrix to DENSE structure */
        void ConvertToDENSE(void);
        /** \brief Convert the matrix to specified matrix ID format */
        void ConvertTo(unsigned int matrix_format, int blockdim = 1);

        /** \brief Perform matrix-vector multiplication, out = this * in; */
        virtual void Apply(const GlobalVector<ValueType>& in, GlobalVector<ValueType>* out) const;
        /** \brief Perform matrix-vector multiplication, out = scalar * this * in; */
        virtual void ApplyAdd(const GlobalVector<ValueType>& in,
                              ValueType                      scalar,
                              GlobalVector<ValueType>*       out) const;

        /** \brief Transpose the matrix */
        virtual void Transpose(void);

        /** \brief Transpose the matrix */
        void Transpose(GlobalMatrix<ValueType>* T) const;

        /** \brief Triple matrix product C=RAP */
        void TripleMatrixProduct(const GlobalMatrix<ValueType>& R,
                                 const GlobalMatrix<ValueType>& A,
                                 const GlobalMatrix<ValueType>& P);

        /** \brief Read matrix from MTX (Matrix Market Format) file */
        void ReadFileMTX(const std::string& filename);
        /** \brief Write matrix to MTX (Matrix Market Format) file */
        void WriteFileMTX(const std::string& filename) const;
        /** \brief Read matrix from CSR (ROCALUTION binary format) file */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#else
        [[deprecated("This function will be removed in a future release.")]]
#endif
        void ReadFileCSR(const std::string& filename);
        /** \brief Write matrix to CSR (ROCALUTION binary format) file */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#else
        [[deprecated("This function will be removed in a future release.")]]
#endif
        void WriteFileCSR(const std::string& filename) const;

        /** \brief Read a matrix from a binary file using rocsparse I/O format */
        ROCALUTION_EXPORT
        void ReadFileRSIO(const std::string& filename, bool maintain_initial_format = false);

        /** \brief Write a matrix to binary file using rocsparse I/O format */
        ROCALUTION_EXPORT
        void WriteFileRSIO(const std::string& filename) const;

        /** \brief Sort the matrix indices
        * \details
        * Sorts the matrix by indices.
        * - For CSR matrices, column values are sorted.
        * - For COO matrices, row indices are sorted.
        */
        void Sort(void);

        /** \brief Extract the diagonal values of the matrix into a GlobalVector */
        void ExtractDiagonal(GlobalVector<ValueType>* vec_diag) const;

        /** \brief Extract the inverse (reciprocal) diagonal values of the matrix into a
        * GlobalVector
        */
        void ExtractInverseDiagonal(GlobalVector<ValueType>* vec_inv_diag) const;

        /** \brief Scale all the values in the matrix */
        void Scale(ValueType alpha);

        /** \brief Initial Pairwise Aggregation scheme */
        void InitialPairwiseAggregation(ValueType         beta,
                                        int&              nc,
                                        LocalVector<int>* G,
                                        int&              Gsize,
                                        int**             rG,
                                        int&              rGsize,
                                        int               ordering) const;
        /** \brief Further Pairwise Aggregation scheme */
        void FurtherPairwiseAggregation(ValueType         beta,
                                        int&              nc,
                                        LocalVector<int>* G,
                                        int&              Gsize,
                                        int**             rG,
                                        int&              rGsize,
                                        int               ordering) const;
        /** \brief Build coarse operator for pairwise aggregation scheme */
        void CoarsenOperator(GlobalMatrix<ValueType>* Ac,
                             int                      nrow,
                             int                      ncol,
                             const LocalVector<int>&  G,
                             int                      Gsize,
                             const int*               rG,
                             int                      rGsize) const;

        /** \brief Create a restriction and prolongation matrix operator based on an int vector map */
        void CreateFromMap(const LocalVector<int>&  map,
                           int64_t                  n,
                           int64_t                  m,
                           GlobalMatrix<ValueType>* pro);

        /** \brief Plain aggregation - Modification of a greedy aggregation scheme from
        * Vanek (1996)
        */
        ROCALUTION_EXPORT
        void AMGGreedyAggregate(ValueType             eps,
                                LocalVector<bool>*    connections,
                                LocalVector<int64_t>* aggregates,
                                LocalVector<int64_t>* aggregate_root_nodes) const;

        /** \brief Parallel aggregation - Parallel maximal independent set aggregation scheme from
        * Bell, Dalton, & Olsen (2012)
        */
        ROCALUTION_EXPORT
        void AMGPMISAggregate(ValueType             eps,
                              LocalVector<bool>*    connections,
                              LocalVector<int64_t>* aggregates,
                              LocalVector<int64_t>* aggregate_root_nodes) const;

        /** \brief Interpolation scheme based on smoothed aggregation from Vanek (1996) */
        ROCALUTION_EXPORT
        void AMGSmoothedAggregation(ValueType                   relax,
                                    const LocalVector<bool>&    connections,
                                    const LocalVector<int64_t>& aggregates,
                                    const LocalVector<int64_t>& aggregate_root_nodes,
                                    GlobalMatrix<ValueType>*    prolong,
                                    int                         lumping_strat = 0) const;

        /** \brief Aggregation-based interpolation scheme */
        ROCALUTION_EXPORT
        void AMGUnsmoothedAggregation(const LocalVector<int64_t>& aggregates,
                                      const LocalVector<int64_t>& aggregate_root_nodes,
                                      GlobalMatrix<ValueType>*    prolong) const;

        /** \brief Ruge Stueben coarsening */
        void RSCoarsening(float eps, LocalVector<int>* CFmap, LocalVector<bool>* S) const;
        /** \brief Parallel maximal independent set coarsening for RS AMG*/
        void RSPMISCoarsening(float eps, LocalVector<int>* CFmap, LocalVector<bool>* S) const;

        /** \brief Ruge Stueben Direct Interpolation */
        void RSDirectInterpolation(const LocalVector<int>&  CFmap,
                                   const LocalVector<bool>& S,
                                   GlobalMatrix<ValueType>* prolong) const;
        /** \brief Ruge Stueben Ext+i Interpolation */
        void RSExtPIInterpolation(const LocalVector<int>&  CFmap,
                                  const LocalVector<bool>& S,
                                  bool                     FF1,
                                  GlobalMatrix<ValueType>* prolong) const;

    protected:
        /** \brief Return true if the object is on the host */
        virtual bool is_host_(void) const;
        /** \brief Return true if the object is on the accelerator */
        virtual bool is_accel_(void) const;

    private:
        void CreateParallelManager_(void);
        void InitCommPattern_(void);

        ParallelManager* pm_self_;

        ValueType* recv_boundary_;
        ValueType* send_boundary_;

        mutable LocalVector<ValueType> recv_buffer_;
        mutable LocalVector<ValueType> send_buffer_;

        LocalVector<int> halo_;

        int64_t nnz_;

        LocalMatrix<ValueType> matrix_interior_;
        LocalMatrix<ValueType> matrix_ghost_;

        friend class GlobalVector<ValueType>;
        friend class LocalMatrix<ValueType>;
        friend class LocalVector<ValueType>;
    };

} // namespace rocalution

#endif // ROCALUTION_GLOBAL_MATRIX_HPP_
