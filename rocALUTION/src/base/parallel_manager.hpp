/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_PARALLEL_MANAGER_HPP_
#define ROCALUTION_PARALLEL_MANAGER_HPP_

#include "base_rocalution.hpp"
#include "rocalution/export.hpp"

#include <complex>
#include <string>

namespace rocalution
{
    template <typename ValueType>
    class GlobalMatrix;
    template <typename ValueType>
    class GlobalVector;
    struct MRequest;

    /** \ingroup backend_module
  * \brief Parallel Manager class
  * \details
  * The parallel manager class handles the communication and the mapping of the global
  * operators. Each global operator and vector need to be initialized with a valid
  * parallel manager in order to perform any operation. For many distributed simulations,
  * the underlying operator is already distributed. This information need to be passed to
  * the parallel manager.
  */
    class ParallelManager : public RocalutionObj
    {
    public:
        ROCALUTION_EXPORT
        ParallelManager();
        ROCALUTION_EXPORT
        ~ParallelManager();

        /** \brief Set the MPI communicator */
        ROCALUTION_EXPORT
        void SetMPICommunicator(const void* comm);
        /** \brief Clear all allocated resources */
        ROCALUTION_EXPORT
        void Clear(void);

        /** \brief Return communicator */
        const void* GetComm(void) const
        {
            return this->comm_;
        }

        /** \brief Return rank */
        int GetRank(void) const
        {
            return this->rank_;
        }

        /** \brief Return the global number of rows */
        ROCALUTION_EXPORT
        int64_t GetGlobalNrow(void) const;
        /** \brief Return the global number of columns */
        ROCALUTION_EXPORT
        int64_t GetGlobalNcol(void) const;
        /** \brief Return the local number of rows */
        ROCALUTION_EXPORT
        int64_t GetLocalNrow(void) const;
        /** \brief Return the local number of columns */
        ROCALUTION_EXPORT
        int64_t GetLocalNcol(void) const;

        /** \brief Return the number of receivers */
        ROCALUTION_EXPORT
        int GetNumReceivers(void) const;
        /** \brief Return the number of senders */
        ROCALUTION_EXPORT
        int GetNumSenders(void) const;
        /** \brief Return the number of involved processes */
        ROCALUTION_EXPORT
        int GetNumProcs(void) const;
        /** \brief Return the global row begin */
        ROCALUTION_EXPORT
        int64_t GetGlobalRowBegin(int rank = -1) const;
        /** \brief Return the global row end */
        ROCALUTION_EXPORT
        int64_t GetGlobalRowEnd(int rank = -1) const;
        /** \brief Return the global column begin */
        ROCALUTION_EXPORT
        int64_t GetGlobalColumnBegin(int rank = -1) const;
        /** \brief Return the global column end */
        ROCALUTION_EXPORT
        int64_t GetGlobalColumnEnd(int rank = -1) const;

        /** \brief Initialize the global number of rows */
        ROCALUTION_EXPORT
        void SetGlobalNrow(int64_t nrow);
        /** \brief Initialize the global number of columns */
        ROCALUTION_EXPORT
        void SetGlobalNcol(int64_t ncol);
        /** \brief Initialize the local number of rows */
        ROCALUTION_EXPORT
        void SetLocalNrow(int64_t nrow);
        /** \brief Initialize the local number of columns */
        ROCALUTION_EXPORT
        void SetLocalNcol(int64_t ncol);

        /** \brief Set all boundary indices of this ranks process */
        ROCALUTION_EXPORT
        void SetBoundaryIndex(int size, const int* index);
        /** \brief Get all boundary indices of this ranks process */
        ROCALUTION_EXPORT
        const int* GetBoundaryIndex(void) const;

        /** \brief Get ghost to global mapping for this rank */
        ROCALUTION_EXPORT
        const int64_t* GetGhostToGlobalMap(void) const;

        /** \brief Number of processes, the current process is receiving data from, array of
      * the processes, the current process is receiving data from and offsets, where the
      * boundary for process 'receiver' starts
      */
        ROCALUTION_EXPORT
        void SetReceivers(int nrecv, const int* recvs, const int* recv_offset);

        /** \brief Number of processes, the current process is sending data to, array of the
      * processes, the current process is sending data to and offsets where the ghost
      * part for process 'sender' starts
      */
        ROCALUTION_EXPORT
        void SetSenders(int nsend, const int* sends, const int* send_offset);

        /** \brief Mapping local to global */
        ROCALUTION_EXPORT
        void LocalToGlobal(int proc, int local, int& global);
        /** \brief Mapping global to local */
        ROCALUTION_EXPORT
        void GlobalToLocal(int global, int& proc, int& local);

        /** \brief Check sanity status of parallel manager */
        ROCALUTION_EXPORT
        bool Status(void) const;

        /** \brief Read file that contains all relevant parallel manager data */
        ROCALUTION_EXPORT
        void ReadFileASCII(const std::string& filename);
        /** \brief Write file that contains all relevant parallel manager data */
        ROCALUTION_EXPORT
        void WriteFileASCII(const std::string& filename) const;

    protected:
        /** \brief Communicate boundary data (async) */
        template <typename ValueType>
        void CommunicateAsync_(ValueType* send_buffer, ValueType* recv_buffer) const;
        /** \brief Synchronize communication */
        void CommunicateSync_(void) const;

        /** \brief Back-communicate boundary data (async) */
        template <typename ValueType>
        void InverseCommunicateAsync_(ValueType* send_buffer, ValueType* recv_buffer) const;
        /** \brief Synchronize communication */
        void InverseCommunicateSync_(void) const;

        /** \brief Communicate CSR matrix data (async) */
        template <typename I, typename J, typename T>
        void CommunicateCSRAsync_(I* send_row_ptr,
                                  J* send_col_ind,
                                  T* send_val,
                                  I* recv_row_ptr,
                                  J* recv_col_ind,
                                  T* recv_val) const;
        /** \brief Synchronize communication */
        void CommunicateCSRSync_(void) const;

        /** \brief Back-communicate CSR matrix data (async) */
        template <typename I, typename J, typename T>
        void InverseCommunicateCSRAsync_(I* send_row_ptr,
                                         J* send_col_ind,
                                         T* send_val,
                                         I* recv_row_ptr,
                                         J* recv_col_ind,
                                         T* recv_val) const;
        /** \brief Synchronize communication */
        void InverseCommunicateCSRSync_(void) const;

        /** \brief Generate parallel manager from ghost columns, mapping and parent PM */
        void GenerateFromGhostColumnsWithParent_(int64_t                nnz,
                                                 const int64_t*         ghost_col,
                                                 const ParallelManager& parent,
                                                 bool                   transposed = false);

        /** \brief Transform global to local ids */
        void BoundaryTransformGlobalToLocal_(void);
        /** \brief Transform global fine points to local coarse points */
        void BoundaryTransformGlobalFineToLocalCoarse_(const int* f2c);

    private:
        // Synchronize all events within this PM
        void Synchronize_(void) const;

        // Communicate global row and column offsets (async)
        void CommunicateGlobalOffsetAsync_(void) const;
        // Synchronize communication
        void CommunicateGlobalOffsetSync_(void) const;

        // Communicate ghost to global mapping (async)
        void CommunicateGhostToGlobalMapAsync_(void) const;
        // Synchronize communication
        void CommunicateGhostToGlobalMapSync_(void) const;

        // Communicator
        const void* comm_;

        // Current process rank
        int rank_;

        // Total number of processes
        int num_procs_;

        // Global sizes
        int64_t global_nrow_;
        int64_t global_ncol_;

        // Local sizes
        int64_t local_nrow_;
        int64_t local_ncol_;

        // Number of total ids, the current process is receiving
        int recv_index_size_;
        // Number of total ids, the current process is sending
        int send_index_size_;

        // Number of processes, the current process receives data from
        int nrecv_;
        // Number of processes, the current process sends data to
        int nsend_;

        // Array of process ids, the current process receives data from
        int* recvs_;
        // Array of process ids, the current process sends data to
        int* sends_;

        // Array of offsets, the current process receives data from
        int* recv_offset_index_;
        // Array of offsets, the current process sends data to
        int* send_offset_index_;

        // Boundary index ids
        int*     boundary_index_;
        int64_t* boundary_buffer_;

        // Flag whether global offsets are available
        mutable bool global_offset_;
        // Global row offsets
        int64_t* global_row_offset_;
        // Global column offsets
        int64_t* global_col_offset_;

        // Flag whether ghost to global mapping is available
        mutable bool ghost_to_global_map_;
        // Ghost to global id mapping
        int64_t* ghost_mapping_;

        // Track ongoing asynchronous communication
        mutable int async_send_;
        mutable int async_recv_;

        // Send/receive events for asynchronous communication
        MRequest* recv_event_;
        MRequest* send_event_;

        friend class GlobalMatrix<double>;
        friend class GlobalMatrix<float>;
        friend class GlobalMatrix<std::complex<double>>;
        friend class GlobalMatrix<std::complex<float>>;
        friend class GlobalVector<double>;
        friend class GlobalVector<float>;
        friend class GlobalVector<std::complex<double>>;
        friend class GlobalVector<std::complex<float>>;
        friend class GlobalVector<int>;
    };

} // namespace rocalution

#endif // ROCALUTION_PARALLEL_MANAGER_HPP_
