/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "parallel_manager.hpp"
#include "../utils/def.hpp"
#include "base_rocalution.hpp"
#include "rocalution/utils/types.hpp"

#include "../utils/allocate_free.hpp"
#include "../utils/log.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

#ifdef SUPPORT_MULTINODE
#include "../utils/communicator.hpp"
#include <mpi.h>
#endif

// LCOV_EXCL_START

namespace rocalution
{

    ParallelManager::ParallelManager()
    {
        this->comm_      = NULL;
        this->rank_      = -1;
        this->num_procs_ = -1;

        this->global_nrow_ = 0;
        this->global_ncol_ = 0;
        this->local_nrow_  = 0;
        this->local_ncol_  = 0;

        this->recv_index_size_ = 0;
        this->send_index_size_ = 0;

        this->nrecv_ = 0;
        this->nsend_ = 0;

        this->recvs_ = NULL;
        this->sends_ = NULL;

        this->recv_offset_index_ = NULL;
        this->send_offset_index_ = NULL;

        this->boundary_index_  = NULL;
        this->boundary_buffer_ = NULL;

        this->ghost_to_global_map_ = false;
        this->ghost_mapping_       = NULL;

        this->global_offset_     = false;
        this->global_row_offset_ = NULL;
        this->global_col_offset_ = NULL;

        this->async_send_ = 0;
        this->async_recv_ = 0;

        this->recv_event_ = NULL;
        this->send_event_ = NULL;

        // if new values are added, also put check into status function
    }

    ParallelManager::~ParallelManager()
    {
        this->Clear();

        free_host(&this->global_row_offset_);
        free_host(&this->global_col_offset_);
    }

    void ParallelManager::SetMPICommunicator(const void* comm)
    {
        assert(comm != NULL);
        this->comm_ = comm;

#ifdef SUPPORT_MULTINODE
        MPI_Comm_rank(*(MPI_Comm*)this->comm_, &this->rank_);
        MPI_Comm_size(*(MPI_Comm*)this->comm_, &this->num_procs_);
#endif

        if(this->global_row_offset_ == NULL)
        {
            allocate_host(this->num_procs_ + 1, &this->global_row_offset_);
        }

        if(this->global_col_offset_ == NULL)
        {
            allocate_host(this->num_procs_ + 1, &this->global_col_offset_);
        }
    }

    void ParallelManager::Clear(void)
    {
        this->global_nrow_ = 0;
        this->global_ncol_ = 0;
        this->local_nrow_  = 0;
        this->local_ncol_  = 0;

        this->global_offset_       = false;
        this->ghost_to_global_map_ = false;

        free_host(&this->recvs_);
        free_host(&this->recv_offset_index_);

        free_host(&this->sends_);
        free_host(&this->send_offset_index_);

#ifdef SUPPORT_MULTINODE
        free_host(&this->recv_event_);
        free_host(&this->send_event_);
#endif

        this->nrecv_ = 0;
        this->nsend_ = 0;

        free_host(&this->boundary_index_);
        free_host(&this->boundary_buffer_);

        this->recv_index_size_ = 0;
        this->send_index_size_ = 0;

        free_host(&this->ghost_mapping_);
    }

    int ParallelManager::GetNumProcs(void) const
    {
        assert(this->Status());

        return this->num_procs_;
    }

    int64_t ParallelManager::GetGlobalRowBegin(int rank) const
    {
        // Default rank is this rank
        rank = (rank < 0 || rank >= this->num_procs_) ? this->rank_ : rank;

        if(this->global_offset_ == false)
        {
            // We need to sync all on-going communication
            this->Synchronize_();

            this->CommunicateGlobalOffsetAsync_();
            this->CommunicateGlobalOffsetSync_();
            this->global_offset_ = true;
        }

        return this->global_row_offset_[rank];
    }

    int64_t ParallelManager::GetGlobalRowEnd(int rank) const
    {
        // Default rank is this rank
        rank = (rank < 0 || rank >= this->num_procs_) ? this->rank_ : rank;

        if(this->global_offset_ == false)
        {
            // We need to sync all on-going communication
            this->Synchronize_();

            this->CommunicateGlobalOffsetAsync_();
            this->CommunicateGlobalOffsetSync_();
            this->global_offset_ = true;
        }

        return this->global_row_offset_[rank + 1];
    }

    int64_t ParallelManager::GetGlobalColumnBegin(int rank) const
    {
        // Default rank is this rank
        rank = (rank < 0 || rank >= this->num_procs_) ? this->rank_ : rank;

        if(this->global_offset_ == false)
        {
            // We need to sync all on-going communication
            this->Synchronize_();

            this->CommunicateGlobalOffsetAsync_();
            this->CommunicateGlobalOffsetSync_();
            this->global_offset_ = true;
        }

        return this->global_col_offset_[rank];
    }

    int64_t ParallelManager::GetGlobalColumnEnd(int rank) const
    {
        // Default rank is this rank
        rank = (rank < 0 || rank >= this->num_procs_) ? this->rank_ : rank;

        if(this->global_offset_ == false)
        {
            // We need to sync all on-going communication
            this->Synchronize_();

            this->CommunicateGlobalOffsetAsync_();
            this->CommunicateGlobalOffsetSync_();
            this->global_offset_ = true;
        }

        return this->global_col_offset_[rank + 1];
    }

    void ParallelManager::SetGlobalNrow(int64_t nrow)
    {
        assert(nrow >= 0);
        assert(nrow >= this->local_nrow_);

        this->global_nrow_ = nrow;
    }

    void ParallelManager::SetGlobalNcol(int64_t ncol)
    {
        assert(ncol >= 0);
        assert(ncol >= this->local_ncol_);

        this->global_ncol_ = ncol;
    }

    void ParallelManager::SetLocalNrow(int64_t nrow)
    {
        assert(nrow >= 0);
        assert(nrow <= this->global_nrow_);

        this->local_nrow_ = nrow;
    }

    void ParallelManager::SetLocalNcol(int64_t ncol)
    {
        assert(ncol >= 0);
        assert(ncol <= this->global_ncol_);

        this->local_ncol_ = ncol;
    }

    int64_t ParallelManager::GetGlobalNrow(void) const
    {
        assert(this->Status());

        return this->global_nrow_;
    }

    int64_t ParallelManager::GetGlobalNcol(void) const
    {
        assert(this->Status());

        return this->global_ncol_;
    }

    int64_t ParallelManager::GetLocalNrow(void) const
    {
        assert(this->Status());

        return this->local_nrow_;
    }

    int64_t ParallelManager::GetLocalNcol(void) const
    {
        assert(this->Status());

        return this->local_ncol_;
    }

    int ParallelManager::GetNumReceivers(void) const
    {
        assert(this->Status());

        return this->recv_index_size_;
    }

    int ParallelManager::GetNumSenders(void) const
    {
        assert(this->Status());

        return this->send_index_size_;
    }

    void ParallelManager::SetBoundaryIndex(int size, const int* index)
    {
        assert(size >= 0);

        if(size > 0)
        {
            assert(index != NULL);
        }

        if(this->send_index_size_ != 0)
        {
            assert(this->send_index_size_ == size);
        }
        else
        {
            this->send_index_size_ = size;
        }

        allocate_host(size, &this->boundary_index_);
        allocate_host(size, &this->boundary_buffer_);

        copy_h2h(size, index, this->boundary_index_);
    }

    const int* ParallelManager::GetBoundaryIndex(void) const
    {
        assert(this->Status());

        return this->boundary_index_;
    }

    const int64_t* ParallelManager::GetGhostToGlobalMap(void) const
    {
        assert(this->Status());

        if(this->ghost_to_global_map_ == false)
        {
            // We need to sync all on-going communication
            this->Synchronize_();

            this->CommunicateGhostToGlobalMapAsync_();
            this->CommunicateGhostToGlobalMapSync_();

            this->ghost_to_global_map_ = true;
        }

        return this->ghost_mapping_;
    }

    void ParallelManager::SetReceivers(int nrecv, const int* recvs, const int* recv_offset)
    {
        assert(nrecv >= 0);
        assert(recv_offset != NULL);

        if(nrecv > 0)
        {
            assert(recvs != NULL);
        }

        this->nrecv_ = nrecv;

        allocate_host(nrecv, &this->recvs_);
        allocate_host(nrecv + 1, &this->recv_offset_index_);

        this->recv_offset_index_[0] = 0;

        copy_h2h(nrecv, recvs, this->recvs_);
        copy_h2h(nrecv, recv_offset + 1, this->recv_offset_index_ + 1);

        this->recv_index_size_ = recv_offset[nrecv];

#ifdef SUPPORT_MULTINODE
        allocate_host(2 * nrecv + 1, &this->recv_event_);
#endif

        if(this->ghost_mapping_ == NULL)
        {
            allocate_host(this->recv_index_size_, &this->ghost_mapping_);
        }
    }

    void ParallelManager::SetSenders(int nsend, const int* sends, const int* send_offset)
    {
        assert(nsend >= 0);
        assert(send_offset != NULL);

        if(nsend > 0)
        {
            assert(sends != NULL);
        }

        this->nsend_ = nsend;

        allocate_host(nsend, &this->sends_);
        allocate_host(nsend + 1, &this->send_offset_index_);

        this->send_offset_index_[0] = 0;

        copy_h2h(nsend, sends, this->sends_);
        copy_h2h(nsend, send_offset + 1, this->send_offset_index_ + 1);

        if(this->send_index_size_ != 0)
        {
            assert(this->send_index_size_ == send_offset[nsend]);
        }
        else
        {
            this->send_index_size_ = send_offset[nsend];
        }

#ifdef SUPPORT_MULTINODE
        allocate_host(2 * nsend + 1, &this->send_event_);
#endif
    }

    bool ParallelManager::Status(void) const
    {
        // clang-format off
        if(this->comm_ == NULL) return false;
        if(this->rank_ < 0) return false;
        if(this->global_nrow_ < 0) return false;
        if(this->global_ncol_ < 0) return false;
        if(this->local_nrow_ < 0) return false;
        if(this->local_ncol_ < 0) return false;
        if(this->nrecv_ < 0) return false;
        if(this->nsend_ < 0) return false;
        if(this->nrecv_ > 0 && this->recvs_ == NULL) return false;
        if(this->nsend_ > 0 && this->sends_ == NULL) return false;
        if(this->nrecv_ > 0 && this->recv_offset_index_ == NULL) return false;
        if(this->nsend_ > 0 && this->send_offset_index_ == NULL) return false;
        if(this->recv_index_size_ < 0) return false;
        if(this->send_index_size_ < 0) return false;
        if(this->send_index_size_ > 0 && this->boundary_index_ == NULL) return false;
        // clang-format on

        return true;
    }

    void ParallelManager::WriteFileASCII(const std::string& filename) const
    {
        log_debug(this, "ParallelManager::WriteFileASCII()", filename);

        assert(this->Status());

        // Master rank writes the global headfile
        if(this->rank_ == 0)
        {
            std::ofstream headfile;

            LOG_INFO("WriteFileASCII: filename=" << filename << "; writing...");

            headfile.open((char*)filename.c_str(), std::ofstream::out);
            if(!headfile.is_open())
            {
                LOG_INFO("Cannot open ParallelManager file [write]: " << filename);
                FATAL_ERROR(__FILE__, __LINE__);
            }

            for(int i = 0; i < this->num_procs_; ++i)
            {
                std::ostringstream rs;
                rs << i;

                std::string name = filename + ".rank." + rs.str();

                headfile << name << "\n";
            }
        }

        std::ostringstream rs;
        rs << this->rank_;

        std::string   name = filename + ".rank." + rs.str();
        std::ofstream file;

        file.open((char*)name.c_str(), std::ifstream::out);

        if(!file.is_open())
        {
            LOG_INFO("Cannot open ParallelManager file [write]:" << name);
            FATAL_ERROR(__FILE__, __LINE__);
        }

        file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        file << "%% ROCALUTION MPI ParallelManager output %%" << std::endl;
        file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        file << "#RANK\n" << this->rank_ << std::endl;
        file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        file << "#GLOBAL_NROW\n" << this->global_nrow_ << std::endl;
        file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        file << "#GLOBAL_NCOL\n" << this->global_ncol_ << std::endl;
        file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        file << "#LOCAL_NROW\n" << this->local_nrow_ << std::endl;
        file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        file << "#LOCAL_NCOL\n" << this->local_ncol_ << std::endl;
        file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        file << "#BOUNDARY_SIZE\n" << this->send_index_size_ << std::endl;
        file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        file << "#NUMBER_OF_RECEIVERS\n" << this->nrecv_ << std::endl;
        file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        file << "#NUMBER_OF_SENDERS\n" << this->nsend_ << std::endl;
        file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        file << "#RECEIVERS_RANK" << std::endl;
        for(int i = 0; i < this->nrecv_; ++i)
        {
            file << this->recvs_[i] << std::endl;
        }
        file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        file << "#SENDERS_RANK" << std::endl;
        for(int i = 0; i < this->nsend_; ++i)
        {
            file << this->sends_[i] << std::endl;
        }
        file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        file << "#RECEIVERS_INDEX_OFFSET" << std::endl;
        for(int i = 0; i < this->nrecv_ + 1; ++i)
        {
            file << this->recv_offset_index_[i] << std::endl;
        }
        file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        file << "#SENDERS_INDEX_OFFSET" << std::endl;
        for(int i = 0; i < this->nsend_ + 1; ++i)
        {
            file << this->send_offset_index_[i] << std::endl;
        }
        file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        file << "#BOUNDARY_INDEX" << std::endl;
        for(int i = 0; i < this->send_index_size_; ++i)
        {
            file << this->boundary_index_[i] << std::endl;
        }

        file.close();

        LOG_INFO("WriteFileASCII: filename=" << name << "; done");
    }

    void ParallelManager::ReadFileASCII(const std::string& filename)
    {
        log_debug(this, "ParallelManager::ReadFileASCII()", filename);

        assert(this->comm_ != NULL);

        // Read header file
        std::ifstream headfile;

        LOG_INFO("ReadFileASCII: filename=" << filename << "; reading...");

        headfile.open((char*)filename.c_str(), std::ifstream::in);
        if(!headfile.is_open())
        {
            LOG_INFO("Cannot open ParallelManager file [read]: " << filename);
            FATAL_ERROR(__FILE__, __LINE__);
        }

        // Go to this ranks line in the headfile
        for(int i = 0; i < this->rank_; ++i)
        {
            headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        std::string name;
        std::getline(headfile, name);

        headfile.close();

        // Extract directory containing the subfiles
        size_t      found = filename.find_last_of("\\/");
        std::string path  = filename.substr(0, found + 1);

        name.erase(remove_if(name.begin(), name.end(), isspace), name.end());
        name = path + name;

        // Open the ranks corresponding file
        std::ifstream file;
        std::string   line;

        file.open(name.c_str(), std::ifstream::in);
        if(!file.is_open())
        {
            LOG_INFO("Cannot open ParallelManager file [read]: " << name);
            FATAL_ERROR(__FILE__, __LINE__);
        }

        this->Clear();
        int rank = -1;

        while(!file.eof())
        {
            std::getline(file, line);

            if(line.find("#RANK") != std::string::npos)
            {
                file >> rank;
            }
            if(line.find("#GLOBAL_SIZE") != std::string::npos)
            {
                file >> this->global_nrow_;
                this->global_ncol_ = this->global_nrow_;
            }
            if(line.find("#GLOBAL_NROW") != std::string::npos)
            {
                file >> this->global_nrow_;
            }
            if(line.find("#GLOBAL_NCOL") != std::string::npos)
            {
                file >> this->global_ncol_;
            }
            if(line.find("#LOCAL_SIZE") != std::string::npos)
            {
                file >> this->local_nrow_;
                this->local_ncol_ = this->local_nrow_;
            }
            if(line.find("#LOCAL_NROW") != std::string::npos)
            {
                file >> this->local_nrow_;
            }
            if(line.find("#LOCAL_NCOL") != std::string::npos)
            {
                file >> this->local_ncol_;
            }
            if(line.find("#BOUNDARY_SIZE") != std::string::npos)
            {
                file >> this->send_index_size_;
            }
            if(line.find("#NUMBER_OF_RECEIVERS") != std::string::npos)
            {
                file >> this->nrecv_;
#ifdef SUPPORT_MULTINODE
                allocate_host(2 * this->nrecv_ + 1, &this->recv_event_);
#endif
            }
            if(line.find("#NUMBER_OF_SENDERS") != std::string::npos)
            {
                file >> this->nsend_;
#ifdef SUPPORT_MULTINODE
                allocate_host(2 * this->nsend_ + 1, &this->send_event_);
#endif
            }
            if(line.find("#RECEIVERS_RANK") != std::string::npos)
            {
                allocate_host(this->nrecv_, &this->recvs_);
                for(int i = 0; i < this->nrecv_; ++i)
                {
                    file >> this->recvs_[i];
                }
            }
            if(line.find("#SENDERS_RANK") != std::string::npos)
            {
                allocate_host(this->nsend_, &this->sends_);
                for(int i = 0; i < this->nsend_; ++i)
                {
                    file >> this->sends_[i];
                }
            }
            if(line.find("#RECEIVERS_INDEX_OFFSET") != std::string::npos)
            {
                assert(this->nrecv_ > -1);
                allocate_host(this->nrecv_ + 1, &this->recv_offset_index_);
                for(int i = 0; i < this->nrecv_ + 1; ++i)
                {
                    file >> this->recv_offset_index_[i];
                }
            }
            if(line.find("#SENDERS_INDEX_OFFSET") != std::string::npos)
            {
                assert(this->nsend_ > -1);
                allocate_host(this->nsend_ + 1, &this->send_offset_index_);
                for(int i = 0; i < this->nsend_ + 1; ++i)
                {
                    file >> this->send_offset_index_[i];
                }
            }
            if(line.find("#BOUNDARY_INDEX") != std::string::npos)
            {
                assert(this->send_index_size_ > -1);
                allocate_host(this->send_index_size_, &this->boundary_index_);
                for(int i = 0; i < this->send_index_size_; ++i)
                {
                    file >> this->boundary_index_[i];
                }
            }
        }

        // Ghost mapping
        if(this->ghost_mapping_ == NULL)
        {
            allocate_host(this->recv_index_size_, &this->ghost_mapping_);
        }

        // Number of nnz we receive
        this->recv_index_size_ = this->recv_offset_index_[this->nrecv_];

        // Number of nnz we send == boundary size
        assert(this->send_index_size_ == this->send_offset_index_[this->nsend_]);

        file.close();

        assert(rank == this->rank_);

        if(this->Status() == false)
        {
            LOG_INFO("Incomplete ParallelManager file");
            FATAL_ERROR(__FILE__, __LINE__);
        }

        LOG_INFO("ReadFileASCII: filename=" << filename << "; done");
    }

    void ParallelManager::Synchronize_(void) const
    {
#ifdef SUPPORT_MULTINODE
        // Sync all events
        communication_syncall(this->async_recv_, this->recv_event_);
        communication_syncall(this->async_send_, this->send_event_);

        // Reset guard
        this->async_recv_ = 0;
        this->async_send_ = 0;
#endif
    }

    template <typename ValueType>
    void ParallelManager::CommunicateAsync_(ValueType* send_buffer, ValueType* recv_buffer) const
    {
        log_debug(
            this, "ParallelManager::CommunicateAsync_()", "#*# begin", send_buffer, recv_buffer);

        assert(this->async_send_ == 0);
        assert(this->async_recv_ == 0);
        assert(this->Status());

        int tag = 0;

        // async recv boundary from neighbors
        for(int n = 0; n < this->nrecv_; ++n)
        {
            // nnz that we receive from process n
            int nnz = this->recv_offset_index_[n + 1] - this->recv_offset_index_[n];

            // if this has ghost values that belong to process i
            if(nnz > 0)
            {
                assert(recv_buffer != NULL);

#ifdef SUPPORT_MULTINODE
                communication_async_recv(recv_buffer + this->recv_offset_index_[n],
                                         nnz,
                                         this->recvs_[n],
                                         tag,
                                         &this->recv_event_[this->async_recv_++],
                                         this->comm_);
#endif
            }
        }

        // async send boundary to neighbors
        for(int n = 0; n < this->nsend_; ++n)
        {
            // nnz that we send to process i
            int nnz = this->send_offset_index_[n + 1] - this->send_offset_index_[n];

            // if process i has ghost values that belong to this
            if(nnz > 0)
            {
                assert(send_buffer != NULL);

#ifdef SUPPORT_MULTINODE
                communication_async_send(send_buffer + this->send_offset_index_[n],
                                         nnz,
                                         this->sends_[n],
                                         tag,
                                         &this->send_event_[this->async_send_++],
                                         this->comm_);
#endif
            }
        }

        log_debug(this, "ParallelManager::CommunicateAsync_()", "#*# end");
    }

    void ParallelManager::CommunicateSync_(void) const
    {
        this->Synchronize_();
    }

    template <typename ValueType>
    void ParallelManager::InverseCommunicateAsync_(ValueType* send_buffer,
                                                   ValueType* recv_buffer) const
    {
        log_debug(this,
                  "ParallelManager::InverseCommunicateAsync_()",
                  "#*# begin",
                  send_buffer,
                  recv_buffer);

        assert(this->async_send_ == 0);
        assert(this->async_recv_ == 0);

        int tag = 0;

        // async recv boundary from neighbors
        for(int n = 0; n < this->nsend_; ++n)
        {
            // nnz that we receive from process n
            int nnz = this->send_offset_index_[n + 1] - this->send_offset_index_[n];

            // if this has ghost values that belong to process i
            if(nnz > 0)
            {
                assert(recv_buffer != NULL);

#ifdef SUPPORT_MULTINODE
                communication_async_recv(recv_buffer + this->send_offset_index_[n],
                                         nnz,
                                         this->sends_[n],
                                         tag,
                                         &this->send_event_[this->async_send_++],
                                         this->comm_);
#endif
            }
        }

        // async send boundary to neighbors
        for(int n = 0; n < this->nrecv_; ++n)
        {
            // nnz that we send to process i
            int nnz = this->recv_offset_index_[n + 1] - this->recv_offset_index_[n];

            // if process i has ghost values that belong to this
            if(nnz > 0)
            {
                assert(send_buffer != NULL);

#ifdef SUPPORT_MULTINODE
                communication_async_send(send_buffer + this->recv_offset_index_[n],
                                         nnz,
                                         this->recvs_[n],
                                         tag,
                                         &this->recv_event_[this->async_recv_++],
                                         this->comm_);
#endif
            }
        }

        log_debug(this, "ParallelManager::InverseCommunicateAsync_()", "#*# end");
    }

    void ParallelManager::InverseCommunicateSync_(void) const
    {
        this->Synchronize_();
    }

    template <typename I, typename J, typename T>
    void ParallelManager::CommunicateCSRAsync_(I* send_row_ptr,
                                               J* send_col_ind,
                                               T* send_val,
                                               I* recv_row_ptr,
                                               J* recv_col_ind,
                                               T* recv_val) const
    {
        log_debug(this,
                  "ParallelManager::CommunicateCSRAsync_()",
                  "#*# begin",
                  send_row_ptr,
                  send_col_ind,
                  send_val,
                  recv_row_ptr,
                  recv_col_ind,
                  recv_val);

        assert(this->Status());
        assert(this->async_send_ == 0);
        assert(this->async_recv_ == 0);

        int tag = 0;

        // Async recv from neighbors
        for(int n = 0; n < this->nrecv_; ++n)
        {
            int first_row = this->recv_offset_index_[n];
            int last_row  = this->recv_offset_index_[n + 1];

            // We expect something, so row ptr cannot be null
            assert(recv_row_ptr != NULL);

            // nnz that we receive from process i
            int nnz = static_cast<int>(recv_row_ptr[last_row] - recv_row_ptr[first_row]);

            // if this has ghost values that belong to process i
            if(nnz > 0)
            {
#ifdef SUPPORT_MULTINODE
                // If col pointer is null, we do not expect anything
                if(recv_col_ind != nullptr)
                {
                    communication_async_recv(recv_col_ind + recv_row_ptr[first_row],
                                             nnz,
                                             this->recvs_[n],
                                             tag,
                                             &this->recv_event_[this->async_recv_++],
                                             this->comm_);
                }

                // If val pointer is null, we do not expect anything
                if(recv_val != nullptr)
                {
                    communication_async_recv(recv_val + recv_row_ptr[first_row],
                                             nnz,
                                             this->recvs_[n],
                                             tag,
                                             &this->recv_event_[this->async_recv_++],
                                             this->comm_);
                }
#endif
            }
        }

        // Async send boundary to neighbors
        for(int n = 0; n < this->nsend_; ++n)
        {
            // nnz that we send to process i
            int first_row = this->send_offset_index_[n];
            int last_row  = this->send_offset_index_[n + 1];

            // We send something, so row ptr cannot be null
            assert(send_row_ptr != NULL);

            int nnz = static_cast<int>(send_row_ptr[last_row] - send_row_ptr[first_row]);

            // if process i has ghost values that belong to this
            if(nnz > 0)
            {
#ifdef SUPPORT_MULTINODE
                // If col pointer is null, we do not send anything
                if(send_col_ind != nullptr)
                {
                    communication_async_send(send_col_ind + send_row_ptr[first_row],
                                             nnz,
                                             this->sends_[n],
                                             tag,
                                             &this->send_event_[this->async_send_++],
                                             this->comm_);
                }

                // If val pointer is null, we do not send anything
                if(send_val != nullptr)
                {
                    communication_async_send(send_val + send_row_ptr[first_row],
                                             nnz,
                                             this->sends_[n],
                                             tag,
                                             &this->send_event_[this->async_send_++],
                                             this->comm_);
                }
#endif
            }
        }

        log_debug(this, "ParallelManager::CommunicateCSRAsync_()", "#*# end");
    }

    void ParallelManager::CommunicateCSRSync_(void) const
    {
        this->Synchronize_();
    }

    template <typename I, typename J, typename T>
    void ParallelManager::InverseCommunicateCSRAsync_(I* send_row_ptr,
                                                      J* send_col_ind,
                                                      T* send_val,
                                                      I* recv_row_ptr,
                                                      J* recv_col_ind,
                                                      T* recv_val) const
    {
        log_debug(this,
                  "ParallelManager::InverseCommunicateCSRAsync_()",
                  "#*# begin",
                  send_row_ptr,
                  send_col_ind,
                  send_val,
                  recv_row_ptr,
                  recv_col_ind,
                  recv_val);

        assert(this->Status());
        assert(this->async_send_ == 0);
        assert(this->async_recv_ == 0);

        int tag = 0;

        // Async recv from neighbors
        for(int n = 0; n < this->nsend_; ++n)
        {
            int first_row = this->send_offset_index_[n];
            int last_row  = this->send_offset_index_[n + 1];

            // We expect something, so row ptr cannot be null
            assert(recv_row_ptr != NULL);

            // nnz that we receive from process i
            int nnz = static_cast<int>(recv_row_ptr[last_row] - recv_row_ptr[first_row]);

            // if this has ghost values that belong to process i
            if(nnz > 0)
            {
#ifdef SUPPORT_MULTINODE
                // If col pointer is null, we do not expect anything
                if(recv_col_ind != nullptr)
                {
                    communication_async_recv(recv_col_ind + recv_row_ptr[first_row],
                                             nnz,
                                             this->sends_[n],
                                             tag,
                                             &this->send_event_[this->async_send_++],
                                             this->comm_);
                }

                // If val pointer is null, we do not expect anything
                if(recv_val != nullptr)
                {
                    communication_async_recv(recv_val + recv_row_ptr[first_row],
                                             nnz,
                                             this->sends_[n],
                                             tag,
                                             &this->send_event_[this->async_send_++],
                                             this->comm_);
                }
#endif
            }
        }

        // Async send boundary to neighbors
        for(int n = 0; n < this->nrecv_; ++n)
        {
            // nnz that we send to process i
            int first_row = this->recv_offset_index_[n];
            int last_row  = this->recv_offset_index_[n + 1];

            // We send something, so row ptr cannot be null
            assert(send_row_ptr != NULL);

            int nnz = static_cast<int>(send_row_ptr[last_row] - send_row_ptr[first_row]);

            // if process i has ghost values that belong to this
            if(nnz > 0)
            {
#ifdef SUPPORT_MULTINODE
                // If col pointer is null, we do not send anything
                if(send_col_ind != nullptr)
                {
                    communication_async_send(send_col_ind + send_row_ptr[first_row],
                                             nnz,
                                             this->recvs_[n],
                                             tag,
                                             &this->recv_event_[this->async_recv_++],
                                             this->comm_);
                }

                // If val pointer is null, we do not send anything
                if(send_val != nullptr)
                {
                    communication_async_send(send_val + send_row_ptr[first_row],
                                             nnz,
                                             this->recvs_[n],
                                             tag,
                                             &this->recv_event_[this->async_recv_++],
                                             this->comm_);
                }
#endif
            }
        }

        log_debug(this, "ParallelManager::InverseCommunicateCSRAsync_()", "#*# end");
    }

    void ParallelManager::InverseCommunicateCSRSync_(void) const
    {
        this->CommunicateCSRSync_();
    }

    void ParallelManager::CommunicateGlobalOffsetAsync_(void) const
    {
        log_debug(this, "ParallelManager::CommunicateGlobalOffsetAsync_()", "#*# begin");

        assert(this->global_row_offset_ != NULL);
        assert(this->global_col_offset_ != NULL);

        // Check guards
        assert(this->async_recv_ <= 2 * this->nrecv_);
        assert(this->async_send_ <= 2 * this->nsend_);

        // Increment guard
        ++this->async_recv_;
        ++this->async_send_;

        int64_t local_nrow = this->local_nrow_;
        int64_t local_ncol = this->local_ncol_;

#ifdef SUPPORT_MULTINODE
        communication_async_allgather_single(
            &local_nrow, this->global_row_offset_ + 1, this->recv_event_, this->comm_);
        communication_async_allgather_single(
            &local_ncol, this->global_col_offset_ + 1, this->send_event_, this->comm_);
#endif

        log_debug(this, "ParallelManager::CommunicateGlobalOffsetAsync_()", "#*# end");
    }

    void ParallelManager::CommunicateGlobalOffsetSync_(void) const
    {
        log_debug(this, "ParallelManager::CommunicateGlobalOffsetSync_()", "#*# begin");

        assert(this->global_row_offset_ != NULL);
        assert(this->global_col_offset_ != NULL);

#ifdef SUPPORT_MULTINODE
        communication_sync(this->recv_event_);
        communication_sync(this->send_event_);
#endif

        // Decrement guard
        --this->async_send_;
        --this->async_recv_;

        this->global_row_offset_[0] = 0;
        this->global_col_offset_[0] = 0;

        for(int n = 0; n < this->num_procs_; ++n)
        {
            this->global_row_offset_[n + 1] += this->global_row_offset_[n];
            this->global_col_offset_[n + 1] += this->global_col_offset_[n];
        }

        log_debug(this, "ParallelManager::CommunicateGlobalOffsetSync_()", "#*# end");
    }

    void ParallelManager::CommunicateGhostToGlobalMapAsync_(void) const
    {
        log_debug(this, "ParallelManager::CommunicateGhostToGlobalMap_()", "#*# begin");

        assert(this->Status());

        // Obtain local indices from neighbors and store them in the map

        // First, get global column begin (this might require communication, so we need to grab it first to avoid deadlock
        int64_t global_col_begin = this->GetGlobalColumnBegin();

        // Prepare send buffer
        for(int i = 0; i < this->send_index_size_; ++i)
        {
            this->boundary_buffer_[i] = this->boundary_index_[i] + global_col_begin;
        }

        // Communicate
        this->CommunicateAsync_(this->boundary_buffer_, this->ghost_mapping_);

        log_debug(this, "ParallelManager::CommunicateGhostToGlobalMap_()", "#*# end");
    }

    void ParallelManager::CommunicateGhostToGlobalMapSync_(void) const
    {
        this->Synchronize_();
    }

    void ParallelManager::GenerateFromGhostColumnsWithParent_(int64_t                nnz,
                                                              const int64_t*         ghost_col,
                                                              const ParallelManager& parent,
                                                              bool                   transposed)
    {
        // Allocate
        std::vector<int>     recv_size(parent.num_procs_, 0);
        std::vector<int64_t> recv_index;
        recv_index.reserve(nnz);

        int64_t last_col = -1;
        for(int64_t i = 0; i < nnz; ++i)
        {
            // Global column index
            int64_t global_col = ghost_col[i];

            // Sanity check
            assert(global_col >= 0);
            assert(global_col < transposed ? parent.global_nrow_ : parent.global_ncol_);

            // Check for duplicates
            if(global_col == last_col)
            {
                continue;
            }

            // Check which process we are expecting to receive this entry from
            for(int n = 0; n < parent.num_procs_; ++n)
            {
                if(n == parent.rank_)
                {
                    continue;
                }

                int64_t col_begin
                    = transposed ? parent.GetGlobalRowBegin(n) : parent.GetGlobalColumnBegin(n);
                int64_t col_end
                    = transposed ? parent.GetGlobalRowEnd(n) : parent.GetGlobalColumnEnd(n);

                if(global_col >= col_begin && global_col < col_end)
                {
                    ++recv_size[n];
                    recv_index.push_back(global_col);

                    break;
                }
            }

            last_col = global_col;
        }

        // Send / receive number of points this rank need to receive from its neighbor
        std::vector<int> send_size(parent.num_procs_);
#ifdef SUPPORT_MULTINODE
        MRequest req;
        communication_async_alltoall_single(recv_size.data(), send_size.data(), parent.comm_, &req);
#endif

        // Determine receiving offsets and sizes
        this->nrecv_ = 0;

        for(int n = 0; n < parent.num_procs_; ++n)
        {
            // Do not communicate with yourself
            if(n == parent.rank_)
            {
                continue;
            }

            // If we need to receive points from n, set up the structure
            if(recv_size[n] > 0)
            {
                ++this->nrecv_;
            }
        }

        allocate_host(this->nrecv_, &this->recvs_);
        allocate_host(this->nrecv_ + 1, &this->recv_offset_index_);

        this->recv_offset_index_[0] = 0;

        for(int n = 0, i = 0; n < parent.num_procs_; ++n)
        {
            // Do not communicate with yourself
            if(n == parent.rank_)
            {
                continue;
            }

            // Number of boundary points we need to share
            int nbp = recv_size[n];

            // If we need to receive points from n, set up the structure
            if(nbp > 0)
            {
                this->recvs_[i]               = n;
                this->recv_offset_index_[++i] = nbp;
            }
        }

        // Exclusive sum to obtain offsets
        for(int n = 0; n < this->nrecv_; ++n)
        {
            this->recv_offset_index_[n + 1] += this->recv_offset_index_[n];
        }

        this->recv_index_size_ = this->recv_offset_index_[this->nrecv_];
#ifdef SUPPORT_MULTINODE
        allocate_host(2 * this->nrecv_ + 1, &this->recv_event_);
#endif

        if(this->ghost_mapping_ == NULL)
        {
            allocate_host(this->recv_index_size_, &this->ghost_mapping_);
        }

        // Synchronize All to All communication
#ifdef SUPPORT_MULTINODE
        communication_sync(&req);
#endif

        // Determine sending offsets and sizes
        this->nsend_ = 0;

        for(int n = 0; n < parent.num_procs_; ++n)
        {
            // Do not communicate with yourself
            if(n == parent.rank_)
            {
                continue;
            }

            if(send_size[n] > 0)
            {
                ++this->nsend_;
            }
        }

        allocate_host(this->nsend_, &this->sends_);
        allocate_host(this->nsend_ + 1, &this->send_offset_index_);

        this->send_offset_index_[0] = 0;

        for(int n = 0, i = 0; n < parent.num_procs_; ++n)
        {
            // Do not communicate with yourself
            if(n == parent.rank_)
            {
                continue;
            }

            // Number of boundary points we need to send
            int nbp = send_size[n];

            // If we need to send points to n, set up the structure
            if(nbp > 0)
            {
                this->sends_[i]               = n;
                this->send_offset_index_[++i] = nbp;
            }
        }

        // Exclusive sum to obtain offsets
        for(int i = 0; i < this->nsend_; ++i)
        {
            this->send_offset_index_[i + 1] += this->send_offset_index_[i];
        }

#ifdef SUPPORT_MULTINODE
        allocate_host(2 * this->nsend_ + 1, &this->send_event_);
#endif

        // Tell neighbors, which boundary points need to be send
        this->send_index_size_ = this->send_offset_index_[this->nsend_];

        allocate_host(this->send_index_size_, &this->boundary_index_);
        allocate_host(this->send_index_size_, &this->boundary_buffer_);

        // Communicate global boundary indices
        this->InverseCommunicateAsync_(recv_index.data(), this->boundary_buffer_);
        this->InverseCommunicateSync_();
    }

    void ParallelManager::BoundaryTransformGlobalToLocal_(void)
    {
        int64_t offset = this->GetGlobalColumnBegin();

        for(int i = 0; i < this->send_index_size_; ++i)
        {
            // Boundary index is a local id, thus fits into 32 bits
            this->boundary_index_[i] = static_cast<int>(this->boundary_buffer_[i] - offset);
        }
    }

    void ParallelManager::BoundaryTransformGlobalFineToLocalCoarse_(const int* f2c)
    {
        int64_t offset = this->GetGlobalRowBegin();

        for(int i = 0; i < this->send_index_size_; ++i)
        {
            int64_t global_fine = this->boundary_buffer_[i];

            // Map from global fine to local fine index
            int64_t local_fine = global_fine - offset;

            // Map from local fine to local coarse
            this->boundary_index_[i] = static_cast<int>(f2c[local_fine]);
        }
    }

    template void ParallelManager::CommunicateAsync_<int>(int*, int*) const;
    template void ParallelManager::CommunicateAsync_<float>(float*, float*) const;
    template void ParallelManager::CommunicateAsync_<double>(double*, double*) const;
    template void
        ParallelManager::CommunicateAsync_<std::complex<float>>(std::complex<float>*,
                                                                std::complex<float>*) const;
    template void
        ParallelManager::CommunicateAsync_<std::complex<double>>(std::complex<double>*,
                                                                 std::complex<double>*) const;

    template void ParallelManager::InverseCommunicateAsync_<bool>(bool*, bool*) const;
    template void ParallelManager::InverseCommunicateAsync_<int>(int*, int*) const;
    template void ParallelManager::InverseCommunicateAsync_<float>(float*, float*) const;
    template void ParallelManager::InverseCommunicateAsync_<double>(double*, double*) const;
    template void
        ParallelManager::InverseCommunicateAsync_<std::complex<float>>(std::complex<float>*,
                                                                       std::complex<float>*) const;
    template void ParallelManager::InverseCommunicateAsync_<std::complex<double>>(
        std::complex<double>*, std::complex<double>*) const;

    template void ParallelManager::CommunicateCSRAsync_<PtrType, int64_t, float>(
        PtrType*, int64_t*, float*, PtrType*, int64_t*, float*) const;
    template void ParallelManager::CommunicateCSRAsync_<PtrType, int64_t, double>(
        PtrType*, int64_t*, double*, PtrType*, int64_t*, double*) const;
    template void ParallelManager::CommunicateCSRAsync_<PtrType, int64_t, std::complex<float>>(
        PtrType*, int64_t*, std::complex<float>*, PtrType*, int64_t*, std::complex<float>*) const;
    template void ParallelManager::CommunicateCSRAsync_<PtrType, int64_t, std::complex<double>>(
        PtrType*, int64_t*, std::complex<double>*, PtrType*, int64_t*, std::complex<double>*) const;
    template void ParallelManager::CommunicateCSRAsync_<PtrType, int, int>(
        PtrType*, int*, int*, PtrType*, int*, int*) const;

    template void ParallelManager::InverseCommunicateCSRAsync_<PtrType, int64_t, float>(
        PtrType*, int64_t*, float*, PtrType*, int64_t*, float*) const;
    template void ParallelManager::InverseCommunicateCSRAsync_<PtrType, int64_t, double>(
        PtrType*, int64_t*, double*, PtrType*, int64_t*, double*) const;
    template void
        ParallelManager::InverseCommunicateCSRAsync_<PtrType, int64_t, std::complex<float>>(
            PtrType*, int64_t*, std::complex<float>*, PtrType*, int64_t*, std::complex<float>*)
            const;
    template void
        ParallelManager::InverseCommunicateCSRAsync_<PtrType, int64_t, std::complex<double>>(
            PtrType*, int64_t*, std::complex<double>*, PtrType*, int64_t*, std::complex<double>*)
            const;

} // namespace rocalution

// LCOV_EXCL_STOP