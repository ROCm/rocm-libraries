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

#include "global_matrix.hpp"
#include "../utils/allocate_free.hpp"
#include "../utils/def.hpp"
#include "../utils/log.hpp"
#include "../utils/math_functions.hpp"
#include "base_matrix.hpp"
#include "base_vector.hpp"
#include "global_vector.hpp"
#include "local_matrix.hpp"
#include "local_vector.hpp"
#include "matrix_formats.hpp"

#ifdef SUPPORT_MULTINODE
#include "../utils/communicator.hpp"
#endif

#include <algorithm>
#include <complex>
#include <limits>
#include <sstream>

#ifndef SUPPORT_MULTINODE
// LCOV_EXCL_START
#endif

namespace rocalution
{
    template <typename ValueType>
    GlobalMatrix<ValueType>::GlobalMatrix()
    {
        log_debug(this, "GlobalMatrix::GlobalMatrix()");

#ifndef SUPPORT_MULTINODE
        LOG_INFO("Multinode support disabled");
        FATAL_ERROR(__FILE__, __LINE__);
#endif

        this->pm_      = NULL;
        this->pm_self_ = NULL;

        this->object_name_ = "";

        this->nnz_ = 0;

        this->recv_boundary_ = NULL;
        this->send_boundary_ = NULL;
    }

    template <typename ValueType>
    GlobalMatrix<ValueType>::GlobalMatrix(const ParallelManager& pm)
    {
        log_debug(this, "GlobalMatrix::GlobalMatrix()", (const void*&)pm);

        assert(pm.Status() == true);

        this->object_name_ = "";

        this->pm_ = &pm;

        this->nnz_ = 0;

        this->recv_boundary_ = NULL;
        this->send_boundary_ = NULL;
    }

    template <typename ValueType>
    GlobalMatrix<ValueType>::~GlobalMatrix()
    {
        log_debug(this, "GlobalMatrix::~GlobalMatrix()");

        this->Clear();

        if(this->pm_self_)
        {
            this->pm_self_->Clear();

            delete this->pm_self_;

            this->pm_      = NULL;
            this->pm_self_ = NULL;
        }
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::Clear(void)
    {
        log_debug(this, "GlobalMatrix::Clear()");

        this->matrix_interior_.Clear();
        this->matrix_ghost_.Clear();
        this->halo_.Clear();
        this->recv_buffer_.Clear();
        this->send_buffer_.Clear();

        this->nnz_ = 0;

        free_pinned(&this->recv_boundary_);
        free_pinned(&this->send_boundary_);
    }

    template <typename ValueType>
    int64_t GlobalMatrix<ValueType>::GetM(void) const
    {
        // Check, if we are actually running multiple processes
        if(this->pm_ != NULL)
        {
            return this->pm_->GetGlobalNrow();
        }
        else
        {
            return this->matrix_interior_.GetM();
        }
    }

    template <typename ValueType>
    int64_t GlobalMatrix<ValueType>::GetN(void) const
    {
        // Check, if we are actually running multiple processes
        if(this->pm_ != NULL)
        {
            return this->pm_->GetGlobalNcol();
        }
        else
        {
            return this->matrix_interior_.GetN();
        }
    }

    template <typename ValueType>
    int64_t GlobalMatrix<ValueType>::GetNnz(void) const
    {
        // Check, if we are actually running multiple processes
        if(this->pm_ == NULL || this->pm_->num_procs_ == 1)
        {
            return this->matrix_interior_.GetNnz();
        }

        return this->nnz_;
    }

    template <typename ValueType>
    int64_t GlobalMatrix<ValueType>::GetLocalM(void) const
    {
        return this->matrix_interior_.GetLocalM();
    }

    template <typename ValueType>
    int64_t GlobalMatrix<ValueType>::GetLocalN(void) const
    {
        return this->matrix_interior_.GetLocalN();
    }

    template <typename ValueType>
    int64_t GlobalMatrix<ValueType>::GetLocalNnz(void) const
    {
        return this->matrix_interior_.GetLocalNnz();
    }

    template <typename ValueType>
    int64_t GlobalMatrix<ValueType>::GetGhostM(void) const
    {
        return this->matrix_ghost_.GetLocalM();
    }

    template <typename ValueType>
    int64_t GlobalMatrix<ValueType>::GetGhostN(void) const
    {
        return this->matrix_ghost_.GetLocalN();
    }

    template <typename ValueType>
    int64_t GlobalMatrix<ValueType>::GetGhostNnz(void) const
    {
        return this->matrix_ghost_.GetLocalNnz();
    }

    template <typename ValueType>
    unsigned int GlobalMatrix<ValueType>::GetFormat(void) const
    {
        return this->matrix_interior_.GetFormat();
    }

    template <typename ValueType>
    const LocalMatrix<ValueType>& GlobalMatrix<ValueType>::GetInterior() const
    {
        log_debug(this, "GlobalMatrix::GetInterior() const");

        return this->matrix_interior_;
    }

    template <typename ValueType>
    const LocalMatrix<ValueType>& GlobalMatrix<ValueType>::GetGhost() const
    {
        log_debug(this, "GlobalMatrix::GetGhost()");

        return this->matrix_ghost_;
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::SetParallelManager(const ParallelManager& pm)
    {
        log_debug(this, "GlobalMatrix::SetParallelManager()", (const void*&)pm);

        assert(pm.Status() == true);

        this->pm_ = &pm;
        this->InitCommPattern_();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::AllocateCSR(const std::string& name,
                                              int64_t            local_nnz,
                                              int64_t            ghost_nnz)
    {
        log_debug(this, "GlobalMatrix::AllocateCSR()", name, local_nnz, ghost_nnz);

        assert(this->pm_ != NULL);
        assert(local_nnz > 0);
        assert(ghost_nnz >= 0);

        this->object_name_        = name;
        std::string interior_name = "Interior of " + name;
        std::string ghost_name    = "Ghost of " + name;

        this->matrix_interior_.AllocateCSR(
            interior_name, local_nnz, this->pm_->GetLocalNrow(), this->pm_->GetLocalNcol());
        this->matrix_ghost_.AllocateCSR(
            ghost_name, ghost_nnz, this->pm_->GetLocalNrow(), this->pm_->GetNumReceivers());

        // Initialize communication pattern
        this->InitCommPattern_();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::AllocateCOO(const std::string& name,
                                              int64_t            local_nnz,
                                              int64_t            ghost_nnz)
    {
        log_debug(this, "GlobalMatrix::AllocateCOO()", name, local_nnz, ghost_nnz);

        assert(this->pm_ != NULL);
        assert(local_nnz > 0);
        assert(ghost_nnz >= 0);

        this->object_name_        = name;
        std::string interior_name = "Interior of " + name;
        std::string ghost_name    = "Ghost of " + name;

        this->matrix_interior_.AllocateCOO(
            interior_name, local_nnz, this->pm_->GetLocalNrow(), this->pm_->GetLocalNcol());
        this->matrix_ghost_.AllocateCOO(
            ghost_name, ghost_nnz, this->pm_->GetLocalNrow(), this->pm_->GetNumReceivers());

        // Initialize communication pattern
        this->InitCommPattern_();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::SetDataPtrCSR(PtrType**   local_row_offset,
                                                int**       local_col,
                                                ValueType** local_val,
                                                PtrType**   ghost_row_offset,
                                                int**       ghost_col,
                                                ValueType** ghost_val,
                                                std::string name,
                                                int64_t     local_nnz,
                                                int64_t     ghost_nnz)
    {
        log_debug(this,
                  "GlobalMatrix::SetDataPtrCSR()",
                  local_row_offset,
                  local_col,
                  local_val,
                  ghost_row_offset,
                  ghost_col,
                  ghost_val,
                  name,
                  local_nnz,
                  ghost_nnz);

        assert(local_row_offset != NULL);
        assert(local_col != NULL);
        assert(local_val != NULL);

        assert(ghost_row_offset != NULL);
        assert(ghost_col != NULL);
        assert(ghost_val != NULL);

        assert(*local_row_offset != NULL);
        assert(*ghost_row_offset != NULL);

        assert(local_nnz >= 0);
        assert(ghost_nnz >= 0);

        if(local_nnz > 0)
        {
            assert(*local_col != NULL);
            assert(*local_val != NULL);
        }

        if(ghost_nnz > 0)
        {
            assert(*ghost_col != NULL);
            assert(*ghost_val != NULL);
        }

        if(*local_col == NULL)
        {
            assert(local_nnz == 0);
            assert(*local_val == NULL);
        }

        if(*local_val == NULL)
        {
            assert(local_nnz == 0);
            assert(*local_col == NULL);
        }

        if(*ghost_col == NULL)
        {
            assert(ghost_nnz == 0);
            assert(*ghost_val == NULL);
        }

        if(*ghost_val == NULL)
        {
            assert(ghost_nnz == 0);
            assert(*ghost_col == NULL);
        }

        assert(this->pm_ != NULL);

        this->Clear();

        this->object_name_        = name;
        std::string interior_name = "Interior of " + name;
        std::string ghost_name    = "Ghost of " + name;

        this->matrix_interior_.SetDataPtrCSR(local_row_offset,
                                             local_col,
                                             local_val,
                                             interior_name,
                                             local_nnz,
                                             this->pm_->GetLocalNrow(),
                                             this->pm_->GetLocalNcol());
        this->matrix_ghost_.SetDataPtrCSR(ghost_row_offset,
                                          ghost_col,
                                          ghost_val,
                                          ghost_name,
                                          ghost_nnz,
                                          this->pm_->GetLocalNrow(),
                                          this->pm_->GetNumReceivers());

        this->matrix_ghost_.ConvertTo(COO);

        // Initialize communication pattern
        this->InitCommPattern_();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::SetDataPtrCOO(int**       local_row,
                                                int**       local_col,
                                                ValueType** local_val,
                                                int**       ghost_row,
                                                int**       ghost_col,
                                                ValueType** ghost_val,
                                                std::string name,
                                                int64_t     local_nnz,
                                                int64_t     ghost_nnz)
    {
        log_debug(this,
                  "GlobalMatrix::SetDataPtrCOO()",
                  local_row,
                  local_col,
                  local_val,
                  ghost_row,
                  ghost_col,
                  ghost_val,
                  name,
                  local_nnz,
                  ghost_nnz);

        assert(local_row != NULL);
        assert(local_col != NULL);
        assert(local_val != NULL);

        assert(ghost_row != NULL);
        assert(ghost_col != NULL);
        assert(ghost_val != NULL);

        assert(*local_row != NULL);
        assert(*local_col != NULL);
        assert(*local_val != NULL);
        assert(local_nnz > 0);

        assert(*ghost_row != NULL);
        assert(*ghost_col != NULL);
        assert(*ghost_val != NULL);
        assert(ghost_nnz > 0);

        assert(this->pm_ != NULL);

        this->Clear();

        this->object_name_        = name;
        std::string interior_name = "Interior of " + name;
        std::string ghost_name    = "Ghost of " + name;

        this->matrix_interior_.SetDataPtrCOO(local_row,
                                             local_col,
                                             local_val,
                                             interior_name,
                                             local_nnz,
                                             this->pm_->GetLocalNrow(),
                                             this->pm_->GetLocalNcol());
        this->matrix_ghost_.SetDataPtrCOO(ghost_row,
                                          ghost_col,
                                          ghost_val,
                                          ghost_name,
                                          ghost_nnz,
                                          this->pm_->GetLocalNrow(),
                                          this->pm_->GetNumReceivers());

        // Initialize communication pattern
        this->InitCommPattern_();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::SetLocalDataPtrCSR(
        PtrType** row_offset, int** col, ValueType** val, std::string name, int64_t nnz)
    {
        log_debug(this, "GlobalMatrix::SetLocalDataPtrCSR()", row_offset, col, val, name, nnz);

        assert(row_offset != NULL);
        assert(col != NULL);
        assert(val != NULL);

        assert(*row_offset != NULL);
        assert(*col != NULL);
        assert(*val != NULL);

        assert(nnz > 0);

        assert(this->pm_ != NULL);

        this->object_name_        = name;
        std::string interior_name = "Interior of " + name;

        this->matrix_interior_.SetDataPtrCSR(row_offset,
                                             col,
                                             val,
                                             interior_name,
                                             nnz,
                                             this->pm_->GetLocalNrow(),
                                             this->pm_->GetLocalNcol());

        // Initialize communication pattern
        this->InitCommPattern_();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::SetLocalDataPtrCOO(
        int** row, int** col, ValueType** val, std::string name, int64_t nnz)
    {
        log_debug(this, "GlobalMatrix::SetLocalDataPtrCOO()", row, col, val, name, nnz);

        assert(row != NULL);
        assert(col != NULL);
        assert(val != NULL);

        assert(*row != NULL);
        assert(*col != NULL);
        assert(*val != NULL);

        assert(nnz > 0);

        assert(this->pm_ != NULL);

        this->object_name_        = name;
        std::string interior_name = "Interior of " + name;

        this->matrix_interior_.SetDataPtrCOO(row,
                                             col,
                                             val,
                                             interior_name,
                                             nnz,
                                             this->pm_->GetLocalNrow(),
                                             this->pm_->GetLocalNcol());

        // Initialize communication pattern
        this->InitCommPattern_();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::SetGhostDataPtrCSR(
        PtrType** row_offset, int** col, ValueType** val, std::string name, int64_t nnz)
    {
        log_debug(this, "GlobalMatrix::SetGhostDataPtrCSR()", row_offset, col, val, name, nnz);

        assert(nnz >= 0);
        assert(row_offset != NULL);
        assert(*row_offset != NULL);

        if(nnz > 0)
        {
            assert(col != NULL);
            assert(val != NULL);
            assert(*col != NULL);
            assert(*val != NULL);
        }

        assert(this->pm_ != NULL);

        std::string ghost_name = "Ghost of " + name;

        this->matrix_ghost_.SetDataPtrCSR(row_offset,
                                          col,
                                          val,
                                          ghost_name,
                                          nnz,
                                          this->pm_->GetLocalNrow(),
                                          this->pm_->GetNumReceivers());

        this->matrix_ghost_.ConvertTo(COO);

        // Initialize communication pattern
        this->InitCommPattern_();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::SetGhostDataPtrCOO(
        int** row, int** col, ValueType** val, std::string name, int64_t nnz)
    {
        log_debug(this, "GlobalMatrix::SetGhostDataPtrCOO()", row, col, val, name, nnz);

        assert(row != NULL);
        assert(col != NULL);
        assert(val != NULL);

        assert(*row != NULL);
        assert(*col != NULL);
        assert(*val != NULL);

        assert(nnz > 0);

        assert(this->pm_ != NULL);

        std::string ghost_name = "Ghost of " + name;

        this->matrix_ghost_.SetDataPtrCOO(row,
                                          col,
                                          val,
                                          ghost_name,
                                          nnz,
                                          this->pm_->GetLocalNrow(),
                                          this->pm_->GetNumReceivers());

        // Sort ghost matrix
        this->matrix_ghost_.Sort();

        // Initialize communication pattern
        this->InitCommPattern_();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::LeaveDataPtrCSR(PtrType**   local_row_offset,
                                                  int**       local_col,
                                                  ValueType** local_val,
                                                  PtrType**   ghost_row_offset,
                                                  int**       ghost_col,
                                                  ValueType** ghost_val)
    {
        log_debug(this,
                  "GlobalMatrix::LeaveDataPtrCSR()",
                  local_row_offset,
                  local_col,
                  local_val,
                  ghost_row_offset,
                  ghost_col,
                  ghost_val);

        assert(*local_row_offset == NULL);
        assert(*local_col == NULL);
        assert(*local_val == NULL);

        assert(*ghost_row_offset == NULL);
        assert(*ghost_col == NULL);
        assert(*ghost_val == NULL);

        assert(this->GetLocalM() > 0);
        assert(this->GetLocalN() > 0);
        assert(this->GetLocalNnz() > 0);

        assert(this->GetGhostM() > 0);
        assert(this->GetGhostN() > 0);
        assert(this->GetGhostNnz() > 0);

        this->matrix_interior_.LeaveDataPtrCSR(local_row_offset, local_col, local_val);
        this->matrix_ghost_.LeaveDataPtrCSR(ghost_row_offset, ghost_col, ghost_val);

        this->nnz_ = 0;
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::LeaveDataPtrCOO(int**       local_row,
                                                  int**       local_col,
                                                  ValueType** local_val,
                                                  int**       ghost_row,
                                                  int**       ghost_col,
                                                  ValueType** ghost_val)
    {
        log_debug(this,
                  "GlobalMatrix::LeaveDataPtrCOO()",
                  local_row,
                  local_col,
                  local_val,
                  ghost_row,
                  ghost_col,
                  ghost_val);

        assert(*local_row == NULL);
        assert(*local_col == NULL);
        assert(*local_val == NULL);

        assert(*ghost_row == NULL);
        assert(*ghost_col == NULL);
        assert(*ghost_val == NULL);

        assert(this->GetLocalM() > 0);
        assert(this->GetLocalN() > 0);
        assert(this->GetLocalNnz() > 0);

        assert(this->GetGhostM() > 0);
        assert(this->GetGhostN() > 0);
        assert(this->GetGhostNnz() > 0);

        this->matrix_interior_.LeaveDataPtrCOO(local_row, local_col, local_val);
        this->matrix_ghost_.LeaveDataPtrCOO(ghost_row, ghost_col, ghost_val);

        this->nnz_ = 0;
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::LeaveLocalDataPtrCSR(PtrType**   row_offset,
                                                       int**       col,
                                                       ValueType** val)
    {
        log_debug(this, "GlobalMatrix::LeaveLocalDataPtrCSR()", row_offset, col, val);

        assert(*row_offset == NULL);
        assert(*col == NULL);
        assert(*val == NULL);

        assert(this->GetLocalM() > 0);
        assert(this->GetLocalN() > 0);
        assert(this->GetLocalNnz() > 0);

        this->matrix_interior_.LeaveDataPtrCSR(row_offset, col, val);

        this->nnz_ = 0;
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::LeaveLocalDataPtrCOO(int** row, int** col, ValueType** val)
    {
        log_debug(this, "GlobalMatrix::LeaveLocalDataPtrCOO()", row, col, val);

        assert(*row == NULL);
        assert(*col == NULL);
        assert(*val == NULL);

        assert(this->GetLocalM() > 0);
        assert(this->GetLocalN() > 0);
        assert(this->GetLocalNnz() > 0);

        this->matrix_interior_.LeaveDataPtrCOO(row, col, val);

        this->nnz_ = 0;
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::LeaveGhostDataPtrCSR(PtrType**   row_offset,
                                                       int**       col,
                                                       ValueType** val)
    {
        log_debug(this, "GlobalMatrix::LeaveGhostDataPtrCSR()", row_offset, col, val);

        assert(*row_offset == NULL);
        assert(*col == NULL);
        assert(*val == NULL);

        assert(this->GetGhostM() > 0);
        assert(this->GetGhostN() > 0);
        assert(this->GetGhostNnz() > 0);

        this->matrix_ghost_.LeaveDataPtrCSR(row_offset, col, val);

        this->nnz_ = 0;
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::LeaveGhostDataPtrCOO(int** row, int** col, ValueType** val)
    {
        log_debug(this, "GlobalMatrix::LeaveGhostDataPtrCOO()", row, col, val);

        assert(*row == NULL);
        assert(*col == NULL);
        assert(*val == NULL);

        assert(this->GetGhostM() > 0);
        assert(this->GetGhostN() > 0);
        assert(this->GetGhostNnz() > 0);

        this->matrix_ghost_.LeaveDataPtrCOO(row, col, val);

        this->nnz_ = 0;
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::MoveToAccelerator(void)
    {
        log_debug(this, "GlobalMatrix::MoveToAccelerator()");

        this->matrix_interior_.MoveToAccelerator();
        this->matrix_ghost_.MoveToAccelerator();
        this->halo_.MoveToAccelerator();
        this->recv_buffer_.MoveToAccelerator();
        this->send_buffer_.MoveToAccelerator();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::MoveToHost(void)
    {
        log_debug(this, "GlobalMatrix::MoveToHost()");

        this->matrix_interior_.MoveToHost();
        this->matrix_ghost_.MoveToHost();
        this->halo_.MoveToHost();
        this->recv_buffer_.MoveToHost();
        this->send_buffer_.MoveToHost();
    }

    template <typename ValueType>
    bool GlobalMatrix<ValueType>::is_host_(void) const
    {
        assert(this->matrix_interior_.is_host_() == this->matrix_ghost_.is_host_());
        assert(this->matrix_interior_.is_host_() == this->halo_.is_host_());
        assert(this->matrix_interior_.is_host_() == this->recv_buffer_.is_host_());
        assert(this->matrix_interior_.is_host_() == this->send_buffer_.is_host_());
        return this->matrix_interior_.is_host_();
    }

    template <typename ValueType>
    bool GlobalMatrix<ValueType>::is_accel_(void) const
    {
        assert(this->matrix_interior_.is_accel_() == this->matrix_ghost_.is_accel_());
        assert(this->matrix_interior_.is_accel_() == this->halo_.is_accel_());
        assert(this->matrix_interior_.is_accel_() == this->recv_buffer_.is_accel_());
        assert(this->matrix_interior_.is_accel_() == this->send_buffer_.is_accel_());
        return this->matrix_interior_.is_accel_();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::Info(void) const
    {
        std::string current_backend_name;

        if(this->is_host_() == true)
        {
            current_backend_name = _rocalution_host_name[0];
        }
        else
        {
            assert(this->is_accel_() == true);
            current_backend_name = _rocalution_backend_name[this->local_backend_.backend];
        }

        std::string format = _matrix_format_names[this->GetFormat()];

        if(this->GetFormat() == CSR)
        {
            std::stringstream sstr;
            sstr << "(" << 8 * sizeof(PtrType) << "," << 8 * sizeof(int) << ")";
            format += sstr.str() + "/" + _matrix_format_names[this->matrix_ghost_.GetFormat()];
        }

        LOG_INFO("GlobalMatrix"
                 << " name=" << this->object_name_ << ";"
                 << " rows=" << this->GetM() << ";"
                 << " cols=" << this->GetN() << ";"
                 << " nnz=" << this->GetNnz() << ";"
                 << " prec=" << 8 * sizeof(ValueType) << "bit;"
                 << " format=" << format << ";"
                 << " subdomains=" << ((this->pm_ != NULL) ? this->pm_->num_procs_ : 1) << ";"
                 << " host backend={" << _rocalution_host_name[0] << "};"
                 << " accelerator backend={"
                 << _rocalution_backend_name[this->local_backend_.backend] << "};"
                 << " current=" << current_backend_name);
    }

    template <typename ValueType>
    bool GlobalMatrix<ValueType>::Check(void) const
    {
        log_debug(this, "GlobalMatrix::Check()");

        bool interior_check = this->matrix_interior_.Check();
        bool ghost_check    = this->matrix_ghost_.Check();

        if(interior_check == true && ghost_check == true)
        {
            return true;
        }

        return false;
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::CloneFrom(const GlobalMatrix<ValueType>& src)
    {
        log_debug(this, "GlobalMatrix::CloneFrom()");

        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::CopyFrom(const GlobalMatrix<ValueType>& src)
    {
        log_debug(this, "GlobalMatrix::CopyFrom()", (const void*&)src);

        assert(this != &src);
        assert(src.GetLocalNnz() != 0);
        assert(src.GetGhostNnz() != 0);
        assert(this->recv_boundary_ != NULL);
        assert(this->send_boundary_ != NULL);

        this->matrix_interior_.CopyFrom(src.GetInterior());
        this->matrix_ghost_.CopyFrom(src.GetGhost());

        this->object_name_ = "Copy from " + src.object_name_;
        this->pm_          = src.pm_;

        this->nnz_ = src.nnz_;
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::ConvertToCSR(void)
    {
        this->ConvertTo(CSR);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::ConvertToMCSR(void)
    {
        this->ConvertTo(MCSR);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::ConvertToBCSR(int blockdim)
    {
        this->ConvertTo(BCSR, blockdim);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::ConvertToCOO(void)
    {
        this->ConvertTo(COO);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::ConvertToELL(void)
    {
        this->ConvertTo(ELL);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::ConvertToDIA(void)
    {
        this->ConvertTo(DIA);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::ConvertToHYB(void)
    {
        this->ConvertTo(HYB);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::ConvertToDENSE(void)
    {
        this->ConvertTo(DENSE);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::ConvertTo(unsigned int matrix_format, int blockdim)
    {
        log_debug(this, "GlobalMatrix::ConverTo()", matrix_format, blockdim);

        this->matrix_interior_.ConvertTo(matrix_format, blockdim);

        // Ghost part remains COO
        this->matrix_ghost_.ConvertTo(COO);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::Apply(const GlobalVector<ValueType>& in,
                                        GlobalVector<ValueType>*       out) const
    {
        log_debug(this, "GlobalMatrix::Apply()", (const void*&)in, out);

        assert(out != NULL);
        assert(&in != out);

        // Calling global routine with single process
        if(this->pm_ == NULL)
        {
            // no PM, do interior apply
            this->matrix_interior_.Apply(in.vector_interior_, &out->vector_interior_);

            return;
        }

        assert(this->GetM() == out->GetSize());
        assert(this->GetN() == in.GetSize());
        assert(this->is_host_() == in.is_host_());
        assert(this->is_host_() == out->is_host_());
        assert(this->is_host_() == this->halo_.is_host_());
        assert(this->is_host_() == this->recv_buffer_.is_host_());
        assert(this->is_host_() == this->send_buffer_.is_host_());

        // Prepare send buffer
        in.vector_interior_.GetIndexValues(this->halo_, &this->send_buffer_);

        // Synchronize default stream
        _rocalution_sync_default();

        // Change to compute mode ghost
        _rocalution_compute_ghost();

        // Make send buffer available for communication
        ValueType* send_buffer = NULL;
        if(this->is_host_() == true)
        {
            // On host, we can directly use the host pointer
            this->send_buffer_.LeaveDataPtr(&send_buffer);
        }
        else
        {
            // On the accelerator, we need to (asynchronously) make the data
            // available on the host
            this->send_buffer_.GetContinuousValues(
                0, this->pm_->GetNumSenders(), this->send_boundary_);

            send_buffer = this->send_boundary_;
        }

        // Change to compute mode interior
        _rocalution_compute_interior();

        // Interior
        this->matrix_interior_.Apply(in.vector_interior_, &out->vector_interior_);

        // Synchronize compute mode ghost
        _rocalution_sync_ghost();

        // Initiate communication
        this->pm_->CommunicateAsync_(send_buffer, this->recv_boundary_);

        // Sync communication
        this->pm_->CommunicateSync_();

        if(this->is_host_() == true)
        {
            // On host, we need to set back the pointer into its structure
            this->send_buffer_.SetDataPtr(&send_buffer, "send buffer", this->pm_->GetNumSenders());
        }

        // Change to compute mode ghost
        _rocalution_compute_ghost();

        // Process receive buffer
        this->recv_buffer_.SetContinuousValues(
            0, this->pm_->GetNumReceivers(), this->recv_boundary_);

        // Change to compute mode default
        _rocalution_compute_default();

        // Ghost
        this->matrix_ghost_.ApplyAdd(
            this->recv_buffer_, static_cast<ValueType>(1), &out->vector_interior_);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::ApplyAdd(const GlobalVector<ValueType>& in,
                                           ValueType                      scalar,
                                           GlobalVector<ValueType>*       out) const
    {
        log_debug(this, "GlobalMatrix::ApplyAdd()", (const void*&)in, scalar, out);

        assert(out != NULL);
        assert(&in != out);

        assert(this->GetM() == out->GetSize());
        assert(this->GetN() == in.GetSize());
        assert(this->is_host_() == in.is_host_());
        assert(this->is_host_() == out->is_host_());

        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::Transpose(void)
    {
        log_debug(this, "GlobalMatrix::Transpose()");

        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::Transpose(GlobalMatrix<ValueType>* T) const
    {
        log_debug(this, "GlobalMatrix::Transpose()", T);

        assert(T != NULL);
        assert(T != this);
        assert(this->is_host_() == T->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif

        // Calling global routine with single process
        if(this->pm_ == NULL || this->pm_->num_procs_ == 1)
        {
            this->matrix_interior_.Transpose(&T->matrix_interior_);

            T->CreateParallelManager_();

            T->pm_self_->SetMPICommunicator(this->pm_->comm_);

            T->pm_self_->SetGlobalNrow(T->matrix_interior_.GetM());
            T->pm_self_->SetGlobalNcol(T->matrix_interior_.GetN());

            T->pm_self_->SetLocalNrow(T->matrix_interior_.GetM());
            T->pm_self_->SetLocalNcol(T->matrix_interior_.GetN());

            return;
        }

        if(this->GetNnz() > 0)
        {
            // Some dummy structures
            LocalVector<int64_t> global_zero;
            global_zero.CloneBackend(*this);

            // Transpose local matrices
            LocalMatrix<ValueType> T_ext;
            T_ext.CloneBackend(*this);

            // Transpose ghost part into T ext
            this->matrix_ghost_.Transpose(&T_ext);

            // Generate T ghost and parallel manager of T

            // T_ext and T ghost MUST be CSR
            assert(T_ext.GetFormat() == CSR);
            assert(T->matrix_ghost_.GetFormat() == CSR);

            // Now we need to extract rows that do not belong to us and thus
            // need to be sent to neighboring processes
            int64_t nrows_send = T_ext.GetM();
            int     nrows_recv = this->pm_->GetNumSenders();

            // Extract row nnz for the rows that we have to send to neighbors
            LocalVector<PtrType> T_ext_row_ptr_send;
            T_ext_row_ptr_send.CloneBackend(*this);
            T_ext_row_ptr_send.Allocate("row nnz send", nrows_send);

            // send number of nonzeros that will be in the ghost restriction rows
            T_ext.matrix_->ExtractExtRowNnz(0, T_ext_row_ptr_send.vector_);

            // Prepare send / receive buffers
            PtrType* pT_ext_row_nnz_send = NULL;
            PtrType* pT_ext_row_ptr_recv = NULL;

            LocalVector<PtrType> T_ext_row_ptr_recv;
#ifdef SUPPORT_RDMA
            T_ext_row_ptr_recv.CloneBackend(*this);
#endif
            T_ext_row_ptr_recv.Allocate("row ptr recv", nrows_recv + 1);

#ifndef SUPPORT_RDMA
            T_ext_row_ptr_send.MoveToHost();
#endif

            // Get host pointers for communication
            T_ext_row_ptr_recv.LeaveDataPtr(&pT_ext_row_ptr_recv);
            T_ext_row_ptr_send.LeaveDataPtr(&pT_ext_row_nnz_send);

            // Initiate communication of nnz per row
            this->pm_->InverseCommunicateAsync_(pT_ext_row_nnz_send, pT_ext_row_ptr_recv);

            // Extract column indices and transform them to global indices (for sending)
            LocalVector<int64_t> T_ext_col_ind_send;
            T_ext_col_ind_send.CloneBackend(*this);
            T_ext_col_ind_send.Allocate("col ind send", T_ext.GetNnz());

            T_ext.matrix_->ExtractGlobalColumnIndices(T_ext.GetN(),
                                                      this->pm_->GetGlobalRowBegin(),
                                                      *global_zero.vector_,
                                                      T_ext_col_ind_send.vector_);

            // Wait for nnz per row communication to finish
            this->pm_->InverseCommunicateSync_();

            // Clean up
            free_host(&pT_ext_row_nnz_send);

            // Put receive buffer back in structure
            T_ext_row_ptr_recv.SetDataPtr(&pT_ext_row_ptr_recv, "row ptr recv", nrows_recv + 1);

#ifndef SUPPORT_RDMA
            T_ext_row_ptr_recv.CloneBackend(*this);
#endif

            // Exclusive sum to obtain recv row pointers
            PtrType nnz_recv = T_ext_row_ptr_recv.ExclusiveSum();

            // Prepare receive buffers
            int64_t*   pT_ext_col_ind_recv = NULL;
            ValueType* pT_ext_val_recv     = NULL;

            LocalVector<int64_t>   T_ext_col_ind_recv;
            LocalVector<ValueType> T_ext_val_recv;

#ifdef SUPPORT_RDMA
            T_ext_col_ind_recv.CloneBackend(*this);
            T_ext_val_recv.CloneBackend(*this);
#endif

            T_ext_col_ind_recv.Allocate("col ind recv", nnz_recv);
            T_ext_val_recv.Allocate("val recv", nnz_recv);

            T_ext_col_ind_recv.LeaveDataPtr(&pT_ext_col_ind_recv);
            T_ext_val_recv.LeaveDataPtr(&pT_ext_val_recv);

            // Prepare send buffers
            int64_t*   pT_ext_col_ind_send_global = NULL;
            PtrType*   pT_ext_row_ptr_send        = NULL;
            int*       pT_ext_col_ind_send        = NULL;
            ValueType* pT_ext_val_send            = NULL;

#ifndef SUPPORT_RDMA
            T_ext_row_ptr_recv.MoveToHost();
            T_ext_col_ind_send.MoveToHost();
            T_ext.MoveToHost();
#endif

            T_ext_row_ptr_recv.LeaveDataPtr(&pT_ext_row_ptr_recv);
            T_ext_col_ind_send.LeaveDataPtr(&pT_ext_col_ind_send_global);
            T_ext.LeaveDataPtrCSR(&pT_ext_row_ptr_send, &pT_ext_col_ind_send, &pT_ext_val_send);

            // Initiate communication of column indices and values
            this->pm_->InverseCommunicateCSRAsync_(pT_ext_row_ptr_send,
                                                   pT_ext_col_ind_send_global,
                                                   pT_ext_val_send,
                                                   pT_ext_row_ptr_recv,
                                                   pT_ext_col_ind_recv,
                                                   pT_ext_val_recv);

            // Perform interior transpose while we do the communication
            this->matrix_interior_.Transpose(&T->matrix_interior_);

            // Wait for additional column indices and values communication to finish
            this->pm_->InverseCommunicateCSRSync_();

            // Now we can clear T ext related send buffers
            free_host(&pT_ext_row_ptr_send);
            free_host(&pT_ext_col_ind_send);
            free_host(&pT_ext_val_send);
            free_host(&pT_ext_col_ind_send_global);

            // Setup manager of T

            // Start with a clean manager
            T->CreateParallelManager_();

            // Same communicator as initial matrix
            T->pm_self_->SetMPICommunicator(this->pm_->comm_);

            // Sizes of the transposed
            T->pm_self_->SetGlobalNrow(this->pm_->global_ncol_);
            T->pm_self_->SetGlobalNcol(this->pm_->global_nrow_);
            T->pm_self_->SetLocalNrow(this->pm_->local_ncol_);
            T->pm_self_->SetLocalNcol(this->pm_->local_nrow_);

            // Put the global column indices back into structure
            T_ext_col_ind_recv.SetDataPtr(&pT_ext_col_ind_recv, "col ind recv", nnz_recv);
            T_ext_col_ind_recv.CloneBackend(*this);

            // To generate the parallel manager, we need to access the sorted global ghost column ids
            LocalVector<int64_t> sorted_ghost_col;
            sorted_ghost_col.CloneBackend(*this);
            sorted_ghost_col.Allocate("sorted global ghost columns", T_ext_col_ind_recv.GetSize());

            // Sort the global ghost columns (we do not need the permutation vector)
            T_ext_col_ind_recv.Sort(&sorted_ghost_col, NULL);

            // Get the sorted ghost columns on host
            int64_t* pghost_col = NULL;
            sorted_ghost_col.MoveToHost();
            sorted_ghost_col.LeaveDataPtr(&pghost_col);

            // Generate the manager from ghost of T and manager of non-transposed
            T->pm_self_->GenerateFromGhostColumnsWithParent_(
                nnz_recv, pghost_col, *this->pm_, true);

            // Communicate global offsets
            T->pm_self_->CommunicateGlobalOffsetAsync_();

            // Clear
            free_host(&pghost_col);

            // Wrap received data into structure
            T_ext_row_ptr_recv.SetDataPtr(&pT_ext_row_ptr_recv, "row ptr recv", nrows_recv + 1);
            T_ext_val_recv.SetDataPtr(&pT_ext_val_recv, "val recv", nnz_recv);

#ifndef SUPPORT_RDMA
            T_ext_row_ptr_recv.CloneBackend(*this);
            T_ext_val_recv.CloneBackend(*this);
#endif

            T->pm_self_->CommunicateGlobalOffsetSync_();

            // Convert global boundary index into local index
            T->pm_self_->BoundaryTransformGlobalToLocal_();

            // Communicate ghost to global map
            T->pm_self_->CommunicateGhostToGlobalMapAsync_();

            // Allocate T ghost
            T->matrix_ghost_.AllocateCSR(
                "ghost", nnz_recv, T->pm_self_->local_nrow_, T->pm_self_->local_ncol_);

            // Finally generate the ghost of T (global columns)
            LocalVector<int64_t> T_ext_cols;
            T_ext_cols.CloneBackend(*this);

            T->matrix_ghost_.matrix_->CopyGhostFromGlobalReceive(*this->halo_.vector_,
                                                                 *T_ext_row_ptr_recv.vector_,
                                                                 *T_ext_col_ind_recv.vector_,
                                                                 *T_ext_val_recv.vector_,
                                                                 T_ext_cols.vector_);

            // Transform global to local ghost columns
            T->matrix_ghost_.matrix_->RenumberGlobalToLocal(*T_ext_cols.vector_);

            // Synchronize
            T->pm_self_->CommunicateGhostToGlobalMapSync_();

            T->SetParallelManager(*T->pm_self_);
        }

#ifdef DEBUG_MODE
        T->Check();
#endif
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::TripleMatrixProduct(const GlobalMatrix<ValueType>& R,
                                                      const GlobalMatrix<ValueType>& A,
                                                      const GlobalMatrix<ValueType>& P)
    {
        log_debug(this,
                  "GlobalMatrix::TripleMatrixProduct()",
                  (const void*&)R,
                  (const void*&)A,
                  (const void*&)P);

        assert(&R != this);
        assert(&A != this);
        assert(&P != this);

        assert(R.GetN() == A.GetM());
        assert(A.GetN() == P.GetM());
        assert(this->is_host_() == R.is_host_());
        assert(this->is_host_() == A.is_host_());
        assert(this->is_host_() == P.is_host_());

#ifdef DEBUG_MODE
        R.Check();
        A.Check();
        P.Check();
#endif

        // Calling global routine with single process
        if(this->pm_ == NULL || this->pm_->num_procs_ == 1)
        {
            this->matrix_interior_.TripleMatrixProduct(
                R.matrix_interior_, A.matrix_interior_, P.matrix_interior_);

            this->CreateParallelManager_();

            this->pm_self_->SetMPICommunicator(A.pm_->comm_);

            this->pm_self_->SetGlobalNrow(this->matrix_interior_.GetM());
            this->pm_self_->SetGlobalNcol(this->matrix_interior_.GetN());

            this->pm_self_->SetLocalNrow(this->matrix_interior_.GetM());
            this->pm_self_->SetLocalNcol(this->matrix_interior_.GetN());

            return;
        }

        // Only CSR matrices are supported
        LocalMatrix<ValueType> R_int;
        LocalMatrix<ValueType> R_gst;
        LocalMatrix<ValueType> A_int;
        LocalMatrix<ValueType> A_gst;
        LocalMatrix<ValueType> P_int;
        LocalMatrix<ValueType> P_gst;

        const LocalMatrix<ValueType>* R_int_ptr = &R.matrix_interior_;
        const LocalMatrix<ValueType>* R_gst_ptr = &R.matrix_ghost_;
        const LocalMatrix<ValueType>* A_int_ptr = &A.matrix_interior_;
        const LocalMatrix<ValueType>* A_gst_ptr = &A.matrix_ghost_;
        const LocalMatrix<ValueType>* P_int_ptr = &P.matrix_interior_;
        const LocalMatrix<ValueType>* P_gst_ptr = &P.matrix_ghost_;

        if(R_int_ptr->GetFormat() != CSR)
        {
            R_int.CloneFrom(*R_int_ptr);
            R_int.ConvertToCSR();
            R_int_ptr = &R_int;
        }

        if(A_int_ptr->GetFormat() != CSR)
        {
            A_int.CloneFrom(*A_int_ptr);
            A_int.ConvertToCSR();
            A_int_ptr = &A_int;
        }

        if(A_gst_ptr->GetFormat() != CSR)
        {
            A_gst.CloneFrom(*A_gst_ptr);
            A_gst.ConvertToCSR();
            A_gst_ptr = &A_gst;
        }

        if(P_int_ptr->GetFormat() != CSR)
        {
            P_int.CloneFrom(*P_int_ptr);
            P_int.ConvertToCSR();
            P_int_ptr = &P_int;
        }

        if(P_gst_ptr->GetFormat() != CSR)
        {
            P_gst.CloneFrom(*P_gst_ptr);
            P_gst.ConvertToCSR();
            P_gst_ptr = &P_gst;
        }

        // Convert to CSR
        unsigned int format   = this->GetFormat();
        int          blockdim = this->matrix_interior_.GetBlockDimension();
        this->ConvertToCSR();

        // Fetch boundary rows of P
        // To compute AP = A * P we need to fetch additional rows of P, such that
        // we can locally access all rows that do not belong to the current rank,
        // but are required due to column dependencies.

        // This means, we need to send all rows of P where A has a dependency on
        // to the neighboring rank
        int P_ext_m_send = A.pm_->GetNumSenders();
        // Number of rows of P we receive from our neighbors
        int P_ext_m_recv = A.pm_->GetNumReceivers();

        // First, count the total number of non-zero entries per boundary row, including
        // ghost part (we need to send full rows)
        LocalVector<PtrType> P_ext_row_ptr_send;
        P_ext_row_ptr_send.CloneBackend(*this);
        P_ext_row_ptr_send.Allocate("P_ext_row_ptr_send", P_ext_m_send + 1);
        P_int_ptr->matrix_->ExtractBoundaryRowNnz(
            P_ext_row_ptr_send.vector_, *A.halo_.vector_, *P_gst_ptr->matrix_);

        // We need the send buffer on host for communication
        PtrType* hP_ext_row_nnz_send = NULL;
        allocate_host(P_ext_m_send + 1, &hP_ext_row_nnz_send);
        P_ext_row_ptr_send.CopyToHostData(hP_ext_row_nnz_send);

        // Receive buffer
        PtrType* hP_ext_row_ptr_recv = NULL;
        allocate_host(P_ext_m_recv + 1, &hP_ext_row_ptr_recv);

        // Initiate communication of nnz per row
        A.pm_->CommunicateAsync_(hP_ext_row_nnz_send, hP_ext_row_ptr_recv);

        // Exclusive sum to obtain row pointers of send buffer
        PtrType P_ext_nnz_send = P_ext_row_ptr_send.ExclusiveSum();

        // We need a copy of P_ext_row_ptr on host for the communication
        PtrType* hP_ext_row_ptr_send = NULL;
        allocate_host(P_ext_m_send + 1, &hP_ext_row_ptr_send);
        P_ext_row_ptr_send.CopyToHostData(hP_ext_row_ptr_send);

        // Extract the ghost mapping of P
        LocalVector<int64_t> P_mapping;
        P_mapping.CloneBackend(*this);
        P_mapping.Allocate("P ghost mapping", P.pm_->GetNumReceivers());
        const int64_t* ghost_mapping = P.pm_->GetGhostToGlobalMap();
        P_mapping.CopyFromHostData(ghost_mapping);

        // Now, extract boundary column indices and values
        LocalVector<int64_t>   P_ext_col_ind_send;
        LocalVector<ValueType> P_ext_val_send;

        P_ext_col_ind_send.CloneBackend(*this);
        P_ext_val_send.CloneBackend(*this);

        P_ext_col_ind_send.Allocate("P_ext_col_ind_send", P_ext_nnz_send);
        P_ext_val_send.Allocate("P_ext_val_send", P_ext_nnz_send);

        P_int_ptr->matrix_->ExtractBoundaryRows(*P_ext_row_ptr_send.vector_,
                                                P_ext_col_ind_send.vector_,
                                                P_ext_val_send.vector_,
                                                P.pm_->GetGlobalColumnBegin(),
                                                *A.halo_.vector_,
                                                *P_mapping.vector_,
                                                *P_gst_ptr->matrix_);

        // We do not need this anymore
        P_ext_row_ptr_send.Clear();

        // Wait for nnz per row communication to finish
        A.pm_->CommunicateSync_();

        // Clean up
        free_host(&hP_ext_row_nnz_send);

        // Obtain row pointer array
        // We can do this on the host, as the number of external rows is typically very small
        // and skips h2d followed by d2h copy
        LocalVector<PtrType> P_ext_row_ptr_recv;
        P_ext_row_ptr_recv.SetDataPtr(&hP_ext_row_ptr_recv, "P_ext_row_ptr_recv", P_ext_m_recv + 1);

        // Exclusive sum
        PtrType P_ext_nnz_recv = P_ext_row_ptr_recv.ExclusiveSum();

        // We need a copy of A_ext_row_ptr on host for the communication
        P_ext_row_ptr_recv.LeaveDataPtr(&hP_ext_row_ptr_recv);

        // We need a copy of P_ext on host for the communication
        int64_t*   hP_ext_col_ind_send = NULL;
        ValueType* hP_ext_val_send     = NULL;

        allocate_host(P_ext_nnz_send, &hP_ext_col_ind_send);
        allocate_host(P_ext_nnz_send, &hP_ext_val_send);

        P_ext_col_ind_send.CopyToHostData(hP_ext_col_ind_send);
        P_ext_val_send.CopyToHostData(hP_ext_val_send);

        // Clean up
        P_ext_col_ind_send.Clear();
        P_ext_val_send.Clear();

        // Receive buffers
        int64_t*   hP_ext_col_ind_recv_global = NULL;
        ValueType* hP_ext_val_recv            = NULL;

        allocate_host(P_ext_nnz_recv, &hP_ext_col_ind_recv_global);
        allocate_host(P_ext_nnz_recv, &hP_ext_val_recv);

        // Initiate communication of column indices and values
        A.pm_->CommunicateCSRAsync_(hP_ext_row_ptr_send,
                                    hP_ext_col_ind_send,
                                    hP_ext_val_send,
                                    hP_ext_row_ptr_recv,
                                    hP_ext_col_ind_recv_global,
                                    hP_ext_val_recv);

        // Now, A interior need to be merged with its ghost part to obtain A_full
        int64_t A_full_m   = A_int_ptr->GetM();
        int64_t A_full_n   = A_int_ptr->GetN() + A_gst_ptr->GetN();
        int64_t A_full_nnz = A_int_ptr->GetNnz() + A_gst_ptr->GetNnz();

        LocalMatrix<ValueType> zero_matrix;
        LocalVector<int>       zero_vector;
        LocalVector<int64_t>   zero_vector64;
        zero_matrix.CloneBackend(*this);
        zero_vector.CloneBackend(*this);
        zero_vector64.CloneBackend(*this);

        LocalMatrix<ValueType> A_full;
        A_full.CloneBackend(*this);
        A_full.AllocateCSR("A full", A_full_nnz, A_full_m, A_full_n);
        A_full.matrix_->MergeToLocal(
            *A_int_ptr->matrix_, *A_gst_ptr->matrix_, *zero_matrix.matrix_, *zero_vector.vector_);

        // Wait for additional column indices and values communication from P to finish
        A.pm_->CommunicateCSRSync_();

        // Clean up
        free_host(&hP_ext_row_ptr_send);
        free_host(&hP_ext_col_ind_send);
        free_host(&hP_ext_val_send);

        // Wrap received data into structure

        // Combine P ghost with the ghost part of the additional rows from neighbors
        // in order to match the column indices when doing SpGEMM on the ghost part
        LocalVector<int>     P_gst_ext_local;
        LocalVector<int64_t> l2g;

        P_gst_ext_local.CloneBackend(*this);
        l2g.CloneBackend(*this);

        LocalVector<int> P_local_col;
        P_local_col.CloneBackend(*this);
        P_local_col.Allocate("local col", P_ext_nnz_recv);

        LocalVector<int64_t> P_ext_col_ind_recv;
        P_ext_col_ind_recv.SetDataPtr(&hP_ext_col_ind_recv_global, "", P_ext_nnz_recv);
        P_ext_col_ind_recv.CloneBackend(*this);
        P_gst_ptr->matrix_->CombineAndRenumber(P_int_ptr->GetN(),
                                               P_ext_nnz_recv,
                                               P.pm_->GetGlobalColumnBegin(),
                                               P.pm_->GetGlobalColumnEnd(),
                                               *P_mapping.vector_,
                                               *P_ext_col_ind_recv.vector_,
                                               P_gst_ext_local.vector_,
                                               l2g.vector_,
                                               P_local_col.vector_);

        // Wrap P ext into structure
        int* hP_ext_col_ind_recv = NULL;
        P_local_col.MoveToHost();
        P_local_col.LeaveDataPtr(&hP_ext_col_ind_recv);

        LocalMatrix<ValueType> P_ext;
        P_ext.SetDataPtrCSR(&hP_ext_row_ptr_recv,
                            &hP_ext_col_ind_recv,
                            &hP_ext_val_recv,
                            "P ext",
                            P_ext_nnz_recv,
                            P_ext_m_recv,
                            INT32_MAX);
        P_ext.CloneBackend(*this);

        // P_ext can potentially be unsorted, which might cause problems later on
        P_ext.Sort();

        int64_t P_full_ghost_n = l2g.GetSize();

        // Merge P with the additional rows of P_ext in order to do the SpGEMM with A
        LocalMatrix<ValueType> P_full;
        P_full.CloneBackend(*this);
        P_full.AllocateCSR("P full",
                           P_int_ptr->GetNnz() + P_gst_ptr->GetNnz() + P_ext_nnz_recv,
                           P_int_ptr->GetM() + P_ext_m_recv,
                           P_int_ptr->GetN() + P_full_ghost_n);
        P_full.matrix_->MergeToLocal(
            *P_int_ptr->matrix_, *P_gst_ptr->matrix_, *P_ext.matrix_, *P_gst_ext_local.vector_);

        P_ext.Clear();
        P_gst_ext_local.Clear();

        // Compute AP = A_full * P_full
        LocalMatrix<ValueType> AP;
        AP.CloneBackend(*this);

        AP.MatrixMult(A_full, P_full);

        // Transpose ghost of P to obtain R ext
        LocalMatrix<ValueType> R_ext;
        R_ext.CloneBackend(*this);
        P_gst_ptr->Transpose(&R_ext);

        // Merge R interior and ghost into R full
        LocalMatrix<ValueType> R_full;
        R_full.CloneBackend(*this);
        R_full.AllocateCSR("R full",
                           P_int_ptr->GetNnz() + P_gst_ptr->GetNnz(),
                           P_int_ptr->GetN() + P_gst_ptr->GetN(),
                           P_int_ptr->GetM());
        R_full.matrix_->MergeToLocal(
            *R_int_ptr->matrix_, *zero_matrix.matrix_, *R_ext.matrix_, *zero_vector.vector_);

        // Compute RAP_full = R_full * AP
        LocalMatrix<ValueType> RAP_full;
        RAP_full.CloneBackend(*this);
        RAP_full.MatrixMult(R_full, AP);

        // Now we need to extract rows that need to be sent to neighboring processes

        // Extract RAP_ext from RAP_full
        int64_t RAP_ext_m_send = RAP_full.GetLocalM() - P_int_ptr->GetN();
        int64_t RAP_ext_m_recv = P.pm_->GetNumSenders();

        LocalVector<PtrType> RAP_ext_row_ptr_send;
        RAP_ext_row_ptr_send.CloneBackend(*this);
        RAP_ext_row_ptr_send.Allocate("RAP ext row ptr send", RAP_ext_m_send + 1);

        RAP_full.matrix_->ExtractExtRowNnz(P_int_ptr->GetN(), RAP_ext_row_ptr_send.vector_);

        // Communication buffers
        PtrType* hRAP_ext_row_nnz_send = NULL;
        PtrType* hRAP_ext_row_ptr_recv = NULL;

        allocate_host(RAP_ext_m_send + 1, &hRAP_ext_row_nnz_send);
        allocate_host(RAP_ext_m_recv + 1, &hRAP_ext_row_ptr_recv);

        RAP_ext_row_ptr_send.CopyToHostData(hRAP_ext_row_nnz_send);

        // Initiate communication of nnz per row
        P.pm_->InverseCommunicateAsync_(hRAP_ext_row_nnz_send, hRAP_ext_row_ptr_recv + 1);

        // Extract RAP ext
        LocalMatrix<ValueType> RAP_ext;
        RAP_ext.CloneBackend(*this);
        RAP_full.ExtractSubMatrix(P_int_ptr->GetN(),
                                  0,
                                  RAP_full.GetLocalM() - P_int_ptr->GetN(),
                                  RAP_full.GetLocalN(),
                                  &RAP_ext);

        // Set new sizes for local RAP ext
        int64_t RAP_local_n   = RAP_full.GetLocalN();
        PtrType RAP_local_nnz = RAP_full.GetLocalNnz() - RAP_ext.GetLocalNnz();

        PtrType*   RAP_csr_row_ptr = NULL;
        int*       RAP_csr_col_ind = NULL;
        ValueType* RAP_csr_val     = NULL;

        RAP_full.LeaveDataPtrCSR(&RAP_csr_row_ptr, &RAP_csr_col_ind, &RAP_csr_val);
        RAP_full.SetDataPtrCSR(&RAP_csr_row_ptr,
                               &RAP_csr_col_ind,
                               &RAP_csr_val,
                               "RAP",
                               RAP_local_nnz,
                               P_int_ptr->GetN(),
                               RAP_local_n);

        // Extract column indices and transform them to global indices (for sending)
        LocalVector<int64_t> RAP_ext_col_ind_send;
        RAP_ext_col_ind_send.CloneBackend(*this);
        RAP_ext_col_ind_send.Allocate("col ind send", RAP_ext.GetNnz());

        RAP_ext.matrix_->ExtractGlobalColumnIndices(P_int_ptr->GetN(),
                                                    P.pm_->GetGlobalColumnBegin(),
                                                    *l2g.vector_,
                                                    RAP_ext_col_ind_send.vector_);

        // Communication buffers
        PtrType*   hRAP_ext_row_ptr_send        = NULL;
        int*       hRAP_ext_col_ind_send        = NULL;
        ValueType* hRAP_ext_val_send            = NULL;
        int64_t*   hRAP_ext_col_ind_send_global = NULL;

        RAP_ext.MoveToHost();
        RAP_ext.LeaveDataPtrCSR(&hRAP_ext_row_ptr_send, &hRAP_ext_col_ind_send, &hRAP_ext_val_send);
        RAP_ext_col_ind_send.MoveToHost();
        RAP_ext_col_ind_send.LeaveDataPtr(&hRAP_ext_col_ind_send_global);

        // Wait for nnz per row communication to finish
        P.pm_->InverseCommunicateSync_();

        // Clean up
        free_host(&hRAP_ext_row_nnz_send);

        // Exclusive sum to obtain recv row pointers
        hRAP_ext_row_ptr_recv[0] = 0;
        for(int64_t i = 0; i < RAP_ext_m_recv; ++i)
        {
            hRAP_ext_row_ptr_recv[i + 1] += hRAP_ext_row_ptr_recv[i];
        }

        // Initiate communication of column indices and values
        PtrType RAP_ext_nnz_recv = hRAP_ext_row_ptr_recv[RAP_ext_m_recv];

        int64_t*   hRAP_ext_col_ind_recv = NULL;
        ValueType* hRAP_ext_val_recv     = NULL;

        allocate_host(RAP_ext_nnz_recv, &hRAP_ext_col_ind_recv);
        allocate_host(RAP_ext_nnz_recv, &hRAP_ext_val_recv);

        P.pm_->InverseCommunicateCSRAsync_(hRAP_ext_row_ptr_send,
                                           hRAP_ext_col_ind_send_global,
                                           hRAP_ext_val_send,
                                           hRAP_ext_row_ptr_recv,
                                           hRAP_ext_col_ind_recv,
                                           hRAP_ext_val_recv);

        // Split RAP into interior and ghost
        LocalMatrix<ValueType> RAP_interior;
        LocalMatrix<ValueType> RAP_ghost;

        RAP_interior.CloneBackend(RAP_full);
        RAP_ghost.CloneBackend(RAP_full);

        RAP_full.matrix_->SplitInteriorGhost(RAP_interior.matrix_, RAP_ghost.matrix_);

        // Wait for additional column indices and values communication to finish
        P.pm_->InverseCommunicateCSRSync_();

        // Clean up
        free_host(&hRAP_ext_row_ptr_send);
        free_host(&hRAP_ext_col_ind_send);
        free_host(&hRAP_ext_val_send);
        free_host(&hRAP_ext_col_ind_send_global);

        LocalMatrix<ValueType> RAP_ext_interior;
        LocalMatrix<ValueType> RAP_ext_ghost;

        RAP_ext_interior.CloneBackend(*this);
        RAP_ext_ghost.CloneBackend(*this);

        // Temporary array to store global ghost columns
        LocalVector<int64_t> ghost_col;
        ghost_col.CloneBackend(*this);

        LocalVector<PtrType>   RAP_ext_row_ptr_recv;
        LocalVector<int64_t>   RAP_ext_col_ind_recv;
        LocalVector<ValueType> RAP_ext_val_recv;

        RAP_ext_row_ptr_recv.SetDataPtr(&hRAP_ext_row_ptr_recv, "", RAP_ext_m_recv + 1);
        RAP_ext_col_ind_recv.SetDataPtr(&hRAP_ext_col_ind_recv, "", RAP_ext_nnz_recv);
        RAP_ext_val_recv.SetDataPtr(&hRAP_ext_val_recv, "", RAP_ext_nnz_recv);

        RAP_ext_row_ptr_recv.CloneBackend(*this);
        RAP_ext_col_ind_recv.CloneBackend(*this);
        RAP_ext_val_recv.CloneBackend(*this);

        RAP_ext_interior.matrix_->CopyFromGlobalReceive(P_int_ptr->GetN(),
                                                        P.pm_->GetGlobalColumnBegin(),
                                                        P.pm_->GetGlobalColumnEnd(),
                                                        *P.halo_.vector_,
                                                        *RAP_ext_row_ptr_recv.vector_,
                                                        *RAP_ext_col_ind_recv.vector_,
                                                        *RAP_ext_val_recv.vector_,
                                                        RAP_ext_ghost.matrix_,
                                                        ghost_col.vector_);

        // Merged global ghost columns
        LocalVector<int64_t> merged_col;
        merged_col.CloneBackend(*this);

        // Merge RAP ghost with RAP ext
        RAP_ghost.CompressAdd(l2g, ghost_col, RAP_ext_ghost, &merged_col);

        // Merge RAP interior with RAP ext
        RAP_interior.CompressAdd(zero_vector64, zero_vector64, RAP_ext_interior, NULL);

        // Communicate local sizes
        int64_t global_nrow;
        int64_t local_nrow = RAP_interior.GetLocalM();

#ifdef SUPPORT_MULTINODE
        MRequest req;
        communication_async_allreduce_single_sum(&local_nrow, &global_nrow, P.pm_->comm_, &req);
#endif

        // Generate PM for RAP
        this->CreateParallelManager_();
        this->pm_self_->SetMPICommunicator(P.pm_->comm_);

        // To generate the parallel manager, we need to access the sorted global ghost column ids
        LocalVector<int64_t> sorted_ghost_col;
        sorted_ghost_col.CloneBackend(*this);
        sorted_ghost_col.Allocate("sorted global ghost columns", merged_col.GetSize());

        // Sort the global ghost columns (we do not need the permutation vector)
        merged_col.Sort(&sorted_ghost_col, NULL);

        // Get the sorted ghost columns on host
        int64_t* pghost_col = NULL;
        sorted_ghost_col.MoveToHost();
        sorted_ghost_col.LeaveDataPtr(&pghost_col);

#ifdef SUPPORT_MULTINODE
        communication_sync(&req);
#endif

        // RAP is a square matrix, we can skip the column allreduce
        this->pm_self_->SetGlobalNrow(global_nrow);
        this->pm_self_->SetGlobalNcol(global_nrow);

        this->pm_self_->SetLocalNrow(RAP_interior.GetLocalM());
        this->pm_self_->SetLocalNcol(RAP_interior.GetLocalN());

        // Generate the manager from ghost of RAP and manager of P
        this->pm_self_->GenerateFromGhostColumnsWithParent_(RAP_ghost.GetNnz(), pghost_col, *P.pm_);

        // Communicate global offsets
        this->pm_self_->CommunicateGlobalOffsetAsync_();

        // Clear
        free_host(&pghost_col);

        // Sync global offsets communication
        this->pm_self_->CommunicateGlobalOffsetSync_();

        // Convert global boundary index to local index
        this->pm_self_->BoundaryTransformGlobalToLocal_();

        // Communicate ghost to global map
        this->pm_self_->CommunicateGhostToGlobalMapAsync_();

        // Renumber ghost columns (from global to local)
        RAP_ghost.matrix_->RenumberGlobalToLocal(*merged_col.vector_);

        // Synchronize
        this->pm_self_->CommunicateGhostToGlobalMapSync_();

        this->SetParallelManager(*this->pm_self_);

        int64_t RAP_ghost_nnz = RAP_ghost.GetLocalNnz();

        PtrType*   hRAP_ghost_csr_row_ptr = NULL;
        int*       hRAP_ghost_csr_col_ind = NULL;
        ValueType* hRAP_ghost_csr_val     = NULL;

        RAP_ghost.LeaveDataPtrCSR(
            &hRAP_ghost_csr_row_ptr, &hRAP_ghost_csr_col_ind, &hRAP_ghost_csr_val);

        // Obtain raw pointers
        int64_t RAP_interior_nnz = RAP_interior.GetLocalNnz();

        PtrType*   RAP_interior_csr_row_ptr = NULL;
        int*       RAP_interior_csr_col_ind = NULL;
        ValueType* RAP_interior_csr_val     = NULL;

        RAP_interior.LeaveDataPtrCSR(
            &RAP_interior_csr_row_ptr, &RAP_interior_csr_col_ind, &RAP_interior_csr_val);

        // Set global coarse matrix
        this->SetDataPtrCSR(&RAP_interior_csr_row_ptr,
                            &RAP_interior_csr_col_ind,
                            &RAP_interior_csr_val,
                            &hRAP_ghost_csr_row_ptr,
                            &hRAP_ghost_csr_col_ind,
                            &hRAP_ghost_csr_val,
                            "RAP",
                            RAP_interior_nnz,
                            RAP_ghost_nnz);

        if(format != CSR || R.GetFormat() != CSR || A.GetFormat() != CSR || P.GetFormat() != CSR)
        {
            LOG_VERBOSE_INFO(
                2, "*** warning: GlobalMatrix::TripleMatrixProduct() is performed in CSR format");

            this->matrix_interior_.ConvertTo(format, blockdim);
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::ReadFileMTX(const std::string& filename)
    {
        log_debug(this, "GlobalMatrix::ReadFileMTX()", filename);

        assert(this->pm_ != NULL);
        assert(this->pm_->Status() == true);

        // Read header file
        std::ifstream headfile(filename.c_str(), std::ifstream::in);

        if(!headfile.is_open())
        {
            LOG_INFO("Cannot open GlobalMatrix file [read]: " << filename);
            FATAL_ERROR(__FILE__, __LINE__);
        }

        // Go to this ranks line in the headfile
        for(int i = 0; i < this->pm_->rank_; ++i)
        {
            headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        std::string interior_name;
        std::string ghost_name;

        std::getline(headfile, interior_name);
        std::getline(headfile, ghost_name);

        headfile.close();

        // Extract directory containing the subfiles
        size_t      found = filename.find_last_of("\\/");
        std::string path  = filename.substr(0, found + 1);

        interior_name.erase(remove_if(interior_name.begin(), interior_name.end(), isspace),
                            interior_name.end());
        ghost_name.erase(remove_if(ghost_name.begin(), ghost_name.end(), isspace),
                         ghost_name.end());

        this->matrix_interior_.ReadFileMTX(path + interior_name);
        this->matrix_ghost_.ReadFileMTX(path + ghost_name);

        // Convert ghost matrix to COO
        this->matrix_ghost_.ConvertToCOO();

        this->object_name_ = filename;

        // Initialize communication pattern
        this->InitCommPattern_();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::WriteFileMTX(const std::string& filename) const
    {
        log_debug(this, "GlobalMatrix::WriteFileMTX()", filename);

        assert(this->pm_ != NULL);

        // Master rank writes the global headfile
        if(this->pm_->rank_ == 0)
        {
            std::ofstream headfile;

            headfile.open((char*)filename.c_str(), std::ofstream::out);
            if(!headfile.is_open())
            {
                LOG_INFO("Cannot open GlobalMatrix file [write]: " << filename);
                FATAL_ERROR(__FILE__, __LINE__);
            }

            for(int i = 0; i < this->pm_->num_procs_; ++i)
            {
                std::ostringstream rs;
                rs << i;

                std::string interior_name = filename + ".interior.rank." + rs.str();
                std::string ghost_name    = filename + ".ghost.rank." + rs.str();

                headfile << interior_name << "\n";
                headfile << ghost_name << "\n";
            }

            headfile.close();
        }

        std::ostringstream rs;
        rs << this->pm_->rank_;

        std::string interior_name = filename + ".interior.rank." + rs.str();
        std::string ghost_name    = filename + ".ghost.rank." + rs.str();

        this->matrix_interior_.WriteFileMTX(interior_name);
        this->matrix_ghost_.WriteFileMTX(ghost_name);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::ReadFileCSR(const std::string& filename)
    {
        log_debug(this, "GlobalMatrix::ReadFileCSR()", filename);

        assert(this->pm_ != NULL);
        assert(this->pm_->Status() == true);

        // Read header file
        std::ifstream headfile(filename.c_str(), std::ifstream::in);

        if(!headfile.is_open())
        {
            LOG_INFO("Cannot open GlobalMatrix file [read]: " << filename);
            FATAL_ERROR(__FILE__, __LINE__);
        }

        // Go to this ranks line in the headfile
        for(int i = 0; i < this->pm_->rank_; ++i)
        {
            headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        std::string interior_name;
        std::string ghost_name;

        std::getline(headfile, interior_name);
        std::getline(headfile, ghost_name);

        headfile.close();

        // Extract directory containing the subfiles
        size_t      found = filename.find_last_of("\\/");
        std::string path  = filename.substr(0, found + 1);

        interior_name.erase(remove_if(interior_name.begin(), interior_name.end(), isspace),
                            interior_name.end());
        ghost_name.erase(remove_if(ghost_name.begin(), ghost_name.end(), isspace),
                         ghost_name.end());

        this->matrix_interior_.ReadFileCSR(path + interior_name);
        this->matrix_ghost_.ReadFileCSR(path + ghost_name);

        // Convert ghost matrix to COO
        this->matrix_ghost_.ConvertToCOO();

        this->object_name_ = filename;

        // Initialize communication pattern
        this->InitCommPattern_();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::WriteFileCSR(const std::string& filename) const
    {
        log_debug(this, "GlobalMatrix::WriteFileCSR()", filename);

        assert(this->pm_ != NULL);

        // Master rank writes the global headfile
        if(this->pm_->rank_ == 0)
        {
            std::ofstream headfile;

            headfile.open((char*)filename.c_str(), std::ofstream::out);
            if(!headfile.is_open())
            {
                LOG_INFO("Cannot open GlobalMatrix file [write]: " << filename);
                FATAL_ERROR(__FILE__, __LINE__);
            }

            for(int i = 0; i < this->pm_->num_procs_; ++i)
            {
                std::ostringstream rs;
                rs << i;

                std::string interior_name = filename + ".interior.rank." + rs.str();
                std::string ghost_name    = filename + ".ghost.rank." + rs.str();

                headfile << interior_name << "\n";
                headfile << ghost_name << "\n";
            }
        }

        std::ostringstream rs;
        rs << this->pm_->rank_;

        std::string interior_name = filename + ".interior.rank." + rs.str();
        std::string ghost_name    = filename + ".ghost.rank." + rs.str();

        this->matrix_interior_.WriteFileCSR(interior_name);
        this->matrix_ghost_.WriteFileCSR(ghost_name);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::ReadFileRSIO(const std::string& filename,
                                               bool               maintain_initial_format)
    {
        log_debug(this, "GlobalMatrix::ReadFileRSIO()", filename);

        assert(this->pm_ != NULL);
        assert(this->pm_->Status() == true);

        // Read header file
        std::ifstream headfile(filename.c_str(), std::ifstream::in);

        if(!headfile.is_open())
        {
            LOG_INFO("Cannot open GlobalMatrix file [read]: " << filename);
            FATAL_ERROR(__FILE__, __LINE__);
        }

        // Go to this ranks line in the headfile
        for(int i = 0; i < this->pm_->rank_; ++i)
        {
            headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        std::string interior_name;
        std::string ghost_name;

        std::getline(headfile, interior_name);
        std::getline(headfile, ghost_name);

        headfile.close();

        // Extract directory containing the subfiles
        size_t      found = filename.find_last_of("\\/");
        std::string path  = filename.substr(0, found + 1);

        interior_name.erase(remove_if(interior_name.begin(), interior_name.end(), isspace),
                            interior_name.end());
        ghost_name.erase(remove_if(ghost_name.begin(), ghost_name.end(), isspace),
                         ghost_name.end());

        this->matrix_interior_.ReadFileRSIO(path + interior_name, maintain_initial_format);
        this->matrix_ghost_.ReadFileRSIO(path + ghost_name);

        // Convert ghost matrix to COO
        this->matrix_ghost_.ConvertToCOO();

        this->object_name_ = filename;

        // Initialize communication pattern
        this->InitCommPattern_();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::WriteFileRSIO(const std::string& filename) const
    {
        log_debug(this, "GlobalMatrix::WriteFileRSIO()", filename);

        assert(this->pm_ != NULL);

        // Master rank writes the global headfile
        if(this->pm_->rank_ == 0)
        {
            std::ofstream headfile;

            headfile.open((char*)filename.c_str(), std::ofstream::out);
            if(!headfile.is_open())
            {
                LOG_INFO("Cannot open GlobalMatrix file [write]: " << filename);
                FATAL_ERROR(__FILE__, __LINE__);
            }

            for(int i = 0; i < this->pm_->num_procs_; ++i)
            {
                std::ostringstream rs;
                rs << i;

                std::string interior_name = filename + ".interior.rank." + rs.str();
                std::string ghost_name    = filename + ".ghost.rank." + rs.str();

                headfile << interior_name << "\n";
                headfile << ghost_name << "\n";
            }
        }

        std::ostringstream rs;
        rs << this->pm_->rank_;

        std::string interior_name = filename + ".interior.rank." + rs.str();
        std::string ghost_name    = filename + ".ghost.rank." + rs.str();

        this->matrix_interior_.WriteFileRSIO(interior_name);
        this->matrix_ghost_.WriteFileRSIO(ghost_name);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::ExtractDiagonal(GlobalVector<ValueType>* vec_diag) const
    {
        log_debug(this, "GlobalMatrix::ExtractDiagonal()", vec_diag);

        assert(vec_diag != NULL);

        this->matrix_interior_.ExtractDiagonal(&vec_diag->vector_interior_);
    }

    template <typename ValueType>
    void
        GlobalMatrix<ValueType>::ExtractInverseDiagonal(GlobalVector<ValueType>* vec_inv_diag) const
    {
        log_debug(this, "GlobalMatrix::ExtractInverseDiagonal()", vec_inv_diag);

        assert(vec_inv_diag != NULL);

        this->matrix_interior_.ExtractInverseDiagonal(&vec_inv_diag->vector_interior_);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::Sort(void)
    {
        log_debug(this, "GlobalMatrix::Sort()");

        this->matrix_interior_.Sort();
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::Scale(ValueType alpha)
    {
        log_debug(this, "GlobalMatrix::Scale()", alpha);

        this->matrix_interior_.Scale(alpha);
        this->matrix_ghost_.Scale(alpha);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::InitialPairwiseAggregation(ValueType         beta,
                                                             int&              nc,
                                                             LocalVector<int>* G,
                                                             int&              Gsize,
                                                             int**             rG,
                                                             int&              rGsize,
                                                             int               ordering) const
    {
        log_debug(this,
                  "GlobalMatrix::InitialPairwiseAggregation()",
                  beta,
                  nc,
                  G,
                  Gsize,
                  rG,
                  rGsize,
                  ordering);

        // Calling global routine with single process
        if(this->pm_ == NULL || this->pm_->num_procs_ == 1)
        {
            this->matrix_interior_.InitialPairwiseAggregation(
                beta, nc, G, Gsize, rG, rGsize, ordering);

            return;
        }

        LocalMatrix<ValueType> tmp;
        tmp.CloneFrom(this->matrix_ghost_);
        tmp.ConvertToCSR();

        this->matrix_interior_.InitialPairwiseAggregation(
            tmp, beta, nc, G, Gsize, rG, rGsize, ordering);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::FurtherPairwiseAggregation(ValueType         beta,
                                                             int&              nc,
                                                             LocalVector<int>* G,
                                                             int&              Gsize,
                                                             int**             rG,
                                                             int&              rGsize,
                                                             int               ordering) const
    {
        log_debug(this,
                  "GlobalMatrix::FurtherPairwiseAggregation()",
                  beta,
                  nc,
                  G,
                  Gsize,
                  rG,
                  rGsize,
                  ordering);

        // Calling global routine with single process
        if(this->pm_ == NULL || this->pm_->num_procs_ == 1)
        {
            this->matrix_interior_.FurtherPairwiseAggregation(
                beta, nc, G, Gsize, rG, rGsize, ordering);

            return;
        }

        LocalMatrix<ValueType> tmp;
        tmp.CloneFrom(this->matrix_ghost_);
        tmp.ConvertToCSR();

        this->matrix_interior_.FurtherPairwiseAggregation(
            tmp, beta, nc, G, Gsize, rG, rGsize, ordering);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::CoarsenOperator(GlobalMatrix<ValueType>* Ac,
                                                  int                      nrow,
                                                  int                      ncol,
                                                  const LocalVector<int>&  G,
                                                  int                      Gsize,
                                                  const int*               rG,
                                                  int                      rGsize) const
    {
        log_debug(this,
                  "GlobalMatrix::CoarsenOperator()",
                  Ac,
                  nrow,
                  ncol,
                  (const void*&)G,
                  Gsize,
                  rG,
                  rGsize);

        assert(Ac != NULL);
        assert(rG != NULL);

        // Calling global routine with single process
        if(this->pm_ == NULL || this->pm_->num_procs_ == 1)
        {
            this->matrix_interior_.CoarsenOperator(
                &Ac->matrix_interior_, nrow, ncol, G, Gsize, rG, rGsize);

            Ac->CreateParallelManager_();

            Ac->pm_self_->SetMPICommunicator(this->pm_->comm_);

            Ac->pm_self_->SetGlobalNrow(Ac->matrix_interior_.GetM());
            Ac->pm_self_->SetGlobalNcol(Ac->matrix_interior_.GetN());

            Ac->pm_self_->SetLocalNrow(Ac->matrix_interior_.GetM());
            Ac->pm_self_->SetLocalNcol(Ac->matrix_interior_.GetN());

            return;
        }

#ifdef SUPPORT_MULTINODE
        // MPI Requests for sync
        std::vector<MRequest> req_mapping(this->pm_->nrecv_ + this->pm_->nsend_);
        std::vector<MRequest> req_offsets(this->pm_->nrecv_ + this->pm_->nsend_);

        // Determine connected pairs for the ghost layer of neighboring ranks
        int** send_ghost_map = new int*[this->pm_->nsend_];
        int** recv_ghost_map = new int*[this->pm_->nrecv_];

        int* send_map_size = NULL;
        int* recv_map_size = NULL;

        allocate_host(this->pm_->nsend_, &send_map_size);
        allocate_host(this->pm_->nrecv_, &recv_map_size);

        // Receive sizes
        for(int n = 0; n < this->pm_->nrecv_; ++n)
        {
            communication_async_recv(
                &recv_map_size[n], 1, this->pm_->recvs_[n], 0, &req_mapping[n], this->pm_->comm_);
        }

        // Loop over neighbor ranks
        for(int n = 0; n < this->pm_->nsend_; ++n)
        {
            send_ghost_map[n]
                = new int[this->pm_->send_offset_index_[n + 1] - this->pm_->send_offset_index_[n]];

            G.ExtractCoarseMapping(this->pm_->send_offset_index_[n],
                                   this->pm_->send_offset_index_[n + 1],
                                   this->pm_->boundary_index_,
                                   nrow,
                                   &send_map_size[n],
                                   send_ghost_map[n]);

            // Send sizes
            communication_async_send(&send_map_size[n],
                                     1,
                                     this->pm_->sends_[n],
                                     0,
                                     &req_mapping[this->pm_->nrecv_ + n],
                                     this->pm_->comm_);
        }

        // Wait for mapping sizes communication to finish
        communication_syncall(this->pm_->nrecv_ + this->pm_->nsend_, &req_mapping[0]);

        // Receive mapping pairs
        for(int n = 0; n < this->pm_->nrecv_; ++n)
        {
            recv_ghost_map[n] = new int[recv_map_size[n]];
            communication_async_recv(recv_ghost_map[n],
                                     recv_map_size[n],
                                     this->pm_->recvs_[n],
                                     0,
                                     &req_mapping[n],
                                     this->pm_->comm_);
        }

        // Send mapping pairs
        for(int n = 0; n < this->pm_->nsend_; ++n)
        {
            communication_async_send(send_ghost_map[n],
                                     send_map_size[n],
                                     this->pm_->sends_[n],
                                     0,
                                     &req_mapping[this->pm_->nrecv_ + n],
                                     this->pm_->comm_);
        }

        // Get coarse boundary of current rank
        int* boundary_index    = NULL;
        int* send_offset_index = NULL;
        int* recv_offset_index = NULL;

        allocate_host(this->pm_->send_index_size_, &boundary_index);
        allocate_host(this->pm_->nsend_ + 1, &send_offset_index);
        allocate_host(this->pm_->nrecv_ + 1, &recv_offset_index);

        send_offset_index[0] = 0;

        int m = 0;
        for(int n = 0; n < this->pm_->nsend_; ++n)
        {
            G.ExtractCoarseBoundary(this->pm_->send_offset_index_[n],
                                    this->pm_->send_offset_index_[n + 1],
                                    this->pm_->boundary_index_,
                                    nrow,
                                    &m,
                                    boundary_index);

            send_offset_index[n + 1] = m;
        }

        // Communicate boundary offsets
        for(int n = 0; n < this->pm_->nrecv_; ++n)
        {
            communication_async_recv(&recv_offset_index[n + 1],
                                     1,
                                     this->pm_->recvs_[n],
                                     0,
                                     &req_offsets[n],
                                     this->pm_->comm_);
        }

        for(int n = 0; n < this->pm_->nsend_; ++n)
        {
            communication_async_send(&send_offset_index[n + 1],
                                     1,
                                     this->pm_->sends_[n],
                                     0,
                                     &req_offsets[this->pm_->nrecv_ + n],
                                     this->pm_->comm_);
        }

        int boundary_size = m;

        // Coarsen interior part of the matrix on the host (no accelerator support)
        LocalMatrix<ValueType> tmp;
        LocalMatrix<ValueType> host_interior;

        if(this->is_accel_())
        {
            host_interior.ConvertTo(this->GetInterior().GetFormat(),
                                    this->GetInterior().GetBlockDimension());
            host_interior.CopyFrom(this->GetInterior());

            LocalVector<int> host_G;
            host_G.CopyFrom(G);

            host_interior.CoarsenOperator(&tmp, nrow, nrow, host_G, Gsize, rG, rGsize);
        }
        else
        {
            this->matrix_interior_.CoarsenOperator(&tmp, nrow, nrow, G, Gsize, rG, rGsize);
        }

        PtrType*   Ac_interior_row_offset = NULL;
        int*       Ac_interior_col        = NULL;
        ValueType* Ac_interior_val        = NULL;

        int64_t nnzc = tmp.GetNnz();
        tmp.LeaveDataPtrCSR(&Ac_interior_row_offset, &Ac_interior_col, &Ac_interior_val);

        // Wait for boundary offset communication to finish
        communication_syncall(this->pm_->nrecv_ + this->pm_->nsend_, &req_offsets[0]);

        recv_offset_index[0] = 0;
        for(int n = 0; n < this->pm_->nrecv_; ++n)
        {
            recv_offset_index[n + 1] += recv_offset_index[n];
        }

        // Wait for mappings communication to finish
        communication_syncall(this->pm_->nrecv_ + this->pm_->nsend_, &req_mapping[0]);

        // Free send mapping buffers and sizes
        for(int n = 0; n < this->pm_->nsend_; ++n)
        {
            delete[] send_ghost_map[n];
        }

        delete[] send_ghost_map;
        free_host(&send_map_size);

        // Prepare ghost G sets
        int* ghost_G = NULL;

        allocate_host(this->pm_->recv_offset_index_[this->pm_->nrecv_], &ghost_G);

        int k = 0;
        for(int n = 0; n < this->pm_->nrecv_; ++n)
        {
            for(int i = 0;
                i < this->pm_->recv_offset_index_[n + 1] - this->pm_->recv_offset_index_[n];
                ++i)
            {
                ghost_G[k]
                    = (i < recv_map_size[n]) ? (recv_offset_index[n] + recv_ghost_map[n][i]) : -1;
                ++k;
            }
        }

        // Free receive mapping buffers and sizes
        for(int n = 0; n < this->pm_->nrecv_; ++n)
        {
            delete[] recv_ghost_map[n];
        }

        delete[] recv_ghost_map;
        free_host(&recv_map_size);

        // Coarsen ghost part of the matrix on the host (no accelerator support)
        LocalVector<int> G_ghost;
        G_ghost.SetDataPtr(&ghost_G, "G ghost", this->pm_->recv_offset_index_[this->pm_->nrecv_]);

        LocalMatrix<ValueType> tmp_ghost;
        LocalMatrix<ValueType> host_ghost;

        if(this->is_accel_())
        {
            host_ghost.ConvertTo(this->GetGhost().GetFormat(),
                                 this->GetGhost().GetBlockDimension());
            host_ghost.CopyFrom(this->GetGhost());

            host_ghost.CoarsenOperator(
                &tmp_ghost, nrow, this->pm_->GetNumReceivers(), G_ghost, Gsize, rG, rGsize);
        }
        else
        {
            this->matrix_ghost_.CoarsenOperator(
                &tmp_ghost, nrow, this->pm_->GetNumReceivers(), G_ghost, Gsize, rG, rGsize);
        }

        G_ghost.Clear();

        PtrType*   Ac_ghost_row_offset = NULL;
        int*       Ac_ghost_col        = NULL;
        ValueType* Ac_ghost_val        = NULL;

        int64_t nnzg = tmp_ghost.GetNnz();
        tmp_ghost.LeaveDataPtrCSR(&Ac_ghost_row_offset, &Ac_ghost_col, &Ac_ghost_val);

        // Clear old Ac
        Ac->Clear();
        bool isaccel = Ac->is_accel_();
        Ac->MoveToHost();

        // Communicator
        Ac->CreateParallelManager_();
        Ac->pm_self_->SetMPICommunicator(this->pm_->comm_);

        // Get the global size
        int64_t local_size = nrow;
        int64_t global_size;
        communication_sync_allreduce_single_sum(&local_size, &global_size, this->pm_->comm_);
        Ac->pm_self_->SetGlobalNrow(global_size);
        Ac->pm_self_->SetGlobalNcol(global_size);

        // Local size
        Ac->pm_self_->SetLocalNrow(local_size);
        Ac->pm_self_->SetLocalNcol(local_size);

        // New boundary and boundary offsets
        Ac->pm_self_->SetBoundaryIndex(boundary_size, boundary_index);
        free_host(&boundary_index);

        Ac->pm_self_->SetReceivers(this->pm_->nrecv_, this->pm_->recvs_, recv_offset_index);
        free_host(&recv_offset_index);

        Ac->pm_self_->SetSenders(this->pm_->nsend_, this->pm_->sends_, send_offset_index);
        free_host(&send_offset_index);

        Ac->SetParallelManager(*Ac->pm_self_);

        Ac->SetDataPtrCSR(&Ac_interior_row_offset,
                          &Ac_interior_col,
                          &Ac_interior_val,
                          &Ac_ghost_row_offset,
                          &Ac_ghost_col,
                          &Ac_ghost_val,
                          "",
                          nnzc,
                          nnzg);

        if(isaccel == true)
        {
            Ac->MoveToAccelerator();
        }
#endif
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::CreateFromMap(const LocalVector<int>&  map,
                                                int64_t                  n,
                                                int64_t                  m,
                                                GlobalMatrix<ValueType>* pro)
    {
        log_debug(this, "GlobalMatrix::CreateFromMap()", (const void*&)map, n, m, pro);

        // P and R are local operators
        this->pm_ = NULL;
        pro->pm_  = NULL;

        this->matrix_interior_.CreateFromMap(map, n, m, &pro->matrix_interior_);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::AMGGreedyAggregate(
        ValueType             eps,
        LocalVector<bool>*    connections,
        LocalVector<int64_t>* aggregates,
        LocalVector<int64_t>* aggregate_root_nodes) const
    {
        log_debug(this,
                  "GlobalMatrix::AMGGreedyAggregate()",
                  connections,
                  aggregates,
                  aggregate_root_nodes);

        assert(connections != NULL);
        assert(aggregates != NULL);
        assert(aggregate_root_nodes != NULL);

        assert(this->is_host_() == connections->is_host_());
        assert(this->is_host_() == aggregates->is_host_());
        assert(this->is_host_() == aggregate_root_nodes->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif
        // Calling global routine with single process
        if(this->pm_ == NULL || this->pm_->num_procs_ == 1)
        {
            this->matrix_interior_.AMGGreedyAggregate(
                eps, connections, aggregates, aggregate_root_nodes);

            return;
        }

        LOG_VERBOSE_INFO(
            2,
            "*** error: GlobalMatrix::AMGGreedyAggregate() is not available on GlobalMatrix "
            "class - use PMIS aggregation instead");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::AMGPMISAggregate(ValueType             eps,
                                                   LocalVector<bool>*    connections,
                                                   LocalVector<int64_t>* aggregates,
                                                   LocalVector<int64_t>* aggregate_root_nodes) const
    {
        log_debug(this,
                  "GlobalMatrix::AMGPMISAggregate()",
                  connections,
                  aggregates,
                  aggregate_root_nodes);

        assert(connections != NULL);
        assert(aggregates != NULL);
        assert(aggregate_root_nodes != NULL);

        assert(this->is_host_() == connections->is_host_());
        assert(this->is_host_() == aggregates->is_host_());
        assert(this->is_host_() == aggregate_root_nodes->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif
        // Calling global routine with single process
        if(this->pm_ == NULL || this->pm_->num_procs_ == 1)
        {
            this->matrix_interior_.AMGPMISAggregate(
                eps, connections, aggregates, aggregate_root_nodes);

            return;
        }

        // Only CSR matrices are supported
        LocalMatrix<ValueType>        csr_int;
        LocalMatrix<ValueType>        csr_gst;
        const LocalMatrix<ValueType>* int_ptr = &this->matrix_interior_;
        const LocalMatrix<ValueType>* gst_ptr = &this->matrix_ghost_;

        if(int_ptr->GetFormat() != CSR)
        {
            csr_int.CloneFrom(*int_ptr);
            csr_int.ConvertToCSR();
            int_ptr = &csr_int;
        }

        if(gst_ptr->GetFormat() != CSR)
        {
            csr_gst.CloneFrom(*gst_ptr);
            csr_gst.ConvertToCSR();
            gst_ptr = &csr_gst;
        }

        if(this->GetNnz() > 0)
        {
            // Communication sizes
            int nsend = this->pm_->GetNumSenders();
            int nrecv = this->pm_->GetNumReceivers();

            // Some communication related buffers
            int*       hisend_buffer   = NULL;
            int*       hirecv_buffer   = NULL;
            int64_t*   hi64send_buffer = NULL;
            int64_t*   hi64recv_buffer = NULL;
            ValueType* hsend_buffer    = NULL;
            ValueType* hrecv_buffer    = NULL;

            allocate_host(nsend, &hisend_buffer);
            allocate_host(nrecv, &hirecv_buffer);
            allocate_host(nsend, &hi64send_buffer);
            allocate_host(nrecv, &hi64recv_buffer);
            allocate_host(nsend, &hsend_buffer);
            allocate_host(nrecv, &hrecv_buffer);

            // Allocate connections array
            connections->Allocate("connections", int_ptr->GetNnz() + gst_ptr->GetNnz());

            // Allocate aggregates array
            aggregates->Allocate("aggregates", int_ptr->GetM() + nrecv);

            // Allocate aggregate types array
            aggregate_root_nodes->Allocate("aggregate types", int_ptr->GetM() + nrecv);

            // Create state array
            LocalVector<int> hash;
            hash.CloneBackend(*this);
            hash.Allocate("hash", int_ptr->GetM() + nrecv);

            // Create state array
            LocalVector<int> state;
            state.CloneBackend(*this);
            state.Allocate("state", int_ptr->GetM() + nrecv);

            // Create max_state array
            LocalVector<int> max_state;
            max_state.CloneBackend(*this);
            max_state.Allocate("max_state", int_ptr->GetM() + nrecv);

            // Create diagonal array
            LocalVector<ValueType> diag;
            diag.CloneBackend(*this);
            diag.Allocate("diag", int_ptr->GetM() + nrecv);

            // Send buffers
            LocalVector<ValueType> send_buffer;
            send_buffer.CloneBackend(*this);
            send_buffer.Allocate("send buffer", nsend);

            LocalVector<int> isend_buffer;
            isend_buffer.CloneBackend(*this);
            isend_buffer.Allocate("isend buffer", nsend);

            LocalVector<int64_t> i64send_buffer;
            i64send_buffer.CloneBackend(*this);
            i64send_buffer.Allocate("i64send buffer", nsend);

            // Obtain the local to global ghost mapping
            LocalVector<int64_t> l2g;
            l2g.CloneBackend(*this);
            l2g.Allocate("l2g ghost map", this->pm_->GetNumReceivers());
            l2g.CopyFromHostData(this->pm_->GetGhostToGlobalMap());

            // Global column offset for this rank
            int64_t global_col_begin = this->pm_->GetGlobalColumnBegin();
            int64_t global_col_end   = this->pm_->GetGlobalColumnEnd();

            int_ptr->matrix_->ExtractDiagonal(diag.vector_);

            // Update diag with received diag from neighbors
            diag.GetIndexValues(this->halo_, &send_buffer);
            send_buffer.CopyToHostData(hsend_buffer);

            // Receive updated diag from neighbors
            this->pm_->CommunicateAsync_(hsend_buffer, hrecv_buffer);
            this->pm_->CommunicateSync_();

            // Update diag with received diag from neighbors
            diag.SetContinuousValues(int_ptr->GetM(), diag.GetSize(), hrecv_buffer);

            // Compute connections
            int_ptr->matrix_->AMGComputeStrongConnections(
                eps, *diag.vector_, *l2g.vector_, connections->vector_, *gst_ptr->matrix_);

            // Initialize max state
            int_ptr->matrix_->AMGPMISInitializeState(global_col_begin,
                                                     *connections->vector_,
                                                     max_state.vector_,
                                                     hash.vector_,
                                                     *gst_ptr->matrix_);

            // Update max_state with received max_state from neighbors
            max_state.GetIndexValues(this->halo_, &isend_buffer);
            isend_buffer.CopyToHostData(hisend_buffer);

            // Receive updated max_state from neighbors
            this->pm_->CommunicateAsync_(hisend_buffer, hirecv_buffer);
            this->pm_->CommunicateSync_();

            // Update max_state with received max_state from neighbors
            max_state.SetContinuousValues(int_ptr->GetM(), max_state.GetSize(), hirecv_buffer);

            // Update hash with received hash from neighbors
            hash.GetIndexValues(this->halo_, &isend_buffer);
            isend_buffer.CopyToHostData(hisend_buffer);

            // Receive updated hash from neighbors
            this->pm_->CommunicateAsync_(hisend_buffer, hirecv_buffer);
            this->pm_->CommunicateSync_();

            // Update hash with received hash from neighbors
            hash.SetContinuousValues(int_ptr->GetM(), hash.GetSize(), hirecv_buffer);

            // Create local vectors
            LocalVector<PtrType> A_ext_row_ptr_send;
            A_ext_row_ptr_send.CloneBackend(*this);
            A_ext_row_ptr_send.Allocate("A ext row ptr send", nsend + 1);

            LocalVector<PtrType> A_ext_row_ptr_recv;
            A_ext_row_ptr_recv.CloneBackend(*this);
            A_ext_row_ptr_recv.Allocate("A ext row ptr recv", nrecv + 1);

            // Create host send and receive buffers
            PtrType* hA_ext_row_ptr_send = NULL;
            allocate_host(nsend + 1, &hA_ext_row_ptr_send);

            PtrType* hA_ext_row_ptr_recv = NULL;
            allocate_host(nrecv + 1, &hA_ext_row_ptr_recv);
            set_to_zero_host(nrecv + 1, hA_ext_row_ptr_recv);

            int_ptr->matrix_->AMGBoundaryNnz(*this->halo_.vector_,
                                             *connections->vector_,
                                             *gst_ptr->matrix_,
                                             A_ext_row_ptr_send.vector_);

            // Transfer to host send buffer
            A_ext_row_ptr_send.CopyToHostData(hA_ext_row_ptr_send);

            // Initiate communication of nnz per row (async)
            this->pm_->CommunicateAsync_(hA_ext_row_ptr_send, hA_ext_row_ptr_recv);

            // Exclusive sum to obtain row pointers
            PtrType A_ext_nnz_send = A_ext_row_ptr_send.ExclusiveSum();

            // Now, extract boundary column indices and values of all strongly connected coarse points
            LocalVector<int64_t> A_ext_col_ind_send;
            A_ext_col_ind_send.CloneBackend(*this);
            A_ext_col_ind_send.Allocate("A ext col ind send", A_ext_nnz_send);

            int64_t* hA_ext_col_ind_send = NULL;
            allocate_host(A_ext_nnz_send, &hA_ext_col_ind_send);

            int_ptr->matrix_->AMGExtractBoundary(global_col_begin,
                                                 *this->halo_.vector_,
                                                 *l2g.vector_,
                                                 *connections->vector_,
                                                 *gst_ptr->matrix_,
                                                 *A_ext_row_ptr_send.vector_,
                                                 A_ext_col_ind_send.vector_);

            // Synchronize communication
            this->pm_->CommunicateSync_();

            // Place our received data into structure
            A_ext_row_ptr_recv.CopyFromHostData(hA_ext_row_ptr_recv);
            PtrType A_ext_nnz_recv = A_ext_row_ptr_recv.ExclusiveSum();
            A_ext_row_ptr_recv.CopyToHostData(hA_ext_row_ptr_recv);

            // We need a copy of A_ext_col_ind_send on host for communication
            A_ext_row_ptr_send.CopyToHostData(hA_ext_row_ptr_send);
            A_ext_col_ind_send.CopyToHostData(hA_ext_col_ind_send);

            LocalVector<int64_t> A_ext_col_ind_recv;
            A_ext_col_ind_recv.CloneBackend(*this);
            A_ext_col_ind_recv.Allocate("A ext col ind recv", A_ext_nnz_recv);

            int64_t* hA_ext_col_ind_recv = NULL;
            allocate_host(A_ext_nnz_recv, &hA_ext_col_ind_recv);

            this->pm_->CommunicateCSRAsync_(hA_ext_row_ptr_send,
                                            hA_ext_col_ind_send,
                                            (ValueType*)NULL,
                                            hA_ext_row_ptr_recv,
                                            hA_ext_col_ind_recv,
                                            (ValueType*)NULL);
            this->pm_->CommunicateCSRSync_();

            A_ext_row_ptr_recv.CopyFromHostData(hA_ext_row_ptr_recv);
            A_ext_col_ind_recv.CopyFromHostData(hA_ext_col_ind_recv);

            LocalVector<int> state_ext_send;
            state_ext_send.CloneBackend(*this);
            state_ext_send.Allocate("state ext send", A_ext_nnz_send);

            LocalVector<int> hash_ext_send;
            hash_ext_send.CloneBackend(*this);
            hash_ext_send.Allocate("hash ext send", A_ext_nnz_send);

            int* hstate_ext_send = NULL;
            allocate_host(A_ext_nnz_send, &hstate_ext_send);

            int* hhash_ext_send = NULL;
            allocate_host(A_ext_nnz_send, &hhash_ext_send);

            int* hstate_ext_recv = NULL;
            allocate_host(A_ext_nnz_recv, &hstate_ext_recv);

            int* hhash_ext_recv = NULL;
            allocate_host(A_ext_nnz_recv, &hhash_ext_recv);

            A_ext_row_ptr_send.CopyToHostData(hA_ext_row_ptr_send);

            LocalVector<int> state_ext_recv;
            state_ext_recv.CloneBackend(*this);
            state_ext_recv.Allocate("state ext recv", A_ext_nnz_recv);

            LocalVector<int> hash_ext_recv;
            hash_ext_recv.CloneBackend(*this);
            hash_ext_recv.Allocate("hash ext recv", A_ext_nnz_recv);

            int iter = 0;
            while(true)
            {
                state.CopyFrom(max_state);

                int_ptr->matrix_->AMGExtractBoundaryState(*A_ext_row_ptr_send.vector_,
                                                          *connections->vector_,
                                                          *max_state.vector_,
                                                          *hash.vector_,
                                                          state_ext_send.vector_,
                                                          hash_ext_send.vector_,
                                                          global_col_begin,
                                                          *this->halo_.vector_,
                                                          *gst_ptr->matrix_);

                state_ext_send.CopyToHostData(hstate_ext_send);
                hash_ext_send.CopyToHostData(hhash_ext_send);

                this->pm_->CommunicateCSRAsync_(hA_ext_row_ptr_send,
                                                hstate_ext_send,
                                                hhash_ext_send,
                                                hA_ext_row_ptr_recv,
                                                hstate_ext_recv,
                                                hhash_ext_recv);
                this->pm_->CommunicateCSRSync_();

                state_ext_recv.CopyFromHostData(hstate_ext_recv);
                hash_ext_recv.CopyFromHostData(hhash_ext_recv);

                bool undecided = false;
                int_ptr->matrix_->AMGPMISFindMaxNeighbourNode(global_col_begin,
                                                              global_col_end,
                                                              undecided,
                                                              *connections->vector_,
                                                              *state.vector_,
                                                              *hash.vector_,
                                                              *A_ext_row_ptr_recv.vector_,
                                                              *A_ext_col_ind_recv.vector_,
                                                              *state_ext_recv.vector_,
                                                              *hash_ext_recv.vector_,
                                                              max_state.vector_,
                                                              aggregates->vector_,
                                                              *gst_ptr->matrix_);

                max_state.GetIndexValues(this->halo_, &isend_buffer);
                isend_buffer.CopyToHostData(hisend_buffer);

                // Receive updated max_state from neighbors
                this->pm_->CommunicateAsync_(hisend_buffer, hirecv_buffer);
                this->pm_->CommunicateSync_();

                // Update max_state with received max_state from neighbors
                max_state.SetContinuousValues(int_ptr->GetM(), max_state.GetSize(), hirecv_buffer);

                // Get global number of undecided vertices
                int local_undecided = undecided;
                int global_undecided;
#ifdef SUPPORT_MULTINODE
                MRequest req;
                communication_async_allreduce_single_max(
                    &local_undecided, &global_undecided, this->pm_->comm_, &req);
                communication_sync(&req);
#endif

                // If no more undecided vertices are left, we are done
                if(global_undecided == 0)
                {
                    break;
                }

                ++iter;

                // Print some warning if number of iteration is getting huge
                if(iter > 20)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: GlobalMatrix::AMGPMISAggregate() Current "
                                     "number of iterations: "
                                         << iter);
                }
            }

            aggregate_root_nodes->SetValues(-1);

            int_ptr->matrix_->AMGPMISInitializeAggregateGlobalIndices(
                global_col_begin, *aggregates->vector_, aggregate_root_nodes->vector_);

            aggregate_root_nodes->GetIndexValues(this->halo_, &i64send_buffer);
            i64send_buffer.CopyToHostData(hi64send_buffer);

            // Receive updated aggregate_root_nodes from neighbors
            this->pm_->CommunicateAsync_(hi64send_buffer, hi64recv_buffer);
            this->pm_->CommunicateSync_();

            // Update aggregate_root_nodes with received aggregate_root_nodes from neighbors
            aggregate_root_nodes->SetContinuousValues(
                int_ptr->GetM(), aggregate_root_nodes->GetSize(), hi64recv_buffer);

            int64_t hsum_send = aggregates->Reduce();
            int64_t hsum_recv = 0;

            aggregates->ExclusiveSum();

#ifdef SUPPORT_MULTINODE
            communication_sync_exscan(&hsum_send, &hsum_recv, 1, this->pm_->comm_);
#endif

            LocalVector<int64_t> ones;
            ones.CloneBackend(*this);
            ones.Allocate("ones", int_ptr->GetM() + nrecv);
            ones.Ones();

            aggregates->AddScale(ones, hsum_recv);

            aggregates->GetIndexValues(this->halo_, &i64send_buffer);
            i64send_buffer.CopyToHostData(hi64send_buffer);

            // Receive updated aggregates from neighbors
            this->pm_->CommunicateAsync_(hi64send_buffer, hi64recv_buffer);
            this->pm_->CommunicateSync_();

            // Update aggregates with received aggregates from neighbors
            aggregates->SetContinuousValues(
                int_ptr->GetM(), aggregates->GetSize(), hi64recv_buffer);

            // Distance 2 aggregations
            for(int k = 0; k < 2; k++)
            {
                state.CopyFrom(max_state);

                int_ptr->matrix_->AMGPMISAddUnassignedNodesToAggregations(
                    global_col_begin,
                    *connections->vector_,
                    *state.vector_,
                    *l2g.vector_,
                    max_state.vector_,
                    aggregates->vector_,
                    aggregate_root_nodes->vector_,
                    *gst_ptr->matrix_);

                aggregates->GetIndexValues(this->halo_, &i64send_buffer);
                i64send_buffer.CopyToHostData(hi64send_buffer);

                // Receive updated aggregates from neighbors
                this->pm_->CommunicateAsync_(hi64send_buffer, hi64recv_buffer);
                this->pm_->CommunicateSync_();

                // Update aggregates with received aggregates from neighbors
                aggregates->SetContinuousValues(
                    int_ptr->GetM(), aggregates->GetSize(), hi64recv_buffer);

                aggregate_root_nodes->GetIndexValues(this->halo_, &i64send_buffer);
                i64send_buffer.CopyToHostData(hi64send_buffer);

                // Receive updated aggregate_root_nodes from neighbors
                this->pm_->CommunicateAsync_(hi64send_buffer, hi64recv_buffer);
                this->pm_->CommunicateSync_();

                // Update aggregate_root_nodes with received aggregate_root_nodes from neighbors
                aggregate_root_nodes->SetContinuousValues(
                    int_ptr->GetM(), aggregate_root_nodes->GetSize(), hi64recv_buffer);

                max_state.GetIndexValues(this->halo_, &isend_buffer);
                isend_buffer.CopyToHostData(hisend_buffer);

                // Receive updated max_state from neighbors
                this->pm_->CommunicateAsync_(hisend_buffer, hirecv_buffer);
                this->pm_->CommunicateSync_();

                // Update max_state with received max_state from neighbors
                max_state.SetContinuousValues(int_ptr->GetM(), max_state.GetSize(), hirecv_buffer);
            }

            free_host(&hisend_buffer);
            free_host(&hirecv_buffer);
            free_host(&hi64send_buffer);
            free_host(&hi64recv_buffer);
            free_host(&hsend_buffer);
            free_host(&hrecv_buffer);

            free_host(&hA_ext_row_ptr_send);
            free_host(&hA_ext_col_ind_send);
            free_host(&hA_ext_row_ptr_recv);
            free_host(&hA_ext_col_ind_recv);

            free_host(&hstate_ext_send);
            free_host(&hhash_ext_send);
            free_host(&hstate_ext_recv);
            free_host(&hhash_ext_recv);
        }

        if(this->GetFormat() != CSR)
        {
            LOG_VERBOSE_INFO(
                2, "*** warning: GlobalMatrix::AMGPMISAggregate() is performed in CSR format");
        }
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::AMGSmoothedAggregation(
        ValueType                   relax,
        const LocalVector<bool>&    connections,
        const LocalVector<int64_t>& aggregates,
        const LocalVector<int64_t>& aggregate_root_nodes,
        GlobalMatrix<ValueType>*    prolong,
        int                         lumping_strat) const
    {
        log_debug(this,
                  "GlobalMatrix::AMGSmoothedAggregation()",
                  relax,
                  (const void*&)connections,
                  (const void*&)aggregates,
                  (const void*&)aggregate_root_nodes,
                  prolong);

        assert(relax > static_cast<ValueType>(0));
        assert(prolong != NULL);
        assert(this != prolong);
        assert(this->is_host_() == connections.is_host_());
        assert(this->is_host_() == aggregates.is_host_());
        assert(this->is_host_() == aggregate_root_nodes.is_host_());
        assert(this->is_host_() == prolong->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif
        // Calling global routine with single process
        if(this->pm_ == NULL || this->pm_->num_procs_ == 1)
        {
            this->matrix_interior_.AMGSmoothedAggregation(relax,
                                                          connections,
                                                          aggregates,
                                                          aggregate_root_nodes,
                                                          &prolong->matrix_interior_,
                                                          lumping_strat);

            // Prolongation PM
            prolong->CreateParallelManager_();
            prolong->pm_self_->SetMPICommunicator(this->pm_->comm_);

            prolong->pm_self_->SetGlobalNrow(prolong->matrix_interior_.GetM());
            prolong->pm_self_->SetGlobalNcol(prolong->matrix_interior_.GetN());

            prolong->pm_self_->SetLocalNrow(prolong->matrix_interior_.GetM());
            prolong->pm_self_->SetLocalNcol(prolong->matrix_interior_.GetN());

            return;
        }

        // Only CSR matrices are supported
        LocalMatrix<ValueType> csr_int;
        LocalMatrix<ValueType> csr_gst;

        const LocalMatrix<ValueType>* int_ptr = &this->matrix_interior_;
        const LocalMatrix<ValueType>* gst_ptr = &this->matrix_ghost_;

        if(int_ptr->GetFormat() != CSR)
        {
            csr_int.CloneFrom(*int_ptr);
            csr_int.ConvertToCSR();
            int_ptr = &csr_int;
        }

        if(gst_ptr->GetFormat() != CSR)
        {
            csr_gst.CloneFrom(*gst_ptr);
            csr_gst.ConvertToCSR();
            gst_ptr = &csr_gst;
        }

        // Start with fresh P operator
        prolong->Clear();

        int64_t global_row_begin = this->pm_->GetGlobalRowBegin();
        int64_t global_row_end   = this->pm_->GetGlobalRowEnd();
        int64_t global_col_begin = this->pm_->GetGlobalColumnBegin();
        int64_t global_col_end   = this->pm_->GetGlobalColumnEnd();

        // Communication sizes
        int nsend = this->pm_->GetNumSenders();
        int nrecv = this->pm_->GetNumReceivers();

        // Obtain the local to global ghost mapping
        LocalVector<int64_t> l2g;
        l2g.CloneBackend(*this);
        l2g.Allocate("l2g ghost map", this->pm_->GetNumReceivers());
        l2g.CopyFromHostData(this->pm_->GetGhostToGlobalMap());

        // fine to coarse map
        LocalVector<int> f2c_map;
        f2c_map.CloneBackend(*this);
        f2c_map.Allocate("f2c map", int_ptr->GetM() + 1);
        f2c_map.Zeros();

        // Determine number of non-zeros for P
        int_ptr->matrix_->AMGSmoothedAggregationProlongNnz(global_col_begin,
                                                           global_col_end,
                                                           *connections.vector_,
                                                           *aggregates.vector_,
                                                           *aggregate_root_nodes.vector_,
                                                           *gst_ptr->matrix_,
                                                           f2c_map.vector_,
                                                           prolong->matrix_interior_.matrix_,
                                                           prolong->matrix_ghost_.matrix_);

        // Temporary array to store global ghost columns
        LocalVector<int64_t> ghost_col;
        ghost_col.CloneBackend(*this);

        // Fill column indices and values of P
        int_ptr->matrix_->AMGSmoothedAggregationProlongFill(global_col_begin,
                                                            global_col_end,
                                                            lumping_strat,
                                                            relax,
                                                            *connections.vector_,
                                                            *aggregates.vector_,
                                                            *aggregate_root_nodes.vector_,
                                                            *l2g.vector_,
                                                            *f2c_map.vector_,
                                                            *gst_ptr->matrix_,
                                                            prolong->matrix_interior_.matrix_,
                                                            prolong->matrix_ghost_.matrix_,
                                                            ghost_col.vector_);

        // Ghost of P MUST be CSR
        assert(prolong->matrix_ghost_.GetFormat() == CSR);

        // Communicate global sizes of P
        int64_t global_ncol;
        int64_t local_ncol = prolong->matrix_interior_.GetN();

#ifdef SUPPORT_MULTINODE
        MRequest req;
        communication_async_allreduce_single_sum(&local_ncol, &global_ncol, this->pm_->comm_, &req);
        communication_sync(&req);
#endif

        // Setup parallel manager of P
        prolong->CreateParallelManager_();
        prolong->pm_self_->SetMPICommunicator(this->pm_->comm_);

        // To generate the parallel manager, we need to access the sorted global ghost column ids
        LocalVector<int64_t> sorted_ghost_col;
        sorted_ghost_col.CloneBackend(*this);
        sorted_ghost_col.Allocate("sorted global ghost columns", ghost_col.GetSize());

        // Sort the global ghost columns (we do not need the permutation vector)
        ghost_col.Sort(&sorted_ghost_col, NULL);

        // Get the sorted ghost columns on host
        int64_t* pghost_col = NULL;
        sorted_ghost_col.MoveToHost();
        sorted_ghost_col.LeaveDataPtr(&pghost_col);

        // Sizes
        prolong->pm_self_->SetGlobalNrow(this->pm_->global_nrow_);
        prolong->pm_self_->SetGlobalNcol(global_ncol);
        prolong->pm_self_->SetLocalNrow(this->pm_->local_nrow_);
        prolong->pm_self_->SetLocalNcol(local_ncol);

        // Generate the PM
        prolong->pm_self_->GenerateFromGhostColumnsWithParent_(
            prolong->matrix_ghost_.GetNnz(), pghost_col, *this->pm_);

        // Communicate offsets
        prolong->pm_self_->CommunicateGlobalOffsetAsync_();

        // This is a prolongation operator, means we need to convert the global
        // fine boundary columns from to local coarse columns
        // Convert local boundary columns from global fine to local coarse
        int* f2c = NULL;
        f2c_map.MoveToHost();
        f2c_map.LeaveDataPtr(&f2c);

        // Clear
        free_host(&pghost_col);

        // Sync global offsets communication
        prolong->pm_self_->CommunicateGlobalOffsetSync_();

        // Convert local boundary columns from global fine to local coarse
        prolong->pm_self_->BoundaryTransformGlobalFineToLocalCoarse_(f2c);

        // Clear
        free_host(&f2c);

        // Communicate ghost to global map
        prolong->pm_self_->CommunicateGhostToGlobalMapAsync_();

        // Finally, renumber ghost columns (from global to local)
        // We couldn't do this earlier, because the parallel manager need
        // to know the global ghost column ids
        prolong->matrix_ghost_.matrix_->RenumberGlobalToLocal(*ghost_col.vector_);

        // Synchronize
        prolong->pm_self_->CommunicateGhostToGlobalMapSync_();

        prolong->SetParallelManager(*prolong->pm_self_);

        // Rename P
        prolong->object_name_ = "Prolongation Operator of " + this->object_name_;

        const int*     boundary_index = this->pm_->GetBoundaryIndex();
        const int64_t* ghost_mapping  = this->pm_->GetGhostToGlobalMap();

        int            prolong_nsend          = prolong->pm_self_->GetNumSenders();
        int            prolong_nrecv          = prolong->pm_self_->GetNumReceivers();
        const int*     prolong_boundary_index = prolong->pm_self_->GetBoundaryIndex();
        const int64_t* prolong_ghost_mapping  = prolong->pm_self_->GetGhostToGlobalMap();

        int64_t prolong_global_row_begin = prolong->pm_->GetGlobalRowBegin();
        int64_t prolong_global_row_end   = prolong->pm_->GetGlobalRowEnd();
        int64_t prolong_global_col_begin = prolong->pm_->GetGlobalColumnBegin();
        int64_t prolong_global_col_end   = prolong->pm_->GetGlobalColumnEnd();

        if(this->GetFormat() != CSR)
        {
            LOG_VERBOSE_INFO(
                2,
                "*** warning: GlobalMatrix::AMGSmoothedAggregation() is performed in CSR format");
        }

#ifdef DEBUG_MODE
        prolong->Check();
#endif
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::AMGUnsmoothedAggregation(
        const LocalVector<int64_t>& aggregates,
        const LocalVector<int64_t>& aggregate_root_nodes,
        GlobalMatrix<ValueType>*    prolong) const
    {
        log_debug(this,
                  "GlobalMatrix::AMGUnsmoothedAggregation()",
                  (const void*&)aggregates,
                  (const void*&)aggregate_root_nodes,
                  prolong);

        assert(prolong != NULL);
        assert(this != prolong);
        assert(this->is_host_() == aggregates.is_host_());
        assert(this->is_host_() == aggregate_root_nodes.is_host_());
        assert(this->is_host_() == prolong->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif
        // Calling global routine with single process
        if(this->pm_ == NULL || this->pm_->num_procs_ == 1)
        {
            this->matrix_interior_.AMGUnsmoothedAggregation(
                aggregates, aggregate_root_nodes, &prolong->matrix_interior_);

            // Prolongation PM
            prolong->CreateParallelManager_();
            prolong->pm_self_->SetMPICommunicator(this->pm_->comm_);

            prolong->pm_self_->SetGlobalNrow(prolong->matrix_interior_.GetM());
            prolong->pm_self_->SetGlobalNcol(prolong->matrix_interior_.GetN());

            prolong->pm_self_->SetLocalNrow(prolong->matrix_interior_.GetM());
            prolong->pm_self_->SetLocalNcol(prolong->matrix_interior_.GetN());

            return;
        }

        // Only CSR matrices are supported
        LocalMatrix<ValueType> csr_int;
        LocalMatrix<ValueType> csr_gst;

        const LocalMatrix<ValueType>* int_ptr = &this->matrix_interior_;
        const LocalMatrix<ValueType>* gst_ptr = &this->matrix_ghost_;

        if(int_ptr->GetFormat() != CSR)
        {
            csr_int.CloneFrom(*int_ptr);
            csr_int.ConvertToCSR();
            int_ptr = &csr_int;
        }

        if(gst_ptr->GetFormat() != CSR)
        {
            csr_gst.CloneFrom(*gst_ptr);
            csr_gst.ConvertToCSR();
            gst_ptr = &csr_gst;
        }

        // Start with fresh P operator
        prolong->Clear();

        int64_t global_row_begin = this->pm_->GetGlobalRowBegin();
        int64_t global_row_end   = this->pm_->GetGlobalRowEnd();
        int64_t global_col_begin = this->pm_->GetGlobalColumnBegin();
        int64_t global_col_end   = this->pm_->GetGlobalColumnEnd();

        // Communication sizes
        int nsend = this->pm_->GetNumSenders();
        int nrecv = this->pm_->GetNumReceivers();

        // fine to coarse map
        LocalVector<int> f2c_map;
        f2c_map.CloneBackend(*this);
        f2c_map.Allocate("f2c map", int_ptr->GetM() + 1);
        f2c_map.Zeros();

        // Determine number of non-zeros for P
        int_ptr->matrix_->AMGUnsmoothedAggregationProlongNnz(global_col_begin,
                                                             global_col_end,
                                                             *aggregates.vector_,
                                                             *aggregate_root_nodes.vector_,
                                                             *gst_ptr->matrix_,
                                                             f2c_map.vector_,
                                                             prolong->matrix_interior_.matrix_,
                                                             prolong->matrix_ghost_.matrix_);

        // Temporary array to store global ghost columns
        LocalVector<int64_t> ghost_col;
        ghost_col.CloneBackend(*this);
        ghost_col.Zeros();

        // Fill column indices and values of P
        int_ptr->matrix_->AMGUnsmoothedAggregationProlongFill(global_col_begin,
                                                              global_col_end,
                                                              *aggregates.vector_,
                                                              *aggregate_root_nodes.vector_,
                                                              *f2c_map.vector_,
                                                              *gst_ptr->matrix_,
                                                              prolong->matrix_interior_.matrix_,
                                                              prolong->matrix_ghost_.matrix_,
                                                              ghost_col.vector_);

        // Ghost of P MUST be CSR
        assert(prolong->matrix_ghost_.GetFormat() == CSR);

        // Communicate global sizes of P
        int64_t global_ncol;
        int64_t local_ncol = prolong->matrix_interior_.GetN();

#ifdef SUPPORT_MULTINODE
        MRequest req;
        communication_async_allreduce_single_sum(&local_ncol, &global_ncol, this->pm_->comm_, &req);
        communication_sync(&req);
#endif

        // Setup parallel manager of P
        prolong->CreateParallelManager_();
        prolong->pm_self_->SetMPICommunicator(this->pm_->comm_);

        // To generate the parallel manager, we need to access the sorted global ghost column ids
        LocalVector<int64_t> sorted_ghost_col;
        sorted_ghost_col.CloneBackend(*this);
        sorted_ghost_col.Allocate("sorted global ghost columns", ghost_col.GetSize());

        // Sort the global ghost columns (we do not need the permutation vector)
        ghost_col.Sort(&sorted_ghost_col, NULL);

        // Get the sorted ghost columns on host
        int64_t* pghost_col = NULL;
        sorted_ghost_col.MoveToHost();
        sorted_ghost_col.LeaveDataPtr(&pghost_col);

        // Sizes
        prolong->pm_self_->SetGlobalNrow(this->pm_->global_nrow_);
        prolong->pm_self_->SetGlobalNcol(global_ncol);
        prolong->pm_self_->SetLocalNrow(this->pm_->local_nrow_);
        prolong->pm_self_->SetLocalNcol(local_ncol);

        // Generate the PM
        prolong->pm_self_->GenerateFromGhostColumnsWithParent_(
            prolong->matrix_ghost_.GetNnz(), pghost_col, *this->pm_);

        // Communicate offsets
        prolong->pm_self_->CommunicateGlobalOffsetAsync_();

        // This is a prolongation operator, means we need to convert the global
        // fine boundary columns from to local coarse columns
        // Convert local boundary columns from global fine to local coarse
        int* f2c = NULL;
        f2c_map.MoveToHost();
        f2c_map.LeaveDataPtr(&f2c);

        // Clear
        free_host(&pghost_col);

        // Sync global offsets communication
        prolong->pm_self_->CommunicateGlobalOffsetSync_();

        // Convert local boundary columns from global fine to local coarse
        prolong->pm_self_->BoundaryTransformGlobalFineToLocalCoarse_(f2c);

        // Clear
        free_host(&f2c);

        // Communicate ghost to global map
        prolong->pm_self_->CommunicateGhostToGlobalMapAsync_();

        // Finally, renumber ghost columns (from global to local)
        // We couldn't do this earlier, because the parallel manager need
        // to know the global ghost column ids
        prolong->matrix_ghost_.matrix_->RenumberGlobalToLocal(*ghost_col.vector_);

        // Synchronize
        prolong->pm_self_->CommunicateGhostToGlobalMapSync_();

        prolong->SetParallelManager(*prolong->pm_self_);

        // Rename P
        prolong->object_name_ = "Prolongation Operator of " + this->object_name_;

        const int*     boundary_index = this->pm_->GetBoundaryIndex();
        const int64_t* ghost_mapping  = this->pm_->GetGhostToGlobalMap();

        int            prolong_nsend          = prolong->pm_self_->GetNumSenders();
        int            prolong_nrecv          = prolong->pm_self_->GetNumReceivers();
        const int*     prolong_boundary_index = prolong->pm_self_->GetBoundaryIndex();
        const int64_t* prolong_ghost_mapping  = prolong->pm_self_->GetGhostToGlobalMap();

        int64_t prolong_global_row_begin = prolong->pm_->GetGlobalRowBegin();
        int64_t prolong_global_row_end   = prolong->pm_->GetGlobalRowEnd();
        int64_t prolong_global_col_begin = prolong->pm_->GetGlobalColumnBegin();
        int64_t prolong_global_col_end   = prolong->pm_->GetGlobalColumnEnd();

        if(this->GetFormat() != CSR)
        {
            LOG_VERBOSE_INFO(
                2,
                "*** warning: GlobalMatrix::AMGSmoothedAggregation() is performed in CSR format");
        }

#ifdef DEBUG_MODE
        prolong->Check();
#endif
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::RSCoarsening(float              eps,
                                               LocalVector<int>*  CFmap,
                                               LocalVector<bool>* S) const
    {
        log_debug(this, "GlobalMatrix::RSCoarsening()", eps, CFmap, S);

        assert(eps < 1.0f);
        assert(eps > 0.0f);
        assert(CFmap != NULL);
        assert(S != NULL);
        assert(this->is_host_() == CFmap->is_host_());
        assert(this->is_host_() == S->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif

        // Calling global routine with single process
        if(this->pm_ == NULL || this->pm_->num_procs_ == 1)
        {
            this->matrix_interior_.RSCoarsening(eps, CFmap, S);

            return;
        }

        LOG_VERBOSE_INFO(2,
                         "*** error: GlobalMatrix::RSCoarsening() is not available on GlobalMatrix "
                         "class - use PMIS coarsening instead");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::RSPMISCoarsening(float              eps,
                                                   LocalVector<int>*  CFmap,
                                                   LocalVector<bool>* S) const
    {
        log_debug(this, "GlobalMatrix::RSPMISCoarsening()", eps, CFmap, S);

        assert(eps < 1.0f);
        assert(eps > 0.0f);
        assert(CFmap != NULL);
        assert(S != NULL);
        assert(this->is_host_() == CFmap->is_host_());
        assert(this->is_host_() == S->is_host_());
        assert(this->is_host_() == this->halo_.is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif

        // Calling global routine with single process
        if(this->pm_ == NULL || this->pm_->num_procs_ == 1)
        {
            this->matrix_interior_.RSPMISCoarsening(eps, CFmap, S);

            return;
        }

        // Only CSR matrices are supported
        LocalMatrix<ValueType> csr_int;
        LocalMatrix<ValueType> csr_gst;

        const LocalMatrix<ValueType>* int_ptr = &this->matrix_interior_;
        const LocalMatrix<ValueType>* gst_ptr = &this->matrix_ghost_;

        if(int_ptr->GetFormat() != CSR)
        {
            csr_int.CloneFrom(*int_ptr);
            csr_int.ConvertToCSR();
            int_ptr = &csr_int;
        }

        if(gst_ptr->GetFormat() != CSR)
        {
            csr_gst.CloneFrom(*gst_ptr);
            csr_gst.ConvertToCSR();
            gst_ptr = &csr_gst;
        }

        if(this->GetNnz() > 0)
        {
            // Communication sizes
            int nsend = this->pm_->GetNumSenders();
            int nrecv = this->pm_->GetNumReceivers();

            // Some communication related buffers
            int*   hisend_buffer = NULL;
            int*   hirecv_buffer = NULL;
            float* hsend_buffer  = NULL;
            float* hrecv_buffer  = NULL;

            allocate_host(nsend, &hisend_buffer);
            allocate_host(nrecv, &hirecv_buffer);
            allocate_host(nsend, &hsend_buffer);
            allocate_host(nrecv, &hrecv_buffer);

            // Allocate S
            S->Allocate("S", int_ptr->GetNnz() + gst_ptr->GetNnz());

            // Sample rng
            LocalVector<float> omega;
            omega.CloneBackend(*this);
            omega.Allocate("omega", int_ptr->GetM() + nrecv);

            // Determine strong influences in the matrix
            unsigned long long seed = this->pm_->GetGlobalRowBegin();
            int_ptr->matrix_->RSPMISStrongInfluences(
                eps, S->vector_, omega.vector_, seed, *gst_ptr->matrix_);

            // Update S, omega and CF mapping from neighbors

            // Prepare send buffer
            omega.GetContinuousValues(int_ptr->GetM(), omega.GetSize(), hrecv_buffer);

            // Send omega ghost to neighbors
            this->pm_->InverseCommunicateAsync_(hrecv_buffer, hsend_buffer);

            // Synchronize
            this->pm_->InverseCommunicateSync_();

            // Update omega with received omega from neighbors
            LocalVector<float> send_buffer;
            send_buffer.CloneBackend(*this);
            send_buffer.Allocate("send buffer", nsend);
            send_buffer.CopyFromHostData(hsend_buffer);

            omega.AddIndexValues(this->halo_, send_buffer);

            // Pack updated omega
            omega.GetIndexValues(this->halo_, &send_buffer);
            send_buffer.CopyToHostData(hsend_buffer);

            // Receive updated omega from neighbors
            this->pm_->CommunicateAsync_(hsend_buffer, hrecv_buffer);

            // Allocate CF mapping
            CFmap->Allocate("CF map", int_ptr->GetM() + nrecv);

            // Mark all vertices as undecided
            CFmap->Zeros();

            LocalVector<bool> marked;
            marked.CloneBackend(*this);
            marked.Allocate("marked coarse", int_ptr->GetM() + nrecv);

            // Synchronize
            this->pm_->CommunicateSync_();

            // Update omega with received omega from neighbors
            omega.SetContinuousValues(int_ptr->GetM(), omega.GetSize(), hrecv_buffer);

            // Iteratively find coarse and fine vertices until all undecided vertices have
            // been marked (JPL approach)
            int iter = 0;

            while(true)
            {
                // First, mark all vertices that have not been assigned yet, as coarse
                int_ptr->matrix_->RSPMISUnassignedToCoarse(
                    CFmap->vector_, marked.vector_, *omega.vector_);

                // Now, correct previously marked vertices with respect to omega
                int_ptr->matrix_->RSPMISCorrectCoarse(CFmap->vector_,
                                                      *S->vector_,
                                                      *marked.vector_,
                                                      *omega.vector_,
                                                      *gst_ptr->matrix_);

                // Communicate ghost CF map to neighbors and update interior CF map with data
                // received from neighbors

                // Prepare send buffer
                CFmap->GetContinuousValues(int_ptr->GetM(), CFmap->GetSize(), hirecv_buffer);

                // Send CF map to neighbors
                this->pm_->InverseCommunicateAsync_(hirecv_buffer, hisend_buffer);

                // Synchronize
                this->pm_->InverseCommunicateSync_();

                // Update interior CF map with data received and pack send buffer
                LocalVector<int> isend_buffer;
                isend_buffer.CloneBackend(*this);
                isend_buffer.Allocate("int send buffer", nsend);
                isend_buffer.CopyFromHostData(hisend_buffer);

                CFmap->vector_->RSPMISUpdateCFmap(*this->halo_.vector_, isend_buffer.vector_);

                isend_buffer.CopyToHostData(hisend_buffer);

                // Receive updated CF map from neighbors
                this->pm_->CommunicateAsync_(hisend_buffer, hirecv_buffer);

                // Synchronize
                this->pm_->CommunicateSync_();

                // Update CFmap with received CFmap from neighbors
                CFmap->SetContinuousValues(int_ptr->GetM(), CFmap->GetSize(), hirecv_buffer);

                // Mark remaining edges of a coarse point to fine
                int_ptr->matrix_->RSPMISCoarseEdgesToFine(
                    CFmap->vector_, *S->vector_, *gst_ptr->matrix_);

                // Pack updated CF map for communication
                CFmap->GetIndexValues(this->halo_, &isend_buffer);
                isend_buffer.CopyToHostData(hisend_buffer);

                // Communicate
                this->pm_->CommunicateAsync_(hisend_buffer, hirecv_buffer);

                // Now, we need to check whether we have vertices left that are marked
                // undecided, in order to restart the loop
                bool undecided;
                int_ptr->matrix_->RSPMISCheckUndecided(undecided, *CFmap->vector_);

                // Get global number of undecided vertices
                int local_undecided = undecided;
                int global_undecided;

#ifdef SUPPORT_MULTINODE
                MRequest req;
                communication_async_allreduce_single_max(
                    &local_undecided, &global_undecided, this->pm_->GetComm(), &req);
#endif

                // Synchronize
                this->pm_->CommunicateSync_();

                // Update CFmap with received CFmap from neighbors
                CFmap->SetContinuousValues(int_ptr->GetM(), CFmap->GetSize(), hirecv_buffer);

#ifdef SUPPORT_MULTINODE
                communication_sync(&req);
#endif

                // If no more undecided vertices are left, we are done
                if(global_undecided == 0)
                {
                    break;
                }

                ++iter;

                // Print some warning if number of iteration is getting huge
                if(iter > 20)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: GlobalMatrix::RSPMISCoarsening() Current "
                                     "number of iterations: "
                                         << iter);
                }
            }

            free_host(&hsend_buffer);
            free_host(&hrecv_buffer);
            free_host(&hisend_buffer);
            free_host(&hirecv_buffer);

            omega.Clear();
        }

        std::string CFmap_name = "CF map of " + this->object_name_;
        std::string S_name     = "S of " + this->object_name_;

        CFmap->object_name_ = CFmap_name;
        S->object_name_     = S_name;

        if(this->GetFormat() != CSR)
        {
            LOG_VERBOSE_INFO(
                2, "*** warning: GlobalMatrix::RSPMISCoarsening() is performed in CSR format");
        }
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::RSDirectInterpolation(const LocalVector<int>&  CFmap,
                                                        const LocalVector<bool>& S,
                                                        GlobalMatrix<ValueType>* prolong) const
    {
        log_debug(this,
                  "GlobalMatrix::RSDirectInterpolation()",
                  (const void*&)CFmap,
                  (const void*&)S,
                  prolong);

        assert(prolong != NULL);
        assert(this != prolong);
        assert(prolong->GetFormat() == CSR);

        assert(this->is_host_() == prolong->is_host_());
        assert(this->is_host_() == CFmap.is_host_());
        assert(this->is_host_() == S.is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif

        // Calling global routine with single process
        if(this->pm_ == NULL || this->pm_->num_procs_ == 1)
        {
            this->matrix_interior_.RSDirectInterpolation(CFmap, S, &prolong->matrix_interior_);

            // Prolongation PM
            prolong->CreateParallelManager_();
            prolong->pm_self_->SetMPICommunicator(this->pm_->comm_);

            prolong->pm_self_->SetGlobalNrow(prolong->matrix_interior_.GetM());
            prolong->pm_self_->SetGlobalNcol(prolong->matrix_interior_.GetN());

            prolong->pm_self_->SetLocalNrow(prolong->matrix_interior_.GetM());
            prolong->pm_self_->SetLocalNcol(prolong->matrix_interior_.GetN());

            return;
        }

        // Only CSR matrices are supported
        LocalMatrix<ValueType> csr_int;
        LocalMatrix<ValueType> csr_gst;

        const LocalMatrix<ValueType>* int_ptr = &this->matrix_interior_;
        const LocalMatrix<ValueType>* gst_ptr = &this->matrix_ghost_;

        if(int_ptr->GetFormat() != CSR)
        {
            csr_int.CloneFrom(*int_ptr);
            csr_int.ConvertToCSR();
            int_ptr = &csr_int;
        }

        if(gst_ptr->GetFormat() != CSR)
        {
            csr_gst.CloneFrom(*gst_ptr);
            csr_gst.ConvertToCSR();
            gst_ptr = &csr_gst;
        }

        // Start with fresh P operator
        prolong->Clear();

        // fine to coarse map
        LocalVector<int> f2c_map;
        f2c_map.CloneBackend(*this);
        f2c_map.Allocate("f2c map", int_ptr->GetM() + 1);

        // Amin/max
        LocalVector<ValueType> Amin;
        LocalVector<ValueType> Amax;

        Amin.CloneBackend(*this);
        Amax.CloneBackend(*this);

        Amin.Allocate("A min", int_ptr->GetM());
        Amax.Allocate("A max", int_ptr->GetM());

        // Determine number of non-zeros for P
        int_ptr->matrix_->RSDirectProlongNnz(*CFmap.vector_,
                                             *S.vector_,
                                             *gst_ptr->matrix_,
                                             Amin.vector_,
                                             Amax.vector_,
                                             f2c_map.vector_,
                                             prolong->matrix_interior_.matrix_,
                                             prolong->matrix_ghost_.matrix_);

        // Obtain the local to global ghost mapping
        LocalVector<int64_t> l2g;
        l2g.CloneBackend(*this);
        l2g.Allocate("l2g ghost map", this->pm_->GetNumReceivers());
        l2g.CopyFromHostData(this->pm_->GetGhostToGlobalMap());

        // Temporary array to store global ghost columns
        LocalVector<int64_t> ghost_col;
        ghost_col.CloneBackend(*this);

        // Fill column indices and values of P
        int_ptr->matrix_->RSDirectProlongFill(*l2g.vector_,
                                              *f2c_map.vector_,
                                              *CFmap.vector_,
                                              *S.vector_,
                                              *gst_ptr->matrix_,
                                              *Amin.vector_,
                                              *Amax.vector_,
                                              prolong->matrix_interior_.matrix_,
                                              prolong->matrix_ghost_.matrix_,
                                              ghost_col.vector_);

        Amin.Clear();
        Amax.Clear();

        // Ghost of P MUST be CSR
        assert(prolong->matrix_ghost_.GetFormat() == CSR);

        // Communicate global sizes of P
        int64_t global_ncol;
        int64_t local_ncol = prolong->GetLocalN();

#ifdef SUPPORT_MULTINODE
        MRequest req;
        communication_async_allreduce_single_sum(&local_ncol, &global_ncol, this->pm_->comm_, &req);
#endif

        // Setup parallel manager of P
        prolong->CreateParallelManager_();
        prolong->pm_self_->SetMPICommunicator(this->pm_->comm_);

        // To generate the parallel manager, we need to access the sorted global ghost column ids
        LocalVector<int64_t> sorted_ghost_col;
        sorted_ghost_col.CloneBackend(*this);
        sorted_ghost_col.Allocate("sorted global ghost columns", ghost_col.GetSize());

        // Sort the global ghost columns (we do not need the permutation vector)
        ghost_col.Sort(&sorted_ghost_col, NULL);

        // Get the sorted ghost columns on host
        int64_t* pghost_col = NULL;
        sorted_ghost_col.MoveToHost();
        sorted_ghost_col.LeaveDataPtr(&pghost_col);

#ifdef SUPPORT_MULTINODE
        communication_sync(&req);
#endif

        // Sizes
        prolong->pm_self_->SetGlobalNrow(this->pm_->global_nrow_);
        prolong->pm_self_->SetGlobalNcol(global_ncol);
        prolong->pm_self_->SetLocalNrow(this->pm_->local_nrow_);
        prolong->pm_self_->SetLocalNcol(local_ncol);

        // Generate the PM
        prolong->pm_self_->GenerateFromGhostColumnsWithParent_(
            prolong->matrix_ghost_.GetNnz(), pghost_col, *this->pm_);

        // Communicate global offsets
        prolong->pm_self_->CommunicateGlobalOffsetAsync_();

        // This is a prolongation operator, means we need to convert the global
        // fine boundary columns from to local coarse columns
        int* f2c = NULL;
        f2c_map.MoveToHost();
        f2c_map.LeaveDataPtr(&f2c);

        // Clear
        free_host(&pghost_col);

        // Sync global offsets communication
        prolong->pm_self_->CommunicateGlobalOffsetSync_();

        // Convert local boundary columns from global fine to local coarse
        prolong->pm_self_->BoundaryTransformGlobalFineToLocalCoarse_(f2c);

        // Communicate ghost to global map
        prolong->pm_self_->CommunicateGhostToGlobalMapAsync_();

        // Clear
        free_host(&f2c);

        // Finally, renumber ghost columns (from global to local)
        // We couldn't do this earlier, because the parallel manager need
        // to know the global ghost column ids
        prolong->matrix_ghost_.matrix_->RenumberGlobalToLocal(*ghost_col.vector_);

        // Synchronize
        prolong->pm_self_->CommunicateGhostToGlobalMapSync_();

        prolong->SetParallelManager(*prolong->pm_self_);

        // Rename P
        prolong->object_name_ = "Prolongation Operator of " + this->object_name_;

        if(this->GetFormat() != CSR)
        {
            LOG_VERBOSE_INFO(
                2, "*** warning: GlobalMatrix::RSDirectInterpolation() is performed in CSR format");
        }

#ifdef DEBUG_MODE
        prolong->Check();
#endif
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::RSExtPIInterpolation(const LocalVector<int>&  CFmap,
                                                       const LocalVector<bool>& S,
                                                       bool                     FF1,
                                                       GlobalMatrix<ValueType>* prolong) const
    {
        log_debug(this,
                  "GlobalMatrix::RSExtPIInterpolation()",
                  (const void*&)CFmap,
                  (const void*&)S,
                  FF1,
                  prolong);

        assert(prolong != NULL);
        assert(this != prolong);

        assert(prolong->GetFormat() == CSR);

        assert(this->is_host_() == prolong->is_host_());
        assert(this->is_host_() == CFmap.is_host_());
        assert(this->is_host_() == S.is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif

        // Calling global routine with single process
        if(this->pm_ == NULL || this->pm_->num_procs_ == 1)
        {
            this->matrix_interior_.RSExtPIInterpolation(CFmap, S, FF1, &prolong->matrix_interior_);

            // Prolongation PM
            prolong->CreateParallelManager_();
            prolong->pm_self_->SetMPICommunicator(this->pm_->comm_);

            prolong->pm_self_->SetGlobalNrow(prolong->matrix_interior_.GetM());
            prolong->pm_self_->SetGlobalNcol(prolong->matrix_interior_.GetN());

            prolong->pm_self_->SetLocalNrow(prolong->matrix_interior_.GetM());
            prolong->pm_self_->SetLocalNcol(prolong->matrix_interior_.GetN());

            return;
        }

        // Only CSR matrices are supported
        LocalMatrix<ValueType> csr_int;
        LocalMatrix<ValueType> csr_gst;

        const LocalMatrix<ValueType>* int_ptr = &this->matrix_interior_;
        const LocalMatrix<ValueType>* gst_ptr = &this->matrix_ghost_;

        if(int_ptr->GetFormat() != CSR)
        {
            csr_int.CloneFrom(*int_ptr);
            csr_int.ConvertToCSR();
            int_ptr = &csr_int;
        }

        if(gst_ptr->GetFormat() != CSR)
        {
            csr_gst.CloneFrom(*gst_ptr);
            csr_gst.ConvertToCSR();
            gst_ptr = &csr_gst;
        }

        // Start with fresh P operator
        prolong->Clear();

        // Communication sizes
        int nsend = this->pm_->GetNumSenders();
        int nrecv = this->pm_->GetNumReceivers();

        // To do Ext+I interpolation, we need to fetch additional rows of A, such that
        // we can locally access all rows that do not belong to the current rank, but
        // are required due to column dependencies.
        // This means, we have to send all rows of A where A has a dependency on to
        // the neighboring rank
        int A_ext_m_send = nsend;

        // First, count the total number of non-zero entries per boundary row that are strongly connected
        // and coarse points, including ghost part (we need to send full rows, not only interior)
        LocalVector<PtrType> A_ext_row_ptr_send;
        A_ext_row_ptr_send.CloneBackend(*this);
        A_ext_row_ptr_send.Allocate("A ext row ptr", A_ext_m_send + 1);

        int_ptr->matrix_->RSExtPIBoundaryNnz(*this->halo_.vector_,
                                             *CFmap.vector_,
                                             *S.vector_,
                                             *gst_ptr->matrix_,
                                             A_ext_row_ptr_send.vector_);

        // Transfer send buffer to host
        PtrType* send_buffer = NULL;
        allocate_host(A_ext_m_send + 1, &send_buffer);
        A_ext_row_ptr_send.CopyToHostData(send_buffer);

        // Number of rows of A we receive from our neighbors
        int A_ext_m_recv = nrecv;

        // Host receive buffer
        PtrType* hA_ext_row_nnz_recv = NULL;
        allocate_host(A_ext_m_recv + 1, &hA_ext_row_nnz_recv);

        // Initiate communication of nnz per row (async)
        this->pm_->CommunicateAsync_(send_buffer, hA_ext_row_nnz_recv);

        // Extract ghost to global map
        LocalVector<int64_t> l2g;
        l2g.CloneBackend(*this);
        l2g.Allocate("A ghost map", nrecv);
        l2g.CopyFromHostData(this->pm_->GetGhostToGlobalMap());

        // Exclusive sum to obtain row pointers
        PtrType A_ext_nnz_send = A_ext_row_ptr_send.ExclusiveSum();

        // Global column offset for this rank
        int64_t global_col_begin = this->pm_->GetGlobalColumnBegin();
        int64_t global_col_end   = this->pm_->GetGlobalColumnEnd();

        // Now, extract boundary column indices and values of all strongly connected coarse points
        LocalVector<int64_t> A_ext_col_ind_send;
        A_ext_col_ind_send.CloneBackend(*this);
        A_ext_col_ind_send.Allocate("A ext col ind send", A_ext_nnz_send);

        int_ptr->matrix_->RSExtPIExtractBoundary(global_col_begin,
                                                 *this->halo_.vector_,
                                                 *l2g.vector_,
                                                 *CFmap.vector_,
                                                 *S.vector_,
                                                 *gst_ptr->matrix_,
                                                 *A_ext_row_ptr_send.vector_,
                                                 A_ext_col_ind_send.vector_);

        // Allocate row pointer array
        LocalVector<PtrType> A_ext_row_ptr_recv;

        // Synchronize communication
        this->pm_->CommunicateSync_();

        // Place our received data into structure
        A_ext_row_ptr_recv.SetDataPtr(&hA_ext_row_nnz_recv, "A ext row ptr", A_ext_m_recv + 1);
        A_ext_row_ptr_recv.CloneBackend(*this);

        // Obtain row pointers
        PtrType A_ext_nnz_recv = A_ext_row_ptr_recv.ExclusiveSum();

        // We need a copy of A_ext_row_ptr on the host for communication
        PtrType* hA_ext_row_ptr_recv = NULL;
        allocate_host(A_ext_m_recv + 1, &hA_ext_row_ptr_recv);
        A_ext_row_ptr_recv.CopyToHostData(hA_ext_row_ptr_recv);

        // We need a copy of A_ext_col_ind_send on host for communication
        int64_t* pA_ext_col_ind_send = NULL;
        A_ext_col_ind_send.MoveToHost();
        A_ext_col_ind_send.LeaveDataPtr(&pA_ext_col_ind_send);

        // Initiate communication of column indices and values
        int64_t* hA_ext_col_ind_recv = NULL;
        allocate_host(A_ext_nnz_recv, &hA_ext_col_ind_recv);

        // We need a copy of A_ext_row_ptr on host for communication
        PtrType* hA_ext_row_ptr_send = NULL;
        allocate_host(A_ext_m_send + 1, &hA_ext_row_ptr_send);
        A_ext_row_ptr_send.CopyToHostData(hA_ext_row_ptr_send);

        this->pm_->CommunicateCSRAsync_(hA_ext_row_ptr_send,
                                        pA_ext_col_ind_send,
                                        (ValueType*)NULL,
                                        hA_ext_row_ptr_recv,
                                        hA_ext_col_ind_recv,
                                        (ValueType*)NULL);

        // fine to coarse map
        LocalVector<int> f2c_map;
        f2c_map.CloneBackend(*this);
        f2c_map.Allocate("f2c map", int_ptr->GetM() + 1);

        // Synchronize communication
        this->pm_->CommunicateCSRSync_();

        // Clear
        free_host(&hA_ext_row_ptr_send);
        free_host(&pA_ext_col_ind_send);

        LocalVector<int64_t> A_ext_col_ind_recv;
        A_ext_col_ind_recv.SetDataPtr(&hA_ext_col_ind_recv, "A ext col ind", A_ext_nnz_recv);
        A_ext_col_ind_recv.CloneBackend(*this);

        // Determine number of non-zeros for P
        int_ptr->RSExtPIProlongNnz(global_col_begin,
                                   global_col_end,
                                   FF1,
                                   l2g,
                                   CFmap,
                                   S,
                                   *gst_ptr,
                                   A_ext_row_ptr_recv,
                                   A_ext_col_ind_recv,
                                   &f2c_map,
                                   &prolong->matrix_interior_,
                                   &prolong->matrix_ghost_);

        // Obtain neighboring rows

        // Count the non-zeros of each boundary row, including the ghost part
        int_ptr->matrix_->ExtractBoundaryRowNnz(
            A_ext_row_ptr_send.vector_, *this->halo_.vector_, *gst_ptr->matrix_);

        // Transfer send buffer to host
        A_ext_row_ptr_send.CopyToHostData(send_buffer);

        // Initiate communication of nnz per row
        this->pm_->CommunicateAsync_(send_buffer, &hA_ext_row_ptr_recv[1]);

        // Exclusive sum to obtain row pointers of send buffer
        A_ext_nnz_send = A_ext_row_ptr_send.ExclusiveSum();

        // Extract the boundary nnz and convert them to global indices
        LocalVector<ValueType> A_ext_val_send;
        A_ext_val_send.CloneBackend(*this);

        LocalVector<int64_t> A_ext_col_ind_send_global;
        A_ext_col_ind_send_global.CloneBackend(*this);
        A_ext_col_ind_send_global.Allocate("A ext col ind send", A_ext_nnz_send);
        A_ext_val_send.Allocate("A ext val send", A_ext_nnz_send);

        int_ptr->matrix_->ExtractBoundaryRows(*A_ext_row_ptr_send.vector_,
                                              A_ext_col_ind_send_global.vector_,
                                              A_ext_val_send.vector_,
                                              global_col_begin,
                                              *this->halo_.vector_,
                                              *l2g.vector_,
                                              *gst_ptr->matrix_);

        // Wait for nnz per row communication to finish
        this->pm_->CommunicateSync_();

        // Clear
        free_host(&send_buffer);

        // Exclusive sum to obtain offsets from what we just received
        hA_ext_row_ptr_recv[0] = 0;
        for(int i = 0; i < A_ext_m_recv; ++i)
        {
            hA_ext_row_ptr_recv[i + 1] += hA_ext_row_ptr_recv[i];
        }

        // Number of non-zeros we are going to receive
        A_ext_nnz_recv = hA_ext_row_ptr_recv[A_ext_m_recv];

        // Prepare communication buffers
        PtrType*   pA_ext_row_ptr_send        = NULL;
        int64_t*   pA_ext_col_ind_send_global = NULL;
        ValueType* pA_ext_val_send            = NULL;

        int64_t*   pA_ext_col_ind_recv_global = NULL;
        ValueType* pA_ext_val_recv            = NULL;

        // Allocate receive buffers
        allocate_host(A_ext_nnz_recv, &pA_ext_col_ind_recv_global);
        allocate_host(A_ext_nnz_recv, &pA_ext_val_recv);

        // Obtain send buffers
        A_ext_row_ptr_send.MoveToHost();
        A_ext_row_ptr_send.LeaveDataPtr(&pA_ext_row_ptr_send);
        A_ext_col_ind_send_global.MoveToHost();
        A_ext_col_ind_send_global.LeaveDataPtr(&pA_ext_col_ind_send_global);
        A_ext_val_send.MoveToHost();
        A_ext_val_send.LeaveDataPtr(&pA_ext_val_send);

        // Initiate communication of column indices and values
        this->pm_->CommunicateCSRAsync_(pA_ext_row_ptr_send,
                                        pA_ext_col_ind_send_global,
                                        pA_ext_val_send,
                                        hA_ext_row_ptr_recv,
                                        pA_ext_col_ind_recv_global,
                                        pA_ext_val_recv);

        // Synchronize communication
        this->pm_->CommunicateCSRSync_();

        // Clean up send buffers
        free_host(&pA_ext_row_ptr_send);
        free_host(&pA_ext_val_send);
        free_host(&pA_ext_col_ind_send_global);

        // Put received data into structure
        LocalVector<PtrType>   A_ext_row_ptr_recv_full;
        LocalVector<int64_t>   A_ext_col_ind_recv_full;
        LocalVector<ValueType> A_ext_val_recv_full;

        A_ext_row_ptr_recv_full.SetDataPtr(&hA_ext_row_ptr_recv, "", A_ext_m_recv + 1);
        A_ext_col_ind_recv_full.SetDataPtr(&pA_ext_col_ind_recv_global, "", A_ext_nnz_recv);
        A_ext_val_recv_full.SetDataPtr(&pA_ext_val_recv, "", A_ext_nnz_recv);

        A_ext_row_ptr_recv_full.CloneBackend(*this);
        A_ext_col_ind_recv_full.CloneBackend(*this);
        A_ext_val_recv_full.CloneBackend(*this);

        // Fill column indices and values of P
        LocalVector<int64_t> ghost_col;
        ghost_col.CloneBackend(*this);

        int_ptr->RSExtPIProlongFill(global_col_begin,
                                    global_col_end,
                                    FF1,
                                    l2g,
                                    f2c_map,
                                    CFmap,
                                    S,
                                    *gst_ptr,
                                    A_ext_row_ptr_recv,
                                    A_ext_col_ind_recv,
                                    A_ext_row_ptr_recv_full,
                                    A_ext_col_ind_recv_full,
                                    A_ext_val_recv_full,
                                    &prolong->matrix_interior_,
                                    &prolong->matrix_ghost_,
                                    &ghost_col);

        // Clear
        A_ext_col_ind_recv.Clear();

        // Finally, generate the parallel manager for prolongation operator
        // For this, we need to access the ghost columns.
        // Additionally, we need the fine to coarse map in order to convert the
        // received boundary indices to coarse.

        // Ghost of P MUST be CSR
        assert(prolong->matrix_ghost_.GetFormat() == CSR);

        // Communicate global sizes of P
        int64_t global_ncol;
        int64_t local_ncol = prolong->matrix_interior_.GetN();

#ifdef SUPPORT_MULTINODE
        MRequest req;
        communication_async_allreduce_single_sum(&local_ncol, &global_ncol, this->pm_->comm_, &req);
#endif

        // Setup parallel manager of P
        prolong->CreateParallelManager_();
        prolong->pm_self_->SetMPICommunicator(this->pm_->comm_);

        // To generate the parallel manager, we need to access the sorted global ghost column ids
        LocalVector<int64_t> sorted_ghost_col;
        sorted_ghost_col.CloneBackend(*this);
        sorted_ghost_col.Allocate("sorted global ghost columns", ghost_col.GetSize());

        // Sort the global ghost columns (we do not need the permutation vector)
        ghost_col.Sort(&sorted_ghost_col, NULL);

        // Get the sorted ghost columns on host
        int64_t* pghost_col = NULL;
        sorted_ghost_col.MoveToHost();
        sorted_ghost_col.LeaveDataPtr(&pghost_col);

#ifdef SUPPORT_MULTINODE
        communication_sync(&req);
#endif

        // Sizes
        prolong->pm_self_->SetGlobalNrow(this->pm_->global_nrow_);
        prolong->pm_self_->SetGlobalNcol(global_ncol);
        prolong->pm_self_->SetLocalNrow(this->pm_->local_nrow_);
        prolong->pm_self_->SetLocalNcol(local_ncol);

        // Generate the PM
        prolong->pm_self_->GenerateFromGhostColumnsWithParent_(
            prolong->matrix_ghost_.GetNnz(), pghost_col, *this->pm_);

        // Communicate offsets
        prolong->pm_self_->CommunicateGlobalOffsetAsync_();

        // This is a prolongation operator, means we need to convert the global
        // fine boundary columns from to local coarse columns
        // Convert local boundary columns from global fine to local coarse
        int* f2c = NULL;
        f2c_map.MoveToHost();
        f2c_map.LeaveDataPtr(&f2c);

        // Clear
        free_host(&pghost_col);

        // Sync global offsets communication
        prolong->pm_self_->CommunicateGlobalOffsetSync_();

        // Convert local boundary columns from global fine to local coarse
        prolong->pm_self_->BoundaryTransformGlobalFineToLocalCoarse_(f2c);

        // Communicate ghost to global map
        prolong->pm_self_->CommunicateGhostToGlobalMapAsync_();

        // Clear
        free_host(&f2c);

        // Finally, renumber ghost columns (from global to local)
        // We couldn't do this earlier, because the parallel manager need
        // to know the global ghost column ids
        prolong->matrix_ghost_.matrix_->RenumberGlobalToLocal(*ghost_col.vector_);

        // Synchronize
        prolong->pm_self_->CommunicateGhostToGlobalMapSync_();

        prolong->SetParallelManager(*prolong->pm_self_);

        // Rename P
        prolong->object_name_ = "Prolongation Operator of " + this->object_name_;

        if(this->GetFormat() != CSR)
        {
            LOG_VERBOSE_INFO(
                2, "*** warning: GlobalMatrix::RSExtPIInterpolation() is performed in CSR format");
        }

#ifdef DEBUG_MODE
        prolong->Check();
#endif
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::CreateParallelManager_(void)
    {
        if(this->pm_self_)
        {
            this->pm_self_->Clear();
        }
        else
        {
            this->pm_self_ = new ParallelManager;
        }

        this->pm_ = this->pm_self_;
    }

    template <typename ValueType>
    void GlobalMatrix<ValueType>::InitCommPattern_(void)
    {
#ifdef SUPPORT_MULTINODE
        int64_t global_nnz_int;
        int64_t global_nnz_gst;
        int64_t local_nnz_int = this->GetLocalNnz();
        int64_t local_nnz_gst = this->GetGhostNnz();

        MRequest req_int;
        MRequest req_gst;

        communication_async_allreduce_single_sum(
            &local_nnz_int, &global_nnz_int, this->pm_->comm_, &req_int);
        communication_async_allreduce_single_sum(
            &local_nnz_gst, &global_nnz_gst, this->pm_->comm_, &req_gst);
#endif

        // Allocate send and receive buffer
        std::string halo_name = "Buffer of " + this->object_name_;
        this->halo_.Allocate(halo_name, this->pm_->GetNumSenders());
        this->halo_.CopyFromHostData(this->pm_->GetBoundaryIndex());

        this->recv_buffer_.Allocate("receive buffer", this->pm_->GetNumReceivers());
        this->send_buffer_.Allocate("send buffer", this->pm_->GetNumSenders());

        if(this->recv_boundary_ == NULL)
        {
            allocate_pinned(this->pm_->GetNumReceivers(), &this->recv_boundary_);
        }

        if(this->send_boundary_ == NULL)
        {
            allocate_pinned(this->pm_->GetNumSenders(), &this->send_boundary_);
        }

#ifdef SUPPORT_MULTINODE
        // Synchronize
        communication_sync(&req_int);
        communication_sync(&req_gst);

        this->nnz_ = global_nnz_int + global_nnz_gst;
#endif
    }

    template class GlobalMatrix<double>;
    template class GlobalMatrix<float>;
#ifdef SUPPORT_COMPLEX
    template class GlobalMatrix<std::complex<double>>;
    template class GlobalMatrix<std::complex<float>>;
#endif

} // namespace rocalution

#ifndef SUPPORT_MULTINODE
// LCOV_EXCL_STOP
#endif