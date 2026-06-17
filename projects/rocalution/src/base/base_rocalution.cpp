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

#include "base_rocalution.hpp"
#include "../utils/def.hpp"
#include "../utils/log.hpp"
#include "parallel_manager.hpp"

namespace rocalution
{
    /// Global obj tracking structure
    Rocalution_Object_Data Rocalution_Object_Data_Tracking;

    RocalutionObj::RocalutionObj()
    {
        log_debug(this, "RocalutionObj::RocalutionObj()");

#ifndef OBJ_TRACKING_OFF
        this->global_obj_id_ = _rocalution_add_obj(this);
#else
        this->global_obj_id_ = 0;
#endif
    }

    RocalutionObj::~RocalutionObj()
    {
        log_debug(this, "RocalutionObj::RocalutionObj()");

#ifndef OBJ_TRACKING_OFF
        bool status = false;
        status      = _rocalution_del_obj(this, this->global_obj_id_);

        if(status != true)
        {
            LOG_INFO("Error: rocALUTION tracking problem");
            FATAL_ERROR(__FILE__, __LINE__);
        }
#else
// nothing
#endif
    }

    template <typename ValueType>
    BaseRocalution<ValueType>::BaseRocalution()
    {
        log_debug(this, "BaseRocalution::BaseRocalution()");

        // copy the backend description
        this->local_backend_ = *_get_backend_descriptor();

        this->asyncf_ = false;

        assert(_get_backend_descriptor()->init == true);
    }

    // LCOV_EXCL_START
    template <typename ValueType>
    BaseRocalution<ValueType>::BaseRocalution(const BaseRocalution<ValueType>& src)
    {
        log_debug(this, "BaseRocalution::BaseRocalution()", (const void*&)src);

        LOG_INFO("no copy constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    BaseRocalution<ValueType>::~BaseRocalution()
    {
        log_debug(this, "BaseRocalution::~BaseRocalution()");
    }

    // LCOV_EXCL_START
    template <typename ValueType>
    BaseRocalution<ValueType>&
        BaseRocalution<ValueType>::operator=(const BaseRocalution<ValueType>& src)
    {
        log_debug(this, "BaseRocalution::operator=()", (const void*&)src);

        LOG_INFO("no overloaded operator=()");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    void BaseRocalution<ValueType>::CloneBackend(const BaseRocalution<ValueType>& src)
    {
        log_debug(this, "BaseRocalution::CloneBackend()", "with the same ValueType");

        assert(this != &src);

        this->local_backend_ = src.local_backend_;
        this->pm_            = src.pm_;

        if(src.is_host_())
        {
            // move to host
            this->MoveToHost();
        }
        else
        {
            assert(src.is_accel_());

            // move to accelerator
            this->MoveToAccelerator();
        }
    }

    template <typename ValueType>
    template <typename ValueType2>
    void BaseRocalution<ValueType>::CloneBackend(const BaseRocalution<ValueType2>& src)
    {
        log_debug(this, "BaseRocalution::CloneBackend()", "with different ValueType");

        this->local_backend_ = src.local_backend_;
        this->pm_            = src.pm_;

        if(src.is_host_())
        {
            // move to host
            this->MoveToHost();
        }
        else
        {
            assert(src.is_accel_());

            // move to accelerator
            this->MoveToAccelerator();
        }
    }

    template <typename ValueType>
    void BaseRocalution<ValueType>::MoveToAcceleratorAsync(void)
    {
        // default call
        this->MoveToAccelerator();
    }

    template <typename ValueType>
    void BaseRocalution<ValueType>::MoveToHostAsync(void)
    {
        // default call
        this->MoveToHost();
    }

    template <typename ValueType>
    void BaseRocalution<ValueType>::Sync(void)
    {
        _rocalution_sync();
        this->asyncf_ = false;
    }

    template class BaseRocalution<double>;
    template class BaseRocalution<float>;
#ifdef SUPPORT_COMPLEX
    template class BaseRocalution<std::complex<double>>;
    template class BaseRocalution<std::complex<float>>;
#endif
    template class BaseRocalution<bool>;
    template class BaseRocalution<int>;
    template class BaseRocalution<int64_t>;

    template void BaseRocalution<bool>::CloneBackend(const BaseRocalution<double>& src);
    template void BaseRocalution<bool>::CloneBackend(const BaseRocalution<float>& src);

    template void BaseRocalution<int>::CloneBackend(const BaseRocalution<double>& src);
    template void BaseRocalution<int>::CloneBackend(const BaseRocalution<float>& src);

    template void BaseRocalution<int64_t>::CloneBackend(const BaseRocalution<double>& src);
    template void BaseRocalution<int64_t>::CloneBackend(const BaseRocalution<float>& src);

    template void BaseRocalution<float>::CloneBackend(const BaseRocalution<double>& src);
    template void BaseRocalution<double>::CloneBackend(const BaseRocalution<float>& src);

#ifdef SUPPORT_COMPLEX
    template void
        BaseRocalution<int>::CloneBackend(const BaseRocalution<std::complex<double>>& src);
    template void BaseRocalution<int>::CloneBackend(const BaseRocalution<std::complex<float>>& src);

    template void
        BaseRocalution<int64_t>::CloneBackend(const BaseRocalution<std::complex<double>>& src);
    template void
        BaseRocalution<int64_t>::CloneBackend(const BaseRocalution<std::complex<float>>& src);

    template void
        BaseRocalution<bool>::CloneBackend(const BaseRocalution<std::complex<double>>& src);
    template void
        BaseRocalution<bool>::CloneBackend(const BaseRocalution<std::complex<float>>& src);

    template void
        BaseRocalution<float>::CloneBackend(const BaseRocalution<std::complex<double>>& src);
    template void
        BaseRocalution<float>::CloneBackend(const BaseRocalution<std::complex<float>>& src);
    template void BaseRocalution<std::complex<float>>::CloneBackend(
        const BaseRocalution<std::complex<double>>& src);
    template void BaseRocalution<std::complex<double>>::CloneBackend(
        const BaseRocalution<std::complex<float>>& src);
#endif

} // namespace rocalution
