/* ************************************************************************
* Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once
#include "auto_testing_bad_arg.hpp"
#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "rocsparse.hpp"
#include "rocsparse_check.hpp"
#include "rocsparse_matrix_factory.hpp"
#include "rocsparse_reproducibility.hpp"
#include "rocsparse_reproducibility_test_save.hpp"
#include "rocsparse_traits.hpp"
#include "rocsparse_vector_utils.hpp"
#include "utility.hpp"

#include "rocsparse_allocator.hpp"
#include <variant>

namespace rocsparse_clients
{

    template <typename T, typename I, typename J>
    struct spmat_csr_descr
    {
        using host_t   = host_csr_matrix<T, I, J>;
        using device_t = device_csr_matrix<T, I, J>;

    protected:
        host_t   m_host{};
        device_t m_device{};

    public:
        inline host_t& host()
        {
            return this->m_host;
        }
        inline device_t& device()
        {
            return this->m_device;
        }
        inline const host_t& host() const
        {
            return this->m_host;
        }
        inline const device_t& device() const
        {
            return this->m_device;
        }
    };

    template <typename T, typename I, typename J>
    struct spmat_csc_descr
    {
        using host_t   = host_csc_matrix<T, I, J>;
        using device_t = device_csc_matrix<T, I, J>;

    protected:
        host_t   m_host{};
        device_t m_device{};

    public:
        inline host_t& host()
        {
            return this->m_host;
        }
        inline device_t& device()
        {
            return this->m_device;
        }
        inline const host_t& host() const
        {
            return this->m_host;
        }
        inline const device_t& device() const
        {
            return this->m_device;
        }
    };

    template <typename T, typename I>
    struct spmat_ell_descr
    {
        using host_t   = host_ell_matrix<T, I>;
        using device_t = device_ell_matrix<T, I>;

    protected:
        host_t   m_host{};
        device_t m_device{};

    public:
        inline host_t& host()
        {
            return this->m_host;
        }
        inline device_t& device()
        {
            return this->m_device;
        }
        inline const host_t& host() const
        {
            return this->m_host;
        }
        inline const device_t& device() const
        {
            return this->m_device;
        }
    };

    template <typename T, typename I>
    struct spmat_bell_descr
    {
        using host_t   = host_bell_matrix<T, I>;
        using device_t = device_bell_matrix<T, I>;

    protected:
        host_t   m_host{};
        device_t m_device{};

    public:
        inline host_t& host()
        {
            return this->m_host;
        }
        inline device_t& device()
        {
            return this->m_device;
        }
        inline const host_t& host() const
        {
            return this->m_host;
        }
        inline const device_t& device() const
        {
            return this->m_device;
        }
    };

    template <typename T, typename I, typename J>
    struct spmat_sell_descr
    {
        using host_t   = host_sell_matrix<T, I, J>;
        using device_t = device_sell_matrix<T, I, J>;

    protected:
        host_t   m_host{};
        device_t m_device{};

    public:
        inline host_t& host()
        {
            return this->m_host;
        }
        inline device_t& device()
        {
            return this->m_device;
        }
        inline const host_t& host() const
        {
            return this->m_host;
        }
        inline const device_t& device() const
        {
            return this->m_device;
        }
    };

    template <typename T, typename I, typename J>
    struct spmat_bsr_descr
    {
        using host_t   = host_gebsr_matrix<T, I, J>;
        using device_t = device_gebsr_matrix<T, I, J>;

    protected:
        host_t   m_host{};
        device_t m_device{};

    public:
        inline host_t& host()
        {
            return this->m_host;
        }
        inline device_t& device()
        {
            return this->m_device;
        }
        inline const host_t& host() const
        {
            return this->m_host;
        }
        inline const device_t& device() const
        {
            return this->m_device;
        }
    };

    template <typename T, typename I>
    struct spmat_coo_aos_descr
    {
        using host_t   = host_coo_aos_matrix<T, I>;
        using device_t = device_coo_aos_matrix<T, I>;

    protected:
        host_t   m_host{};
        device_t m_device{};

    public:
        inline host_t& host()
        {
            return this->m_host;
        }
        inline device_t& device()
        {
            return this->m_device;
        }
        inline const host_t& host() const
        {
            return this->m_host;
        }
        inline const device_t& device() const
        {
            return this->m_device;
        }
    };

    template <typename T, typename I>
    struct spmat_coo_descr
    {
        using host_t   = host_coo_matrix<T, I>;
        using device_t = device_coo_matrix<T, I>;

    protected:
        host_t   m_host{};
        device_t m_device{};

    public:
        inline host_t& host()
        {
            return this->m_host;
        }
        inline device_t& device()
        {
            return this->m_device;
        }
        inline const host_t& host() const
        {
            return this->m_host;
        }
        inline const device_t& device() const
        {
            return this->m_device;
        }
    };

    template <typename T, typename I, typename J = I>
    struct spmat_descr
    {
    private:
        using coo_t     = spmat_coo_descr<T, I>;
        using coo_aos_t = spmat_coo_aos_descr<T, I>;
        using csr_t     = spmat_csr_descr<T, I, J>;
        using csc_t     = spmat_csc_descr<T, I, J>;
        using ell_t     = spmat_ell_descr<T, I>;
        using bell_t    = spmat_bell_descr<T, I>;
        using bsr_t     = spmat_bsr_descr<T, I, J>;
        using sell_t    = spmat_sell_descr<T, I, J>;

        std::variant<coo_t, coo_aos_t, csr_t, csc_t, ell_t, bell_t, bsr_t, sell_t> m_data;

        static std::variant<coo_t, coo_aos_t, csr_t, csc_t, ell_t, bell_t, bsr_t, sell_t>
            make_variant(rocsparse_format format)
        {
            switch(format)
            {
            case rocsparse_format_coo:
            {
                return coo_t{};
            }
            case rocsparse_format_coo_aos:
            {
                return coo_aos_t{};
            }
            case rocsparse_format_csr:
            {
                return csr_t{};
            }
            case rocsparse_format_csc:
            {
                return csc_t{};
            }
            case rocsparse_format_ell:
            {
                return ell_t{};
            }
            case rocsparse_format_bell:
            {
                return bell_t{};
            }
            case rocsparse_format_bsr:
            {
                return bsr_t{};
            }
            case rocsparse_format_sell:
            {
                return sell_t{};
            }
            }
        }

    protected:
        rocsparse_format       m_format{};
        rocsparse_local_spmat* m_local_descr{};
        int64_t                m_batch_count{1};
        int64_t                m_batch_val_stride{0};
        void                   init(const Arguments& arg, int64_t batch_count, bool full_rank);

    public:
        ~spmat_descr();

        bool             is_square() const;
        int64_t          get_nrows() const;
        int64_t          get_stride() const;
        int64_t          get_batch_count() const;
        rocsparse_format get_format() const;
        int64_t          get_size_values() const;
        int64_t          get_size_rows() const;
        int64_t          get_size_cols() const;
        void             reinit_values();

        const device_dense_vector<T>& get_device_values() const;
        const host_vector<T>&         get_host_values() const;
        device_dense_vector<T>&       get_device_values();
        host_vector<T>&               get_host_values();

        template <rocsparse_format FORMAT>
        auto& as()
        {
            return std::get<FORMAT>(this->m_data);
        }

        template <rocsparse_format FORMAT>
        auto& host()
        {
            return std::get<FORMAT>(this->m_data).host();
        }

        template <rocsparse_format FORMAT>
        auto& device()
        {
            return std::get<FORMAT>(this->m_data).device();
        }

        explicit spmat_descr(const Arguments& arg, int64_t batch_count, bool full_rank);

        void near_check_values(const host_dense_vector<int64_t>& symbolic,
                               const host_dense_vector<int64_t>& numeric);

        operator rocsparse_spmat_descr&();
        operator const rocsparse_spmat_descr&() const;
    };

}
