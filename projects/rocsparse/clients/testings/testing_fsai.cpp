/*! \file */
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

#include "testing_fsai.hpp"
#include "rocsparse_clients_objects.hpp"
#include "rocsparse_clients_spmat_descr.hpp"
#include "rocsparse_enum.hpp"
#include "testing.hpp"

namespace rocsparse_clients
{
    class local_fsai_descr
    {
    private:
        rocsparse_handle     m_handle{};
        rocsparse_fsai_descr m_descr{};

    public:
        local_fsai_descr(rocsparse_handle handle)
            : m_handle(handle)
        {
            ROCSPARSE_CLIENTS_ROUTINE_TRACE;
            rocsparse_error*       p_error = nullptr;
            const rocsparse_status status
                = rocsparse_fsai_descr_create(handle, &this->m_descr, p_error);
            if(status != rocsparse_status_success)
            {
                throw(status);
            }
        }

        ~local_fsai_descr()
        {
            ROCSPARSE_CLIENTS_ROUTINE_TRACE;
            rocsparse_error* p_error = nullptr;
            std::ignore = rocsparse_fsai_descr_destroy(this->m_handle, this->m_descr, p_error);
        }

        inline operator rocsparse_fsai_descr&()
        {
            return this->m_descr;
        }

        inline operator const rocsparse_fsai_descr&() const
        {
            return this->m_descr;
        }
    };
}

template <typename I, typename J, typename T>
void testing_fsai_bad_arg(const Arguments& arg)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;

    // Test invalid handle
    {
        rocsparse_fsai_descr descr = nullptr;
        rocsparse_error*     p_error = nullptr;
        EXPECT_ROCSPARSE_STATUS(rocsparse_fsai_descr_create(nullptr, &descr, p_error),
                                rocsparse_status_invalid_handle);
    }

    // Test invalid descriptor pointer
    {
        rocsparse_error* p_error = nullptr;
        EXPECT_ROCSPARSE_STATUS(rocsparse_fsai_descr_create(handle, nullptr, p_error),
                                rocsparse_status_invalid_pointer);
    }

    // Create valid descriptor for further tests
    rocsparse_fsai_descr descr = nullptr;
    {
        rocsparse_error* p_error = nullptr;
        EXPECT_ROCSPARSE_STATUS(rocsparse_fsai_descr_create(handle, &descr, p_error),
                                rocsparse_status_success);
    }

    // Test set_input with invalid handle
    {
        rocsparse_fsai_alg alg = rocsparse_fsai_alg_default;
        rocsparse_error*   p_error = nullptr;
        EXPECT_ROCSPARSE_STATUS(rocsparse_fsai_descr_set_input(nullptr,
                                                               descr,
                                                               rocsparse_fsai_input_alg,
                                                               &alg,
                                                               sizeof(alg),
                                                               p_error),
                                rocsparse_status_invalid_handle);
    }

    // Test set_input with invalid descriptor
    {
        rocsparse_fsai_alg alg = rocsparse_fsai_alg_default;
        rocsparse_error*   p_error = nullptr;
        EXPECT_ROCSPARSE_STATUS(rocsparse_fsai_descr_set_input(handle,
                                                               nullptr,
                                                               rocsparse_fsai_input_alg,
                                                               &alg,
                                                               sizeof(alg),
                                                               p_error),
                                rocsparse_status_invalid_pointer);
    }

    // Test set_input with invalid data pointer
    {
        rocsparse_error* p_error = nullptr;
        EXPECT_ROCSPARSE_STATUS(rocsparse_fsai_descr_set_input(handle,
                                                               descr,
                                                               rocsparse_fsai_input_alg,
                                                               nullptr,
                                                               sizeof(rocsparse_fsai_alg),
                                                               p_error),
                                rocsparse_status_invalid_pointer);
    }

    // Test set_input with invalid size
    {
        rocsparse_fsai_alg alg = rocsparse_fsai_alg_default;
        rocsparse_error*   p_error = nullptr;
        EXPECT_ROCSPARSE_STATUS(rocsparse_fsai_descr_set_input(handle,
                                                               descr,
                                                               rocsparse_fsai_input_alg,
                                                               &alg,
                                                               0, // invalid size
                                                               p_error),
                                rocsparse_status_invalid_size);
    }

    // Clean up
    {
        rocsparse_error* p_error = nullptr;
        EXPECT_ROCSPARSE_STATUS(rocsparse_fsai_descr_destroy(handle, descr, p_error),
                                rocsparse_status_success);
    }
}

void testing_fsai_extra(const Arguments& arg)
{
    // Extra tests for FSAI (e.g., edge cases, special configurations)
    // Currently empty - expand as needed
}

template <typename I, typename J, typename T>
void testing_fsai(const Arguments& arg)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // TODO: Implement full FSAI testing once the compute kernel is integrated
    // This is a placeholder that tests basic descriptor functionality

    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;

    rocsparse_clients::local_fsai_descr fsai(handle);

    // Set algorithm
    {
        rocsparse_fsai_alg alg = rocsparse_fsai_alg_default;
        rocsparse_error*   p_error = nullptr;
        EXPECT_ROCSPARSE_STATUS(rocsparse_fsai_descr_set_input(handle,
                                                               fsai,
                                                               rocsparse_fsai_input_alg,
                                                               &alg,
                                                               sizeof(alg),
                                                               p_error),
                                rocsparse_status_success);
    }

    // Set compute datatype
    {
        rocsparse_datatype compute_datatype = rocsparse_datatype_f64_r;
        rocsparse_error*   p_error = nullptr;
        EXPECT_ROCSPARSE_STATUS(rocsparse_fsai_descr_set_input(handle,
                                                               fsai,
                                                               rocsparse_fsai_input_compute_datatype,
                                                               &compute_datatype,
                                                               sizeof(compute_datatype),
                                                               p_error),
                                rocsparse_status_success);
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                         \
    template void testing_fsai_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_fsai<ITYPE, JTYPE, TTYPE>(const Arguments& arg);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

#undef INSTANTIATE
