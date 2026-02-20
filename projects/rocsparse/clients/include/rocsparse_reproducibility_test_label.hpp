/*! \file */
/* ************************************************************************
* Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_enum_name.hpp"
#include <iostream>
#include <vector>
namespace rocsparse_reproducibility_utils
{
    template <typename T>
    inline void test_label_record(std::vector<std::pair<std::string, std::string>>& out, T item)
    {
    }

    template <typename T>
    inline void test_label_print(uint16_t& itemIndex, std::ostream& out, T item)
    {
    }

#define TEST_LABEL_PRINT_SPECIALIZE(ENUM_TYPE)                                                     \
    template <>                                                                                    \
    inline void test_label_print<ENUM_TYPE>(                                                       \
        uint16_t & itemIndex, std::ostream & out, ENUM_TYPE item)                                  \
    {                                                                                              \
        if(itemIndex > 0)                                                                          \
        {                                                                                          \
            out << ",";                                                                            \
        }                                                                                          \
        out << "\"" << #ENUM_TYPE << "\" : \"" << rocsparse_enum_name(item) << "\"";               \
        ++itemIndex;                                                                               \
    }                                                                                              \
    template <>                                                                                    \
    inline void test_label_record<ENUM_TYPE>(                                                      \
        std::vector<std::pair<std::string, std::string>> & out, ENUM_TYPE item)                    \
    {                                                                                              \
        out.push_back(std::pair<std::string, std::string>(#ENUM_TYPE, rocsparse_enum_name(item))); \
    }

    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_indextype);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_datatype);
#if 0
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_index_base);
#endif
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_operation);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_matrix_type);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_diag_type);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_fill_mode);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_storage_mode);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_action);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_hyb_partition);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_analysis_policy);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_solve_policy);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_direction);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_order);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_format);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_sddmm_alg);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_spmv_alg);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_spsv_alg);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_sptrsv_alg);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_spitsv_alg);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_spsm_alg);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_sptrsm_alg);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_spmm_alg);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_spgemm_alg);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_spgeam_alg);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_sparse_to_dense_alg);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_dense_to_sparse_alg);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_gtsv_interleaved_alg);
    TEST_LABEL_PRINT_SPECIALIZE(rocsparse_gpsv_interleaved_alg);

#undef TEST_LABEL_PRINT_SPECIALIZE

    template <typename T>
    inline void test_label_impl_remain(uint16_t& itemIndex, std::ostream& out, T item)
    {
        test_label_print(itemIndex, out, item);
    }

    inline void test_label_impl_remain(uint16_t& itemIndex, std::ostream& out) {}

    template <typename T, typename... R>
    inline void test_label_impl_remain(uint16_t& itemIndex, std::ostream& out, T item, R... remains)
    {
        test_label_print(itemIndex, out, item);
        test_label_impl_remain(itemIndex, out, remains...);
    }

    template <typename T, typename... R>
    inline void test_label_impl(uint16_t& itemIndex, std::ostream& out, T item, R... remains)
    {
        test_label_print(itemIndex, out, item);
        test_label_impl_remain(itemIndex, out, remains...);
    }

    template <typename T, typename... R>
    inline void test_label(std::ostream& out, T item, R... remains)
    {
        uint16_t itemIndex = 0;
        test_label_impl(itemIndex, out, item, remains...);
    }

    template <typename T>
    inline void record_impl_remain(std::vector<std::pair<std::string, std::string>>& out, T item)
    {
        test_label_record(out, item);
    }

    inline void record_impl_remain(std::vector<std::pair<std::string, std::string>>& out) {}

    template <typename T, typename... R>
    inline void record_impl_remain(std::vector<std::pair<std::string, std::string>>& out,
                                   T                                                 item,
                                   R... remains)
    {
        test_label_record(out, item);
        record_impl_remain(out, remains...);
    }

    template <typename T, typename... R>
    inline void
        record_impl(std::vector<std::pair<std::string, std::string>>& out, T item, R... remains)
    {
        test_label_record(out, item);
        record_impl_remain(out, remains...);
    }

    template <typename T, typename... R>
    inline void record(std::vector<std::pair<std::string, std::string>>& out, T item, R... remains)
    {
        record_impl(out, item, remains...);
    }

}
