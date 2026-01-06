/*! \file */
/* ************************************************************************
* Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_test_enum.hpp"

template <typename T>
inline void rocsparse_test_name_suffix_generator_print(std::ostream& s, T item)
{
    s << item;
}

#define ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(ENUM_TYPE, TOSTRING)      \
    template <>                                                                         \
    inline void rocsparse_test_name_suffix_generator_print<ENUM_TYPE>(std::ostream & s, \
                                                                      ENUM_TYPE item)   \
    {                                                                                   \
        s << TOSTRING(item);                                                            \
    }

ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_matrix_init,
                                                      rocsparse_matrix2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_indextype,
                                                      rocsparse_indextype2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_datatype,
                                                      rocsparse_datatype2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_index_base,
                                                      rocsparse_indexbase2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_operation,
                                                      rocsparse_operation2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_matrix_type,
                                                      rocsparse_matrixtype2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_diag_type,
                                                      rocsparse_diagtype2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_fill_mode,
                                                      rocsparse_fillmode2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_storage_mode,
                                                      rocsparse_storagemode2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_action, rocsparse_action2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_hyb_partition,
                                                      rocsparse_partition2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_analysis_policy,
                                                      rocsparse_analysis2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_solve_policy,
                                                      rocsparse_solve2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_direction,
                                                      rocsparse_direction2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_order, rocsparse_order2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_format, rocsparse_format2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_sddmm_alg,
                                                      rocsparse_sddmmalg2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_spmv_alg, rocsparse_spmvalg2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_spsv_alg, rocsparse_spsvalg2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_sptrsv_alg,
                                                      rocsparse_sptrsvalg2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_spitsv_alg,
                                                      rocsparse_spitsvalg2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_spsm_alg, rocsparse_spsmalg2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_sptrsm_alg,
                                                      rocsparse_sptrsmalg2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_spmm_alg, rocsparse_spmmalg2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_spgemm_alg,
                                                      rocsparse_spgemmalg2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_spgeam_alg,
                                                      rocsparse_spgeamalg2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_sparse_to_dense_alg,
                                                      rocsparse_sparsetodensealg2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_dense_to_sparse_alg,
                                                      rocsparse_densetosparsealg2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_gtsv_interleaved_alg,
                                                      rocsparse_gtsvinterleavedalg2string);
ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocsparse_gpsv_interleaved_alg,
                                                      rocsparse_gpsvalg2string);

#undef ROCSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE

template <typename T>
inline void rocsparse_test_name_suffix_generator_remain(std::ostream& s, T item)
{
    rocsparse_test_name_suffix_generator_print(s << "_", item);
}

inline void rocsparse_test_name_suffix_generator_remain(std::ostream& s) {}
template <typename T, typename... R>
inline void rocsparse_test_name_suffix_generator_remain(std::ostream& s, T item, R... remains)
{
    rocsparse_test_name_suffix_generator_print(s << "_", item);
    rocsparse_test_name_suffix_generator_remain(s, remains...);
}

template <typename T, typename... R>
inline void rocsparse_test_name_suffix_generator(std::ostream& s, T item, R... remains)
{
    rocsparse_test_name_suffix_generator_print(s, item);
    rocsparse_test_name_suffix_generator_remain(s, remains...);
}
