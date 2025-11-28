/*! \file */
/* ************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "hipsparse_test_enum.hpp"

template <typename T>
inline void hipsparse_test_name_suffix_generator_print(std::ostream& s, T item)
{
    s << item;
}

#define HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(ENUM_TYPE, TOSTRING)      \
    template <>                                                                         \
    inline void hipsparse_test_name_suffix_generator_print<ENUM_TYPE>(std::ostream & s, \
                                                                      ENUM_TYPE item)   \
    {                                                                                   \
        s << TOSTRING(item);                                                            \
    }

HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseIndexType_t,
                                                      hipsparse_indextype2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipDataType, hipsparse_datatype2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseIndexBase_t,
                                                      hipsparse_indexbase2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseOperation_t,
                                                      hipsparse_operation2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseDiagType_t,
                                                      hipsparse_diagtype2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseFillMode_t,
                                                      hipsparse_fillmode2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseAction_t, hipsparse_action2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseHybPartition_t,
                                                      hipsparse_partition2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseSolvePolicy_t,
                                                      hipsparse_solvepolicy2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseDirection_t,
                                                      hipsparse_direction2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseOrder_t, hipsparse_order2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseFormat_t, hipsparse_format2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseSDDMMAlg_t,
                                                      hipsparse_sddmmalg2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseSpMVAlg_t, hipsparse_spmvalg2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseSpSVAlg_t, hipsparse_spsvalg2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseSpSMAlg_t, hipsparse_spsmalg2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseSpMMAlg_t, hipsparse_spmmalg2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseSpGEMMAlg_t,
                                                      hipsparse_spgemmalg2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseSparseToDenseAlg_t,
                                                      hipsparse_sparsetodensealg2string);
HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(hipsparseDenseToSparseAlg_t,
                                                      hipsparse_densetosparsealg2string);
#undef HIPSPARSE_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE

template <typename T>
inline void hipsparse_test_name_suffix_generator_remain(std::ostream& s, T item)
{
    hipsparse_test_name_suffix_generator_print(s << "_", item);
}

inline void hipsparse_test_name_suffix_generator_remain(std::ostream& s) {}
template <typename T, typename... R>
inline void hipsparse_test_name_suffix_generator_remain(std::ostream& s, T item, R... remains)
{
    hipsparse_test_name_suffix_generator_print(s << "_", item);
    hipsparse_test_name_suffix_generator_remain(s, remains...);
}

template <typename T, typename... R>
inline void hipsparse_test_name_suffix_generator(std::ostream& s, T item, R... remains)
{
    hipsparse_test_name_suffix_generator_print(s, item);
    hipsparse_test_name_suffix_generator_remain(s, remains...);
}