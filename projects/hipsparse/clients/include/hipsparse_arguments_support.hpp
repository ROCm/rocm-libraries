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

#include <vector>

#include <hipsparse.h>

#if(defined(CUDART_VERSION))
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
static void print_cuda_12_0_0_to_12_5_1_support_string()
{
    std::cout << "Warning: You are using CUDA version: " << TOSTRING(CUDART_VERSION)
              << " but this routine is not supported. See CUDA support table for this"
              << " routine below: " << std::endl;
    std::string table = "                      CUDA Version                    \n"
                        "|11.7.1|11.8.0|12.0.0|12.0.1|...|12.4.1|12.5.0|12.5.1|\n"
                        "              |<--------------supported------------->|  ";
    std::cout << table << std::endl;
}

static void print_cuda_11_3_1_to_12_5_1_support_string()
{
    std::cout << "Warning: You are using CUDA version: " << TOSTRING(CUDART_VERSION)
              << " but this routine is not supported. See CUDA support table for this"
              << " routine below: " << std::endl;
    std::string table = "                      CUDA Version                    \n"
                        "|11.2.2|11.3.0|11.3.1|11.4.0|...|12.4.1|12.5.0|12.5.1|\n"
                        "              |<--------------supported------------->|  ";
    std::cout << table << std::endl;
}

static void print_cuda_11_2_0_to_12_5_1_support_string()
{
    std::cout << "Warning: You are using CUDA version: " << TOSTRING(CUDART_VERSION)
              << " but this routine is not supported. See CUDA support table for this"
              << " routine below: " << std::endl;
    std::string table = "                      CUDA Version                    \n"
                        "|11.1.0|11.1.1|11.2.0|11.2.1|...|12.4.1|12.5.0|12.5.1|\n"
                        "              |<--------------supported------------->|  ";
    std::cout << table << std::endl;
}

static void print_cuda_10_0_0_to_12_5_1_support_string()
{
    std::cout << "Warning: You are using CUDA version: " << TOSTRING(CUDART_VERSION)
              << " but this routine is not supported. See CUDA support table for this"
              << " routine below: " << std::endl;

    std::string table = "                  CUDA Version               \n"
                        "|10.0|10.1|10.2|....|12.4.1|12.5.0|12.5.1|...\n"
                        "|<---------------supported-------------->|     ";
    std::cout << table << std::endl;
}

static void print_cuda_10_0_0_to_11_8_0_support_string()
{
    std::cout << "Warning: You are using CUDA version: " << TOSTRING(CUDART_VERSION)
              << " but this routine is not supported. See CUDA support table for this"
              << " routine below: " << std::endl;
    std::string table = "                        CUDA Version                       \n"
                        "|10.0|10.1|10.2|....|11.7.0|11.7.1|11.8.0|12.0.0|12.0.1|...\n"
                        "|<--------------supported--------------->|     ";
    std::cout << table << std::endl;
}

static void print_cuda_10_0_0_to_10_2_0_support_string()
{
    std::cout << "Warning: You are using CUDA version: " << TOSTRING(CUDART_VERSION)
              << " but this routine is not supported. See CUDA support table for this"
              << " routine below: " << std::endl;

    std::string table = "                 CUDA Version                    \n"
                        "|10.0|10.1|10.1.1|10.1.2|10.2.0|11.0.1|11.0.2|...\n"
                        "|<---------supported---------->|                   ";
    std::cout << table << std::endl;
}
#undef STRINGIFY
#undef TOSTRING
#endif

struct routine_support
{
    // Level 1
    static bool is_axpyi_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
        return true;
#else
        return false;
#endif
    }
    static bool is_doti_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
        return true;
#else
        return false;
#endif
    }
    static bool is_dotci_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
        return true;
#else
        return false;
#endif
    }
    static bool is_gthr_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
        return true;
#else
        return false;
#endif
    }
    static bool is_gthrz_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
        return true;
#else
        return false;
#endif
    }
    static bool is_roti_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
        return true;
#else
        return false;
#endif
    }
    static bool is_sctr_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
        return true;
#else
        return false;
#endif
    }

    // Level2
    static bool is_bsrsv2_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
        return true;
#else
        return false;
#endif
    }
    static bool is_coomv_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
        return true;
#else
        return false;
#endif
    }
    static bool is_csrmv_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
        return true;
#else
        return false;
#endif
    }
    static bool is_csrsv_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
        return true;
#else
        return false;
#endif
    }
    static bool is_gemvi_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
        return true;
#else
        return false;
#endif
    }
    static bool is_hybmv_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
        return true;
#else
        return false;
#endif
    }

    // Level3
    static bool is_bsrmm_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
        return true;
#else
        return false;
#endif
    }
    static bool is_bsrsm2_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
        return true;
#else
        return false;
#endif
    }
    static bool is_coomm_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
        return true;
#else
        return false;
#endif
    }
    static bool is_cscmm_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
        return true;
#else
        return false;
#endif
    }
    static bool is_csrmm_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
        return true;
#else
        return false;
#endif
    }
    static bool is_coosm_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
        return true;
#else
        return false;
#endif
    }
    static bool is_csrsm_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
        return true;
#else
        return false;
#endif
    }
    static bool is_gemmi_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
        return true;
#else
        return false;
#endif
    }
    // Extra
    static bool is_csrgeam_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
        return true;
#else
        return false;
#endif
    }
    static bool is_csrgemm_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
        return true;
#else
        return false;
#endif
    }
    // Precond
    static bool is_bsric02_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
        return true;
#else
        return false;
#endif
    }
    static bool is_bsrilu02_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
        return true;
#else
        return false;
#endif
    }
    static bool is_csric02_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
        return true;
#else
        return false;
#endif
    }
    static bool is_csrilu02_supported()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
        return true;
#else
        return false;
#endif
    }
    static bool is_gtsv2_supported()
    {
        return true;
    }
    static bool is_gtsv2_nopivot_supported()
    {
        return true;
    }
    static bool is_gtsv2_strided_batch_supported()
    {
        return true;
    }
    static bool is_gtsv_interleaved_batch_supported()
    {
        return true;
    }
    static bool is_gpsv_interleaved_batch_supported()
    {
        return true;
    }
    // Conversion
    static bool is_bsr2csr_supported()
    {
        return true;
    }
    static bool is_csr2coo_supported()
    {
        return true;
    }
    static bool is_csr2csc_supported()
    {
        return true;
    }
    static bool is_csr2hyb_supported()
    {
        return true;
    }
    static bool is_csr2bsr_supported()
    {
        return true;
    }
    static bool is_csr2gebsr_supported()
    {
        return true;
    }
    static bool is_csr2csr_compress_supported()
    {
        return true;
    }
    static bool is_coo2csr_supported()
    {
        return true;
    }
    static bool is_hyb2csr_supported()
    {
        return true;
    }
    static bool is_csr2dense_supported()
    {
        return true;
    }
    static bool is_csc2dense_supported()
    {
        return true;
    }
    static bool is_coo2dense_supported()
    {
        return true;
    }
    static bool is_dense2csr_supported()
    {
        return true;
    }
    static bool is_dense2csc_supported()
    {
        return true;
    }
    static bool is_dense2coo_supported()
    {
        return true;
    }
    static bool is_gebsr2csr_supported()
    {
        return true;
    }
    static bool is_gebsr2gebsc_supported()
    {
        return true;
    }
    static bool is_gebsr2gebsr_supported()
    {
        return true;
    }

    // Level 1
    static void print_axpyi_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_11_8_0_support_string();
#endif
    }
    static void print_doti_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_10_2_0_support_string();
#endif
    }
    static void print_dotci_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_10_2_0_support_string();
#endif
    }
    static void print_gthr_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_11_8_0_support_string();
#endif
    }
    static void print_gthrz_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_11_8_0_support_string();
#endif
    }
    static void print_roti_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_11_8_0_support_string();
#endif
    }
    static void print_sctr_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_11_8_0_support_string();
#endif
    }
    // Level 2
    static void print_bsrsv2_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_coomv_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_csrmv_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_csrsv_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_11_8_0_support_string();
#endif
    }
    static void print_gemvi_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_11_8_0_support_string();
#endif
    }
    static void print_hybmv_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_10_2_0_support_string();
#endif
    }
    // Level 3
    static void print_bsrmm_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_bsrsm2_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_coomm_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_cscmm_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_csrmm_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_coosm_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_11_3_1_to_12_5_1_support_string();
#endif
    }
    static void print_csrsm_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_11_3_1_to_12_5_1_support_string();
#endif
    }
    static void print_gemmi_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_11_8_0_support_string();
#endif
    }
    // Extra
    static void print_csrgeam_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_10_2_0_support_string();
#endif
    }
    static void print_csrgemm_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_10_2_0_support_string();
#endif
    }
    // Precond
    static void print_bsric02_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_bsrilu02_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_csric02_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_csrilu02_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_gtsv2_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_gtsv2_nopivot_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_gtsv2_strided_batch_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_gtsv_interleaved_batch_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_gpsv_interleaved_batch_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    // Conversion
    static void print_bsr2csr_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_csr2coo_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_csr2csc_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_csr2hyb_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_csr2bsr_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_csr2gebsr_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_csr2csr_compress_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_coo2csr_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_hyb2csr_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_csr2dense_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_11_2_0_to_12_5_1_support_string();
#endif
    }
    static void print_csc2dense_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_11_2_0_to_12_5_1_support_string();
#endif
    }
    static void print_coo2dense_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_11_2_0_to_12_5_1_support_string();
#endif
    }
    static void print_dense2csr_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_11_2_0_to_12_5_1_support_string();
#endif
    }
    static void print_dense2csc_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_11_2_0_to_12_5_1_support_string();
#endif
    }
    static void print_dense2coo_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_11_2_0_to_12_5_1_support_string();
#endif
    }
    static void print_gebsr2csr_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_gebsr2gebsc_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
    static void print_gebsr2gebsr_support_warning()
    {
#if(defined(CUDART_VERSION))
        print_cuda_10_0_0_to_12_5_1_support_string();
#endif
    }
};

struct csr2csc_alg_support
{
    static hipsparseCsr2CscAlg_t get_default_algorithm()
    {
#if(!defined(CUDART_VERSION))
        return HIPSPARSE_CSR2CSC_ALG_DEFAULT;
#else
#if(CUDART_VERSION >= 12000)
        return HIPSPARSE_CSR2CSC_ALG_DEFAULT;
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 12000)
        return HIPSPARSE_CSR2CSC_ALG1;
#else
        return -1;
#endif
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION))
        return "Indicates what algorithm to use when running csr2csc. Possible choices are "
               "default: 0, Alg1: 1, Alg2: 2 (default:0)";
#else
#if(CUDART_VERSION >= 12000)
        return "Indicates what algorithm to use when running csr2csc. Possible choices are "
               "default: 0, Alg1: 1 (default:0)";
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 12000)
        return "Indicates what algorithm to use when running csr2csc. Possible choices are "
               "default: 1, Alg1: 1, Alg2: 2 (default:1)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
#endif
    }

    static std::vector<int> get_supported_algorithms()
    {
#if(!defined(CUDART_VERSION))
        return std::vector<int>(
            {HIPSPARSE_CSR2CSC_ALG_DEFAULT, HIPSPARSE_CSR2CSC_ALG1, HIPSPARSE_CSR2CSC_ALG2});
#else
#if(CUDART_VERSION >= 12000)
        return std::vector<int>({HIPSPARSE_CSR2CSC_ALG_DEFAULT, HIPSPARSE_CSR2CSC_ALG1});
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 12000)
        return std::vector<int>({HIPSPARSE_CSR2CSC_ALG1, HIPSPARSE_CSR2CSC_ALG2});
#endif
#endif
        return std::vector<int>();
    }
};

struct dense2sparse_alg_support
{
    static hipsparseDenseToSparseAlg_t get_default_algorithm()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
        return HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT;
#else
        return -1;
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
        return "Indicates what algorithm to use when running dense2sparse. Possible choices are "
               "default: 0 (default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
    }
};

struct sparse2dense_alg_support
{
    static hipsparseSparseToDenseAlg_t get_default_algorithm()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
        return HIPSPARSE_SPARSETODENSE_ALG_DEFAULT;
#else
        return -1;
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
        return "Indicates what algorithm to use when running sparse2dense. Possible choices are "
               "default: 0 (default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
    }
};

struct sddmm_alg_support
{
    static hipsparseSDDMMAlg_t get_default_algorithm()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11022)
        return HIPSPARSE_SDDMM_ALG_DEFAULT;
#else
        return -1;
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11022)
        return "Indicates what algorithm to use when running sddmm. Possible choices are default: "
               "0 (default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
    }
};

struct spgemm_alg_support
{
    static hipsparseSpGEMMAlg_t get_default_algorithm()
    {
#if(!defined(CUDART_VERSION))
        return HIPSPARSE_SPGEMM_DEFAULT;
#else
#if(CUDART_VERSION >= 12000)
        return HIPSPARSE_SPGEMM_DEFAULT;
#elif(CUDART_VERSION >= 11031 && CUDART_VERSION < 12000)
        return HIPSPARSE_SPGEMM_DEFAULT;
#elif(CUDART_VERSION >= 11000)
        return HIPSPARSE_SPGEMM_DEFAULT;
#else
        return -1;
#endif
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION))
        return "Indicates what algorithm to use when running spgemm. Possible choices are default: "
               "0, Deterministic: 1, Non-Deterministic: 2, Alg1: 3, Alg2: 4, Alg3: 5 (default:0)";
#else
#if(CUDART_VERSION >= 12000)
        return "Indicates what algorithm to use when running spgemm. Possible choices are default: "
               "0, Deterministic: 1, Non-Deterministic: 2, Alg1: 3, Alg2: 4, Alg3: 5 (default:0)";
#elif(CUDART_VERSION >= 11031 && CUDART_VERSION < 12000)
        return "Indicates what algorithm to use when running spgemm. Possible choices are default: "
               "0, Deterministic: 1, Non-Deterministic: 2 (default:0)";
#elif(CUDART_VERSION >= 11000)
        return "Indicates what algorithm to use when running spgemm. Possible choices are default: "
               "0 (default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
#endif
    }
};

struct spmm_alg_support
{
    static hipsparseSpMMAlg_t get_default_algorithm()
    {
#if(!defined(CUDART_VERSION))
        return HIPSPARSE_SPMM_ALG_DEFAULT;
#else
#if(CUDART_VERSION >= 12000)
        return HIPSPARSE_SPMM_ALG_DEFAULT;
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
        return HIPSPARSE_SPMM_ALG_DEFAULT;
#elif(CUDART_VERSION >= 11003 && CUDART_VERSION < 11021)
        return HIPSPARSE_SPMM_ALG_DEFAULT;
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11003)
        return HIPSPARSE_MM_ALG_DEFAULT;
#else
        return -1;
#endif
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION))
        return "Indicates what algorithm to use when running spmm. Possible choices are default: "
               "0, COO Alg1: 1, COO Alg2: 2, COO Alg3: 3, CSR Alg1: 4, COO Alg4: 5, CSR Alg2: 6, "
               "CSR Alg3: 12, Blocked ELL Alg1: 13 (default:0)";
#else
#if(CUDART_VERSION >= 12000)
        return "Indicates what algorithm to use when running spmm. Possible choices are default: "
               "0, COO Alg1: 1, COO Alg2: 2, COO Alg3: 3, CSR Alg1: 4, COO Alg4: 5 CSR Alg2: 6, "
               "CSR Alg3: 12, Blocked ELL Alg1: 13 (default:0)";
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
        return "Indicates what algorithm to use when running spmm. Possible choices are default: "
               "0, COO Alg1: 1, COO Alg2: 2, COO Alg3: 3, CSR Alg1: 4, COO Alg4: 5, CSR Alg2: 6, "
               "CSR Alg3: 12, Blocked ELL Alg1: 13 (default:0)";
#elif(CUDART_VERSION >= 11003 && CUDART_VERSION < 11021)
        return "Indicates what algorithm to use when running spmm. Possible choices are default: "
               "0, COO Alg1: 1, COO Alg2: 2, COO Alg3: 3, CSR Alg1: 4, COO Alg4: 5, CSR Alg2: 6, "
               "Blocked ELL Alg1: 13 (default:0)";
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11003)
        return "Indicates what algorithm to use when running spmm. Possible choices are default: "
               "0, COO Alg1: 1, COO Alg2: 2, COO Alg3: 3, CSR Alg1: 4  (default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
#endif
    }
};

struct spmv_alg_support
{
    static hipsparseSpMVAlg_t get_default_algorithm()
    {
#if(!defined(CUDART_VERSION))
        return HIPSPARSE_SPMV_ALG_DEFAULT;
#else
#if(CUDART_VERSION >= 12000)
        return HIPSPARSE_SPMV_ALG_DEFAULT;
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
        return HIPSPARSE_SPMV_ALG_DEFAULT;
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11021)
        return HIPSPARSE_MV_ALG_DEFAULT;
#else
        return -1;
#endif
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION))
        return "Indicates what algorithm to use when running spmv. Possible choices are default: "
               "0, COO Alg1: 1, CSR Alg1: 2, CSR Alg2: 3, COO Alg2: 4 (default:0)";
#else
#if(CUDART_VERSION >= 12000)
        return "Indicates what algorithm to use when running spmv. Possible choices are default: "
               "0, COO Alg1: 1, CSR Alg1: 2, CSR Alg2: 3, COO Alg2: 4 (default:0)";
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
        return "Indicates what algorithm to use when running spmv. Possible choices are default: "
               "0, COO Alg1: 1, CSR Alg1: 2, CSR Alg2: 3, COO Alg2: 4 (default:0)";
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11021)
        return "Indicates what algorithm to use when running spmv. Possible choices are default: "
               "0, COO Alg: 1, CSR Alg1: 2, CSR Alg2: 3 (default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
#endif
    }
};

struct spsm_alg_support
{
    static hipsparseSpSMAlg_t get_default_algorithm()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
        return HIPSPARSE_SPSM_ALG_DEFAULT;
#else
        return -1;
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
        return "Indicates what algorithm to use when running spsm. Possible choices are default: 0 "
               "(default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
    }
};

struct spsv_alg_support
{
    static hipsparseSpSVAlg_t get_default_algorithm()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
        return HIPSPARSE_SPSV_ALG_DEFAULT;
#else
        return -1;
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
        return "Indicates what algorithm to use when running spsv. Possible choices are default: 0 "
               "(default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
    }
};
