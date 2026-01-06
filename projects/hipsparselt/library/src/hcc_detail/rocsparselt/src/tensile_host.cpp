/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// The implementation of the rocsparselt<->Tensile interface layer.

#include "tensile_host.hpp"
#include "activation.hpp"
#include "definitions.h"
#include "rocsparselt_spmm_utils.hpp"
#include "status.h"
#include "utility.hpp"
/*****************************************************************************
 * This is the only file in rocsparselt which should #include Tensile headers    *
 * or reference Tensile identifiers. tensile_host.hpp defines the interface. *
 *****************************************************************************/

//#include <Tensile/AMDGPU.hpp>
#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/PlaceholderLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>
#include <atomic>
#include <complex>
#include <exception>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include <glob.h>
#include <libgen.h>
#include <link.h>
#include <unistd.h>

#define ROCSPARSELT_LIB_PATH "/opt/rocm/hipsparselt/lib"

namespace
{
 #ifndef WIN32
    int rocsparselt_dl_iterate_phdr_callback(struct dl_phdr_info* hdr_info, size_t size, void* data)
    {
        std::pair<std::string, std::string>* typedData
            = reinterpret_cast<std::pair<std::string, std::string>*>(data);
        if(hdr_info->dlpi_name && strstr(hdr_info->dlpi_name, typedData->second.c_str()))
        {
            typedData->first.assign(hdr_info->dlpi_name);
            return 1;
        }
        return 0;
    }
#endif

    std::string rocsparselt_internal_get_so_path(const std::string& keyword)
    {
        std::pair<std::string, std::string> result{"", keyword};
        dl_iterate_phdr(rocsparselt_dl_iterate_phdr_callback, &result);
        return result.first;
    }

    /******************************************************
     * Map a rocsparselt type to a corresponding Tensile type *
     ******************************************************/
    template <typename T>
    struct rocsparselt_to_tensile_type
    {
        using tensile_type = T;
    };

    template <>
    struct rocsparselt_to_tensile_type<__half>
    {
        using tensile_type = TensileLite::Half;
    };

    template <>
    struct rocsparselt_to_tensile_type<hip_bfloat16>
    {
        using tensile_type = TensileLite::BFloat16;
    };

    // int8_t -> int8_t (supported for MI-kernel) / rocsparselt_int8x4 -> PackedInt8x4
    template <>
    struct rocsparselt_to_tensile_type<int8_t>
    {
        using tensile_type = int8_t;
    };

    /********************************************************************
     * Variable template to map a rocsparselt type into a rocisa::DataType *
     ********************************************************************/
    template <typename>
    constexpr auto tensile_datatype = nullptr;

    // int8_t -> int8_t (supported for MI-kernel) / rocsparselt_int8x4 -> PackedInt8x4
    template <>
    constexpr auto tensile_datatype<int8_t> = rocisa::DataType::Int8;

    template <>
    constexpr auto tensile_datatype<__half> = rocisa::DataType::Half;

    template <>
    constexpr auto tensile_datatype<hip_bfloat16> = rocisa::DataType::BFloat16;

    template <>
    constexpr auto tensile_datatype<float> = rocisa::DataType::Float;

    template <>
    constexpr auto tensile_datatype<__hip_fp8_e4m3> = rocisa::DataType::Float8;

    template <>
    constexpr auto tensile_datatype<__hip_fp8_e5m2> = rocisa::DataType::BFloat8;

    /*************************************************************************
     * Class for converting alpha and beta between rocsparselt and Tensile types *
     * By default, alpha and beta are the same type as Tc compute_type       *
     *************************************************************************/
    template <typename Ti, typename To = Ti, typename Tc = To>
    struct AlphaBeta
    {
        using tensile_type = typename rocsparselt_to_tensile_type<Tc>::tensile_type;
        static void copy(tensile_type* dst, const Tc* src)
        {
            static_assert(sizeof(*src) == sizeof(*dst),
                          "Tensile and rocsparselt types are not the same size");
            static_assert(std::is_standard_layout<tensile_type>{} && std::is_standard_layout<Tc>{},
                          "Tensile or rocsparselt types are not standard layout types");
            memcpy(dst, src, sizeof(*dst));
        }
    };

    /******************************************************
    * Map a rocsparselt data type to a corresponding Tensile type *
    ******************************************************/
    inline rocisa::DataType hipDataType_to_tensile_type(hipDataType type)
    {
        switch(type)
        {
        case HIP_R_16F:
            return rocisa::DataType::Half;
        case HIP_R_32F:
            return rocisa::DataType::Float;
        case HIP_R_16BF:
            return rocisa::DataType::BFloat16;
        case HIP_R_8I:
            return rocisa::DataType::Int8;
#if HIP_FP8_TYPE_OCP
        case HIP_R_8F_E4M3:
            return rocisa::DataType::Float8;
        case HIP_R_8F_E5M2:
            return rocisa::DataType::BFloat8;
#endif
        default:
            assert(!"hipblasltDatatype_to_tensile_type: non-supported type");
            return rocisa::DataType::None;
        }
    }

    /****************************************************************
     * Construct a Tensile Problem from a RocsparseltContractionProblem *
     ****************************************************************/
    template <typename Ti, typename To, typename Tc>
    auto ConstructTensileProblem(const RocsparseltContractionProblem<Ti, To, Tc>& prob,
                                 int                                              useBias          = 0,
                                 int                                              useScaleAlphaVec = 0)
    {
        // Tensile DataTypes corresponding to rocsparselt data types
        static constexpr rocisa::DataType Tensile_Ti = tensile_datatype<Ti>;
        static constexpr rocisa::DataType Tensile_To = tensile_datatype<To>;
        static constexpr rocisa::DataType Tensile_Tc = tensile_datatype<Tc>;

        // Tensor descriptors for a, b
        TensileLite::TensorDescriptor a, b;

        // Tensile Indices for contraction problem
        TensileLite::ContractionProblemGemm::FreeIndices  freeIndex(2);
        TensileLite::ContractionProblemGemm::BoundIndices boundIndex(1);
        TensileLite::ContractionProblemGemm::BatchIndices batchIndex{{2, 2, 2, 2}};

        // Set up GEMM indices
        freeIndex[0].isA = true;
        freeIndex[1].isA = false;
        freeIndex[0].c = freeIndex[0].d = 0;
        freeIndex[1].c = freeIndex[1].d = 1;

        // We set K=0 when alpha==0.
        // This makes alpha==0 a change in the problem, and not just a change in the inputs.
        // It optimizes all problems with alpha==0 into K=0 and alpha=(don't care)
        auto k = prob.k && (prob.alpha_vector_scaling || *prob.alpha) ? prob.k : 0;

        // clang-format off

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_a != rocsparselt_operation_none)
        {
            a = {
                    "a",
                    Tensile_Ti,
                    {k, prob.m, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a}
                };
            freeIndex[0].i  = 1;
            boundIndex[0].a = 0;
        }
        else
        {
            a = {
                    "a",
                    Tensile_Ti,
                    {prob.m, k, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a}
                };
            freeIndex[0].i  = 0;
            boundIndex[0].a = 1;
        }

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_b != rocsparselt_operation_none)
        {
            b = {
                    "b",
                    Tensile_Ti,
                    {prob.n, k, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b}
                };
            freeIndex[1].i  = 0;
            boundIndex[0].b = 1;
        }
        else
        {
            b = {
                    "b",
                    Tensile_Ti,
                    {k, prob.n, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b}
                };
            freeIndex[1].i  = 1;
            boundIndex[0].b = 0;
        }

        // clang-format on

        // Descriptor for input matrix C
        TensileLite::TensorDescriptor c{"c",
                                    Tensile_To,
                                    {prob.m, prob.n, prob.batch_count},
                                    {prob.row_stride_c, prob.col_stride_c, prob.batch_stride_c}};

        // Descriptor for output matrix D
        TensileLite::TensorDescriptor d{"d",
                                    Tensile_To,
                                    {prob.m, prob.n, prob.batch_count},
                                    {prob.row_stride_d, prob.col_stride_d, prob.batch_stride_d}};

        TensileLite::TensorDescriptor e{"e"};
        TensileLite::TensorDescriptor bias{"bias"};
        TensileLite::TensorDescriptor scaleA{"scaleA"};
        TensileLite::TensorDescriptor scaleB{"scaleB"};
        TensileLite::TensorDescriptor scaleC{"scaleC"};
        TensileLite::TensorDescriptor scaleD{"scaleD"};
        TensileLite::TensorDescriptor scaleAlphaVec{"scaleAlphaVec"};

        // The ContractionProblemGemm
        TensileLite::ContractionProblemGemm tensileProblem{a,
                                                       b,
                                                       c,
                                                       d,
                                                       e,
                                                       bias,
                                                       scaleA,
                                                       scaleB,
                                                       scaleC,
                                                       scaleD,
                                                       scaleAlphaVec,
                                                       freeIndex,
                                                       batchIndex,
                                                       boundIndex,
                                                       *prob.beta,
                                                       prob.workspaceSize};
        tensileProblem.setComputeInputType(Tensile_Ti);
        tensileProblem.setAlphaType(Tensile_Tc);
        tensileProblem.setBetaType(Tensile_Tc);

        // HPA is active iff sizeof(compute type) > sizeof(input type)
        tensileProblem.setHighPrecisionAccumulate(sizeof(Tc) > sizeof(Ti));

        // set batch mode
        tensileProblem.setStridedBatched(prob.strided_batch);

        // alpha and beta are stored by value in TensileLite::TypedContractionInputs
        // alpha and beta are copied from host to TensileLite::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set tensileAlpha=0
        // Not positive if this is necessary here as well
        typename AlphaBeta<Ti, To, Tc>::tensile_type tensileAlpha;
        const Tc                                     ALPHAONE = static_cast<Tc>(1);
        if(prob.k)
        {
            if(prob.alpha_vector_scaling)
                AlphaBeta<Ti, To, Tc>::copy(&tensileAlpha, &ALPHAONE);
            else
                AlphaBeta<Ti, To, Tc>::copy(&tensileAlpha, prob.alpha);
        }
        else
            memset(&tensileAlpha, 0, sizeof(tensileAlpha));
        tensileProblem.setAlphaRestriction(TensileLite::toScalarValueEnum(tensileAlpha));

        // Add problem predicates for CEqualsD
        tensileProblem.setCEqualsD(prob.C == prob.D);

        tensileProblem.setSparse(prob.sparseA ? 1 : 2);

        // set Actvation
        tensileProblem.setActivationType(TensileLite::ActivationType::All);
        tensileProblem.setActivationComputeType(Tensile_Tc);
        TensileLite::ActivationType tensileAct = TensileLite::ActivationType::None;

        switch(prob.act_type)
        {
        case hipsparselt_activation_type::abs:
            tensileAct = TensileLite::ActivationType::Abs;
            break;
        case hipsparselt_activation_type::clippedrelu:
            tensileAct = TensileLite::ActivationType::Clippedrelu;
            break;
        case hipsparselt_activation_type::gelu:
            if(prob.act_arg0 == 1.f)
                tensileAct = TensileLite::ActivationType::Gelu;
            else
                tensileAct = TensileLite::ActivationType::Geluscaling;
            break;
        case hipsparselt_activation_type::leakyrelu:
            tensileAct = TensileLite::ActivationType::Leakyrelu;
            break;
        case hipsparselt_activation_type::relu:
            tensileAct = TensileLite::ActivationType::Relu;
            break;
        case hipsparselt_activation_type::sigmoid:
            tensileAct = TensileLite::ActivationType::Sigmoid;
            break;
        case hipsparselt_activation_type::tanh:
            tensileAct = TensileLite::ActivationType::Tanh;
            break;
        default:
            break;
        }
        tensileProblem.setParams().setActivationEnum(tensileAct);

        assert((useBias == useScaleAlphaVec) || (useBias == 0 || useScaleAlphaVec == 0));

        int guessedFactorDim = (prob.order == rocsparselt_order_row) ? 2 : 1;
        // set bias mode
        if(prob.bias_vector != nullptr || useBias > 0)
        {
            tensileProblem.setUseBias(useBias > 0 ? useBias : guessedFactorDim);
            tensileProblem.setBias(hipDataType_to_tensile_type(prob.bias_type),
                                   prob.order == rocsparselt_order_row ? d.sizes()[1]
                                                                       : d.sizes()[0],
                                   prob.bias_stride,
                                   false,
                                   TensileLite::ContractionProblemGemm::TENSOR::D,
                                   prob.order == rocsparselt_order_row);
        }

        if(prob.alpha_vector_scaling || useScaleAlphaVec > 0)
        {

            tensileProblem.setUseScaleAlphaVec(useScaleAlphaVec > 0 ? useScaleAlphaVec
                                                                    : guessedFactorDim);
            tensileProblem.setScaleAlphaVec(Tensile_Tc,
                                            prob.order == rocsparselt_order_row ? d.sizes()[1]
                                                                                : d.sizes()[0],
                                            prob.order == rocsparselt_order_row);
        }
        return tensileProblem;
    }

    /***************************************************************
     * Construct the inputs to a Tensile ContractionProblemGemm        *
     ***************************************************************/
    template <typename Ti, typename To, typename Tc>
    auto GetTensileInputs(const RocsparseltContractionProblem<Ti, To, Tc>& prob)
    {
        // Tensile types corresponding to Ti, To, Tc
        using Tensile_Ti          = typename rocsparselt_to_tensile_type<Ti>::tensile_type;
        using Tensile_To          = typename rocsparselt_to_tensile_type<To>::tensile_type;
        using Tensile_Talpha_beta = typename AlphaBeta<Ti, To, Tc>::tensile_type;

        // Make sure rocsparselt and Tensile types are compatible
        // (Even if Ti=rocsparselt_int8x4, Tensile_Ti=Int8x4, they are both 32-byte)
        static_assert(sizeof(Tensile_Ti) == sizeof(Ti) && sizeof(Tensile_To) == sizeof(To),
                      "Tensile and rocsparselt types are not the same size");

        static_assert(std::is_standard_layout<Ti>{} && std::is_standard_layout<Tensile_Ti>{}
                          && std::is_standard_layout<To>{} && std::is_standard_layout<Tensile_To>{},
                      "Tensile or rocsparselt types are not standard layout types");

        // Structure describing the inputs (A, B, C, D, alpha, beta)
        TensileLite::ContractionInputs inputs;

        // Set the A, B, C, D matrices pointers in Tensile
        inputs.a = reinterpret_cast<const void*>(prob.A);
        inputs.b = reinterpret_cast<const void*>(prob.B);
        inputs.c = reinterpret_cast<const void*>(prob.C);
        inputs.d = reinterpret_cast<void*>(prob.D);

        inputs.compressed = prob.sparseA ? inputs.a : inputs.b;

        inputs.batchA = reinterpret_cast<void const* const*>(prob.batch_A);
        inputs.batchB = reinterpret_cast<void const* const*>(prob.batch_B);
        inputs.batchC = reinterpret_cast<void const* const*>(prob.batch_C);
        inputs.batchD = reinterpret_cast<void* const*>(prob.batch_D);

        // Set the GSU workspace
        inputs.ws = prob.workspace;
        inputs.Synchronizer = prob.workspace;

        // set bias vector
        inputs.bias = reinterpret_cast<const void*>(prob.bias_vector);
        if(prob.alpha_vector_scaling)
            inputs.scaleAlphaVec = reinterpret_cast<const void*>(prob.alpha);

        // alpha and beta are stored by value in TensileLite::TypedContractionInputs
        // alpha and beta are copied from host to TensileLite::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set inputs.alpha=0
        if(prob.k)
        {
            if(prob.alpha_vector_scaling)
                inputs.alpha = static_cast<Tensile_Talpha_beta>(1);
            else
                inputs.alpha = static_cast<Tensile_Talpha_beta>((*prob.alpha));
        }
        else
            inputs.alpha = static_cast<Tensile_Talpha_beta>(0);
        inputs.beta = static_cast<Tensile_Talpha_beta>((*prob.beta));

        inputs.metadata = reinterpret_cast<const unsigned char*>(prob.metadata);

        // push 2 activation arguments
        inputs.activationArgs.push_back(static_cast<Tensile_Talpha_beta>(prob.act_arg0));
        inputs.activationArgs.push_back(static_cast<Tensile_Talpha_beta>(prob.act_arg1));

        return inputs;
    }

    TensileLite::LazyLoadingInit getLazyLoadingArch(int deviceID)
    {
        hipDeviceProp_t deviceProperties;
        HIP_CHECK_EXC(hipGetDeviceProperties(&deviceProperties, deviceID));
        // strip out xnack/ecc from name
        std::string deviceFullString(deviceProperties.gcnArchName);
        std::string deviceString = deviceFullString.substr(0, deviceFullString.find(":"));

        if(deviceString.find("gfx942") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx942;
        }
        else if(deviceString.find("gfx950") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx950;
        }
        return TensileLite::LazyLoadingInit::None;
    }

    /**************************************************
     * The TensileHost struct interfaces with Tensile *
     **************************************************/
    class TensileHost
    {
        // The library object
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>> m_library;
        std::unordered_set<TensileLite::LazyLoadingInit>                  m_deviceSet;
        std::unordered_map<std::string, std::shared_ptr<hipDeviceProp_t>> m_devicePropMap;

        // The adapter object. mutable is used to allow adapters to be modified
        // even when they are stored in a const vector which is immutable in size
        struct adapter_s
        {
            mutable std::atomic<TensileLite::hip::SolutionAdapter*> adapter{nullptr};
            mutable std::mutex                                  mutex;
        };

        // Each device contains an adapter
        std::vector<adapter_s> const m_adapters;

    public:
        TensileHost()
            : m_adapters(GetDeviceCount())
        {
            // We mark TensileHost as initialized. This is so that CI tests can
            // verify that the initialization occurs in the "multiheaded" tests
            rocsparselt_internal_tensile_is_initialized() = true;
        }

        // TensileHost is not copyable or assignable
        TensileHost(const TensileHost&)            = delete;
        TensileHost& operator=(const TensileHost&) = delete;

        // Get the number of devices
        static int GetDeviceCount()
        {
            int count;
            if(hipGetDeviceCount(&count) != hipSuccess)
            {
                hipsparselt_cerr
                    << "\nhipsparselt_error: Could not initialize Tensile host: No devices found"
                    << std::endl;
                hipsparselt_abort();
            }
            return count;
        }

        ~TensileHost()
        {
            for(auto& a : m_adapters)
                delete a.adapter;
        }

        auto& get_library() const
        {
            return m_library;
        }

        auto& get_device_property(const std::string& deviceName) const
        {
            return m_devicePropMap.at(deviceName);
        }

        auto& get_adapters() const
        {
            return m_adapters;
        }

        /*******************************************************
         * Testpath() tests that a path exists and is readable *
         *******************************************************/
        static bool TestPath(const std::string& path)
        {
#ifdef WIN32
            return ((_access(path.c_str(), 4) != -1) || (_access(path.c_str(), 6) != -1));
#else
            return access(path.c_str(), R_OK) == 0;
#endif
        }

        /*********************************************************************
         * Initialize adapter and library according to environment variables *
         * and default paths based on librocsparselt.so location and GPU         *
         *********************************************************************/
        void initialize(TensileLite::hip::SolutionAdapter& adapter, int32_t deviceId)
        {
            std::string path;
#ifndef WIN32
            path.reserve(PATH_MAX);
#endif

            // The name of the current GPU platform
            std::string processor = rocsparselt_internal_get_arch_name();

            const char* env = getenv("ROCSPARSELT_TENSILE_LIBPATH");
            if(env)
            {
                path = env;
            }
            else
            {
                path = ROCSPARSELT_LIB_PATH;

                // Find the location of librocsparselt.so
                // Fall back on hard-coded path if static library or not found

#ifndef HIPSPARSELT_STATIC_LIB
                auto rocsparselt_so_path = rocsparselt_internal_get_so_path("hipsparselt");
                if(rocsparselt_so_path.size())
                    path = std::string{dirname(&rocsparselt_so_path[0])};
#endif // ifndef HIPSPARSELT_STATIC_LIB

                // Find the location of the libraries
                // The first path below is for new build system where `hipblaslt` is added as a subdirectory
                if(TestPath(path + "/../hipblaslt/Tensile/library"))
                    path += "/../hipblaslt/Tensile/library";
                else if(TestPath(path + "/../Tensile/library"))
                    path += "/../Tensile/library";
                else if(TestPath(path + "../hipsparselt/library"))
                    path += "/../hipsparselt/library";
                else
                    path += "/hipsparselt/library";

                if(TestPath(path + "/" + processor))
                    path += "/" + processor;
            }

            // only load modules for the current architecture
            auto dir = path + "/*" + processor + "*co";
#if ROCSPARSELT_TENSILE_LAZY_LOAD == 0
            bool no_match = false;
#ifdef WIN32
            std::replace(dir.begin(), dir.end(), '/', '\\');
            WIN32_FIND_DATAA finddata;
            HANDLE           hfine = FindFirstFileA(dir.c_str(), &finddata);
            if(hfine != INVALID_HANDLE_VALUE)
            {
                do
                {
                    std::string codeObjectFile = path + "\\" + finddata.cFileName;
                    adapter.loadCodeObjectFile(codeObjectFile.c_str());
                } while(FindNextFileA(hfine, &finddata));
            }
            else
            {
                no_match = true;
            }
            FindClose(hfine);
#else
            glob_t glob_result{};
            int    g = glob(dir.c_str(), GLOB_NOSORT, nullptr, &glob_result);
            if(!g)
            {
                for(size_t i = 0; i < glob_result.gl_pathc; ++i)
                    (void)adapter.loadCodeObjectFile(glob_result.gl_pathv[i]);
            }
            else if(g == GLOB_NOMATCH)
            {
                no_match = true;
            }
            else
            {
                // clang-format off
                static hipsparselt_internal_ostream& once = hipsparselt_cerr
                                    << "\nrocsparselt warning: glob(\"" << dir << "\", ...) returned "
                                    << (g == GLOB_ABORTED ? "GLOB_ABORTED"
                                                          : g == GLOB_NOSPACE ? "GLOB_NOSPACE"
                                                                              : "an unknown error")
                                    << "." << std::endl;
                (void)once;
                // clang-format on
            }
            globfree(&glob_result);
#endif
            if(no_match)
            {
                static hipsparselt_internal_ostream& once
                    = hipsparselt_cerr
                      << "\nrocsparselt warning: No paths matched " << dir
                      << ". Make sure that ROCSPARSELT_TENSILE_LIBPATH is set correctly."
                      << std::endl;
                (void)once;
            }
#endif // ROCSPARSELT_TENSILE_LAZY_LOAD == 0
            // We initialize a local static variable with a lambda function call to avoid
            // race conditions when multiple threads with different device IDs try to
            // initialize library. This ensures that only one thread initializes library,
            // and other threads trying to initialize library wait for it to complete.
            static int once = [&] {
                // Determine library path
                std::string tensileLibPath;
#if ROCSPARSELT_TENSILE_LAZY_LOAD
#ifdef TENSILE_YAML
                tensileLibPath = path + "/TensileLibrary_lazy_" + processor + ".yaml";
#else
                tensileLibPath = path + "/TensileLibrary_lazy_" + processor + ".dat";
#endif
#else
#ifdef TENSILE_YAML
                tensileLibPath = path + "/TensileLibrary_" + processor + ".yaml";
#else
                tensileLibPath = path + "/TensileLibrary_" + processor + ".dat";
#endif
#endif
                if(!TestPath(tensileLibPath))
                {
                    hipsparselt_cerr << "\nhipsparselt_error: Cannot read " << tensileLibPath << ": "
                                     << strerror(errno) << std::endl;
                    //rocsparselt_abort();
                }

                // Get devices
                hipDeviceProp_t prop;
                int             count;
                HIP_CHECK_EXC(hipGetDeviceCount(&count));
                for(int devId = 0; devId < count; devId++)
                {
                    auto deviceArch = getLazyLoadingArch(devId);
                    if(m_deviceSet.find(deviceArch) == m_deviceSet.end())
                    {
                        // populate the arch list for lazy loading
                        m_deviceSet.insert(deviceArch);
                        // populate device property map, used in finding solutions based on arch
                        HIP_CHECK_EXC(hipGetDeviceProperties(&prop, devId));
                        // strip out xnack/ecc from name
                        std::string deviceFullString(prop.gcnArchName);
                        std::string deviceString
                            = deviceFullString.substr(0, deviceFullString.find(":"));
                        m_devicePropMap[deviceString] = std::make_shared<hipDeviceProp_t>(prop);
                    }
                }

                auto lib = TensileLite::LoadLibraryFile<TensileLite::ContractionProblemGemm>(tensileLibPath);
                if(!lib)
                {
                    hipsparselt_cerr << "\nhipsparselt_error: Could not load " << tensileLibPath << std::endl;
                    return -1;
                }
                else
                {
                    using MSL = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;
                    m_library = std::dynamic_pointer_cast<MSL>(lib);
                    if(!m_library->initLibraryMapping(tensileLibPath))
                    {
                        std::cerr << "\nrocblaslt error: Could not initialize Tensile library "
                                     "mapping"
                                  << std::endl;
                    }
                }
                return 0;
            }();

            static_cast<void>(adapter.initializeLazyLoading(processor, path));


            if(!m_library && once != 0)
            {
                hipsparselt_cerr << "\nhipsparselt_error: Could not initialize Tensile library"
                                 << std::endl;
                //rocsparselt_abort();
            }
        }
    };

    // Return the library and adapter for the current HIP device
    auto& get_library_and_adapter(
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>* library
        = nullptr,
        std::shared_ptr<hipDeviceProp_t>* deviceProp = nullptr,
        int                               device     = -1)
    try
    {
        // TensileHost is initialized on the first call
        static TensileHost host;

        if(device == -1)
            if(hipGetDevice(&device) != hipSuccess)
                throw "Invalid Device";

        // Adapter entry for the current HIP device ID
        auto& a       = host.get_adapters().at(device);
        auto* adapter = a.adapter.load(std::memory_order_acquire);

        // Once set, a.adapter contains the adapter for the current HIP device ID
        if(!adapter)
        {
            // Lock so that only one thread performs initialization of the adapter
            std::lock_guard<std::mutex> lock(a.mutex);

            adapter = a.adapter.load(std::memory_order_relaxed);
            if(!adapter)
            {
                // Allocate a new adapter using the current HIP device
                adapter = new TensileLite::hip::SolutionAdapter;

                // Initialize the adapter and possibly the library
                host.initialize(*adapter, device);

                // Atomically change the adapter stored for this device ID
                a.adapter.store(adapter, std::memory_order_release);
            }
        }

        // If an adapter is found, it is assumed that the library is initialized
        if(library)
            *library = host.get_library();
        if(deviceProp)
            *deviceProp = host.get_device_property(rocsparselt_internal_get_arch_name());

        return *adapter;
    }
    catch(const std::exception& e)
    {
        hipsparselt_cerr << "\nhipsparselt_error: Could not initialize Tensile host:\n"
                         << e.what() << std::endl;
        hipsparselt_abort();
    }
    catch(...)
    {
        hipsparselt_cerr
            << "\nhipsparselt_error: Could not initialize Tensile host:\nUnknown exception thrown"
            << std::endl;
        hipsparselt_abort();
    }

    /**************************************************************************
    * We normally print error messages only once, to avoid excessive logging *
    **************************************************************************/
    void print_once(const hipsparselt_internal_ostream& msg)
    {
        if(rocsparselt_suppress_tensile_error_messages())
            return;
        static constexpr char varname[] = "ROCSPARSELT_VERBOSE_TENSILE_ERROR";
        static const char*    verbose   = getenv(varname);
        if(!verbose)
        {
            static auto& once = hipsparselt_cerr
                                << msg
                                << "\nThis message will be only be displayed once, unless the "
                                << varname << " environment variable is set." << std::endl;
            (void)once;
        }
        else
            hipsparselt_cerr << msg << std::endl;
    }

} // namespace

/******************************************************************************
 * runContractionProblem calls Tensile to run a contraction problem described *
 * by RocsparseltContractionProblem                                               *
 ******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocsparselt_status runContractionProblem(const RocsparseltContractionProblem<Ti, To, Tc>& prob,
                                         _rocsparselt_matmul_config*                      configs,
                                         int*                                             config_id,
                                         const int config_max_id,
                                         const int search_iterations)
{
    rocsparselt_status                            status = rocsparselt_status_internal_error;
    std::shared_ptr<TensileLite::ContractionSolution> solution;

    try
    {
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>> library;
        std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
        std::shared_ptr<TensileLite::Hardware>                                               hardware;

        auto& adapter = get_library_and_adapter(&library, &deviceProp, prob.handle->device);

        hardware = TensileLite::hip::GetDevice(*deviceProp);

        if(!config_max_id || configs == nullptr)
        {
            hipsparselt_internal_ostream msg;
            print_once(msg << "\nhipsparselt_error: No Tensile solution found for " << prob);
            status = rocsparselt_status_not_implemented;
        }
        else
        {
            auto tensile_prob = ConstructTensileProblem(
                prob, configs[*config_id].use_bias, configs[*config_id].use_scale_alpha_vec);

            auto tensile_inputs = GetTensileInputs(prob);

            if(!search_iterations)
            {
                if(configs[*config_id].max_workspace_bytes + configs[*config_id].synchronizer_bytes > prob.workspaceSize
                   || (configs[*config_id].max_workspace_bytes + configs[*config_id].synchronizer_bytes > 0 && prob.workspace == nullptr))
                {
                    hipsparselt_cerr << "config " << *config_id << " need extra workspace "
                                     << configs[*config_id].max_workspace_bytes << " bytes."
                                     << std::endl;
                    return rocsparselt_status_internal_error;
                }

                solution = library->getSolutionByIndex(
                    tensile_prob, *hardware, configs[*config_id].index);
                if(!solution)
                {
                    hipsparselt_cerr << "Solution of config:" << *config_id
                                     << " does not exists - skip" << std::endl;
                    return rocsparselt_status_not_implemented;
                }
                tensile_inputs.ws = (uint8_t*) tensile_inputs.Synchronizer + configs[*config_id].synchronizer_bytes;
                RETURN_IF_HIP_ERROR(
                    adapter.launchKernels(solution->solve(tensile_prob, tensile_inputs, *hardware),
                                          prob.streams[0],
                                          nullptr,
                                          nullptr));
            }
            else
            {
                float      min_ms = std::numeric_limits<float>::max();
                hipEvent_t startEvent, stopEvent;
                float      ms, sum_ms;
                RETURN_IF_HIP_ERROR(hipEventCreate(&startEvent));
                RETURN_IF_HIP_ERROR(hipEventCreate(&stopEvent));
                for(int id = 0; id < config_max_id; id++)
                {
                    if(configs[id].max_workspace_bytes + configs[id].synchronizer_bytes > prob.workspaceSize
                       || (configs[id].max_workspace_bytes + configs[id].synchronizer_bytes> 0 && prob.workspace == nullptr))
                    {
                        hipsparselt_cerr << "config " << id << " need extra workspace "
                                         << configs[id].max_workspace_bytes << " bytes - skip."
                                         << std::endl;
                        continue;
                    }

                    solution
                        = library->getSolutionByIndex(tensile_prob, *hardware, configs[id].index);
                    if(!solution)
                    {
                        hipsparselt_cerr << "Solution of config:" << id << " does not exists - skip"
                                         << std::endl;
                        continue;
                    }
                    tensile_inputs.ws = (uint8_t*) tensile_inputs.Synchronizer + configs[id].synchronizer_bytes;
                    //warm up
                    RETURN_IF_HIP_ERROR(adapter.launchKernels(
                        solution->solve(tensile_prob, tensile_inputs, *hardware),
                        prob.streams[0],
                        nullptr,
                        nullptr));

                    sum_ms = 0.0f;
                    for(int i = 0; i < search_iterations; i++)
                    {
                        RETURN_IF_HIP_ERROR(adapter.launchKernels(
                            solution->solve(tensile_prob, tensile_inputs, *hardware),
                            prob.streams[0],
                            startEvent,
                            stopEvent));
                        RETURN_IF_HIP_ERROR(hipEventSynchronize(stopEvent));
                        RETURN_IF_HIP_ERROR(hipEventElapsedTime(&ms, startEvent, stopEvent));
                        sum_ms += ms;
                    }

                    if(sum_ms < min_ms)
                    {
                        min_ms     = sum_ms;
                        *config_id = id;
                    }
                }
                RETURN_IF_HIP_ERROR(hipEventDestroy(startEvent));
                RETURN_IF_HIP_ERROR(hipEventDestroy(stopEvent));

                if(min_ms == std::numeric_limits<float>::max())
                    return rocsparselt_status_internal_error;
            }

            status = rocsparselt_status_success;
        }
    }
    catch(const std::exception& e)
    {
        hipsparselt_internal_ostream msg;
        print_once(msg << "\nhipsparselt_error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
    }
    catch(...)
    {
        hipsparselt_internal_ostream msg;
        print_once(msg << "\nhipsparselt_error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
    }

    return status;
}

/******************************************************************************
 * getBestSolutions calls Tensile's findTopSolutions and converts to          *
 * _rocsparselt_matmul_config.                                                *
 ******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocsparselt_status getBestSolutions(const RocsparseltContractionProblem<Ti, To, Tc>& prob,
                                    int                                              requestConfigs,
                                    _rocsparselt_matmul_config*                      configs,
                                    int*                                             foundConfigs)
{
    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>> library;
    std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
    std::shared_ptr<TensileLite::Hardware>                                               hardware;

    // auto &adapter =
    get_library_and_adapter(&library, &deviceProp, prob.handle->device);

    if(!library)
    {
        return rocsparselt_status_invalid_pointer;
    }

    hardware          = TensileLite::hip::GetDevice(*deviceProp);
    auto tensile_prob = ConstructTensileProblem(prob);
    // auto handle = prob.handle;
    auto solutions = library->findTopSolutions(tensile_prob, *hardware, requestConfigs);

    *foundConfigs = std::min((int)solutions.size(), requestConfigs);

    // Finding alternative solutions.
    auto findAlternativeSolution = [&](int useBias, int useScaleAlphaVec) {
        tensile_prob  = ConstructTensileProblem(prob, useBias, useScaleAlphaVec);
        solutions     = library->findTopSolutions(tensile_prob, *hardware, requestConfigs);
        *foundConfigs = std::min((int)solutions.size(), requestConfigs);
    };

    if(*foundConfigs == 0)
    {
        log_info(prob.handle, __func__, "No solution founds, try to find alternative solutions");

        for(int useBias = tensile_prob.useBias(); useBias < 4; useBias++)
        {
            for(int useScaleAlphaVec = tensile_prob.useScaleAlphaVec(); useScaleAlphaVec < 4;
                useScaleAlphaVec++)
            {
                if(useBias != 0 && useScaleAlphaVec != 0 && useScaleAlphaVec != useBias)
                    continue; //useBias and useScaleAlphaVec must in the same dimension.

                if(useBias == tensile_prob.useBias()
                   && useScaleAlphaVec == tensile_prob.useScaleAlphaVec())
                    continue; // already try in the first time.

                findAlternativeSolution(useBias, useScaleAlphaVec);
                if(*foundConfigs != 0)
                    break;
            }
            if(*foundConfigs != 0)
                break;
        }

        if(*foundConfigs != 0)
        {
            log_info(prob.handle, __func__, *foundConfigs, " alternative solutions found");
        }
    }

    for(size_t i = 0; i < *foundConfigs; i++)
    {
        auto solution                  = solutions[i];
        configs[i].index               = solution->index;
        configs[i].max_workspace_bytes = solution->requiredWorkspaceSize(tensile_prob, *hardware);
        configs[i].use_bias            = tensile_prob.useBias();
        configs[i].use_scale_alpha_vec = tensile_prob.useScaleAlphaVec();
        configs[i].synchronizer_bytes  = std::ceil(solution->requiredSynchronizerSize(tensile_prob, *hardware) / 16) * 16; // align 16
    }
    return rocsparselt_status_success;
}

/***************************************************************
 * ! \brief  Initialize rocsparselt for the current HIP device, to *
 * avoid costly startup time at the first call on that device. *
 ***************************************************************/
extern "C" void rocsparselt_initialize()
{
    get_library_and_adapter();
}

/***********************************************************************************
 * Whether Tensile has been initialized for at least one device (used for testing) *
 ***********************************************************************************/
std::atomic_bool& rocsparselt_internal_tensile_is_initialized()
{
    static std::atomic_bool init;
    return init;
}

/******************************************************************************
 * Intantiate the cases of runContractionProblem which are needed to satisfy  *
 * rocsparselt dependencies. This file's template functions are not defined in a  *
 * header file, in order to keep Tensile and rocsparselt separate.                *
 ******************************************************************************/
#define GENERATE_DEFINITIONS(Ti, To, Tc)                           \
    template rocsparselt_status runContractionProblem<Ti, To, Tc>( \
        const RocsparseltContractionProblem<Ti, To, Tc>&,          \
        _rocsparselt_matmul_config*,                               \
        int*,                                                      \
        const int,                                                 \
        const int);                                                \
    template rocsparselt_status getBestSolutions<Ti, To, Tc>(      \
        const RocsparseltContractionProblem<Ti, To, Tc>&, int, _rocsparselt_matmul_config*, int*);

GENERATE_DEFINITIONS(__half, __half, float)
GENERATE_DEFINITIONS(hip_bfloat16, hip_bfloat16, float)
GENERATE_DEFINITIONS(int8_t, int8_t, float)
GENERATE_DEFINITIONS(int8_t, __half, float)
GENERATE_DEFINITIONS(int8_t, hip_bfloat16, float)
GENERATE_DEFINITIONS(__hip_fp8_e4m3, float, float)
GENERATE_DEFINITIONS(__hip_fp8_e5m2, float, float)

#undef GENERATE_DEFINITIONS
