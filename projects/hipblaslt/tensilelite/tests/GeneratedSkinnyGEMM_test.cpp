// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Runs the skinny MXFP8 GEMM kernels that TensileCreateLibrary generates from the
// SkinnyGEMM logic fixture, through the same path hipBLASLt uses: lazy library
// loading, predicate matching, solve and launchKernels. The library directory is
// named by TENSILELITE_GENERATED_SKINNY_GEMM_LIBRARY; without it the tests skip.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>

namespace
{
    using DataType = rocisa::DataType;
    using Problem  = TensileLite::ContractionProblemGemm;
    using Solution = TensileLite::ContractionSolution;
    using Library  = TensileLite::MasterSolutionLibrary<Problem>;

    constexpr const char* libraryEnvironment = "TENSILELITE_GENERATED_SKINNY_GEMM_LIBRARY";
    constexpr const char* kernelPrefix       = "gemm_BM";
    constexpr const char* generatedSuffix    = "_Generated_gfx950.co";

    constexpr int64_t maxExtent = int64_t(1) << 20;
    constexpr int64_t maxWide   = int64_t(1) << 30;
    constexpr int64_t mxBlock   = 32;

    // A generated solution and the K it serves: kLow < K <= kHigh and K a multiple of kMultiple.
    // The useBeta variants read C, the others only serve beta == 0.
    struct GeneratedKernel
    {
        const char* name;
        int64_t     kLow;
        int64_t     kHigh;
        int64_t     kMultiple;
        bool        useBeta;
    };

    // clang-format off
    const std::array<GeneratedKernel, 10> generatedKernels = {{
        {"gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS2_PARTIAL_STEPS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA0",
         4096,
         8192,
         4096,
         false},
        {"gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS2_PARTIAL_STEPS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1",
         4096,
         8192,
         4096,
         true},
        {"gemm_BM1_TN1_LANES_K16_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA0",
         0,
         maxExtent,
         1024,
         false},
        {"gemm_BM1_TN1_LANES_K16_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1",
         0,
         maxExtent,
         1024,
         true},
        {"gemm_BM2_TN2_LANES_K16_WAVES4_UP_ITERS0_PARTIAL_STEPS1_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA0",
         0,
         maxExtent,
         1,
         false},
        {"gemm_BM2_TN2_LANES_K16_WAVES4_UP_ITERS0_PARTIAL_STEPS1_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1",
         0,
         maxExtent,
         1,
         true},
        {"gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS2_PARTIAL_STEPS0_PF2_GROUP1_NUM_XCDS1_GRID_3D0_USE_BETA0",
         4096,
         8192,
         4096,
         false},
        {"gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS2_PARTIAL_STEPS0_PF2_GROUP1_NUM_XCDS1_GRID_3D0_USE_BETA1",
         4096,
         8192,
         4096,
         true},
        {"gemm_BM1_TN2_LANES_K16_WAVES2_UP_ITERS3_PARTIAL_STEPS1_PF2_GROUP2_NUM_XCDS8_GRID_3D0_USE_BETA0",
         2048,
         3072,
         1,
         false},
        {"gemm_BM1_TN2_LANES_K16_WAVES2_UP_ITERS3_PARTIAL_STEPS1_PF2_GROUP2_NUM_XCDS8_GRID_3D0_USE_BETA1",
         2048,
         3072,
         1,
         true},
    }};
    // clang-format on

    int64_t alignUp(int64_t value, int64_t multiple)
    {
        return (value + multiple - 1) / multiple * multiple;
    }

    int64_t ceilDiv(int64_t value, int64_t divisor)
    {
        return (value + divisor - 1) / divisor;
    }

    // A TN MXFP8 problem of the shape the generated kernels serve, and the features that make
    // it one they do not.
    struct ProblemSpec
    {
        int64_t m     = 0;
        int64_t n     = 0;
        int64_t k     = 0;
        int64_t batch = 1;
        int64_t ldA   = 0;
        int64_t ldB   = 0;
        int64_t ldC   = 0;
        int64_t ldD   = 0;

        double alpha = 1.0;
        double beta  = 0.0;
        // C is D itself: the same buffer, so ldC must equal ldD.
        bool aliasC = false;

        bool useBias       = false;
        bool activation    = false;
        bool scaleAB       = false;
        bool scaleAlphaVec = false;
        bool outputAmaxD   = false;
    };

    // Leading dimensions larger than the extent, so that a kernel reading or writing the padding
    // fails the checks. ldA and ldB are multiples of 4, ldC and ldD need no alignment.
    ProblemSpec makeSpec(int64_t m, int64_t n, int64_t k)
    {
        ProblemSpec spec;
        spec.m   = m;
        spec.n   = n;
        spec.k   = k;
        spec.ldA = alignUp(k, 4) + 16;
        spec.ldB = alignUp(k, 4) + 4;
        spec.ldD = alignUp(m, 4) + 4;
        spec.ldC = spec.ldD;
        return spec;
    }

    void setK(ProblemSpec& spec, int64_t k)
    {
        spec.k   = k;
        spec.ldA = alignUp(k, 4) + 16;
        spec.ldB = alignUp(k, 4) + 4;
    }

    std::string describe(ProblemSpec const& spec)
    {
        std::ostringstream stream;
        stream << "m=" << spec.m << " n=" << spec.n << " k=" << spec.k << " batch=" << spec.batch
               << " ldA=" << spec.ldA << " ldB=" << spec.ldB << " ldC=" << spec.ldC
               << " ldD=" << spec.ldD << " alpha=" << spec.alpha << " beta=" << spec.beta
               << " aliasC=" << spec.aliasC;
        return stream.str();
    }

    bool usesUnsupportedFeature(ProblemSpec const& spec)
    {
        return spec.useBias || spec.activation || spec.scaleAB || spec.scaleAlphaVec;
    }

    // What the SkinnyGEMM logic fixture declares for each generated kernel.
    bool expectedAccepts(GeneratedKernel const& kernel, ProblemSpec const& spec)
    {
        if(usesUnsupportedFeature(spec) || spec.batch != 1)
            return false;
        if(spec.ldA % 4 != 0 || spec.ldB % 4 != 0)
            return false;
        if(!kernel.useBeta && spec.beta != 0.0)
            return false;
        if(spec.m > maxExtent || spec.n > maxWide)
            return false;
        if(spec.k <= kernel.kLow || spec.k > kernel.kHigh)
            return false;
        return spec.k % kernel.kMultiple == 0;
    }

    void setBias(Problem& problem, ProblemSpec const& spec)
    {
        problem.setUseBias(1);
        problem.setBias(DataType::BFloat16, spec.m, 0);
        problem.setParams().setBiasEnum(DataType::BFloat16);
    }

    Problem makeProblem(ProblemSpec const& spec)
    {
        Problem problem = Problem::GEMM_Strides(true,
                                                false,
                                                DataType::Float8,
                                                DataType::Float8,
                                                DataType::BFloat16,
                                                DataType::BFloat16,
                                                spec.m,
                                                spec.n,
                                                spec.k,
                                                spec.batch,
                                                spec.ldA,
                                                spec.ldA * spec.m,
                                                spec.ldB,
                                                spec.ldB * spec.n,
                                                spec.ldC,
                                                spec.ldC * spec.n,
                                                spec.ldD,
                                                spec.ldD * spec.n,
                                                spec.beta);
        problem.setComputeInputTypeA(DataType::Float8);
        problem.setComputeInputTypeB(DataType::Float8);
        problem.setAlphaType(DataType::Float);
        problem.setBetaType(DataType::Float);
        problem.setHighPrecisionAccumulate(true);
        problem.setStridedBatched(true);
        problem.setActivationType(spec.activation ? TensileLite::ActivationType::Hipblaslt_all
                                                  : TensileLite::ActivationType::None);
        problem.setActivationComputeType(DataType::Float);
        problem.setUseScaleAB(spec.scaleAB ? "Scalar" : "");
        problem.setUseScaleAlphaVec(spec.scaleAlphaVec ? 1 : 0);
        problem.setScaleAlphaVec(DataType::Float, spec.m);
        problem.setOutputAmaxD(spec.outputAmaxD);
        problem.setAmaxD(DataType::Float, true);
        if(spec.useBias)
            setBias(problem, spec);
        problem.setMXScaleA(DataType::E8, mxBlock);
        problem.setMXScaleB(DataType::E8, mxBlock);
        return problem;
    }

    // The library and adapter, loaded the way hipBLASLt loads a lazy library.
    struct LoadedLibrary
    {
        std::shared_ptr<Library>                           library;
        std::unique_ptr<TensileLite::hip::SolutionAdapter> adapter;
        std::shared_ptr<TensileLite::Hardware>             hardware;
        std::string                                        skipReason;
    };

    std::string masterLibraryPath(std::filesystem::path const& directory)
    {
        std::filesystem::path yaml = directory / "TensileLibrary_lazy_gfx950.yaml";
        if(std::filesystem::exists(yaml))
            return yaml.string();
        return (directory / "TensileLibrary_lazy_gfx950.dat").string();
    }

    std::string skipReasonForDevice()
    {
        int count = 0;
        if(hipGetDeviceCount(&count) != hipSuccess || count == 0)
            return "no HIP device";
        hipDeviceProp_t properties;
        if(hipGetDeviceProperties(&properties, 0) != hipSuccess)
            return "cannot query device 0";
        if(std::string(properties.gcnArchName).rfind("gfx950", 0) != 0)
            return std::string("device 0 is ") + properties.gcnArchName + ", not gfx950";
        return "";
    }

    // The solution index to shard mapping exists only in msgpack libraries, YAML ones
    // resolve a placeholder by its file name.
    bool initializeMapping([[maybe_unused]] Library& library, [[maybe_unused]] std::string const& path)
    {
#ifdef TENSILE_MSGPACK
        return library.initLibraryMapping(path);
#else
        return true;
#endif
    }

    void loadLibrary(LoadedLibrary& loaded, std::filesystem::path const& directory)
    {
        std::string path = masterLibraryPath(directory);
        std::shared_ptr<TensileLite::SolutionLibrary<Problem>> lib
            = TensileLite::LoadLibraryFilePreload<Problem>(
                path, std::vector<TensileLite::LazyLoadingInit>{});
        loaded.library = std::dynamic_pointer_cast<Library>(lib);
        if(!loaded.library || !initializeMapping(*loaded.library, path))
        {
            loaded.library.reset();
            return;
        }
        loaded.adapter = std::make_unique<TensileLite::hip::SolutionAdapter>();
        static_cast<void>(loaded.adapter->initializeLazyLoading("gfx950", directory.string()));
        loaded.hardware = TensileLite::hip::GetDevice(0);
    }

    LoadedLibrary& loadedLibrary()
    {
        static LoadedLibrary loaded = [] {
            LoadedLibrary result;
            const char*   directory = std::getenv(libraryEnvironment);
            if(directory == nullptr)
            {
                result.skipReason = std::string(libraryEnvironment) + " is not set";
                return result;
            }
            result.skipReason = skipReasonForDevice();
            if(result.skipReason.empty())
                loadLibrary(result, directory);
            return result;
        }();
        return loaded;
    }

    // Host operands.
    constexpr uint8_t  nanByte      = 0x7F;
    constexpr uint16_t sentinel     = 0x5A5B;
    constexpr uint16_t nanBf16      = 0x7FC0;
    constexpr int64_t  guardCount   = 64;
    constexpr int64_t  scaleRows    = 32;
    constexpr int64_t  scaleColumns = 8;

    struct Operand
    {
        int64_t              cols = 0;
        int64_t              k    = 0;
        std::vector<uint8_t> codes; // cols x k, each column contiguous.
        std::vector<uint8_t> scales; // cols x ceil(k / 32).
        std::vector<uint8_t> image; // The device buffer, ld padded with NaN.
        std::vector<uint8_t> swizzled; // The device scale buffer.
    };

    // Finite E4M3 codes with exponents 4..10, i.e. magnitudes from 0.125 to about 15.
    uint8_t randomCode(std::mt19937_64& rng)
    {
        uint8_t sign     = rng() & 1;
        uint8_t exponent = 4 + rng() % 7;
        uint8_t mantissa = rng() % 8;
        return (sign << 7) | (exponent << 3) | mantissa;
    }

    // The byte offset of the scale of row r, scale column c in preSwizzleScalesGFX950.
    int64_t swizzledOffset(int64_t r, int64_t c, int64_t paddedCols)
    {
        return (r / 32) * (paddedCols / 8) * 256 + (c / 8) * 256 + (c % 4) * 64 + (r % 16) * 4
               + ((c / 4) % 2) * 2 + (r / 16) % 2;
    }

    std::vector<uint8_t> preSwizzleScales(std::vector<uint8_t> const& scales, int64_t rows, int64_t k)
    {
        int64_t              scaleCols  = ceilDiv(k, mxBlock);
        int64_t              paddedCols = alignUp(scaleCols, scaleColumns);
        std::vector<uint8_t> bytes(ceilDiv(rows, scaleRows) * paddedCols * scaleRows, 0);
        for(int64_t r = 0; r < rows; r++)
            for(int64_t c = 0; c < scaleCols; c++)
                bytes[swizzledOffset(r, c, paddedCols)] = scales[r * scaleCols + c];
        return bytes;
    }

    // Every column is padded to a multiple of 4 bytes, the last one not to ld.
    Operand makeOperand(std::mt19937_64& rng, int64_t cols, int64_t k, int64_t ld)
    {
        Operand operand;
        operand.cols = cols;
        operand.k    = k;
        operand.codes.resize(cols * k);
        for(uint8_t& code : operand.codes)
            code = randomCode(rng);
        operand.scales.resize(cols * ceilDiv(k, mxBlock));
        for(uint8_t& scale : operand.scales)
            scale = 125 + rng() % 5;
        operand.image.assign((cols - 1) * ld + alignUp(k, 4), nanByte);
        for(int64_t col = 0; col < cols; col++)
            std::memcpy(&operand.image[col * ld], &operand.codes[col * k], k);
        operand.swizzled = preSwizzleScales(operand.scales, cols, k);
        return operand;
    }

    // E4M3FN: bias 7, exponent 0 is subnormal (mantissa * 2^-9).
    double decodeE4M3(uint8_t code)
    {
        double sign     = (code >> 7) ? -1.0 : 1.0;
        int    exponent = (code >> 3) & 0xF;
        int    mantissa = code & 0x7;
        if(exponent == 0)
            return sign * mantissa * std::ldexp(1.0, -9);
        return sign * (1.0 + mantissa / 8.0) * std::ldexp(1.0, exponent - 7);
    }

    // The 32-element block sums are exact in double, so the order of summation does not matter.
    double dotProduct(Operand const& a, int64_t i, Operand const& b, int64_t j)
    {
        int64_t scaleCols = ceilDiv(a.k, mxBlock);
        double  sum       = 0;
        for(int64_t block = 0; block < scaleCols; block++)
        {
            double partial = 0;
            int64_t end    = std::min(a.k, (block + 1) * mxBlock);
            for(int64_t kk = block * mxBlock; kk < end; kk++)
                partial += decodeE4M3(a.codes[i * a.k + kk]) * decodeE4M3(b.codes[j * b.k + kk]);
            int exponent = int(a.scales[i * scaleCols + block]) + int(b.scales[j * scaleCols + block]) - 254;
            sum += std::ldexp(partial, exponent);
        }
        return sum;
    }

    // Column-major [n][m] reference of D = A^T B.
    std::vector<float> referenceProduct(Operand const& a, Operand const& b)
    {
        std::vector<float> expected(a.cols * b.cols);
        for(int64_t j = 0; j < b.cols; j++)
            for(int64_t i = 0; i < a.cols; i++)
                expected[j * a.cols + i] = static_cast<float>(dotProduct(a, i, b, j));
        return expected;
    }

    float bf16ToFloat(uint16_t bits)
    {
        uint32_t wide = static_cast<uint32_t>(bits) << 16;
        float    value;
        std::memcpy(&value, &wide, sizeof(value));
        return value;
    }

    uint16_t truncateToBf16(float value)
    {
        uint32_t wide;
        std::memcpy(&wide, &value, sizeof(value));
        return static_cast<uint16_t>(wide >> 16);
    }

    // Device memory.
    template <typename T>
    int64_t byteCount(std::vector<T> const& host)
    {
        return static_cast<int64_t>(host.size() * sizeof(T));
    }

    class DeviceBuffer
    {
    public:
        explicit DeviceBuffer(int64_t bytes)
        {
            EXPECT_EQ(hipMalloc(&m_pointer, static_cast<size_t>(bytes)), hipSuccess);
        }

        DeviceBuffer(DeviceBuffer const&)            = delete;
        DeviceBuffer& operator=(DeviceBuffer const&) = delete;

        ~DeviceBuffer()
        {
            static_cast<void>(hipFree(m_pointer));
        }

        void* get() const
        {
            return m_pointer;
        }

        template <typename T>
        void upload(std::vector<T> const& host)
        {
            ASSERT_EQ(hipMemcpy(m_pointer, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
                      hipSuccess);
        }

        template <typename T>
        void download(std::vector<T>& host) const
        {
            ASSERT_EQ(hipMemcpy(host.data(), m_pointer, host.size() * sizeof(T), hipMemcpyDeviceToHost),
                      hipSuccess);
        }

    private:
        void* m_pointer = nullptr;
    };

    int64_t dElements(ProblemSpec const& spec)
    {
        return (spec.n - 1) * spec.ldD + spec.m;
    }

    int64_t cElements(ProblemSpec const& spec)
    {
        return (spec.n - 1) * spec.ldC + spec.m;
    }

    // C holds multiples of 1/8 that bf16 represents exactly. With beta == 0 it is all NaN, so a
    // kernel that reads it produces NaN; the padding and the guard keep the sentinel otherwise.
    std::vector<uint16_t> makeC(std::mt19937_64& rng, ProblemSpec const& spec)
    {
        if(spec.beta == 0.0)
            return std::vector<uint16_t>(cElements(spec) + guardCount, nanBf16);
        std::vector<uint16_t> c(cElements(spec) + guardCount, sentinel);
        for(int64_t j = 0; j < spec.n; j++)
            for(int64_t i = 0; i < spec.m; i++)
                c[j * spec.ldC + i] = truncateToBf16((static_cast<int>(rng() % 33) - 16) / 8.0f);
        return c;
    }

    // alpha * A^T B + beta * C, in the column-major [n][m] layout of the reference.
    void applyScalars(std::vector<float>&          expected,
                      std::vector<uint16_t> const& c,
                      ProblemSpec const&           spec)
    {
        for(int64_t j = 0; j < spec.n; j++)
            for(int64_t i = 0; i < spec.m; i++)
            {
                double value = spec.alpha * expected[j * spec.m + i];
                if(spec.beta != 0.0)
                    value += spec.beta * bf16ToFloat(c[j * spec.ldC + i]);
                expected[j * spec.m + i] = static_cast<float>(value);
            }
    }

    // Checks that the ld padding and the guard kept the sentinel and that D matches the
    // reference up to the truncation of the f32 result to bf16.
    void verifyD(std::vector<uint16_t> const& d,
                 std::vector<float> const&    expected,
                 ProblemSpec const&           spec)
    {
        std::vector<bool> written(d.size(), false);
        double            maxAbs = 1.0;
        for(int64_t j = 0; j < spec.n; j++)
            for(int64_t i = 0; i < spec.m; i++)
            {
                written[j * spec.ldD + i] = true;
                maxAbs                    = std::max(maxAbs, double(std::fabs(expected[j * spec.m + i])));
            }
        double atol = 1e-4 * maxAbs;
        for(int64_t index = 0; index < static_cast<int64_t>(d.size()); index++)
            if(!written[index])
                ASSERT_EQ(d[index], sentinel) << "padding or guard element " << index << " was written";
        int mismatches = 0;
        for(int64_t j = 0; j < spec.n; j++)
            for(int64_t i = 0; i < spec.m; i++)
            {
                double actual = bf16ToFloat(d[j * spec.ldD + i]);
                double wanted = bf16ToFloat(truncateToBf16(expected[j * spec.m + i]));
                if(std::fabs(actual - wanted) <= atol + 1e-2 * std::fabs(wanted))
                    continue;
                if(mismatches++ < 5)
                    ADD_FAILURE() << "D(" << i << ", " << j << ") = " << actual << ", expected " << wanted;
            }
        EXPECT_EQ(mismatches, 0);
    }

    // The generated solutions of the library.
    std::vector<std::shared_ptr<Solution>> generatedSolutions(LoadedLibrary& loaded,
                                                               Problem const& problem)
    {
        std::set<std::shared_ptr<Solution>> found = loaded.library->findAllSolutions(
            problem, *loaded.hardware, TensileLite::SolutionLibrarySearchType::GEMM_TYPE_ONLY);
        std::vector<std::shared_ptr<Solution>> solutions;
        for(std::shared_ptr<Solution> const& solution : found)
            if(solution->KernelName().rfind(kernelPrefix, 0) == 0)
                solutions.push_back(solution);
        std::sort(solutions.begin(),
                  solutions.end(),
                  [](std::shared_ptr<Solution> const& lhs, std::shared_ptr<Solution> const& rhs) {
                      return lhs->index < rhs->index;
                  });
        return solutions;
    }

    GeneratedKernel const* findKernel(std::string const& name)
    {
        for(GeneratedKernel const& kernel : generatedKernels)
            if(name == kernel.name)
                return &kernel;
        return nullptr;
    }

    // The solution's problem predicate agrees with the fixture for every generated kernel.
    void checkPredicates(LoadedLibrary& loaded, ProblemSpec const& spec, std::vector<std::string>* accepted)
    {
        Problem problem = makeProblem(spec);
        std::vector<std::shared_ptr<Solution>> solutions = generatedSolutions(loaded, problem);
        ASSERT_EQ(static_cast<int64_t>(solutions.size()), static_cast<int64_t>(generatedKernels.size()));
        for(std::shared_ptr<Solution> const& solution : solutions)
        {
            SCOPED_TRACE(solution->KernelName() + ", " + describe(spec));
            GeneratedKernel const* kernel = findKernel(solution->KernelName());
            ASSERT_NE(kernel, nullptr) << "unexpected generated solution";
            EXPECT_TRUE((*solution->hardwarePredicate)(*loaded.hardware));
            bool actual = (*solution->problemPredicate)(problem);
            if(actual != expectedAccepts(*kernel, spec))
            {
                std::ostringstream trace;
                solution->problemPredicate->debugEval(problem, trace);
                ADD_FAILURE() << "problem predicate returned " << actual << "\n" << trace.str();
            }
            if(actual && accepted != nullptr)
                accepted->push_back(solution->KernelName());
        }
    }

    void launch(LoadedLibrary&       loaded,
                Solution const&      solution,
                Problem const&       problem,
                TensileLite::ContractionInputs const& inputs)
    {
        std::vector<TensileLite::KernelInvocation> kernels
            = solution.solve(problem, inputs, *loaded.hardware);
        ASSERT_EQ(static_cast<int64_t>(kernels.size()), 1);
        EXPECT_EQ(kernels[0].kernelName, solution.KernelName());
        EXPECT_NE(kernels[0].codeObjectFile.find(generatedSuffix), std::string::npos)
            << kernels[0].codeObjectFile;
        ASSERT_EQ(loaded.adapter->launchKernels(kernels, nullptr, nullptr, nullptr), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }

    struct ProblemData
    {
        ProblemSpec        spec;
        Operand            a;
        Operand               b;
        std::vector<uint16_t> c;
        std::vector<float>    expected;
    };

    ProblemData makeData(ProblemSpec const& spec)
    {
        std::mt19937_64 rng(1234 + spec.m * 31 + spec.n);
        ProblemData     data;
        data.spec     = spec;
        data.a        = makeOperand(rng, spec.m, spec.k, spec.ldA);
        data.b        = makeOperand(rng, spec.n, spec.k, spec.ldB);
        data.c        = makeC(rng, spec);
        data.expected = referenceProduct(data.a, data.b);
        applyScalars(data.expected, data.c, spec);
        return data;
    }

    void runSolution(LoadedLibrary& loaded, Solution const& solution, ProblemData const& data)
    {
        ProblemSpec const& spec = data.spec;
        Problem            problem = makeProblem(spec);

        std::vector<uint16_t> d(dElements(spec) + guardCount, sentinel);
        if(spec.aliasC)
            d = data.c;
        DeviceBuffer          deviceA(byteCount(data.a.image));
        DeviceBuffer          deviceB(byteCount(data.b.image));
        DeviceBuffer          deviceScaleA(byteCount(data.a.swizzled));
        DeviceBuffer          deviceScaleB(byteCount(data.b.swizzled));
        DeviceBuffer          deviceC(byteCount(data.c));
        DeviceBuffer          deviceD(byteCount(d));
        deviceA.upload(data.a.image);
        deviceB.upload(data.b.image);
        deviceScaleA.upload(data.a.swizzled);
        deviceScaleB.upload(data.b.swizzled);
        deviceC.upload(data.c);
        deviceD.upload(d);

        TensileLite::ContractionInputs inputs;
        inputs.a     = deviceA.get();
        inputs.b     = deviceB.get();
        inputs.c     = spec.aliasC ? deviceD.get() : deviceC.get();
        inputs.d     = deviceD.get();
        inputs.mxsa  = deviceScaleA.get();
        inputs.mxsb  = deviceScaleB.get();
        inputs.alpha = static_cast<float>(spec.alpha);
        inputs.beta  = static_cast<float>(spec.beta);

        launch(loaded, solution, problem, inputs);
        deviceD.download(d);
        verifyD(d, data.expected, spec);
    }

    class GeneratedSkinnyGEMMTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if(!loadedLibrary().skipReason.empty())
                GTEST_SKIP() << loadedLibrary().skipReason;
            ASSERT_NE(loadedLibrary().library, nullptr) << "cannot load the generated library";
        }
    };
}

TEST_F(GeneratedSkinnyGEMMTest, LibraryHoldsExactlyTheFixtureSolutions)
{
    ProblemSpec spec     = makeSpec(5000, 1, 8192);
    std::set<std::string> names;
    for(std::shared_ptr<Solution> const& solution : generatedSolutions(loadedLibrary(), makeProblem(spec)))
        names.insert(solution->KernelName());

    std::set<std::string> expected;
    for(GeneratedKernel const& kernel : generatedKernels)
        expected.insert(kernel.name);
    EXPECT_EQ(names, expected);
}

enum class CLayout
{
    SameAsD, // ldC == ldD, separate buffers.
    Padded, // ldC larger than ldD.
    OddLd, // Odd ldC and ldD, ldC larger than ldD.
    Aliased, // C is D.
};

struct NumericsCase
{
    const char* name;
    int64_t     m;
    int64_t     n;
    int64_t     k;
    double      alpha;
    double      beta;
    CLayout     cLayout;
    int64_t     acceptedCount;
};

static ProblemSpec makeNumericsSpec(NumericsCase const& param)
{
    ProblemSpec spec = makeSpec(param.m, param.n, param.k);
    spec.alpha       = param.alpha;
    spec.beta        = param.beta;
    spec.aliasC      = param.cLayout == CLayout::Aliased;
    if(param.cLayout == CLayout::Padded)
        spec.ldC = spec.ldD + 8;
    if(param.cLayout == CLayout::OddLd)
    {
        spec.ldD = alignUp(param.m, 4) + 5;
        spec.ldC = spec.ldD + 2;
    }
    return spec;
}

class GeneratedSkinnyGEMMNumericsTest : public GeneratedSkinnyGEMMTest,
                                        public ::testing::WithParamInterface<NumericsCase>
{
};

TEST_P(GeneratedSkinnyGEMMNumericsTest, AcceptedSolutionsComputeD)
{
    NumericsCase const& param = GetParam();
    ProblemSpec         spec  = makeNumericsSpec(param);
    SCOPED_TRACE(describe(spec));

    std::vector<std::string> accepted;
    ASSERT_NO_FATAL_FAILURE(checkPredicates(loadedLibrary(), spec, &accepted));
    ASSERT_EQ(static_cast<int64_t>(accepted.size()), param.acceptedCount);

    ProblemData data = makeData(spec);
    Problem     problem = makeProblem(spec);
    for(std::shared_ptr<Solution> const& solution : generatedSolutions(loadedLibrary(), problem))
    {
        if(std::find(accepted.begin(), accepted.end(), solution->KernelName()) == accepted.end())
            continue;
        SCOPED_TRACE(solution->KernelName());
        runSolution(loadedLibrary(), *solution, data);
    }
}

// The useBeta=0 solutions only serve beta == 0, so beta != 0 runs the useBeta=1 half of each shape.
INSTANTIATE_TEST_SUITE_P(
    GeneratedSkinnyGEMMProblems,
    GeneratedSkinnyGEMMNumericsTest,
    ::testing::Values(
        NumericsCase{"P1_m5000_n1_k8192", 5000, 1, 8192, 1.0, 0.0, CLayout::SameAsD, 8},
        NumericsCase{"P2_m1_n5000_k8192", 1, 5000, 8192, 1.0, 0.0, CLayout::SameAsD, 8},
        NumericsCase{"P3_m777_n3_k3000", 777, 3, 3000, 1.0, 0.0, CLayout::SameAsD, 4},
        NumericsCase{"P4_m3_n777_k3072", 3, 777, 3072, 1.0, 0.0, CLayout::SameAsD, 6},
        NumericsCase{"P1_alpha0p75_beta0", 5000, 1, 8192, 0.75, 0.0, CLayout::SameAsD, 8},
        NumericsCase{"P1_alpha2neg_beta0", 5000, 1, 8192, -2.0, 0.0, CLayout::SameAsD, 8},
        NumericsCase{"P1_alpha1_beta1", 5000, 1, 8192, 1.0, 1.0, CLayout::SameAsD, 4},
        NumericsCase{"P1_alpha1_betaNegHalf", 5000, 1, 8192, 1.0, -0.5, CLayout::SameAsD, 4},
        NumericsCase{"P1_alpha0p75_beta1", 5000, 1, 8192, 0.75, 1.0, CLayout::SameAsD, 4},
        NumericsCase{"P1_alpha0p75_betaNegHalf", 5000, 1, 8192, 0.75, -0.5, CLayout::SameAsD, 4},
        NumericsCase{"P1_alpha2neg_beta1", 5000, 1, 8192, -2.0, 1.0, CLayout::SameAsD, 4},
        NumericsCase{"P1_alpha2neg_betaNegHalf", 5000, 1, 8192, -2.0, -0.5, CLayout::SameAsD, 4},
        NumericsCase{"P2_alpha0p75_betaNegHalf", 1, 5000, 8192, 0.75, -0.5, CLayout::SameAsD, 4},
        NumericsCase{"P3_alpha2neg_betaNegHalf", 777, 3, 3000, -2.0, -0.5, CLayout::SameAsD, 2},
        NumericsCase{"P4_alpha0p75_beta1", 3, 777, 3072, 0.75, 1.0, CLayout::SameAsD, 3},
        NumericsCase{"P1_paddedC_beta0", 5000, 1, 8192, 0.75, 0.0, CLayout::Padded, 8},
        NumericsCase{"P2_paddedC_beta1", 1, 5000, 8192, 0.75, 1.0, CLayout::Padded, 4},
        NumericsCase{"P4_paddedC_betaNegHalf", 3, 777, 3072, -2.0, -0.5, CLayout::Padded, 3},
        NumericsCase{"P1_oddLd_beta0", 5000, 1, 8192, 0.75, 0.0, CLayout::OddLd, 8},
        NumericsCase{"P2_oddLd_beta1", 1, 5000, 8192, 0.75, 1.0, CLayout::OddLd, 4},
        NumericsCase{"P3_oddLd_betaNegHalf", 777, 3, 3000, -2.0, -0.5, CLayout::OddLd, 2},
        NumericsCase{"P1_aliasedC_beta1", 5000, 1, 8192, 0.75, 1.0, CLayout::Aliased, 4},
        NumericsCase{"P2_aliasedC_betaNegHalf", 1, 5000, 8192, -2.0, -0.5, CLayout::Aliased, 4},
        NumericsCase{"P3_aliasedC_beta1", 777, 3, 3000, 1.0, 1.0, CLayout::Aliased, 2}),
    [](::testing::TestParamInfo<NumericsCase> const& info) {
        return std::string(info.param.name);
    });

struct RejectionCase
{
    const char* name;
    void (*modify)(ProblemSpec&);
};

class GeneratedSkinnyGEMMRejectionTest : public GeneratedSkinnyGEMMTest,
                                         public ::testing::WithParamInterface<RejectionCase>
{
};

TEST_P(GeneratedSkinnyGEMMRejectionTest, PredicatesFollowTheFixture)
{
    ProblemSpec spec = makeSpec(5000, 1, 8192);
    GetParam().modify(spec);
    SCOPED_TRACE(GetParam().name);
    ASSERT_NO_FATAL_FAILURE(checkPredicates(loadedLibrary(), spec, nullptr));
}

INSTANTIATE_TEST_SUITE_P(
    GeneratedSkinnyGEMMFeatures,
    GeneratedSkinnyGEMMRejectionTest,
    ::testing::Values(
        RejectionCase{"Baseline", [](ProblemSpec&) {}},
        RejectionCase{"Bias", [](ProblemSpec& spec) { spec.useBias = true; }},
        RejectionCase{"Activation", [](ProblemSpec& spec) { spec.activation = true; }},
        RejectionCase{"ScaleAB", [](ProblemSpec& spec) { spec.scaleAB = true; }},
        RejectionCase{"ScaleAlphaVec", [](ProblemSpec& spec) { spec.scaleAlphaVec = true; }},
        RejectionCase{"LdaNotMultipleOf4", [](ProblemSpec& spec) { spec.ldA = 8194; }},
        RejectionCase{"LdbNotMultipleOf4", [](ProblemSpec& spec) { spec.ldB = 8195; }},
        // ldC and ldD need no alignment: odd values keep every solution.
        RejectionCase{"LddOdd", [](ProblemSpec& spec) { spec.ldD = 5003; }},
        RejectionCase{"K8193", [](ProblemSpec& spec) { setK(spec, 8193); }},
        // The lower window boundaries are exclusive: S1 and S4 reject K=4096, S5 rejects K=2048.
        RejectionCase{"K4096", [](ProblemSpec& spec) { setK(spec, 4096); }},
        RejectionCase{"K2048", [](ProblemSpec& spec) { setK(spec, 2048); }},
        RejectionCase{"Batch2", [](ProblemSpec& spec) { spec.batch = 2; }},
        // beta != 0 rejects every useBeta=0 solution; alpha is not asserted.
        RejectionCase{"Beta1", [](ProblemSpec& spec) { spec.beta = 1.0; }},
        RejectionCase{"BetaNegHalf", [](ProblemSpec& spec) { spec.beta = -0.5; }},
        RejectionCase{"Alpha", [](ProblemSpec& spec) { spec.alpha = -2.0; }},
        RejectionCase{"PaddedLdc", [](ProblemSpec& spec) { spec.ldC = spec.ldD + 8; }},
        RejectionCase{"LdcOdd", [](ProblemSpec& spec) { spec.ldC = 5003; }},
        RejectionCase{"LdcOddBeta1",
                      [](ProblemSpec& spec) {
                          spec.ldC  = 5003;
                          spec.beta = 1.0;
                      }}),
    [](::testing::TestParamInfo<RejectionCase> const& info) { return std::string(info.param.name); });

TEST_F(GeneratedSkinnyGEMMTest, OutputAmaxDMatchesNoGeneratedSolution)
{
    ProblemSpec spec   = makeSpec(5000, 1, 8192);
    spec.outputAmaxD   = true;
    Problem problem    = makeProblem(spec);
    std::set<std::shared_ptr<Solution>> found = loadedLibrary().library->findAllSolutions(
        problem, *loadedLibrary().hardware, TensileLite::SolutionLibrarySearchType::DEFAULT);
    for(std::shared_ptr<Solution> const& solution : found)
        EXPECT_NE(solution->KernelName().rfind(kernelPrefix, 0), std::string::size_type(0)) << solution->KernelName();
}

namespace
{
    // The name of the solution the library picks by its own heuristic rows, empty for none.
    std::string pickedKernel(ProblemSpec const& spec)
    {
        LoadedLibrary&                  loaded = loadedLibrary();
        std::shared_ptr<Solution> const picked
            = loaded.library->findBestSolution(makeProblem(spec), *loaded.hardware);
        return picked ? picked->KernelName() : std::string();
    }
}

// A Range table answers with the first row whose range matches, so a routed call with beta != 0
// reaches the useBeta=0 solution, fails its predicate and gets no solution; hipBLASLt then falls
// through to the next library.
TEST_F(GeneratedSkinnyGEMMTest, RoutedRangesReachTheUseBetaZeroSolutionOnlyAtBetaZero)
{
    ProblemSpec spec = makeSpec(5000, 1, 8192);
    EXPECT_EQ(pickedKernel(spec), generatedKernels[0].name);
    spec.alpha = -2.0;
    EXPECT_EQ(pickedKernel(spec), generatedKernels[0].name);
    for(double beta : {1.0, -0.5})
    {
        spec.beta = beta;
        EXPECT_EQ(pickedKernel(spec), "") << describe(spec);
    }

    spec = makeSpec(777, 3, 3000);
    EXPECT_EQ(pickedKernel(spec), generatedKernels[8].name);
    spec.beta = 1.0;
    EXPECT_EQ(pickedKernel(spec), "") << describe(spec);
}

// The enumeration-only rows have an empty range, so no problem is routed to them.
TEST_F(GeneratedSkinnyGEMMTest, UnroutedRangesPickNoGeneratedSolution)
{
    ProblemSpec spec = makeSpec(5000, 1, 1024);
    EXPECT_EQ(pickedKernel(spec), "");
    spec.beta = 1.0;
    EXPECT_EQ(pickedKernel(spec), "");
}
