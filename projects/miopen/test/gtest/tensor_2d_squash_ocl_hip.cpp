/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <miopen/datatype.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor_ops.hpp>
#include <miopen/kernel_build_params.hpp>
#include <tensor_util.hpp>

#include "gtest_common.hpp"
#include <gtest/gtest.h>

#include "perf_helper.hpp"

namespace {
constexpr bool PERF_ENABLE = false;

constexpr size_t H_FOR_PERF = 32 * 1024 * 1024;
constexpr size_t H          = PERF_ENABLE ? (H_FOR_PERF) : 4;

struct TestCase
{
    using tensorlen_t    = std::vector<size_t>;
    using tensorstride_t = std::vector<size_t>;

    using alphabeta_t = std::array<float, 3>;
    using offsets_t   = std::array<int64_t, 3>;

    tensorlen_t tensorlens_ac;
    tensorlen_t tensorlens_b;
    offsets_t offsets;
    tensorstride_t stride_a;
    tensorstride_t stride_b;
    tensorstride_t stride_c;
    alphabeta_t alphabeta;
    bool packed;
    miopenTensorOp_t operation;
};

// tensor A
std::vector<TestCase::tensorlen_t> tensorALensArr{{1, 1, H}, {1, 1, H * 2}, {1, 1, H * 8}};
std::vector<TestCase::tensorstride_t> tensorAStridesArr{
    {H, H, 1}, {H * 2, H * 2, 1}, {H * 8, H * 8, 1}};

// tensor B
std::vector<TestCase::tensorlen_t> tensorBLensArr{{1, 16, H}, {1, 32, H * 2}, {1, 8, H * 8}};
std::vector<TestCase::tensorstride_t> tensorBStridesArr{
    {16 * H, H * 1, 1}, {32 * H * 2, H * 2 * 1, 1}, {8 * H * 8, H * 8 * 1, 1}};

constexpr std::array<TestCase::offsets_t, 4> offsetsArr{
    {{0, 0, 0}, {64, 32, 16}, {32, 16, 32}, {32, 16, 32}}};

constexpr std::array<TestCase::alphabeta_t, 6> alphabetaArr{{{-1.0, 1.0, 1.0},
                                                             {0.0, 1.0, 1.0},
                                                             {1.0, 0.0, 1.0},
                                                             {1.0, 0.5, 0.0},
                                                             {0.0, 0.0, 1.0},
                                                             {0.0, 0.0, 0.0}}};

constexpr std::array packedArr = {true, false};

constexpr std::array operationArr = {
    miopenTensorOpAdd, miopenTensorOpMul, miopenTensorOpMin, miopenTensorOpMax};

} // namespace

template <typename T>
struct Op2dTensorSquashTest : public testing::TestWithParam<TestCase>
{
    void SetUp() override { prng::reset_seed(); }

    void Run()
    {
        CreateTensors();

        const auto tensorOCL = runOCL();
        const auto tensorHIP = runHIP();

        CompareResults(tensorHIP, tensorOCL);
    }

private:
    void CreateTensors()
    {
        const TestCase& testCase = GetParam();

        tensorA = CreateTensor(
            testCase.tensorlens_ac, testCase.stride_a, testCase.offsets[0], testCase.packed);
        tensorB = CreateTensor(
            testCase.tensorlens_b, testCase.stride_b, testCase.offsets[1], testCase.packed);
        tensorC = CreateTensor(
            testCase.tensorlens_ac, testCase.stride_c, testCase.offsets[2], testCase.packed);
    }

    tensor<T> CreateTensor(const std::vector<size_t>& lens,
                           const std::vector<size_t>& strides,
                           int64_t offset,
                           bool isPacked)
    {
        uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

        if(!isPacked)
        {
            std::vector<size_t> real_strides(strides.begin() + (strides.size() - lens.size()),
                                             strides.end());
            auto r = tensor<T>{lens, real_strides}.generate(tensor_elem_gen_integer{max_value});
            r.data.resize(r.data.size() + offset);
            return r;
        }
        else
        {
            return tensor<T>{lens}.generate(tensor_elem_gen_integer{max_value});
        }
    }

    using kernelRunner_t = std::function<void()>;
    void profile(miopen::Handle& handle, kernelRunner_t& kernelRunner, const std::string& engine)
    {
        if constexpr(!PERF_ENABLE)
        {
            return;
        }

        handle.EnableProfiling();
        handle.ResetKernelTime();

        std::vector<float> meas;

        for(int i = 0; i < 10; ++i)
        {
            kernelRunner();
            meas.push_back(handle.GetKernelTime());
            handle.ResetKernelTime();
        }

        handle.ResetKernelTime();
        writeProfileResults(meas, engine);
    }

    void writeProfileResults(std::vector<float>& meas, const std::string& engine)
    {
        std::ofstream file;
        const std::string filename{engine + "_tensor_2d_squash.csv"};
        file.open(filename, std::ios::app);

        if(!file.is_open())
        {
            throw std::runtime_error("Failed to open file");
        }

        if(miopen::fs::file_size(filename) == 0)
        {
            file << "type,ASize,BSize,a0,a1,b,OP,e0,e1,e2,e3,e4,e5,e6,e7,e8,e9\n";
        }

        miopenDataType_t data_type = tensorB.desc.GetType();
        const TestCase& testCase   = GetParam();
        const auto alpha0          = testCase.alphabeta[0];
        const auto alpha1          = testCase.alphabeta[1];
        const auto beta            = testCase.alphabeta[2];

        file << miopen::GetDataType(data_type) << ","
             << std::to_string(testCase.tensorlens_ac[2] / (1024 * 1024)) << ","
             << std::to_string(testCase.tensorlens_b[2] / (1024 * 1024)) << ","
             << std::to_string(alpha0) << "," << std::to_string(alpha1) << ","
             << std::to_string(beta) << "," << op2string(testCase.operation);

        for(auto m : meas)
        {
            file << "," << m;
        }
        file << "\n";
        file.close();
    }

    const std::string op2string(const miopenTensorOp_t op)
    {
        static const std::string ops[]{"miopenAdd", "miopenMul", "miopenMin", "miopenMax"};

        return ops[op];
    };

    std::tuple<int, int, int> GetBitmapAndWgInfo(const std::vector<size_t>& blens,
                                                 const std::vector<size_t>& clens)
    {
        // first_not_one is incorrect if btensor size equal to 1
        auto first_not_one =
            std::find_if(blens.rbegin(), blens.rend(), [](int i) { return i != 1; });
        auto d = std::distance(blens.begin(), first_not_one.base());

        // quick fix
        int num_wg = first_not_one != blens.rend()
                         ? static_cast<int>(*first_not_one == 0 ? 1 : *first_not_one)
                         : 1;

        int work_per_wg =
            std::accumulate(clens.begin() + d, clens.end(), 1, std::multiplies<int>());

        unsigned int bitmap = 0;
        // update bitmap for first_not_one
        bitmap |= (1 << (blens.size() - d));

        for(int i = (d - 2); i >= 0; i--)
        {
            if(blens[i] != 1)
            {
                bitmap |= (1 << (blens.size() - (i + 1)));
                num_wg *= blens[i];
            }
            else
            {
                work_per_wg *= clens[i];
            }
        }

        return std::make_tuple(num_wg, work_per_wg, bitmap);
    };

    std::tuple<size_t, std::string> GetRDBLCKandREADTYPE(size_t len, miopenDataType_t type)
    {
        const std::string data_type = miopen::GetDataType(type);
        size_t RD_BLCK              = (len % 4 == 0) ? 4 : (len % 2 == 0) ? 2 : 1;

        if(data_type == "half" && RD_BLCK == 4)
        {
            RD_BLCK = 2;
        }

        return std::make_tuple(RD_BLCK,
                               (RD_BLCK == 1) ? data_type : data_type + std::to_string(RD_BLCK));
    };

    std::vector<T> runOCL()
    {
        const TestCase& testCase = GetParam();

        auto&& handle = get_handle();

        auto a_dev = handle.Write(tensorA.data);
        auto b_dev = handle.Write(tensorB.data);
        auto c_dev = handle.Write(tensorC.data);

        miopenDataType_t data_type = tensorB.desc.GetType();

        auto&& [num_wg, work_per_wg, bitmap] =
            GetBitmapAndWgInfo(tensorB.desc.GetLengths(), tensorC.desc.GetLengths());

        const int max_num_wg = 4096;
        num_wg               = num_wg > max_num_wg ? max_num_wg : num_wg;

        const size_t local_threads = 256;

        auto&& [RD_BLCK, READ_TYPE] = GetRDBLCKandREADTYPE(tensorC.desc.GetLengths()[2], data_type);

        const size_t total_work = std::max(tensorC.desc.GetLengths()[2] / RD_BLCK, size_t(1));
        size_t grp_sz           = (total_work + local_threads - 1) / local_threads;

        grp_sz                = std::min(size_t(max_num_wg), grp_sz);
        size_t global_threads = local_threads * grp_sz;

        const std::vector<size_t> vld{local_threads, 1, 1};
        const std::vector<size_t> vgd{global_threads, 1, 1};

        std::string network_config =
            std::to_string(data_type) + "-" + op2string(testCase.operation) + "-" +
            std::to_string(global_threads) + "-" + std::to_string(local_threads) + "-ocl";

        std::string params = " -DMIOPEN_TYPE=" + miopen::GetDataType(data_type) + " " +
                             miopen::GetDataTypeKBP(data_type).GenerateFor(miopen::kbp::OpenCL{}) +
                             " -DMIOPEN_TENSOR_OP=" + op2string(testCase.operation) +
                             " -DUSE_2D_TENSOR_SQUASH" + " -DRD_BLCK=" + std::to_string(RD_BLCK) +
                             " -DREAD_TYPE=" + READ_TYPE;

        std::string program_name = "MIOpenTensorKernels.cl";

        miopen::as_float<const T> asT;
        const auto alpha0 = asT(testCase.alphabeta[0]);
        const auto alpha1 = asT(testCase.alphabeta[1]);
        const auto beta   = asT(testCase.alphabeta[2]);

        auto kernel = handle.AddKernel(
            "Op2dTensorSquash", network_config, program_name, "Op2dTensorSquash", vld, vgd, params);

        kernelRunner_t kernelRunner = [&]() {
            kernel(a_dev.get(),
                   b_dev.get(),
                   static_cast<int>(tensorB.desc.GetLengths()[1]),
                   static_cast<int>(tensorB.desc.GetStrides()[1]),
                   c_dev.get(),
                   alpha0,
                   alpha1,
                   beta,
                   static_cast<int64_t>(testCase.offsets[0]),
                   static_cast<int64_t>(testCase.offsets[1]),
                   static_cast<int64_t>(testCase.offsets[2]),
                   static_cast<int64_t>(total_work),
                   static_cast<int>(!miopen::float_equal(alpha0, 0.0)),
                   static_cast<int>(!miopen::float_equal(alpha1, 0.0)),
                   static_cast<int>(!miopen::float_equal(beta, 0.0)));
        };

        kernelRunner();
        auto res = handle.Read<T>(c_dev, tensorC.data.size());

        profile(handle, kernelRunner, "OCL");

        return res;
    }

    std::vector<T> runHIP()
    {
        const TestCase& testCase = GetParam();

        auto&& handle = get_handle();

        auto a_dev = handle.Write(tensorA.data);
        auto b_dev = handle.Write(tensorB.data);
        auto c_dev = handle.Write(tensorC.data);

        const int max_num_wg       = 4096;
        const size_t local_threads = 256;
        miopenDataType_t data_type = tensorB.desc.GetType();

        auto&& [RD_BLCK, READ_TYPE] = GetRDBLCKandREADTYPE(tensorC.desc.GetLengths()[2], data_type);
        auto&& [num_wg, work_per_wg, bitmap] =
            GetBitmapAndWgInfo(tensorB.desc.GetLengths(), tensorC.desc.GetLengths());

        num_wg = num_wg > max_num_wg ? max_num_wg : num_wg;

        const size_t total_work = std::max(tensorC.desc.GetLengths()[2] / RD_BLCK, size_t(1));
        size_t grp_sz           = (total_work + local_threads - 1) / local_threads;
        grp_sz                  = std::min(size_t(max_num_wg), grp_sz);

        size_t global_threads = local_threads * grp_sz;

        const std::vector<size_t> vld{local_threads, 1, 1};
        const std::vector<size_t> vgd{global_threads, 1, 1};

        std::string network_config = std::to_string(data_type) + "-miopenTensorOpAdd-" +
                                     std::to_string(global_threads) + "-" +
                                     std::to_string(local_threads) + "-hip";

        std::string params = " -DMIOPEN_TYPE=" + miopen::GetDataType(data_type) + " " +
                             miopen::GetDataTypeKBP(data_type).GenerateFor(miopen::kbp::HIP{}) +
                             " -DMIOPEN_TENSOR_OP=" + op2string(testCase.operation) +
                             " -DUSE_2D_TENSOR_SQUASH" + " -DRD_BLCK=" + std::to_string(RD_BLCK) +
                             " -DREAD_TYPE=" + READ_TYPE;

        std::string program_name = "MIOpenTensorKernelsHip.cpp";

        miopen::as_float<const T> asT;
        const auto alpha0 = asT(testCase.alphabeta[0]);
        const auto alpha1 = asT(testCase.alphabeta[1]);
        const auto beta   = asT(testCase.alphabeta[2]);

        auto kernel = handle.AddKernel(
            "Op2dTensorSquash", network_config, program_name, "Op2dTensorSquash", vld, vgd, params);

        kernelRunner_t kernelRunner = [&]() {
            kernel(a_dev.get(),
                   b_dev.get(),
                   static_cast<int>(tensorB.desc.GetLengths()[1]),
                   static_cast<int>(tensorB.desc.GetStrides()[1]),
                   c_dev.get(),
                   alpha0,
                   alpha1,
                   beta,
                   static_cast<int64_t>(testCase.offsets[0]),
                   static_cast<int64_t>(testCase.offsets[1]),
                   static_cast<int64_t>(testCase.offsets[2]),
                   static_cast<int64_t>(total_work),
                   static_cast<int>(!miopen::float_equal(alpha0, 0.0)),
                   static_cast<int>(!miopen::float_equal(alpha1, 0.0)),
                   static_cast<int>(!miopen::float_equal(beta, 0.0)));
        };

        kernelRunner();
        auto res = handle.Read<T>(c_dev, tensorC.data.size());

        profile(handle, kernelRunner, "HIP");

        return res;
    }

    void CompareResults(const std::vector<T>& valA, const std::vector<T>& valB)
    {
        const TestCase& testCase = GetParam();

        double tolerance = 1;
        if(std::is_same_v<T, half_float::half>)
        {
            // taken from original c-test
            tolerance = 80;
        }

        const double threshold = std::numeric_limits<T>::epsilon() * tolerance;
        const double error     = miopen::rms_range(valB, valA);

        ASSERT_LE(error, threshold)
            << "TensorOp: " << testCase.operation << std::endl
            << "A tensor: " << tensorA.desc.ToString() << std::endl
            << "B tensor: " << tensorB.desc.ToString() << std::endl
            << "IsPacked: " << testCase.packed << std::endl
            << "Offsets: " << testCase.offsets[0] << "," << testCase.offsets[1] << ","
            << testCase.offsets[2] << std::endl;
    }

private:
    tensor<T> tensorA;
    tensor<T> tensorB;
    tensor<T> tensorC;
};

namespace {

void AddTestCases(std::vector<TestCase>& testCases,
                  const TestCase::tensorlen_t& tensorALens,
                  const TestCase::tensorlen_t& tensorBLens,
                  const TestCase::tensorstride_t& stride_a,
                  const TestCase::tensorstride_t& stride_b,
                  const TestCase::tensorstride_t& stride_c)
{
    for(bool packed : packedArr)
    {
        for(const auto& offsets : offsetsArr)
        {
            std::array<int64_t, 3> final_offsets{0, 0, 0};
            if(!packed)
            {
                if(std::any_of(offsets.begin(), offsets.end(), [](int64_t o) { return o < 0; }))
                    continue;

                final_offsets = offsets;
            }

            auto checkStride = [p = packed](const TestCase::tensorlen_t& lens,
                                            const TestCase::tensorstride_t& strides) {
                if(p)
                    return true;

                if(lens.size() > strides.size())
                    return false;

                // only sparsed case allowed, since all the kernels do not support the last
                // dimension strides
                if(strides.back() == 1)
                {
                    // we use float here for all types because strides are independent to type
                    auto packedStrides =
                        miopen::TensorDescriptor(miopen_type<float>{}, lens).GetStrides();

                    return std::equal(packedStrides.rbegin(),
                                      packedStrides.rend(),
                                      strides.rbegin(),
                                      [](size_t ps, size_t s) { return s >= ps; });
                }

                // currently tensor operations do not support non-one stride in the last dimention.
                return false;
            };

            if(!checkStride(tensorALens, stride_a) || !checkStride(tensorBLens, stride_b) ||
               !checkStride(tensorALens, stride_c))
            {
                FAIL() << "Incorrect stride";
            }

            for(const auto& operation : operationArr)
            {
                for(const auto& alphabeta : alphabetaArr)
                {
                    TestCase& testCase = testCases.emplace_back();

                    testCase.tensorlens_ac = tensorALens;
                    testCase.tensorlens_b  = tensorBLens;
                    testCase.alphabeta     = alphabeta;
                    testCase.offsets       = final_offsets;
                    testCase.packed        = packed;
                    testCase.operation     = operation;
                    testCase.stride_a      = stride_a;
                    testCase.stride_b      = stride_b;
                    testCase.stride_c      = stride_c;
                }
            }
        }
    }
}

std::vector<TestCase> GenCases()
{
    std::vector<TestCase> testCases;

    for(int i = 0, s = tensorALensArr.size(); i < s; ++i)
    {
        AddTestCases(testCases,
                     tensorALensArr[i],
                     tensorBLensArr[i],
                     tensorAStridesArr[i],
                     tensorBStridesArr[i],
                     tensorAStridesArr[i]);
    }

    return testCases;
}

inline auto GetCases()
{
    static const auto cases = testing::ValuesIn(GenCases());
    return cases;
}
} // namespace

using Op2dTensorSquashTest_FP16 = Op2dTensorSquashTest<half_float::half>;
using Op2dTensorSquashTest_FP32 = Op2dTensorSquashTest<float>;
using Op2dTensorSquashTest_FP64 = Op2dTensorSquashTest<double>;

TEST_P(Op2dTensorSquashTest_FP16, PortTest) { this->Run(); }
TEST_P(Op2dTensorSquashTest_FP32, PortTest) { this->Run(); }
TEST_P(Op2dTensorSquashTest_FP64, PortTest) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, Op2dTensorSquashTest_FP16, GetCases());
INSTANTIATE_TEST_SUITE_P(Smoke, Op2dTensorSquashTest_FP32, GetCases());
INSTANTIATE_TEST_SUITE_P(Smoke, Op2dTensorSquashTest_FP64, GetCases());
