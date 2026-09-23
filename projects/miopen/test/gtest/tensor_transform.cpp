/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#include <miopen/tensor_ops.hpp>

#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "tensor_util.hpp"
#include "test_parameter_name_generator.hpp"

namespace {

#define MIO_TRANSFORM_DEBUG 0

using TestCase = std::tuple<NamedContainer<std::vector<int>>,
                            NamedContainer<std::vector<float>>,
                            NamedContainer<std::vector<int>>,
                            NamedContainer<std::vector<int>>,
                            NamedContainer<std::vector<int>>,
                            NamedParameter<int>>;

#if(MIO_TRANSFORM_DEBUG)
template <class T>
static void show_tensor(const tensor<T>& ten)
{
    // initialize lengths
    int batch_size = 1;
    int depth      = 1;
    int channels   = 1;
    int height     = 1;
    int width      = 1;
    // get the underlying array
    std::vector<size_t> lens = ten.desc.GetLengths();
    int dim                  = ten.desc.GetLengths().size();

    switch(dim)
    {
    case 1: width = lens.at(0); break;
    case 2:
        height = lens.at(0);
        width  = lens.at(1);
        break;
    case 3:
        channels = lens.at(0);
        height   = lens.at(1);
        width    = lens.at(2);
        break;
    case 4:
        depth    = lens.at(0);
        channels = lens.at(1);
        height   = lens.at(2);
        width    = lens.at(3);
        break;
    case 5:
        batch_size = lens.at(0);
        depth      = lens.at(1);
        channels   = lens.at(2);
        height     = lens.at(3);
        width      = lens.at(4);
        break;
    default: return;
    }

    // show data layout

    for(int n = 0; n < batch_size; n++)
    {
        for(int d = 0; d < depth; d++)
        {
            std::cout << d << " --" << std::endl;
            for(int c = 0; c < channels; c++)
            {
                for(int h = 0; h < height; h++)
                {
                    for(int w = 0; w < width; w++)
                    {
                        std::cout
                            << std::setprecision(5)
                            << double(
                                   ten[(((n * depth + d) * channels + c) * height + h) * width + w])
                            << ",";
                    }
                    std::cout << "\n";
                }
                std::cout << std::endl;
            }
        }
    }
}
#endif

template <class T>
struct verify_tensor_transform_layout
{
    miopen::TensorDescriptor srcDesc;
    miopen::TensorDescriptor dstDesc;
    tensor<T> srcSuper;
    tensor<T> dstSuper;
    float alpha;
    float beta;

    verify_tensor_transform_layout(const tensor<T>& psrc_super,
                                   const tensor<T>& pdst_super,
                                   const miopen::TensorDescriptor& psd,
                                   const miopen::TensorDescriptor& pdd,
                                   const float palpha,
                                   const float pbeta)
    {
        srcDesc  = psd;
        dstDesc  = pdd;
        srcSuper = psrc_super;
        dstSuper = pdst_super;
        alpha    = palpha;
        beta     = pbeta;
    }

    tensor<T> cpu() const
    {
        tensor<T> dstSuperCpu = dstSuper;

        if(dstDesc.GetType() == miopenInt8 && srcDesc.GetType() == miopenInt8)
        {
            int x_c;
            std::tie(std::ignore, x_c, std::ignore, std::ignore) =
                miopen::tien<4>(srcDesc.GetLengths());

            int y_n, y_c, y_h, y_w;
            std::tie(y_n, y_c, y_h, y_w) = miopen::tien<4>(dstDesc.GetLengths());

            miopen::par_ford(y_n, y_c, y_h, y_w)([&](int l, int k, int i, int j) {
                if(k < x_c)
                {
                    float tmp_fp =
                        alpha * float(srcSuper(l, k, i, j)) + beta * float(dstSuperCpu(l, k, i, j));
                    int8_t tmp_int          = tmp_fp >= 127    ? 127
                                              : tmp_fp <= -128 ? -128
                                                               : int8_t(std::lround(tmp_fp));
                    dstSuperCpu(l, k, i, j) = tmp_int;
                }
                else
                {
                    dstSuperCpu(l, k, i, j) = T(0);
                }
            });
        }

        return dstSuperCpu;
    }

    tensor<T> gpu() const
    {
        tensor<T> dstSuperGpu = dstSuper;

        auto&& handle     = get_handle();
        auto dstSuper_dev = handle.Write(dstSuperGpu.data);
        auto srcSuper_dev = handle.Write(srcSuper.data);

        miopen::TransformTensor(
            handle, &alpha, srcDesc, srcSuper_dev.get(), &beta, dstDesc, dstSuper_dev.get());

        dstSuperGpu.data = handle.Read<T>(dstSuper_dev, dstSuperGpu.data.size());

        return dstSuperGpu;
    }

    void fail(float = 0)
    {
        std::cout << "Tensor Copy: " << std::endl;
        std::cout << "src super-tensor: " << srcSuper.desc.ToString() << std::endl;
        std::cout << "dst super-tensor: " << dstSuper.desc.ToString() << std::endl;
        std::cout << "src sub-tensor: " << srcDesc.ToString() << std::endl;
        std::cout << "dst sub-tensor: " << dstDesc.ToString() << std::endl;
    }
};

template <class T>
struct verify_tensor_transform_scale
{
    miopen::TensorDescriptor subDesc_src;
    tensor<T> super_src;
    miopen::TensorDescriptor subDesc_dst;
    tensor<T> super_dst;
    size_t src_offset;
    size_t dst_offset;
    T alpha;
    T beta;

    verify_tensor_transform_scale(const tensor<T>& rSuper_src,
                                  const miopen::TensorDescriptor& rSubDesc_src,
                                  const tensor<T>& rSuper_dst,
                                  const miopen::TensorDescriptor& rSubDesc_dst,
                                  const size_t rOffset_src,
                                  const size_t rOffset_dst,
                                  const T alphaIn,
                                  const T betaIn)
    {
        super_src   = rSuper_src;
        super_dst   = rSuper_dst;
        subDesc_src = rSubDesc_src;
        subDesc_dst = rSubDesc_dst;
        src_offset  = rOffset_src;
        dst_offset  = rOffset_dst;
        alpha       = alphaIn;
        beta        = betaIn;
    }

    tensor<T> cpu() const
    {

        tensor<T> superCpu_src = super_src;
        tensor<T> superCpu_dst = super_dst;

        operate_over_subtensor(
            [a = alpha, b = beta](auto& dst, auto src) {
                dst = static_cast<T>(src) * a + static_cast<T>(dst) * b;
            },
            superCpu_dst,
            superCpu_src,
            subDesc_dst,
            subDesc_src,
            dst_offset,
            src_offset);

#if(MIO_TRANSFORM_DEBUG)
        printf("\n CPU: \n");
        show_tensor(superCpu_dst);
        printf("\n");
#endif

        return superCpu_dst;
    }

    tensor<T> gpu() const
    {
        tensor<T> superGpu_src = super_src;
        tensor<T> superGpu_dst = super_dst;

        auto&& handle      = get_handle();
        auto super_dev_src = handle.Write(superGpu_src.data);
        auto super_dev_dst = handle.Write(superGpu_dst.data);

        miopen::TransformTensor(handle,
                                &alpha,
                                subDesc_src,
                                super_dev_src.get(),
                                &beta,
                                subDesc_dst,
                                super_dev_dst.get(),
                                src_offset,
                                dst_offset);

        superGpu_dst.data = handle.Read<T>(super_dev_dst, superGpu_dst.data.size());
#if(MIO_TRANSFORM_DEBUG)
        printf("\n GPU: \n");
        show_tensor(superGpu_dst);
        printf("\n");
#endif
        return superGpu_dst;
    }

    void fail(float = 0)
    {
        std::cout << "Tensor Set: " << super_src << std::endl;
        std::cout << "super-tensor: " << super_dst.desc.ToString() << std::endl;
        std::cout << "sub-tensor: " << subDesc_dst.ToString() << std::endl;
    }
};

inline auto GenSmokeTestCases()
{
    static const std::vector<std::vector<int>> all_lens{
        {32, 11, 32, 16}, {16, 30, 16, 16}, {15, 1, 14, 14}, {10, 16, 7, 7}, {1, 1, 1, 1}};

    static const std::vector<std::vector<float>> all_scales{{1.0f, 0.0f}, {0.5f, 0.5f}};

#if(MIO_TRANSFORM_DEBUG)
#define NROWS 6
#define NCOLS 6
#define LENS_CONTENT NROWS, NCOLS
#define SUBLENS_CONTENT      \
    {                        \
        NROWS - 2, NCOLS - 2 \
    }
#define OFFSET_CONTENT 0
#else
#define LENS_CONTENT 32, 32, 16, 16, 16
#define SUBLENS_CONTENT \
    {                   \
        32, 8, 10       \
    }
#define OFFSET_CONTENT 7
#endif

    static const std::vector<int> lens{LENS_CONTENT};
    static const std::vector<std::vector<int>> all_superlens_src{lens};
    static const std::vector<std::vector<int>> all_sublens{SUBLENS_CONTENT};
    static const std::vector<std::vector<int>> all_superlens_dst{lens};
    static const std::vector<int> tensor_offsets{OFFSET_CONTENT};

    return testing::Combine(
        MakeNamedParameterCollectionValues<std::vector<int>>("srcLens", all_lens, "x"),
        MakeNamedParameterCollectionValues<std::vector<float>>("scales", all_scales, ","),
        MakeNamedParameterCollectionValues<std::vector<int>>(
            "superLens_src", all_superlens_src, "x"),
        MakeNamedParameterCollectionValues<std::vector<int>>("subLens", all_sublens, "x"),
        MakeNamedParameterCollectionValues<std::vector<int>>(
            "superLens_dst", all_superlens_dst, "x"),
        MakeNamedParameterCollectionValues<int>("offset", tensor_offsets));
}

inline auto GenFullTestCases()
{
    static const std::vector<std::vector<int>> all_lens{
        {32, 11, 32, 16}, {16, 30, 16, 16}, {15, 1, 14, 14}, {10, 16, 7, 7}, {1, 1, 1, 1}};

    static const std::vector<std::vector<float>> all_scales{
        {1.0f, 0.0f}, {1.0f, 0.5f}, {0.5f, 0.0f}, {0.5f, 0.5f}};

#if(MIO_TRANSFORM_DEBUG)
#define NROWS 6
#define NCOLS 6
#define LENS_CONTENT NROWS, NCOLS
#else
#define LENS_CONTENT 32, 32, 16, 16, 16
#endif

    static const std::vector<int> lens{LENS_CONTENT};
    static const std::vector<std::vector<int>> all_superlens_src{lens};
    static const std::vector<std::vector<int>> all_sublens{get_sub_tensor()};
    static const std::vector<std::vector<int>> all_superlens_dst{lens};
    static const std::vector<int> tensor_offsets{get_tensor_offset<int>()};

    return testing::Combine(
        MakeNamedParameterCollectionValues<std::vector<int>>("srcLens", all_lens, "x"),
        MakeNamedParameterCollectionValues<std::vector<float>>("scales", all_scales, ","),
        MakeNamedParameterCollectionValues<std::vector<int>>(
            "superLens_src", all_superlens_src, "x"),
        MakeNamedParameterCollectionValues<std::vector<int>>("subLens", all_sublens, "x"),
        MakeNamedParameterCollectionValues<std::vector<int>>(
            "superLens_dst", all_superlens_dst, "x"),
        MakeNamedParameterCollectionValues<int>("offset", tensor_offsets));
}

inline auto GetSmokeTestCases()
{
    static const auto cases = GenSmokeTestCases();
    return cases;
}

inline auto GetFullTestCases()
{
    static const auto cases = GenFullTestCases();
    return cases;
}

} // namespace

template <class T>
struct tensor_transform_test : public testing::TestWithParam<TestCase>
{
    // Params for tensor layout transform functionality for low-precision computation
    tensor<T> srcSuper_pad;
    tensor<T> dstSuper_pad;
    tensor<T> srcSuper_depad;
    tensor<T> dstSuper_depad;
    std::vector<int> srcLens;

    miopen::TensorDescriptor srcDesc;
    miopen::TensorDescriptor dstDesc;
    std::vector<float> scales;

    // Params for tensor scale addition
    tensor<T> super_src;
    std::vector<int> superLens_src;
    miopen::TensorDescriptor subDesc_src;
    std::vector<int> subLens;

    tensor<T> super_dst;
    std::vector<int> superLens_dst;
    miopen::TensorDescriptor subDesc_dst;
    size_t offset = 0;

    void SetUp() override
    {
        prng::reset_seed();
        std::tie(srcLens, scales, superLens_src, subLens, superLens_dst, offset) = GetParam();
    }

    void Run()
    {
        const float alpha = scales[0];
        const float beta  = scales[1];

        const uint64_t max_value = miopen_type<T>{} == miopenHalf   ? 5
                                   : miopen_type<T>{} == miopenInt8 ? 127
                                                                    : 17;

        const bool skip_layout = !(miopen::float_equal(static_cast<const float>(alpha), 1.0) &&
                                   miopen::float_equal(static_cast<const float>(beta), 0.0) &&
                                   std::is_same<T, int8_t>{});
        if(!skip_layout)
        {
            // Test tensor layout transform
            srcSuper_pad   = tensor<T>{srcLens}.generate(tensor_elem_gen_integer{max_value});
            dstSuper_depad = tensor<T>{srcLens}.generate(tensor_elem_gen_integer{max_value});
            srcDesc        = miopen::TensorDescriptor(miopen_type<T>{}, srcLens);

            srcLens[1]     = (srcLens[1] % 4 == 0) ? srcLens[1] : ((srcLens[1] + 3) / 4) * 4;
            dstSuper_pad   = tensor<T>{srcLens}.generate(tensor_elem_gen_integer{max_value});
            srcSuper_depad = tensor<T>{srcLens}.generate(tensor_elem_gen_integer{max_value});
            dstDesc        = miopen::TensorDescriptor(miopen_type<T>{}, srcLens);

            if(srcDesc.GetLengths().size() == dstDesc.GetLengths().size())
            {
                VerifyEquals(verify_tensor_transform_layout<T>{
                    srcSuper_pad, dstSuper_pad, srcDesc, dstDesc, alpha, beta});

                VerifyEquals(verify_tensor_transform_layout<T>{
                    srcSuper_depad, dstSuper_depad, dstDesc, srcDesc, alpha, beta});
            }
        }

        // Test tensor scale addition
        if(miopen_type<T>{} == miopenInt8)
            return;

        super_src = tensor<T>{superLens_src}.generate(tensor_elem_gen_integer{max_value});
        super_dst = tensor<T>{superLens_dst}.generate(tensor_elem_gen_integer{max_value});

#if(MIO_TRANSFORM_DEBUG)
        printf("\n SRC: \n");
        show_tensor(super_src);
        printf("\n DST: \n");
        show_tensor(super_dst);
#endif
        const std::vector<size_t> superStrides_src = super_src.desc.GetStrides();
        const std::vector<size_t> superStrides_dst = super_dst.desc.GetStrides();
        const std::vector<int> subStrides_src(superStrides_src.begin() +
                                                  (super_src.desc.GetNumDims() - subLens.size()),
                                              superStrides_src.end());
        const std::vector<int> subStrides_dst(superStrides_dst.begin() +
                                                  (super_dst.desc.GetNumDims() - subLens.size()),
                                              superStrides_dst.end());

        subDesc_src = miopen::TensorDescriptor(miopen_type<T>{}, subLens, subStrides_src);
        subDesc_dst = miopen::TensorDescriptor(miopen_type<T>{}, subLens, subStrides_dst);

        VerifyEquals(verify_tensor_transform_scale<T>{
            super_src, subDesc_src, super_dst, subDesc_dst, offset, offset, T(alpha), T(beta)});
    }

private:
    void VerifyEquals(auto&& v)
    {
        const auto cpu = v.cpu();
        const auto gpu = v.gpu();
        const auto idx = miopen::mismatch_idx(cpu, gpu, miopen::float_equal);

        ASSERT_GE(idx, miopen::range_distance(cpu)) << (v.fail(), "");
    }
};

class TestNameGenerator
{
public:
    std::string operator()(const auto& testCase)
    {
        std::stringstream ss;
        const auto& [srcLens, scales, superLens_src, subLens, superLens_dst, offset] =
            testCase.param;

        ss << "srcLens_" << GetRangeAsString(srcLens(), "x") << "_scales_"
           << GetRangeAsString(scales(), ",") << "_superLens_src_"
           << GetRangeAsString(superLens_src(), "x") << "_subLens_"
           << GetRangeAsString(subLens(), "x") << "_superLens_dst_"
           << GetRangeAsString(superLens_dst(), "x") << "_offset_" << offset() << "_test_id_"
           << testCase.index;

        std::string testName(ss.str());

        std::transform(testName.begin(), testName.end(), testName.begin(), [](char c) -> char {
            return (c == '.') ? 'p' : (std::isalnum(c) ? c : '_');
        });

        return testName;
    }
};

using GPU_TensorTransform_I8   = tensor_transform_test<int8_t>;
using GPU_TensorTransform_FP16 = tensor_transform_test<half_float::half>;
using GPU_TensorTransform_FP32 = tensor_transform_test<float>;

TEST_P(GPU_TensorTransform_I8, TestInt8) { this->Run(); }
TEST_P(GPU_TensorTransform_FP16, TestFloat16) { this->Run(); }
TEST_P(GPU_TensorTransform_FP32, TestFloat32) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_TensorTransform_I8, GetSmokeTestCases(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_TensorTransform_FP16, GetSmokeTestCases(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_TensorTransform_FP32, GetSmokeTestCases(), TestNameGenerator{});

INSTANTIATE_TEST_SUITE_P(Full, GPU_TensorTransform_I8, GetFullTestCases(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full, GPU_TensorTransform_FP16, GetFullTestCases(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full, GPU_TensorTransform_FP32, GetFullTestCases(), TestNameGenerator{});
