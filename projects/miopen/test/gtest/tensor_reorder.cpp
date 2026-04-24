/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2026 Advanced Micro Devices, Inc.
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

#include <miopen/invoker.hpp>
#include <miopen/tensor_reorder_util.hpp>
#include <miopen/tensor_layout.hpp>

#include "gtest_common.hpp"
#include "random.hpp"
#include "test_parameter_name_generator.hpp"
#include "workspace.hpp"

namespace {

using TestCase = std::tuple<NamedContainer<std::vector<uint32_t>>,
                            NamedParameter<uint32_t>,
                            NamedParameter<uint32_t>,
                            NamedParameter<uint32_t>,
                            NamedParameter<uint32_t>>;

template <typename T>
void cpu_tensor_reorder(T* dst,
                        const T* src,
                        uint64_t dim_0,
                        uint64_t dim_1,
                        uint64_t dim_2,
                        uint64_t dim_3,
                        uint64_t order_0,
                        uint64_t order_1,
                        uint64_t order_2,
                        uint64_t order_3)
{
    const uint64_t src_dim[4] = {dim_0, dim_1, dim_2, dim_3};
    const uint64_t dst_dim[4] = {
        src_dim[order_0], src_dim[order_1], src_dim[order_2], src_dim[order_3]};

    const uint64_t src_stride[4] = {
        src_dim[1] * src_dim[2] * src_dim[3], src_dim[2] * src_dim[3], src_dim[3], 1};
    const uint64_t dst_stride[4] = {
        dst_dim[1] * dst_dim[2] * dst_dim[3], dst_dim[2] * dst_dim[3], dst_dim[3], 1};

    uint64_t itr_src_dim[4] = {0, 0, 0, 0};
    uint64_t itr_dst_dim[4] = {0, 0, 0, 0};

    for(itr_src_dim[0] = 0; itr_src_dim[0] < src_dim[0]; itr_src_dim[0]++)
    {
        for(itr_src_dim[1] = 0; itr_src_dim[1] < src_dim[1]; itr_src_dim[1]++)
        {
            for(itr_src_dim[2] = 0; itr_src_dim[2] < src_dim[2]; itr_src_dim[2]++)
            {
                for(itr_src_dim[3] = 0; itr_src_dim[3] < src_dim[3]; itr_src_dim[3]++)
                {
                    itr_dst_dim[0] = itr_src_dim[order_0];
                    itr_dst_dim[1] = itr_src_dim[order_1];
                    itr_dst_dim[2] = itr_src_dim[order_2];
                    itr_dst_dim[3] = itr_src_dim[order_3];

                    const uint64_t idx_src =
                        itr_src_dim[0] * src_stride[0] + itr_src_dim[1] * src_stride[1] +
                        itr_src_dim[2] * src_stride[2] + itr_src_dim[3] * src_stride[3];
                    const uint64_t idx_dst =
                        itr_dst_dim[0] * dst_stride[0] + itr_dst_dim[1] * dst_stride[1] +
                        itr_dst_dim[2] * dst_stride[2] + itr_dst_dim[3] * dst_stride[3];

                    dst[idx_dst] = src[idx_src];
                }
            }
        }
    }
}

template <typename T>
struct cpu_reorder
{
    static void run(T* dst,
                    const T* src,
                    uint64_t dim_0,
                    uint64_t dim_1,
                    uint64_t dim_2,
                    uint64_t dim_3,
                    uint64_t order_0,
                    uint64_t order_1,
                    uint64_t order_2,
                    uint64_t order_3)
    {
        cpu_tensor_reorder<T>(
            dst, src, dim_0, dim_1, dim_2, dim_3, order_0, order_1, order_2, order_3);
    }
};

struct reorder_str
{
    static std::string get(uint32_t order_0, uint32_t order_1, uint32_t order_2, uint32_t order_3)
    {
        return ("r" + std::to_string(order_0) + std::to_string(order_1) + std::to_string(order_2) +
                std::to_string(order_3));
    }
};

std::string
supported_reorder_to_string(uint32_t order_0, uint32_t order_1, uint32_t order_2, uint32_t order_3)
{
    std::string layout_string("N/A");
    // NOLINTBEGIN(*-braces-around-statements)
    if((order_0 == 0) && (order_1 == 1) && (order_2 == 3) && (order_3 == 2))
        layout_string = "r0132";
    else if((order_0 == 0) && (order_1 == 2) && (order_2 == 1) && (order_3 == 3))
        layout_string = "r0213";
    else if((order_0 == 0) && (order_1 == 2) && (order_2 == 3) && (order_3 == 1))
        layout_string = "r0231";
    else if((order_0 == 0) && (order_1 == 3) && (order_2 == 1) && (order_3 == 2))
        layout_string = "r0312";
    else if((order_0 == 0) && (order_1 == 3) && (order_2 == 2) && (order_3 == 1))
        layout_string = "r0321";
    else if((order_0 == 1) && (order_1 == 0) && (order_2 == 2) && (order_3 == 3))
        layout_string = "r1023";
    else if((order_0 == 1) && (order_1 == 0) && (order_2 == 3) && (order_3 == 2))
        layout_string = "r1032";
    else if((order_0 == 1) && (order_1 == 2) && (order_2 == 0) && (order_3 == 3))
        layout_string = "r1203";
    else if((order_0 == 1) && (order_1 == 2) && (order_2 == 3) && (order_3 == 0))
        layout_string = "r1230";
    else if((order_0 == 1) && (order_1 == 3) && (order_2 == 0) && (order_3 == 2))
        layout_string = "r1302";
    else if((order_0 == 1) && (order_1 == 3) && (order_2 == 2) && (order_3 == 0))
        layout_string = "r1320";
    else if((order_0 == 2) && (order_1 == 0) && (order_2 == 1) && (order_3 == 3))
        layout_string = "r2013";
    else if((order_0 == 2) && (order_1 == 0) && (order_2 == 3) && (order_3 == 1))
        layout_string = "r2031";
    else if((order_0 == 2) && (order_1 == 1) && (order_2 == 0) && (order_3 == 3))
        layout_string = "r2103";
    else if((order_0 == 2) && (order_1 == 1) && (order_2 == 3) && (order_3 == 0))
        layout_string = "r2130";
    else if((order_0 == 2) && (order_1 == 3) && (order_2 == 0) && (order_3 == 1))
        layout_string = "r2301";
    else if((order_0 == 2) && (order_1 == 3) && (order_2 == 1) && (order_3 == 0))
        layout_string = "r2310";
    else if((order_0 == 3) && (order_1 == 0) && (order_2 == 1) && (order_3 == 2))
        layout_string = "r3012";
    else if((order_0 == 3) && (order_1 == 0) && (order_2 == 2) && (order_3 == 1))
        layout_string = "r3021";
    else if((order_0 == 3) && (order_1 == 1) && (order_2 == 0) && (order_3 == 2))
        layout_string = "r3102";
    else if((order_0 == 3) && (order_1 == 1) && (order_2 == 2) && (order_3 == 0))
        layout_string = "r3120";
    else if((order_0 == 3) && (order_1 == 2) && (order_2 == 0) && (order_3 == 1))
        layout_string = "r3201";
    else if((order_0 == 3) && (order_1 == 2) && (order_2 == 1) && (order_3 == 0))
        layout_string = "r3210";
    else
        MIOPEN_THROW("Unsupported reorder layout");
    // NOLINTEND(*-braces-around-statements)
    return layout_string;
}

static const constexpr int RAND_INTEGER_MAX = 120;
static const constexpr int RAND_INTEGER_MIN = -88;

template <typename T>
void rand_tensor_integer(tensor<T>& t, int max = RAND_INTEGER_MAX, int min = RAND_INTEGER_MIN)
{
    // use integer to random.
    for(size_t i = 0; i < t.data.size(); i++)
        t[i] = static_cast<T>(prng::gen_A_to_B(min, max));
}

template <typename T>
bool compare_equal(T r1, T r2)
{
    return r1 == r2;
}

template <>
bool compare_equal<double>(double r1, double r2)
{
    return miopen::float_equal(r1, r2);
}

template <>
bool compare_equal<float>(float r1, float r2)
{
    return miopen::float_equal(r1, r2);
}

struct reorder_invoke_param : public miopen::InvokeParams
{
    ConstData_t src = nullptr;
    Data_t dst      = nullptr;

    [[maybe_unused]] reorder_invoke_param(ConstData_t src_, Data_t dst_) : src(src_), dst(dst_) {}
    [[maybe_unused]] reorder_invoke_param(miopen::InvokeType type_, ConstData_t src_, Data_t dst_)
        : InvokeParams{type_}, src(src_), dst(dst_)
    {
    }

    Data_t GetWorkspace() const { return nullptr; }
    std::size_t GetWorkspaceSize() const { return 0; }
};

inline auto GenCases()
{
    static const std::vector<std::vector<uint32_t>> all_possible_order{
        {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3},
        {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0}, {2, 0, 1, 3},
        {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2},
        {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0}};

    return testing::Combine(
        MakeNamedParameterCollectionValues<std::vector<uint32_t>>("order", all_possible_order, ","),
        MakeNamedParameterValues<uint32_t>(
            "dim0", 1U, 2U, static_cast<uint32_t>(prng::gen_off_range(3, 4))),
        MakeNamedParameterValues<uint32_t>(
            "dim1", 3U, 8U, static_cast<uint32_t>(prng::gen_off_range(15, 13))),
        MakeNamedParameterValues<uint32_t>(
            "dim2", 1U, 9U, static_cast<uint32_t>(prng::gen_off_range(29, 13))),
        MakeNamedParameterValues<uint32_t>(
            "dim3", 1U, 9U, static_cast<uint32_t>(prng::gen_off_range(29, 13))));
}

inline auto GetCases()
{
    static const auto cases = GenCases();
    return cases;
}

} // namespace

template <typename T>
struct tensor_reorder_test : public testing::TestWithParam<TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        std::tie(order, dim0, dim1, dim2, dim3) = GetParam();
    }

    // NOLINTBEGIN(clang-analyzer-cplusplus.NewDeleteLeaks)
    void Run()
    {
        const int tensor_sz = dim0 * dim1 * dim2 * dim3;
        const std::vector<int> tensor_len({static_cast<int>(dim0),
                                           static_cast<int>(dim1),
                                           static_cast<int>(dim2),
                                           static_cast<int>(dim3)});

        std::vector<int> tensor_strides;

        std::string layout_default = miopen::tensor_layout_get_default(4);
        std::string layout_string  = miopen::TensorDescriptor::LayoutEnumToStr(miopenTensorNCHW);
        std::string reorder_string =
            supported_reorder_to_string(order[0], order[1], order[2], order[3]);

        miopen::tensor_layout_to_strides(tensor_len, layout_default, layout_string, tensor_strides);

        tensor<T> t_src(tensor_len, tensor_strides);
        tensor<T> t_dst(tensor_len, tensor_strides);
        tensor<T> t_dst_gpu(tensor_len, tensor_strides);
        rand_tensor_integer(t_src);

        auto& handle = get_handle();
        miopen::ExecutionContext ctx;
        ctx.SetStream(&handle);
        const auto reorder_sol = MakeTensorReorderAttributes(
            ctx, miopen_type<T>{}, dim0, dim1, dim2, dim3, order[0], order[1], order[2], order[3]);
        ASSERT_NE(reorder_sol, nullptr);
        size_t workspace_size =
            reorder_sol->IsSkippable() ? sizeof(T) * tensor_sz : reorder_sol->GetOutputTensorSize();
        Workspace wspace{workspace_size};

        auto src_dev = handle.Write(t_src.data);

        const auto invoke_param         = reorder_invoke_param{src_dev.get(), wspace.ptr()};
        std::vector<OpKernelArg> opArgs = reorder_sol->GetKernelArg();
        std::optional<miopen::InvokerFactory> invoker_factory(
            [=](const std::vector<miopen::Kernel>& kernels) mutable {
                return [=](const miopen::Handle& handle_,
                           const miopen::AnyInvokeParams& primitive_param) mutable {
                    decltype(auto) invoke_params = primitive_param.CastTo<reorder_invoke_param>();
                    const auto k                 = handle_.Run(kernels[0]);
                    opArgs[0]                    = OpKernelArg(invoke_params.dst);
                    opArgs[1]                    = OpKernelArg(invoke_params.src);
                    k(opArgs);
                };
            });
        const std::vector<miopen::solver::KernelInfo> construction_params{
            reorder_sol->GetKernelInfo()};

        if(invoker_factory.has_value())
        {
            const auto invoker = handle.PrepareInvoker(*invoker_factory, construction_params);
            // run gpu
            invoker(handle, invoke_param);
            // run cpu
            cpu_reorder<T>::run(t_dst.data.data(),
                                t_src.data.data(),
                                dim0,
                                dim1,
                                dim2,
                                dim3,
                                order[0],
                                order[1],
                                order[2],
                                order[3]);
        }
        invoker_factory.reset();

        t_dst_gpu.data = wspace.Read<decltype(t_dst_gpu.data)>();

        // we expect excact match, since use integer
        VerifyTensor(t_dst_gpu, t_dst);
    }
    // NOLINTEND(clang-analyzer-cplusplus.NewDeleteLeaks)

private:
    std::vector<uint32_t> order;
    uint32_t dim0{0};
    uint32_t dim1{0};
    uint32_t dim2{0};
    uint32_t dim3{0};

    void VerifyTensor(const tensor<T>& t_gpu, const tensor<T>& t_cpu)
    {
        ASSERT_EQ(t_gpu.data.size(), t_cpu.data.size());

        const auto idx          = miopen::mismatch_idx(t_gpu.data, t_cpu.data, compare_equal<T>);
        const bool valid_result = idx >= static_cast<size_t>(miopen::range_distance(t_cpu));

        std::cout << "[" << reorder_str::get(order[0], order[1], order[2], order[3]) << ", b"
                  << (sizeof(T) * 8) << " ] " << "dim0:" << dim0 << ", dim1:" << dim1
                  << ", dim2:" << dim2 << ", dim3:" << dim3 << ", valid:" << valid_result
                  << std::endl;

        EXPECT_TRUE(valid_result) << "diff at:" << idx << ", gpu:" << t_gpu[idx]
                                  << ", cpu:" << t_cpu[idx] << std::endl;
    }
};

struct TestNameGenerator
{
    std::string operator()(const auto& info)
    {
        const auto& [order, dim0, dim1, dim2, dim3] = info.param;
        std::stringstream ss;
        std::string str;

        ss << "order_" << GetRangeAsString(order(), ",") << "_dim0_" << dim0() << "_dim1_" << dim1()
           << "_dim2_" << dim2() << "_dim3_" << dim3() << "_test_id_" << info.index;

        str = ss.str();

        // Name format only supports letters, numbers and underscores.
        std::transform(str.begin(), str.end(), str.begin(), [](char c) -> char {
            return (c == '.') ? 'p' : (std::isalnum(c) ? c : '_');
        });

        return str;
    }
};

using GPU_TensorReorder_I8   = tensor_reorder_test<int8_t>;
using GPU_TensorReorder_FP16 = tensor_reorder_test<half_float::half>;
using GPU_TensorReorder_FP32 = tensor_reorder_test<float>;
using GPU_TensorReorder_FP64 = tensor_reorder_test<double>;

TEST_P(GPU_TensorReorder_I8, TestInt8) { this->Run(); }
TEST_P(GPU_TensorReorder_FP16, TestFloat16) { this->Run(); }
TEST_P(GPU_TensorReorder_FP32, TestFloat32) { this->Run(); }
TEST_P(GPU_TensorReorder_FP64, TestFloat64) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_TensorReorder_I8, GetCases(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_TensorReorder_FP16, GetCases(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_TensorReorder_FP32, GetCases(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_TensorReorder_FP64, GetCases(), TestNameGenerator{});
