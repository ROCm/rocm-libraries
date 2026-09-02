// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/batched_transpose_sol.hpp>
#include <miopen/datatype.hpp>
#include <miopen/op_kernel_args.hpp>
#include <miopen/subbuffers.hpp>
#include <miopen/tensor_layout.hpp>

namespace miopen {
namespace solver {

template <class Element = std::size_t>
inline static std::array<Element, 5> GetNCDHW(const std::vector<std::size_t>& values)
{
    const auto cast = [](auto v) { return static_cast<Element>(v); };
    std::size_t n = 1, c = 1, d = 1, h = 1, w = 1;

    switch(values.size())
    {
    case 5: std::tie(n, c, d, h, w) = tien<5>(values); break;
    case 4: std::tie(n, c, h, w) = tien<4>(values); break;
    default: MIOPEN_THROW(miopenStatusBadParm);
    }

    return {cast(n), cast(c), cast(d), cast(h), cast(w)};
}

struct TransposeProblem
{
    TensorDescriptor input;
    const char* layout;
    std::string target_layout;

    TensorDescriptor GetOutputDescriptor() const
    {
        const auto labels    = tensor_layout_get_default(input.GetNumDims());
        auto derived_strides = std::vector<size_t>{};
        auto target          = target_layout;
        if(strlen(layout) < 5)
            target = ReplaceString(target, "D", "");
        tensor_layout_to_strides(input.GetLengths(), labels, target, derived_strides);
        return {input.GetType(), input.GetLengths(), derived_strides};
    }
};

using OldStyleTransposeProblem = std::tuple<const ExecutionContext*, const TransposeProblem*>;

struct TransposeInvokeParams : InvokeParams
{
    ConstData_t in;
    Data_t out;
    TensorDescriptor in_desc;
    TensorDescriptor out_desc;

    TransposeInvokeParams(ConstData_t in_,
                          Data_t out_,
                          TensorDescriptor in_desc_,
                          TensorDescriptor out_desc_)
        : in(in_), out(out_), in_desc(in_desc_), out_desc(out_desc_)
    {
    }

    std::size_t GetWorkspaceSize() const noexcept { return 0; }
    Data_t GetWorkspace() const noexcept { return nullptr; }
};

struct TransposePseudoSolver
{
    virtual ~TransposePseudoSolver()                                        = default;
    virtual std::string GetTranspose() const                                = 0;
    virtual bool IsApplicable(const TransposeProblem& problem) const        = 0;
    virtual ConvSolution GetSolution(const ExecutionContext& ctx,
                                     const TransposeProblem& problem) const = 0;

protected:
    TransposePseudoSolver()                             = default;
    TransposePseudoSolver(const TransposePseudoSolver&) = default;
};

template <class Derived, class Interface>
struct AnyImplementation
{
    AnyImplementation() : buffer(), copy(nullptr), p(nullptr) {}

    template <class Implementation>
    AnyImplementation(const Implementation& impl)
    {
        static_assert(sizeof(Implementation) == sizeof(Interface),
                      "Implementation must be stateless");
        static_assert(std::is_base_of<Interface, Implementation>{},
                      "Not derived class of the interface");
        copy = +[](const Storage& src, Storage& dst, Interface** interface) {
            new(std::addressof(dst)) Implementation(*StorageCast<const Implementation>(src));
            *interface = static_cast<Interface*>(StorageCast<Implementation>(dst));
        };

        new(std::addressof(buffer)) Implementation(impl);
        p = static_cast<Interface*>(StorageCast<Implementation>(buffer));
    }

    AnyImplementation(const Derived& rhs) : buffer(), copy(rhs.copy), p(nullptr)
    {
        copy(rhs.buffer, buffer, &p);
    }

    const Interface* get() const noexcept { return p; }
    const Interface& operator*() const noexcept { return *get(); }
    const Interface* operator->() const noexcept { return get(); }

    ~AnyImplementation() noexcept
    {
        if(p)
            p->~Interface();
    }

private:
    struct alignas(Interface) Storage
    {
        unsigned char data[sizeof(Interface)];
    };
    using Cloner = void (*)(const Storage&, Storage&, Interface**);

    template <class T, class S>
    static T* StorageCast(S&& s)
    {
        return reinterpret_cast<T*>(std::addressof(s));
    }

    Storage buffer;
    Cloner copy;
    Interface* p;
};

struct AnyTransposePseudoSolver : AnyImplementation<AnyTransposePseudoSolver, TransposePseudoSolver>
{
    AnyTransposePseudoSolver() = default;

    template <class Transpose>
    AnyTransposePseudoSolver(const Transpose& s)
        : AnyImplementation<AnyTransposePseudoSolver, TransposePseudoSolver>(s)
    {
    }

    AnyTransposePseudoSolver(const AnyTransposePseudoSolver& rhs)
        : AnyImplementation<AnyTransposePseudoSolver, TransposePseudoSolver>(rhs)
    {
    }
};

struct UniversalTransposeSolver : TransposePseudoSolver
{
    std::string GetTranspose() const override { return "*-*"; }

    bool IsApplicable(const TransposeProblem&) const override { return true; }

    ConvSolution GetSolution(const ExecutionContext& ctx,
                             const TransposeProblem& problem) const override
    {
        auto sln = ConvSolution{};

        auto create_kernel = [&](const std::string& index_type) {
            const auto tensor_space = problem.input.GetElementSize();
            const auto cus          = ctx.GetStream().GetMaxComputeUnits();
            auto build_params       = GetDataTypeKBP(problem.input.GetType());

            constexpr std::size_t max_block_size = 256;
            const auto local_size                = max_block_size;
            // Note: std::max/min without explicit template params to avoid cppcheck false
            // positives when testing with -Dmax/-Dmin preprocessor defines
            const auto num_blocks =
                std::max(std::size_t{1}, (tensor_space + local_size - 1) / local_size);
            const auto capped_blocks = std::min(num_blocks, cus * 4);
            const auto global_size   = capped_blocks * local_size;

            build_params.Define("INDEX_TYPE", index_type);

            auto transposeKernel = KernelInfo{};
            transposeKernel.g_wk = {global_size, 1, 1};
            transposeKernel.l_wk = {local_size, 1, 1};

            transposeKernel.kernel_file  = "UniversalTranspose.cpp";
            transposeKernel.kernel_name  = "UniversalTranspose";
            transposeKernel.comp_options = build_params.GenerateFor(kbp::HIP{});

            return transposeKernel;
        };

        sln.construction_params.emplace_back(create_kernel("uint32_t"));
        sln.construction_params.emplace_back(create_kernel("uint64_t"));

        sln.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [kernels](const Handle& handle, const AnyInvokeParams& any_params) {
                const auto& params = any_params.CastTo<TransposeInvokeParams>();

                uint64_t max_index =
                    std::max(params.in_desc.GetElementSpace(), params.out_desc.GetElementSpace());

                // Limit using uint32_t indices to about 4M elements as this seems to be the limit
                // for where it helps performance
                constexpr uint64_t u32_threshold = 4ull * 1024ull * 1024ull;
                if(max_index <= u32_threshold)
                {
                    const auto& lens        = GetNCDHW<uint32_t>(params.in_desc.GetLengths());
                    const auto& in_strides  = GetNCDHW<uint32_t>(params.in_desc.GetStrides());
                    const auto& out_strides = GetNCDHW<uint32_t>(params.out_desc.GetStrides());

                    // clang-format off
                    handle.Run(kernels[0])(
                        params.in, params.out,
                        lens[0],        lens[1],        lens[2],        lens[3],        lens[4],
                        in_strides[0],  in_strides[1],  in_strides[2],  in_strides[3],  in_strides[4],
                        out_strides[0], out_strides[1], out_strides[2], out_strides[3], out_strides[4]
                    );
                    // clang-format on
                }
                else
                {
                    const auto& lens        = GetNCDHW<uint64_t>(params.in_desc.GetLengths());
                    const auto& in_strides  = GetNCDHW<uint64_t>(params.in_desc.GetStrides());
                    const auto& out_strides = GetNCDHW<uint64_t>(params.out_desc.GetStrides());

                    // clang-format off
                    handle.Run(kernels[1])(
                        params.in, params.out,
                        lens[0],        lens[1],        lens[2],        lens[3],        lens[4],
                        in_strides[0],  in_strides[1],  in_strides[2],  in_strides[3],  in_strides[4],
                        out_strides[0], out_strides[1], out_strides[2], out_strides[3], out_strides[4]
                    );
                    // clang-format on
                }
            };
        };

        return sln;
    }
};

template <std::size_t TileSizeX, std::size_t TileSizeY, std::size_t BlockSize>
struct TiledTransposeSolver : TransposePseudoSolver
{
    std::string GetTranspose() const override { return "*-*"; }

    bool IsApplicable(const TransposeProblem& problem) const override
    {
        const auto& in_strides  = problem.input.GetStrides();
        const auto out_desc     = problem.GetOutputDescriptor();
        const auto& out_strides = out_desc.GetStrides();

        const auto in_stride_w  = in_strides.back();
        const auto out_stride_w = out_strides.back();
        const auto in_stride_c  = in_strides.size() >= 2 ? in_strides[1] : 1;
        const auto out_stride_c = out_strides.size() >= 2 ? out_strides[1] : 1;

        const bool nchw_to_nhwc = (in_stride_w == 1 && out_stride_c == 1);
        const bool nhwc_to_nchw = (in_stride_c == 1 && out_stride_w == 1);

        return nchw_to_nhwc || nhwc_to_nchw;
    }

    ConvSolution GetSolution(const ExecutionContext& ctx,
                             const TransposeProblem& problem) const override
    {
        auto sln = ConvSolution{};

        auto create_kernel = [&](const std::string& index_type) {
            const auto& lens  = problem.input.GetLengths();
            const auto cus    = ctx.GetStream().GetMaxComputeUnits();
            auto build_params = GetDataTypeKBP(problem.input.GetType());

            const auto n = lens.size() >= 1 ? lens[0] : 1;
            const auto c = lens.size() >= 2 ? lens[1] : 1;
            const auto d = lens.size() >= 5 ? lens[2] : 1;
            const auto h = lens.size() >= 4 ? lens[lens.size() - 2] : 1;
            const auto w = lens.size() >= 3 ? lens[lens.size() - 1] : 1;

            const auto tiles_c   = (c + TileSizeY - 1) / TileSizeY;
            const auto tiles_w   = (w + TileSizeX - 1) / TileSizeX;
            const auto num_tiles = n * d * h * tiles_c * tiles_w;

            // Cap blocks at 8 waves per CU
            const auto capped_blocks = std::min(num_tiles, cus * 8);
            const auto global_size   = capped_blocks * BlockSize;

            build_params.Define("INDEX_TYPE", index_type);
            build_params.Define("TILE_SIZE_X", TileSizeX);
            build_params.Define("TILE_SIZE_Y", TileSizeY);
            build_params.Define("BLOCK_SIZE", BlockSize);

            auto transposeKernel = KernelInfo{};
            transposeKernel.g_wk = {global_size, 1, 1};
            transposeKernel.l_wk = {BlockSize, 1, 1};

            transposeKernel.kernel_file  = "UniversalTranspose.cpp";
            transposeKernel.kernel_name  = "TiledTranspose";
            transposeKernel.comp_options = build_params.GenerateFor(kbp::HIP{});
            return std::move(transposeKernel);
        };

        sln.construction_params.emplace_back(create_kernel("uint32_t"));
        sln.construction_params.emplace_back(create_kernel("uint64_t"));

        sln.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [kernels](const Handle& handle, const AnyInvokeParams& any_params) {
                const auto& params = any_params.CastTo<TransposeInvokeParams>();

                uint64_t max_index =
                    std::max(params.in_desc.GetElementSpace(), params.out_desc.GetElementSpace());

                constexpr uint64_t u32_threshold = 4ull * 1024ull * 1024ull;
                if(max_index <= u32_threshold)
                {
                    const auto& lens    = GetNCDHW<uint32_t>(params.in_desc.GetLengths());
                    const auto& in_str  = GetNCDHW<uint32_t>(params.in_desc.GetStrides());
                    const auto& out_str = GetNCDHW<uint32_t>(params.out_desc.GetStrides());

                    // clang-format off
                    handle.Run(kernels[0])(
                        params.in, params.out,
                        lens[0],    lens[1],    lens[2],    lens[3],    lens[4],
                        in_str[0],  in_str[1],  in_str[2],  in_str[3],  in_str[4],
                        out_str[0], out_str[1], out_str[2], out_str[3], out_str[4]
                    );
                    // clang-format on
                }
                else
                {
                    const auto& lens    = GetNCDHW<uint64_t>(params.in_desc.GetLengths());
                    const auto& in_str  = GetNCDHW<uint64_t>(params.in_desc.GetStrides());
                    const auto& out_str = GetNCDHW<uint64_t>(params.out_desc.GetStrides());

                    // clang-format off
                    handle.Run(kernels[1])(
                        params.in, params.out,
                        lens[0],    lens[1],    lens[2],    lens[3],    lens[4],
                        in_str[0],  in_str[1],  in_str[2],  in_str[3],  in_str[4],
                        out_str[0], out_str[1], out_str[2], out_str[3], out_str[4]
                    );
                    // clang-format on
                }
            };
        };

        return sln;
    }
};

template <std::size_t VectorSize, std::size_t BlockSize>
struct VectorizedTransposeSolver : TransposePseudoSolver
{
    static_assert(VectorSize == 1 || VectorSize == 2 || VectorSize == 4,
                  "Vector size must be 1, 2, or 4");

    std::string GetTranspose() const override { return "*-*"; }

    bool IsApplicable(const TransposeProblem& problem) const override
    {
        const auto& in_strides  = problem.input.GetStrides();
        const auto& lens        = problem.input.GetLengths();
        const auto out_desc     = problem.GetOutputDescriptor();
        const auto& out_strides = out_desc.GetStrides();
        const auto type         = problem.input.GetType();
        if(type != miopenFloat && type != miopenHalf && type != miopenBFloat16)
            return false;
        if(in_strides.empty() || lens.empty())
            return false;

        const auto w_len        = lens.back();
        const auto in_stride_w  = in_strides.back();
        const auto out_stride_w = out_strides.back();

        const bool can_vectorize_in  = (in_stride_w == 1) && (w_len % VectorSize == 0);
        const bool can_vectorize_out = (out_stride_w == 1) && (w_len % VectorSize == 0);

        return can_vectorize_in || can_vectorize_out;
    }

    ConvSolution GetSolution(const ExecutionContext& ctx,
                             const TransposeProblem& problem) const override
    {
        auto sln = ConvSolution{};

        auto create_kernel = [&](const std::string& index_type) {
            const auto tensor_space = problem.input.GetElementSize();
            const auto cus          = ctx.GetStream().GetMaxComputeUnits();
            auto build_params       = GetDataTypeKBP(problem.input.GetType());

            const auto num_blocks =
                std::max(std::size_t{1}, (tensor_space + BlockSize - 1) / BlockSize);
            // Cap blocks at 4 waves per CU
            const auto capped_blocks = std::min(num_blocks, cus * 4);
            const auto global_size   = capped_blocks * BlockSize;

            build_params.Define("INDEX_TYPE", index_type);
            build_params.Define("VECTOR_SIZE", VectorSize);
            build_params.Define("BLOCK_SIZE", BlockSize);

            auto transposeKernel = KernelInfo{};
            transposeKernel.g_wk = {global_size, 1, 1};
            transposeKernel.l_wk = {BlockSize, 1, 1};

            transposeKernel.kernel_file  = "UniversalTranspose.cpp";
            transposeKernel.kernel_name  = "VectorizedTranspose";
            transposeKernel.comp_options = build_params.GenerateFor(kbp::HIP{});
            return std::move(transposeKernel);
        };

        sln.construction_params.emplace_back(create_kernel("uint32_t"));
        sln.construction_params.emplace_back(create_kernel("uint64_t"));

        sln.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [kernels](const Handle& handle, const AnyInvokeParams& any_params) {
                const auto& params = any_params.CastTo<TransposeInvokeParams>();

                uint64_t max_index =
                    std::max(params.in_desc.GetElementSpace(), params.out_desc.GetElementSpace());

                constexpr uint64_t u32_threshold = 4ull * 1024ull * 1024ull;
                if(max_index <= u32_threshold)
                {
                    const auto& lens        = GetNCDHW<uint32_t>(params.in_desc.GetLengths());
                    const auto& in_strides  = GetNCDHW<uint32_t>(params.in_desc.GetStrides());
                    const auto& out_strides = GetNCDHW<uint32_t>(params.out_desc.GetStrides());
                    const bool can_vectorize_in =
                        (in_strides[4] == 1) && (lens[4] % VectorSize == 0);
                    const bool can_vectorize_out =
                        (out_strides[4] == 1) && (lens[4] % VectorSize == 0);

                    // clang-format off
                    handle.Run(kernels[0])(
                        params.in, params.out,
                        lens[0],        lens[1],        lens[2],        lens[3],        lens[4],
                        in_strides[0],  in_strides[1],  in_strides[2],  in_strides[3],  in_strides[4],
                        out_strides[0], out_strides[1], out_strides[2], out_strides[3], out_strides[4],
                        can_vectorize_in, can_vectorize_out
                    );
                    // clang-format on
                }
                else
                {
                    const auto& lens        = GetNCDHW<uint64_t>(params.in_desc.GetLengths());
                    const auto& in_strides  = GetNCDHW<uint64_t>(params.in_desc.GetStrides());
                    const auto& out_strides = GetNCDHW<uint64_t>(params.out_desc.GetStrides());
                    const bool can_vectorize_in =
                        (in_strides[4] == 1) && (lens[4] % VectorSize == 0);
                    const bool can_vectorize_out =
                        (out_strides[4] == 1) && (lens[4] % VectorSize == 0);

                    // clang-format off
                    handle.Run(kernels[1])(
                        params.in, params.out,
                        lens[0],        lens[1],        lens[2],        lens[3],        lens[4],
                        in_strides[0],  in_strides[1],  in_strides[2],  in_strides[3],  in_strides[4],
                        out_strides[0], out_strides[1], out_strides[2], out_strides[3], out_strides[4],
                        can_vectorize_in, can_vectorize_out
                    );
                    // clang-format on
                }
            };
        };

        return sln;
    }
};

/// \brief Traits for batched transpose layout transformations
/// Provides layout string for each transpose solution type
template <typename TransposeSolution>
struct BatchedTransposeTraits;

template <>
struct BatchedTransposeTraits<TransposeSolutionDefault2Nhwc>
{
    static constexpr const char* layout_transform = "NCHW-NHWC";
    static constexpr int ndims                    = 4;
};

template <>
struct BatchedTransposeTraits<TransposeSolutionNhwc2Default>
{
    static constexpr const char* layout_transform = "NHWC-NCHW";
    static constexpr int ndims                    = 4;
};

template <>
struct BatchedTransposeTraits<TransposeSolutionDefault2Ndhwc>
{
    static constexpr const char* layout_transform = "NCDHW-NDHWC";
    static constexpr int ndims                    = 5;
};

template <>
struct BatchedTransposeTraits<TransposeSolutionNdhwc2Default>
{
    static constexpr const char* layout_transform = "NDHWC-NCDHW";
    static constexpr int ndims                    = 5;
};

/// \brief Generic batched transpose solver template
/// Eliminates code duplication between NCHW<->NHWC transpose solvers by parameterizing
/// on the TransposeSolution type. Uses traits to provide the layout transformation string.
/// \tparam TransposeSolution The specific batched transpose solution class to use
///         (e.g., TransposeSolutionDefault2Nhwc, TransposeSolutionNhwc2Default)
template <typename TransposeSolution>
struct BatchedTransposeSolverImpl : TransposePseudoSolver
{
    std::string GetTranspose() const override
    {
        return BatchedTransposeTraits<TransposeSolution>::layout_transform;
    }

    bool IsApplicable(const TransposeProblem& problem) const override
    {
        auto pair = std::string(problem.layout) + "-" + problem.target_layout;
        if(pair != BatchedTransposeTraits<TransposeSolution>::layout_transform)
            return false;

        const auto& desc = problem.input;
        const auto& lens = desc.GetLengths();

        return BatchedTransposeSolution::IsApplicable(desc.GetType(), lens);
    }

    ConvSolution GetSolution(const ExecutionContext& ctx,
                             const TransposeProblem& problem) const override
    {
        auto transpose_sol = CreateTransposeSolution(ctx, problem.input);

        auto sln = ConvSolution{};
        sln.construction_params.push_back(transpose_sol.GetKernelInfo());

        // Capture kernel args by value for the invoker
        auto kernel_args = transpose_sol.GetKernelArg();

        sln.invoker_factory = [kernel_args](const std::vector<Kernel>& kernels) mutable {
            return [kernel_args, kernel = kernels.front()](
                       const Handle& handle, const AnyInvokeParams& any_params) mutable {
                const auto& params = any_params.CastTo<TransposeInvokeParams>();

                // Update src/dst pointers in kernel args
                kernel_args[0] = OpKernelArg(params.out); // dst
                kernel_args[1] = OpKernelArg(params.in);  // src

                handle.Run(kernel)(kernel_args);
            };
        };

        return sln;
    }

private:
    static TransposeSolution CreateTransposeSolution(const ExecutionContext& ctx,
                                                     const TensorDescriptor& desc)
    {
        const auto& lens     = desc.GetLengths();
        constexpr int n_dims = BatchedTransposeTraits<TransposeSolution>::ndims;

        const uint32_t n = static_cast<uint32_t>(lens[0]);
        const uint32_t c = static_cast<uint32_t>(lens[1]);

        if constexpr(n_dims == 4)
        {
            const uint32_t h = static_cast<uint32_t>(lens[2]);
            const uint32_t w = static_cast<uint32_t>(lens[3]);
            return TransposeSolution(ctx, desc.GetType(), n, c, h, w);
        }
        else // n_dims == 5
        {
            const uint32_t d = static_cast<uint32_t>(lens[2]);
            const uint32_t h = static_cast<uint32_t>(lens[3]);
            const uint32_t w = static_cast<uint32_t>(lens[4]);
            return TransposeSolution(ctx, desc.GetType(), n, c, d, h, w);
        }
    }
};

/// \brief High-performance NCHW to NHWC transpose using LDS-tiled batched transpose kernel
/// Uses TransposeSolutionDefault2Nhwc which provides coalesced memory access and shared memory
/// tiling for significantly better performance than the naive UniversalTransposeSolver.
using BatchedNchw2NhwcTransposeSolver = BatchedTransposeSolverImpl<TransposeSolutionDefault2Nhwc>;
using BatchedNcdhw2NdhwcTransposeSolver =
    BatchedTransposeSolverImpl<TransposeSolutionDefault2Ndhwc>;

/// \brief High-performance NHWC to NCHW transpose using LDS-tiled batched transpose kernel
/// Uses TransposeSolutionNhwc2Default which provides coalesced memory access and shared memory
/// tiling for significantly better performance than the naive UniversalTransposeSolver.
using BatchedNhwc2NchwTransposeSolver = BatchedTransposeSolverImpl<TransposeSolutionNhwc2Default>;
using BatchedNdhwc2NcdhwTransposeSolver =
    BatchedTransposeSolverImpl<TransposeSolutionNdhwc2Default>;

class SegmentedGpuBuffer
{
public:
    SegmentedGpuBuffer(const Handle& handle_, Data_t memory_, std::size_t offset_ = 0)
        : handle(&handle_), memory(memory_), offset(offset_)
    {
        assert(handle);
    }

    SegmentedGpuBuffer(SegmentedGpuBuffer&)             = delete;
    SegmentedGpuBuffer(SegmentedGpuBuffer&&)            = delete;
    SegmentedGpuBuffer& operator=(SegmentedGpuBuffer&)  = delete;
    SegmentedGpuBuffer& operator=(SegmentedGpuBuffer&&) = delete;

    miopen::shared<Data_t> operator()(std::size_t size)
    {
        const auto align = GetSubbufferAlignment(handle);
        offset += (align - offset) % align;

        auto subbuffer = handle->CreateSubBuffer(memory, offset, size);
        offset += size;

        return subbuffer;
    }

private:
    const Handle* handle;
    Data_t memory;
    std::size_t offset;
};

inline std::string SyncLayoutDims(const char* from, const char* to)
{
    if(strlen(from) < 5)
        return ReplaceString(to, "D", "");
    return to;
}

template <class Problem, class InvokeParams>
struct ProblemTensorTransposeDescriptor
{
    using ConstDescriptorGetter = const TensorDescriptor& (Problem::*)() const;

    ConstDescriptorGetter cdescriptor;
    TensorDescriptor InvokeParams::*rt_descriptor;

    ConstData_t InvokeParams::*as_input = nullptr;
    Data_t InvokeParams::*as_output     = nullptr;

    const char* to;
    bool is_input;

    template <class Problem_> // to deal with constParameter invalid warning
    inline void Transpose(const Problem& src, Problem_& dest) const
    {
        const auto& desc_from = (src.*cdescriptor)();
        // Use const_cast on the copy (dest) only - this is safe because we're mutating a copy,
        // not the original problem. The copy is owned by the transposing solver and not shared.
        auto& desc_to = const_cast<TensorDescriptor&>((dest.*cdescriptor)());
        desc_to       = Transpose(desc_from);
    }

    inline void Transpose(const InvokeParams& src, InvokeParams& dest) const
    {
        const auto& desc_from = src.*rt_descriptor;
        auto& desc_to         = dest.*rt_descriptor;
        desc_to               = Transpose(desc_from);
    }

    inline TensorDescriptor Transpose(const TensorDescriptor& in) const
    {
        const auto labels    = tensor_layout_get_default(in.GetNumDims());
        auto derived_strides = std::vector<size_t>{};
        tensor_layout_to_strides(
            in.GetLengths(), labels, SyncLayoutDims(labels.c_str(), to), derived_strides);
        return {in.GetType(), in.GetLengths(), derived_strides};
    }
};

class ProblemTensorTransposeInvoke
{
public:
    template <class Problem, class InvokeParams>
    ProblemTensorTransposeInvoke(
        SegmentedGpuBuffer& allocator,
        const ProblemTensorTransposeDescriptor<Problem, InvokeParams>& descriptor,
        const Invoker& invoker_,
        const InvokeParams& invoke_params,
        InvokeParams& transposed_params)
        : invoker(invoker_)
    {
        // Transpose runtime tensor descriptor
        descriptor.Transpose(invoke_params, transposed_params);

        const auto& orig_descriptor       = invoke_params.*(descriptor.rt_descriptor);
        const auto& transposed_descriptor = transposed_params.*(descriptor.rt_descriptor);

        // Allocate subbuffer in the workspace
        const auto e_size      = get_data_size(transposed_descriptor.GetType());
        const auto buffer_size = transposed_descriptor.GetElementSpace() * e_size;
        buffer                 = allocator(buffer_size);

        if(descriptor.is_input)
            transposed_params.*(descriptor.as_input) = buffer.get();
        else
            transposed_params.*(descriptor.as_output) = buffer.get();

        if(!descriptor.is_input)
        {
            // Prepare output transpose invoker
            const auto& out = invoke_params.*(descriptor.as_output);

            transpose_params =
                TransposeInvokeParams{buffer.get(), out, transposed_descriptor, orig_descriptor};
            return;
        }

        // Transpose input tensor
        const auto& in = invoke_params.*(descriptor.as_input);

        transpose_params =
            TransposeInvokeParams{in, buffer.get(), orig_descriptor, transposed_descriptor};
    }

    void operator()(const Handle& handle) const
    {
        const auto time = handle.GetKernelTime();
        invoker(handle, transpose_params);
        handle.AccumKernelTime(time);
    }

private:
    miopen::shared<Data_t> buffer;
    Invoker invoker;
    AnyInvokeParams transpose_params;
};

class ProblemTensorTransposeGroup
{
public:
    template <class Problem, class InvokeParams>
    ProblemTensorTransposeGroup(
        const Handle& handle_,
        SegmentedGpuBuffer& allocator,
        const std::vector<
            std::tuple<ProblemTensorTransposeDescriptor<Problem, InvokeParams>, Invoker>>& inputs_,
        const std::vector<
            std::tuple<ProblemTensorTransposeDescriptor<Problem, InvokeParams>, Invoker>>& outputs_,
        const InvokeParams& invoke_params,
        InvokeParams& transposed_params)
        : handle(&handle_)
    {
        std::transform(
            inputs_.begin(), inputs_.end(), std::back_inserter(inputs), [&](auto&& params) {
                return ProblemTensorTransposeInvoke(allocator,
                                                    std::get<0>(params),
                                                    std::get<1>(params),
                                                    invoke_params,
                                                    transposed_params);
            });

        std::transform(
            outputs_.begin(), outputs_.end(), std::back_inserter(outputs), [&](auto&& params) {
                return ProblemTensorTransposeInvoke(allocator,
                                                    std::get<0>(params),
                                                    std::get<1>(params),
                                                    invoke_params,
                                                    transposed_params);
            });

        MIOPEN_LOG_I2("Executing the input transpose");
        for(const auto& transpose : inputs)
            transpose(*handle);
    }

    ProblemTensorTransposeGroup(ProblemTensorTransposeGroup&)             = delete;
    ProblemTensorTransposeGroup(ProblemTensorTransposeGroup&&)            = delete;
    ProblemTensorTransposeGroup& operator=(ProblemTensorTransposeGroup&)  = delete;
    ProblemTensorTransposeGroup& operator=(ProblemTensorTransposeGroup&&) = delete;

    ~ProblemTensorTransposeGroup()
    {
        MIOPEN_LOG_I2("Executing the output transpose");
        for(const auto& transpose : outputs)
            transpose(*handle);
    }

private:
    const Handle* handle;
    std::vector<ProblemTensorTransposeInvoke> inputs;
    std::vector<ProblemTensorTransposeInvoke> outputs;
};

/// Helper base that provides the correct GetSolution override based on whether
/// the solver Base is tunable or non-tunable. Tunable solvers have
/// GetSolution(ctx, problem, config) while non-tunable solvers have
/// GetSolution(ctx, problem). This class inherits from Base so that
/// TransposingSolver can inherit from it and get both Base and GetSolution.
template <class Derived,
          class Base,
          class Problem,
          class InvokeParams,
          class Inner,
          bool IsTunable = std::is_base_of<TunableSolverTrait, Base>::value>
struct TransposingSolverGetSolution;

// Forward declaration
template <class Derived, class Base, class Problem, class InvokeParams, class Inner>
struct TransposingSolver;

/// Non-tunable specialization: provides GetSolution(ctx, problem)
template <class Derived, class Base, class Problem, class InvokeParams, class Inner>
struct TransposingSolverGetSolution<Derived, Base, Problem, InvokeParams, Inner, false> : Base
{
    ConvSolution GetSolution(const ExecutionContext& ctx, const Problem& problem) const override
    {
        auto transposed_problem = Derived::Transpose(problem);
        ConvSolution sln        = Inner{}.GetSolution(ctx, transposed_problem);
        return static_cast<const Derived*>(this)->WrapSolutionWithTranspose(
            ctx, problem, transposed_problem, std::move(sln));
    }
};

/// Tunable specialization: provides GetSolution(ctx, problem, config) and delegates
/// GetDefaultPerformanceConfig, IsValidPerformanceConfig, and Search to the inner solver.
template <class Derived, class Base, class Problem, class InvokeParams, class Inner>
struct TransposingSolverGetSolution<Derived, Base, Problem, InvokeParams, Inner, true> : Base
{
    using PerformanceConfigType = typename Inner::PerformanceConfigType;

    PerformanceConfigType GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                                      const Problem& problem) const override
    {
        auto transposed_problem = Derived::Transpose(problem);
        return Inner{}.GetDefaultPerformanceConfig(ctx, transposed_problem);
    }

    bool IsValidPerformanceConfig(const ExecutionContext& ctx,
                                  const Problem& problem,
                                  const PerformanceConfigType& config) const override
    {
        auto transposed_problem = Derived::Transpose(problem);
        return Inner{}.IsValidPerformanceConfig(ctx, transposed_problem, config);
    }

    PerformanceConfigType Search(const ExecutionContext& ctx,
                                 const Problem& problem,
                                 const AnyInvokeParams& invoke_ctx) const override
    {
        auto transposed_problem = Derived::Transpose(problem);
        return Inner{}.Search(ctx, transposed_problem, invoke_ctx);
    }

    ConvSolution GetSolution(const ExecutionContext& ctx,
                             const Problem& problem,
                             const PerformanceConfigType& config) const override
    {
        auto transposed_problem = Derived::Transpose(problem);
        ConvSolution sln        = Inner{}.GetSolution(ctx, transposed_problem, config);
        return static_cast<const Derived*>(this)->WrapSolutionWithTranspose(
            ctx, problem, transposed_problem, std::move(sln));
    }
};

template <class Derived, class Base, class Problem, class InvokeParams, class Inner>
struct TransposingSolver : TransposingSolverGetSolution<Derived, Base, Problem, InvokeParams, Inner>
{
    // Allow the GetSolution helper to access Transpose and WrapSolutionWithTranspose
    friend struct TransposingSolverGetSolution<Derived,
                                               Base,
                                               Problem,
                                               InvokeParams,
                                               Inner,
                                               std::is_base_of<TunableSolverTrait, Base>::value>;

    using TransposeDescriptor = ProblemTensorTransposeDescriptor<Problem, InvokeParams>;

    /// TransposingSolver always needs workspace for transpose buffers.
    bool MayNeedWorkspace() const override { return true; }

    /// Convert from API invoke params to TransposingSolver invoke params.
    /// Override in derived class if API passes a different type than InvokeParams.
    /// Default: cast AnyInvokeParams directly to InvokeParams.
    static InvokeParams ConvertFromApiParams(const AnyInvokeParams& any_params)
    {
        return any_params.CastTo<InvokeParams>();
    }

    /// Convert invoke params for inner solver invocation.
    /// Override in derived class if inner solver expects a different params type.
    /// Default: return params as AnyInvokeParams (pass-through).
    static AnyInvokeParams ConvertForInnerSolver(const InvokeParams& params) { return params; }

    static std::vector<AnyTransposePseudoSolver> GetTransposeSolvers()
    {
        return {BatchedNchw2NhwcTransposeSolver{},
                BatchedNhwc2NchwTransposeSolver{},
                BatchedNcdhw2NdhwcTransposeSolver{},
                BatchedNdhwc2NcdhwTransposeSolver{},
                TiledTransposeSolver<16, 16, 256>{},
                VectorizedTransposeSolver<4, 64>{},
                VectorizedTransposeSolver<2, 64>{},
                UniversalTransposeSolver{}};
    }

    static const AnyTransposePseudoSolver*
    FindApplicableSolver(const TransposeProblem& problem,
                         const std::vector<AnyTransposePseudoSolver>& solvers)
    {
        for(const auto& solver : solvers)
        {
            if(solver->IsApplicable(problem))
                return &solver;
        }
        return nullptr;
    }

    // Check applicability for transposing solvers that wraps an inner solver
    bool IsApplicable(const ExecutionContext& ctx, const Problem& problem) const override
    {
        const auto transpose_solvers = Derived::GetTransposeSolvers();
        auto any_difference          = false;

        for(auto transpose : Derived::GetTransposes(problem))
        {
            decltype(auto) descriptor = (problem.*(transpose.cdescriptor))();
            const auto layout         = descriptor.GetLayout_str();
            const auto to             = SyncLayoutDims(layout.c_str(), transpose.to);

            if(layout == to)
                continue;

            any_difference = true;

            const auto transpose_problem = TransposeProblem{descriptor, layout.c_str(), to};
            if(FindApplicableSolver(transpose_problem, transpose_solvers) == nullptr)
                return false;
        }

        if(!any_difference)
        {
            MIOPEN_LOG_I("No layout difference detected for solver, skipping transpose");
            return false;
        }

        // Use Derived::Transpose to allow derived classes to override (CRTP pattern)
        const auto transposed_problem = Derived::Transpose(problem);

        const bool inner_applicable = Inner{}.IsApplicable(ctx, transposed_problem);

        return inner_applicable;
    }

    std::size_t GetWorkspaceSize(const ExecutionContext& ctx, const Problem& problem) const override
    {
        // Use Derived::Transpose to allow derived classes to override (CRTP pattern)
        const auto transposed_problem = Derived::Transpose(problem);

        // Calculate workspace needed by inner solver plus transpose buffers
        auto ws_size = Inner{}.GetWorkspaceSize(ctx, transposed_problem);
        for(const auto& transpose : Derived::GetTransposes(problem))
        {
            const auto& descriptor = (transposed_problem.*(transpose.cdescriptor))();
            const auto e_size      = get_data_size(descriptor.GetType());
            const auto tensor_size = descriptor.GetElementSpace() * e_size;
            ws_size += tensor_size;
        }
        return ws_size;
    }

    bool IsDynamic() const override
    {
        // Transposed solvers ALWAYS require workspace for transpose buffers,
        // so they can ONLY work in Find mode (not immediate mode with workspace=0).
        return false;
    }

    float GetWti(const ExecutionContext& ctx, const Problem& problem) const override
    {
        // Delegate to inner solver's GetWti() to ensure transposed solvers
        // are included in WTI fallback path when TunaNet doesn't predict them
        const auto transposed_problem = Derived::Transpose(problem);
        return Inner{}.GetWti(ctx, transposed_problem);
    }

    /// Wraps an inner solver's ConvSolution with transpose kernels.
    /// Called by the GetSolution helper specializations.
    ConvSolution WrapSolutionWithTranspose(const ExecutionContext& ctx,
                                           const Problem& problem,
                                           Problem& transposed_problem,
                                           ConvSolution sln) const
    {
        // NOLINTNEXTLINE (bugprone-unchecked-optional-access)
        auto old_factory             = sln.invoker_factory.value();
        const auto old_kernels_end   = sln.construction_params.size();
        const auto transpose_solvers = Derived::GetTransposeSolvers();

        std::vector<std::tuple<TransposeDescriptor, InvokerFactory>> in_transpose_ifs,
            out_transpose_ifs;

        for(auto transpose : Derived::GetTransposes(problem))
        {
            // For input transposes: use original problem's layout as source
            // For output transposes: use transposed problem's layout as source
            const auto& src_problem = transpose.is_input ? problem : transposed_problem;
            const auto& dst_problem = transpose.is_input ? transposed_problem : problem;

            const auto& src_descriptor = (src_problem.*(transpose.cdescriptor))();
            const auto& dst_descriptor = (dst_problem.*(transpose.cdescriptor))();

            const auto layout = src_descriptor.GetLayout_str();
            const auto to = SyncLayoutDims(layout.c_str(), dst_descriptor.GetLayout_str().c_str());

            if(layout == to)
            {
                MIOPEN_LOG_I("TransposingSolver: skipping - layout already matches target");
                continue;
            }

            const auto transpose_problem = TransposeProblem{src_descriptor, layout.c_str(), to};
            const auto* solver = FindApplicableSolver(transpose_problem, transpose_solvers);
            if(solver == nullptr)
                MIOPEN_THROW("No applicable transpose solver found for layout transformation: " +
                             std::string(layout) + " -> " + std::string(to));

            auto transpose_sln = (*solver)->GetSolution(ctx, transpose_problem);

            const auto kernels_begin = sln.construction_params.size();
            sln.construction_params.insert(sln.construction_params.end(),
                                           transpose_sln.construction_params.begin(),
                                           transpose_sln.construction_params.end());
            const auto kernels_end         = sln.construction_params.size();
            const auto raw_invoker_factory = transpose_sln.invoker_factory;

            auto transpose_invoker_factory = [kernels_begin, kernels_end, raw_invoker_factory](
                                                 const std::vector<Kernel>& kernels) {
                auto segment = std::vector<Kernel>{};
                segment.reserve(kernels_end - kernels_begin);
                for(auto i = kernels_begin; i < kernels_end; ++i)
                    segment.push_back(kernels[i]);
                return (*raw_invoker_factory)(segment);
            };

            (transpose.is_input ? in_transpose_ifs : out_transpose_ifs)
                .emplace_back(transpose, std::move(transpose_invoker_factory));
        }

        if(in_transpose_ifs.size() + out_transpose_ifs.size() == 0)
            return sln;

        // Inner solver workspace is used as the starting offset for SegmentedGpuBuffer.
        // The allocator carves out transpose buffers starting after the inner solver's region.
        // Total workspace (inner + transpose) is reported via sln.workspace_sz so the caller
        // allocates enough memory for both.
        const auto inner_ws_size = Inner{}.GetWorkspaceSize(ctx, transposed_problem);
        const auto total_ws_size = this->GetWorkspaceSize(ctx, problem);
        sln.invoker_factory =
            [old_factory, old_kernels_end, in_transpose_ifs, out_transpose_ifs, inner_ws_size](
                const std::vector<Kernel>& kernels) {
                const auto inner_kernels =
                    std::vector<Kernel>{kernels.begin(), kernels.begin() + old_kernels_end};
                std::vector<std::tuple<TransposeDescriptor, Invoker>> in_transpose_invokers,
                    out_transpose_invokers;

                std::transform(in_transpose_ifs.begin(),
                               in_transpose_ifs.end(),
                               std::back_inserter(in_transpose_invokers),
                               [&](const auto& params) {
                                   return std::make_tuple(std::get<0>(params),
                                                          std::get<1>(params)(kernels));
                               });

                std::transform(out_transpose_ifs.begin(),
                               out_transpose_ifs.end(),
                               std::back_inserter(out_transpose_invokers),
                               [&](const auto& params) {
                                   return std::make_tuple(std::get<0>(params),
                                                          std::get<1>(params)(kernels));
                               });

                auto invoker = old_factory(inner_kernels);

                return [invoker, in_transpose_invokers, out_transpose_invokers, inner_ws_size](
                           const Handle& handle, const AnyInvokeParams& any_params) {
                    const auto invoke_params = Derived::ConvertFromApiParams(any_params);
                    auto transposed_params   = invoke_params;

                    handle.ResetKernelTime();

                    // Start allocating transpose buffers after the inner solver's workspace.
                    // SegmentedGpuBuffer third parameter is the starting OFFSET, not size.
                    SegmentedGpuBuffer allocator{handle, invoke_params.workspace, inner_ws_size};

                    ProblemTensorTransposeGroup transposeGroup{handle,
                                                               allocator,
                                                               in_transpose_invokers,
                                                               out_transpose_invokers,
                                                               invoke_params,
                                                               transposed_params};

                    // Use Derived::ConvertForInnerSolver to convert params if needed
                    MIOPEN_LOG_I2("Executing the inner solver invoker");
                    const auto time = handle.GetKernelTime();
                    invoker(handle, Derived::ConvertForInnerSolver(transposed_params));
                    handle.AccumKernelTime(time);
                };
            };

        sln.workspace_sz = total_ws_size;

        return sln;
    }

protected:
    inline static Problem Transpose(const Problem& problem)
    {
        auto transposed_problem = problem;
        for(const auto& transpose : Derived::GetTransposes(problem))
            transpose.Transpose(problem, transposed_problem);
        return transposed_problem;
    }
};

} // namespace solver
} // namespace miopen
