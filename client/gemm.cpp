
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/Serialization/YAML.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/HIPTimer.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "../../test/unit/Utilities.hpp"

#include "include/Parser.hpp"

using namespace rocRoller;

struct GEMMProblem
{
    std::string name;

    int   M;
    int   N;
    int   K;
    int   mac_m;
    int   mac_n;
    int   mac_k;
    float alpha;
    float beta;

    int numWarmUp;
    int numOuter;
    int numInner;
};

struct GEMMResult : GEMMProblem
{
    GEMMResult(GEMMProblem const& prob)
        : GEMMProblem(prob)
    {
    }

    size_t              kernelGenerate;
    size_t              kernelAssemble;
    std::vector<size_t> kernelExecute;
};

template <typename IO>
struct rocRoller::Serialization::
    MappingTraits<GEMMResult, IO, rocRoller::Serialization::EmptyContext>
{
    static const bool flow = false;
    using iot              = IOTraits<IO>;

    static void mapping(IO& io, GEMMResult& result)
    {
        iot::mapRequired(io, "M", result.M);
        iot::mapRequired(io, "N", result.N);
        iot::mapRequired(io, "K", result.K);
        iot::mapRequired(io, "alpha", result.alpha);
        iot::mapRequired(io, "beta", result.beta);
        iot::mapRequired(io, "numWarmUp", result.numWarmUp);
        iot::mapRequired(io, "numOuter", result.numOuter);
        iot::mapRequired(io, "numInner", result.numInner);
        iot::mapRequired(io, "kernelGenerate", result.kernelGenerate);
        iot::mapRequired(io, "kernelAssemble", result.kernelAssemble);
        iot::mapRequired(io, "kernelExecute", result.kernelExecute);
    }

    static void mapping(IO& io, GEMMResult& arch, EmptyContext& ctx)
    {
        mapping(io, arch);
    }
};

// D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
GEMMResult GEMM(GEMMProblem prob, bool checkResult)
{
    GEMMResult result(prob);

    AssertFatal(result.M % result.mac_m == 0, "MacroTile size mismatch (M)");
    AssertFatal(result.N % result.mac_n == 0, "MacroTile size mismatch (N)");

    // wave tile sizes
    int wave_m = 32;
    int wave_n = 32;
    int wave_k = 2;
    int wave_b = 1;

    AssertFatal(result.mac_m % wave_m == 0, "WaveTile size mismatch (M)");
    AssertFatal(result.mac_n % wave_n == 0, "WaveTile size mismatch (N)");

    uint wavefront_size   = 64;
    uint workgroup_size_x = wavefront_size * (result.mac_m * result.mac_n / wave_m / wave_n);
    uint workgroup_size_y = 1;

    // one macro tile per workgroup
    uint num_workgroup_x = result.M / result.mac_m;
    uint num_workgroup_y = result.N / result.mac_n;

    auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
    auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
    auto NZ = std::make_shared<Expression::Expression>(1u);

    // Host data
    RandomGenerator    random(31415u);
    std::vector<float> h_A = random.vector<float>(result.M * result.K, -100, 100);
    std::vector<float> h_B = random.vector<float>(result.K * result.N, -100, 100);
    std::vector<float> h_C = random.vector<float>(result.M * result.N, -100, 100);

    // Device data
    std::shared_ptr<float> d_A = make_shared_device(h_A);
    std::shared_ptr<float> d_B = make_shared_device(h_B);
    std::shared_ptr<float> d_C = make_shared_device(h_C);
    std::shared_ptr<float> d_D = make_shared_device<float>(result.M * result.N, 0);

    auto command = std::make_shared<Command>();

    command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
        rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 0))); // A
    command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
        rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 1))); // B
    command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
        rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 2))); // C
    command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
        rocRoller::Operations::T_Load_Scalar(DataType::Float, 3))); // alpha
    command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
        rocRoller::Operations::T_Load_Scalar(DataType::Float, 4))); // beta

    command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
        rocRoller::Operations::T_Mul(5, 0, 1))); // A * B

    rocRoller::Operations::T_Execute execute;
    execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
        rocRoller::Operations::E_Mul(6, 3, 5))); // alpha * (A * B)
    execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
        rocRoller::Operations::E_Mul(7, 4, 2))); // beta * C
    execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
        rocRoller::Operations::E_Add(8, 6, 7))); // alpha * (A * B) + beta * C
    command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

    command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
        rocRoller::Operations::T_Store_Tiled(DataType::Float, 2, 8))); // D

    KernelArguments runtimeArgs;

    runtimeArgs.append("A", d_A.get());
    runtimeArgs.append("d_a_limit", (size_t)result.M * result.K);
    runtimeArgs.append("d_a_size_0", (size_t)result.M);
    runtimeArgs.append("d_a_size_1", (size_t)result.K);
    runtimeArgs.append("d_a_stride_0", (size_t)1);
    runtimeArgs.append("d_a_stride_1", (size_t)result.M);

    runtimeArgs.append("B", d_B.get());
    runtimeArgs.append("d_b_limit", (size_t)result.K * result.N);
    runtimeArgs.append("d_b_size_0", (size_t)result.K);
    runtimeArgs.append("d_b_size_1", (size_t)result.N);
    runtimeArgs.append("d_b_stride_0", (size_t)1);
    runtimeArgs.append("d_b_stride_1", (size_t)result.K);

    runtimeArgs.append("C", d_C.get());
    runtimeArgs.append("d_c_limit", (size_t)result.M * result.N);
    runtimeArgs.append("d_c_size_0", (size_t)result.M);
    runtimeArgs.append("d_c_size_1", (size_t)result.N);
    runtimeArgs.append("d_c_stride_0", (size_t)1);
    runtimeArgs.append("d_c_stride_1", (size_t)result.M);

    runtimeArgs.append("alpha", result.alpha);

    runtimeArgs.append("beta", result.beta);

    runtimeArgs.append("D", d_D.get());
    runtimeArgs.append("d_d_limit", (size_t)result.M * result.N);
    runtimeArgs.append("d_d_stride_0", (size_t)1);
    runtimeArgs.append("d_d_stride_1", (size_t)result.M);

    // TODO: remove this when for loop indexing is fixed
    command->allocateArgument(DataType::UInt32, DataDirection::ReadOnly, "UINT_MAT_K");
    runtimeArgs.append("UINT_MAT_K", static_cast<uint>(result.K));

    auto params = std::make_shared<CommandParameters>();
    params->setManualKernelDimension(2);

    auto mac_tile_0 = KernelGraph::CoordinateTransform::MacroTile( // A
        0,
        {result.mac_m, result.mac_k},
        LayoutType::MATRIX_A,
        {wave_m, wave_n, wave_k, wave_b});
    auto mac_tile_1 = KernelGraph::CoordinateTransform::MacroTile( // B
        1,
        {result.mac_k, result.mac_n},
        LayoutType::MATRIX_B,
        {wave_m, wave_n, wave_k, wave_b});
    auto mac_tile_2 = KernelGraph::CoordinateTransform::MacroTile( // C
        2,
        {result.mac_m, result.mac_n},
        LayoutType::MATRIX_ACCUMULATOR,
        {wave_m, wave_n, wave_k, wave_b});
    auto mac_tile_5 = KernelGraph::CoordinateTransform::MacroTile( // A * B
        5,
        {result.mac_m, result.mac_n},
        LayoutType::MATRIX_ACCUMULATOR,
        {wave_m, wave_n, wave_k, wave_b});
    auto mac_tile_6 = KernelGraph::CoordinateTransform::MacroTile( // alpha * (A * B)
        6,
        {result.mac_m, result.mac_n},
        LayoutType::MATRIX_ACCUMULATOR,
        {wave_m, wave_n, wave_k, wave_b});
    auto mac_tile_7 = KernelGraph::CoordinateTransform::MacroTile( // beta * C
        7,
        {result.mac_m, result.mac_n},
        LayoutType::MATRIX_ACCUMULATOR,
        {wave_m, wave_n, wave_k, wave_b});
    auto mac_tile_8 = KernelGraph::CoordinateTransform::MacroTile( // D
        8,
        {result.mac_m, result.mac_n},
        LayoutType::MATRIX_ACCUMULATOR,
        {wave_m, wave_n, wave_k, wave_b},
        true);

    params->setDimensionInfo(mac_tile_0);
    params->setDimensionInfo(mac_tile_1);
    params->setDimensionInfo(mac_tile_2);
    params->setDimensionInfo(mac_tile_5);
    params->setDimensionInfo(mac_tile_6);
    params->setDimensionInfo(mac_tile_7);
    params->setDimensionInfo(mac_tile_8);

    auto one = Expression::literal(1u);
    auto wavefront_n
        = Expression::literal(static_cast<uint>(result.mac_m * result.mac_n / wave_m / wave_n));
    auto wavefront_nx = Expression::literal(static_cast<uint>(result.mac_m / wave_m));
    auto wavefront_ny = Expression::literal(static_cast<uint>(result.mac_n / wave_n));

    std::vector<int> ctags_with_wavefronts = {0, 1, 2, 8};
    for(auto ctag : ctags_with_wavefronts)
    {
        bool output = ctag == 8;
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(ctag, -1, wavefront_n, one, output));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(ctag, 0, wavefront_nx, one, output));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(ctag, 1, wavefront_ny, one, output));
    }

    params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
    params->setManualWorkitemCount({NX, NY, NZ});

    // Build GEMM kernel
    CommandKernel commandKernel(command, "GEMM", params);

    // Warmup runs
    for(int i = 0; i < result.numWarmUp; ++i)
    {
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());
    }

    // Benchmark runs
    for(int outer = 0; outer < result.numOuter; ++outer)
    {
        HIP_TIMER(t_kernel, "GEMM");
        HIP_TIC(t_kernel);
        for(int inner = 0; inner < result.numInner; ++inner)
        {
            commandKernel.launchKernel(runtimeArgs.runtimeArguments());
        }
        HIP_TOC(t_kernel);
        HIP_SYNC(t_kernel);
        result.kernelExecute.push_back(t_kernel.nanoseconds());
    }

    if(checkResult)
    {
        // Device result
        std::vector<float> h_D(result.M * result.N, 0);
        AssertFatal(
            hipMemcpy(
                h_D.data(), d_D.get(), result.M * result.N * sizeof(float), hipMemcpyDeviceToHost)
            == HIP_SUCCESS);

        // Host result
        std::vector<float> h_result(result.M * result.N, 0);
        rocRoller::CPUMM(h_result,
                         h_C,
                         h_A,
                         h_B,
                         result.M,
                         result.N,
                         result.K,
                         result.alpha,
                         result.beta,
                         false);

        double rnorm = relativeNorm(h_D, h_result);

        std::string result = rnorm > 1e-5 ? "Incorrect" : "Correct";
        std::cout << "Result : " << result << std::endl;
    }

    // TODO loop
    double totalTime   = result.kernelExecute[0] / 1.e9;
    double averageTime = totalTime / result.numInner;

    std::cout << "Average runtime (s): " << averageTime << std::endl;
    std::cout << "Average GFLOPS: "
              << (double)result.M * result.N * result.K * 2.0 / averageTime * 1.e-9 << std::endl;

    result.kernelAssemble = TimerPool::nanoseconds("CommandKernel::assembleKernel");
    result.kernelGenerate = TimerPool::nanoseconds("CommandKernel::generateKernel");

    return result;
}

int main(int argc, const char* argv[])
{
    ParseOptions po("GEMM Driver: D (MxN) = alpha * A (MxK) * B (KxN) + beta * C (MxN)");

    po.addArg("M", Arg({"M"}, "Tensor Size M"));
    po.addArg("N", Arg({"N"}, "Tensor Size N"));
    po.addArg("K", Arg({"K"}, "Tensor Size K"));
    po.addArg("mac_m", Arg({"mac_m"}, "Macro Tile Size M"));
    po.addArg("mac_n", Arg({"mac_n"}, "Macro Tile Size N"));
    po.addArg("mac_k", Arg({"mac_k"}, "Macro Tile Size K"));
    po.addArg("alpha", Arg({"a", "alpha"}, "Alpha scalar"));
    po.addArg("beta", Arg({"b", "beta"}, "Beta scalar"));
    po.addArg("yaml", Arg({"o", "yaml"}, "Results"));
    po.addArg("num_warmup", Arg({"num_warmup"}, "Number of warm-up runs."));
    po.addArg("num_outer", Arg({"num_outer"}, "Number of outer runs."));
    po.addArg("num_inner", Arg({"num_inner"}, "Number of innter runs."));

    po.parse_args(argc, argv);

    GEMMProblem prob;

    prob.M     = po.get("M", 3072);
    prob.N     = po.get("N", 4096);
    prob.K     = po.get("K", 4096);
    prob.mac_m = po.get("mac_m", 64);
    prob.mac_n = po.get("mac_n", 64);
    prob.mac_k = po.get("mac_k", 64);
    prob.alpha = po.get("alpha", 2.0f);
    prob.beta  = po.get("beta", 0.5f);

    prob.numWarmUp = po.get("num_warmup", 3);
    prob.numOuter  = po.get("num_outer", 5);
    prob.numInner  = po.get("num_inner", 2);

    auto result = GEMM(prob, true);

    std::string filename = po.get("yaml", std::string());
    if(!filename.empty())
    {
        std::string rv;
#ifdef ROCROLLER_USE_LLVM
        llvm::raw_string_ostream sout(rv);
        llvm::yaml::Output       yout(sout);
        yout << result;
#elif ROCROLLER_USE_YAML_CPP
        YAML::Emitter                emitter;
        Serialization::EmitterOutput yout(&emitter);
        yout.outputDoc(result);
        rv = emitter.c_str();
#endif
        std::ofstream yaml;
        yaml.open(filename);
        yaml << rv;
        yaml.close();
    }

    return 0;
}
