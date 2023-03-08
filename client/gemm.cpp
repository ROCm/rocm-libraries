
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Serialization/YAML.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/HIPTimer.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "../../test/unit/Utilities.hpp"

#include "include/Parser.hpp"
#include "include/visualize.hpp"

using namespace rocRoller;

const int wavefront_size = 64;

struct GEMMProblem
{
    std::string name;

    // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
    int   M;
    int   N;
    int   K;
    float alpha;
    float beta;

    // Macro Tile Size
    int mac_m;
    int mac_n;
    int mac_k;

    // Number of wave tiles to execute per workgroup
    int workgroup_size_x = 64;
    int workgroup_size_y = 1;

    bool loadLDS_A  = true;
    bool loadLDS_B  = true;
    bool storeLDS_D = true;

    // Unroll Options
    unsigned int unroll_x = 0;
    unsigned int unroll_y = 0;

    // Datatype of inputs and outputs
    std::string type_A;
    std::string type_B;
    std::string type_C;
    std::string type_D;
    std::string type_acc;

    int numWarmUp;
    int numOuter;
    int numInner;

    std::string trans_A; // N or T
    std::string trans_B; // N or T

    std::string scheduler;
    bool        match_memory_access;
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
        iot::mapRequired(io, "client", result.name);
        iot::mapRequired(io, "M", result.M);
        iot::mapRequired(io, "N", result.N);
        iot::mapRequired(io, "K", result.K);
        iot::mapRequired(io, "alpha", result.alpha);
        iot::mapRequired(io, "beta", result.beta);
        iot::mapRequired(io, "trans_A", result.trans_A);
        iot::mapRequired(io, "trans_B", result.trans_B);

        iot::mapRequired(io, "type_A", result.type_A);
        iot::mapRequired(io, "type_B", result.type_B);
        iot::mapRequired(io, "type_C", result.type_C);
        iot::mapRequired(io, "type_D", result.type_D);
        iot::mapRequired(io, "type_acc", result.type_acc);

        iot::mapRequired(io, "mac_m", result.mac_m);
        iot::mapRequired(io, "mac_n", result.mac_n);
        iot::mapRequired(io, "mac_k", result.mac_k);
        iot::mapRequired(io, "workgroup_size_x", result.workgroup_size_x);
        iot::mapRequired(io, "workgroup_size_y", result.workgroup_size_y);
        iot::mapRequired(io, "unroll_x", result.unroll_x);
        iot::mapRequired(io, "unroll_y", result.unroll_y);
        iot::mapRequired(io, "loadLDS_A", result.loadLDS_A);
        iot::mapRequired(io, "loadLDS_B", result.loadLDS_B);
        iot::mapRequired(io, "storeLDS_D", result.storeLDS_D);
        iot::mapRequired(io, "scheduler", result.scheduler);

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

std::string gemmKernelName(GEMMResult const& result, std::shared_ptr<KernelOptions> const& options)
{
    std::ostringstream rv;
    rv << "GEMM_" << result.trans_A << result.trans_B;

    rv << "_";
    for(auto const& t :
        {result.type_A, result.type_B, result.type_C, result.type_D, result.type_acc})
        rv << t.substr(0, 1);

    rv << "_MT";
    streamJoin(rv, std::vector{result.mac_m, result.mac_n, result.mac_k}, "x");

    rv << "_WG";
    streamJoin(rv, std::vector{result.workgroup_size_x, result.workgroup_size_y}, "x");

    rv << "_LDS";
    streamJoin(rv, std::vector{result.loadLDS_A, result.loadLDS_B, result.storeLDS_D}, "");

    rv << "_UNROLL";
    streamJoin(rv, std::vector{result.unroll_x, result.unroll_y}, "x");

    rv << "_" << result.scheduler;

    return rv.str();
}

// D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
template <typename A, typename B, typename C, typename D>
GEMMResult GEMM(GEMMProblem prob, bool checkResult, bool doVisualize)
{
    GEMMResult result(prob);

    AssertFatal(result.M % result.mac_m == 0, "MacroTile size mismatch (M)");
    AssertFatal(result.N % result.mac_n == 0, "MacroTile size mismatch (N)");

    AssertFatal(result.workgroup_size_x % wavefront_size == 0,
                "Workgroup Size X must be multiply of wave front size");

    int wave_m, wave_n, wave_k, wave_b = 0;

    if constexpr(std::is_same_v<A, float> && std::is_same_v<B, float>)
    {
        // wave tile sizes
        wave_m = 32;
        wave_n = 32;
        wave_k = 2;
        wave_b = 1;
    }
    else if constexpr(std::is_same_v<A, Half> && std::is_same_v<B, Half>)
    {
        // wave tile sizes
        wave_m = 32;
        wave_n = 32;
        wave_k = 8;
        wave_b = 1;
    }
    else
    {
        Throw<FatalError>("Unsupported datatype combination in client");
    }

    uint wavetile_per_wavefront_m
        = wavefront_size * result.mac_m / wave_m / result.workgroup_size_x;
    uint wavetile_per_wavefront_n = result.mac_n / wave_n / result.workgroup_size_y;

    AssertFatal(result.mac_m % (wave_m * wavetile_per_wavefront_m) == 0,
                "WaveTile size mismatch (M)",
                ShowValue(result.mac_m),
                ShowValue(wave_m),
                ShowValue(wavetile_per_wavefront_m));
    AssertFatal(result.mac_n % (wave_n * wavetile_per_wavefront_n) == 0,
                "WaveTile size mismatch (N)",
                ShowValue(result.mac_n),
                ShowValue(wave_n),
                ShowValue(wavetile_per_wavefront_n));

    uint workgroup_size_x = result.workgroup_size_x * result.workgroup_size_y;
    uint workgroup_size_y = 1;

    // one macro tile per workgroup
    uint num_workgroup_x = result.M / result.mac_m;
    uint num_workgroup_y = result.N / result.mac_n;

    auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
    auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
    auto NZ = std::make_shared<Expression::Expression>(1u);

    // Host data
    RandomGenerator random(31415u);
    std::vector<A>  h_A = random.vector<A>(result.M * result.K, -1.0, 1.0);
    std::vector<B>  h_B = random.vector<B>(result.K * result.N, -1.0, 1.0);
    std::vector<C>  h_C = random.vector<C>(result.M * result.N, -1.0, 1.0);

    // Device data
    std::shared_ptr<A> d_A = make_shared_device(h_A);
    std::shared_ptr<B> d_B = make_shared_device(h_B);
    std::shared_ptr<C> d_C = make_shared_device(h_C);
    std::shared_ptr<D> d_D = make_shared_device<D>(result.M * result.N, 0.0);

    auto command = std::make_shared<Command>();

    //TODO: Handle transposed matrices more elegantly
    if(result.trans_A == "T")
    {
        command->addOperation(
            std::make_shared<rocRoller::Operations::Operation>(rocRoller::Operations::T_Load_Tiled(
                TypeInfo<A>::Var.dataType, 2, 0, {(size_t)0, (size_t)1}))); // A
    }
    else
    {
        command->addOperation(
            std::make_shared<rocRoller::Operations::Operation>(rocRoller::Operations::T_Load_Tiled(
                TypeInfo<A>::Var.dataType, 2, 0, {(size_t)1}))); // A
    }
    //TODO: Handle transposed matrices more elegantly
    if(result.trans_B == "T")
    {
        command->addOperation(
            std::make_shared<rocRoller::Operations::Operation>(rocRoller::Operations::T_Load_Tiled(
                TypeInfo<B>::Var.dataType, 2, 1, {(size_t)0, (size_t)1}))); // B
    }
    else
    {
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(TypeInfo<B>::Var.dataType,
                                                2,
                                                1,
                                                {
                                                    (size_t)1,
                                                }))); // B
    }
    command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
        rocRoller::Operations::T_Load_Tiled(TypeInfo<C>::Var.dataType, 2, 2, {(size_t)1}))); // C
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
        rocRoller::Operations::T_Store_Tiled(TypeInfo<D>::Var.dataType, 2, 8))); // D

    bool            logArgs = Log::getLogger()->should_log(spdlog::level::debug);
    KernelArguments runtimeArgs(logArgs);

    runtimeArgs.append("A", d_A.get());
    runtimeArgs.append("d_a_limit", (size_t)result.M * result.K);

    runtimeArgs.append("d_a_size_0", (size_t)result.M);
    runtimeArgs.append("d_a_size_1", (size_t)result.K);

    //TODO: Handle transposed matrices more elegantly
    if(result.trans_A == "T")
    {
        runtimeArgs.append("d_a_stride_0", (size_t)result.K);
        runtimeArgs.append("d_a_stride_1", (size_t)1);
    }
    else
    {
        runtimeArgs.append("d_a_stride_0", (size_t)1);
        runtimeArgs.append("d_a_stride_1", (size_t)result.M);
    }

    runtimeArgs.append("B", d_B.get());
    runtimeArgs.append("d_b_limit", (size_t)result.K * result.N);

    runtimeArgs.append("d_b_size_0", (size_t)result.K);
    runtimeArgs.append("d_b_size_1", (size_t)result.N);

    //TODO: Handle transposed matrices more elegantly
    if(result.trans_B == "T")
    {
        runtimeArgs.append("d_b_stride_0", (size_t)result.N);
        runtimeArgs.append("d_b_stride_1", (size_t)1);
    }
    else
    {
        runtimeArgs.append("d_b_stride_0", (size_t)1);
        runtimeArgs.append("d_b_stride_1", (size_t)result.K);
    }

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

    if(logArgs)
        Log::getLogger()->debug(runtimeArgs.toString());

    auto params = std::make_shared<CommandParameters>();
    params->setManualKernelDimension(2);
    // TODO: Calculate these values internally based on workgroup sizes.
    params->setWaveTilesPerWavefront(wavetile_per_wavefront_m, wavetile_per_wavefront_n);

    auto mac_tile_A = KernelGraph::CoordinateGraph::MacroTile({result.mac_m, result.mac_k},
                                                              LayoutType::MATRIX_A,
                                                              {wave_m, wave_n, wave_k, wave_b},
                                                              result.loadLDS_A ? MemoryType::LDS
                                                                               : MemoryType::WAVE);
    auto mac_tile_B = KernelGraph::CoordinateGraph::MacroTile({result.mac_k, result.mac_n},
                                                              LayoutType::MATRIX_B,
                                                              {wave_m, wave_n, wave_k, wave_b},
                                                              result.loadLDS_B ? MemoryType::LDS
                                                                               : MemoryType::WAVE);
    auto mac_tile_C = KernelGraph::CoordinateGraph::MacroTile({result.mac_m, result.mac_n},
                                                              LayoutType::MATRIX_ACCUMULATOR,
                                                              {wave_m, wave_n, wave_k, wave_b});
    auto mac_tile_D = KernelGraph::CoordinateGraph::MacroTile({result.mac_m, result.mac_n},
                                                              LayoutType::MATRIX_ACCUMULATOR,
                                                              {wave_m, wave_n, wave_k, wave_b},
                                                              result.storeLDS_D ? MemoryType::LDS
                                                                                : MemoryType::WAVE);

    params->setDimensionInfo(4, mac_tile_A);
    params->setDimensionInfo(11, mac_tile_B);
    params->setDimensionInfo(18, mac_tile_C);
    params->setDimensionInfo(30, mac_tile_C);
    params->setDimensionInfo(32, mac_tile_C);
    params->setDimensionInfo(34, mac_tile_D);

    params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
    params->setManualWorkitemCount({NX, NY, NZ});

    auto schedulerValue = fromString<Scheduling::SchedulerProcedure>(result.scheduler);
    if(result.scheduler != "")
        Settings::getInstance()->set(Settings::Scheduler, schedulerValue);

    auto postParams = std::make_shared<CommandParameters>();

    auto one         = Expression::literal(1u);
    auto wavefront_n = Expression::literal(
        static_cast<uint>(result.mac_m * result.mac_n / wave_m / wave_n / wavetile_per_wavefront_m
                          / wavetile_per_wavefront_n));
    auto wavefront_nx
        = Expression::literal(static_cast<uint>(result.mac_m / wave_m / wavetile_per_wavefront_m));
    auto wavefront_ny
        = Expression::literal(static_cast<uint>(result.mac_n / wave_n / wavetile_per_wavefront_n));

    auto WF  = KernelGraph::CoordinateGraph::Wavefront(-1, wavefront_n, one);
    auto WFX = KernelGraph::CoordinateGraph::Wavefront(0, wavefront_nx, one);
    auto WFY = KernelGraph::CoordinateGraph::Wavefront(1, wavefront_ny, one);

    std::vector<int> wavefront_ids = {58, 91, 124, 173};
    for(auto id : wavefront_ids)
    {
        postParams->setDimensionInfo(id, WF);
        postParams->setDimensionInfo(id - 2, WFX);
        postParams->setDimensionInfo(id - 1, WFY);
    }

    auto kernelOptions     = std::make_shared<KernelOptions>();
    kernelOptions->unrollX = result.unroll_x;
    kernelOptions->unrollY = result.unroll_y;

    auto kernelName = gemmKernelName(result, kernelOptions);

    if(result.match_memory_access)
    {
        kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_A] = result.trans_A == "T";
        kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_B] = result.trans_B == "T";
    }

    // Build GEMM kernel
    CommandKernel commandKernel(command, kernelName, params, postParams, kernelOptions);

    if(doVisualize)
    {
        Client::visualize(command, commandKernel, runtimeArgs);
    }

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
        std::vector<D> h_D(result.M * result.N, 0.0);
        AssertFatal(
            hipMemcpy(h_D.data(), d_D.get(), result.M * result.N * sizeof(D), hipMemcpyDeviceToHost)
            == (hipError_t)HIP_SUCCESS);

        // Host result
        std::vector<D> h_result(result.M * result.N, 0.0);
        rocRoller::CPUMM(h_result,
                         h_C,
                         h_A,
                         h_B,
                         result.M,
                         result.N,
                         result.K,
                         result.alpha,
                         result.beta,
                         result.trans_A == "T",
                         result.trans_B == "T");

        double rnorm = relativeNorm(h_D, h_result);

        bool isCorrect = rnorm < 3e-5;
        std::cout << "Result: " << (isCorrect ? "Correct" : "Incorrect") << std::endl;
        std::cout << "RNorm: " << rnorm << std::endl;
        if(!isCorrect)
        {
            std::cerr << "WARNING: Result incorrect. RNorm too large: " << rnorm << std::endl;
        }
    }

    double totalTime = 0;
    for(auto ke : result.kernelExecute)
        totalTime += static_cast<double>(ke) / 1.e9;
    double averageTime = totalTime / (result.numInner * result.numOuter);

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

    // Problem definition
    po.addArg("M", Arg({"M"}, "Tensor Size M"));
    po.addArg("N", Arg({"N"}, "Tensor Size N"));
    po.addArg("K", Arg({"K"}, "Tensor Size K"));
    po.addArg("trans_A",
              Arg({"trans_A"}, "N: A is not to be transposed. T: A is to be transposed."));
    po.addArg("trans_B",
              Arg({"trans_B"}, "N: B is not to be transposed. T: B is to be transposed."));
    po.addArg("alpha", Arg({"a", "alpha"}, "Alpha scalar"));
    po.addArg("beta", Arg({"b", "beta"}, "Beta scalar"));
    po.addArg("type_A", Arg({"type_A"}, "Datatype of A Matrix [float | half]"));
    po.addArg("type_B", Arg({"type_B"}, "Datatype of B Matrix [float | half]"));
    po.addArg("type_C", Arg({"type_C"}, "Datatype of C Matrix [float | half]"));
    po.addArg("type_D", Arg({"type_D"}, "Datatype of D Matrix [float | half]"));
    po.addArg("type_acc", Arg({"type_acc"}, "Datatype of accumulation [float]"));

    // Kernel options
    po.addArg("mac_m", Arg({"mac_m"}, "Macro Tile Size M"));
    po.addArg("mac_n", Arg({"mac_n"}, "Macro Tile Size N"));
    po.addArg("mac_k", Arg({"mac_k"}, "Macro Tile Size K"));
    po.addArg("workgroup_size_x", Arg({"workgroup_size_x"}, "Workgroup size in the x dimension"));
    po.addArg("workgroup_size_y", Arg({"workgroup_size_y"}, "Workgroup size in the y dimension"));
    po.addArg("unroll_x", Arg({"unroll_x"}, "Unroll Size in X"));
    po.addArg("unroll_y", Arg({"unroll_y"}, "Unroll Size in Y"));
    po.addArg("loadLDS_A", Arg({"loadLDS_A"}, "Use LDS when loading A Matrix"));
    po.addArg("loadLDS_B", Arg({"loadLDS_B"}, "Use LDS when loading B Matrix"));
    po.addArg("storeLDS_D", Arg({"storeLDS_D"}, "Use LDS when storing D Matrix"));
    po.addArg("scheduler", Arg({"scheduler"}, "Which scheduler to use."));
    po.addArg("match_memory_access",
              Arg({"match_memory_access"},
                  "Match memory access to transpose. "
                  "Currently decreases performance."));

    // Benchmarking options
    po.addArg("yaml", Arg({"o", "yaml"}, "Results"));
    po.addArg("num_warmup", Arg({"num_warmup"}, "Number of warm-up runs."));
    po.addArg("num_outer", Arg({"num_outer"}, "Number of outer runs."));
    po.addArg("num_inner", Arg({"num_inner"}, "Number of inner runs."));
    po.addArg("visualize",
              Arg({"visualize"}, "Dump out volumes describing memory access patterns."));

    po.parse_args(argc, argv);

    GEMMProblem prob;
    prob.name                = "GEMMv00";
    prob.M                   = po.get("M", 3072);
    prob.N                   = po.get("N", 4096);
    prob.K                   = po.get("K", 4096);
    prob.alpha               = po.get("alpha", 2.0f);
    prob.beta                = po.get("beta", 0.5f);
    prob.mac_m               = po.get("mac_m", 64);
    prob.mac_n               = po.get("mac_n", 64);
    prob.mac_k               = po.get("mac_k", 64);
    prob.workgroup_size_x    = po.get("workgroup_size_x", wavefront_size * 2);
    prob.workgroup_size_y    = po.get("workgroup_size_y", 2);
    prob.unroll_x            = po.get("unroll_x", 0);
    prob.unroll_y            = po.get("unroll_y", 0);
    prob.loadLDS_A           = po.get("loadLDS_A", true);
    prob.loadLDS_B           = po.get("loadLDS_B", true);
    prob.storeLDS_D          = po.get("storeLDS_D", true);
    prob.type_A              = po.get("type_A", std::string("float"));
    prob.type_B              = po.get("type_B", std::string("float"));
    prob.type_C              = po.get("type_C", std::string("float"));
    prob.type_D              = po.get("type_D", std::string("float"));
    prob.type_acc            = po.get("type_acc", std::string("float"));
    prob.trans_A             = po.get("trans_A", std::string("N"));
    prob.trans_B             = po.get("trans_B", std::string("N"));
    prob.scheduler           = po.get("scheduler", std::string("Priority"));
    prob.match_memory_access = po.get("match_memory_access", false);

    prob.numWarmUp = po.get("num_warmup", 3);
    prob.numOuter  = po.get("num_outer", 5);
    prob.numInner  = po.get("num_inner", 2);

    bool doVisualize = po.get("visualize", false);

    AssertFatal(prob.type_acc == "float");

    GEMMResult result(prob);

    if(prob.type_A == "float" && prob.type_B == "float" && prob.type_C == "float"
       && prob.type_D == "float")
    {
        result = GEMM<float, float, float, float>(prob, true, doVisualize);
    }
    else if(prob.type_A == "half" && prob.type_B == "half" && prob.type_C == "half"
            && prob.type_D == "half")
    {
        result = GEMM<Half, Half, Half, Half>(prob, true, doVisualize);
    }
    else
    {
        Throw<FatalError>("Unsupported combination of datatypes for GEMM");
    }

    std::string filename = po.get("yaml", std::string());
    if(!filename.empty())
    {
        std::ofstream file(filename);
        Serialization::writeYAML(file, result);
    }

    return 0;
}
