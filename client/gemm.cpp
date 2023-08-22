
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Serialization/YAML.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/HIPTimer.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "../../test/unit/Utilities.hpp"

#include "include/Parser.hpp"
#include "include/visualize.hpp"

using namespace rocRoller;

const int wavefront_size = 64;

struct GEMMProblem
{
    std::string name;

    int device;

    // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
    int   m;
    int   n;
    int   k;
    float alpha;
    float beta;

    // Macro Tile Size
    int macM;
    int macN;
    int macK;

    // Number of wave tiles to execute per workgroup
    int workgroupSizeX = 64;
    int workgroupSizeY = 1;

    bool loadLDSA  = true;
    bool loadLDSB  = true;
    bool storeLDSD = true;

    bool prefetch          = false;
    int  prefetchInFlight  = 2;
    int  prefetchLDSFactor = 0;
    bool betaInFma         = true;

    // Unroll Options
    unsigned int unrollX = 0;
    unsigned int unrollY = 0;

    // Datatype of inputs and outputs
    std::string typeA;
    std::string typeB;
    std::string typeC;
    std::string typeD;
    std::string typeAcc;

    int numWarmUp;
    int numOuter;
    int numInner;

    std::string transA; // N or T
    std::string transB; // N or T

    std::string scheduler;
    bool        matchMemoryAccess;
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
    bool                checked = false;
    bool                correct = true;
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
        iot::mapRequired(io, "device", result.device);
        iot::mapRequired(io, "M", result.m);
        iot::mapRequired(io, "N", result.n);
        iot::mapRequired(io, "K", result.k);
        iot::mapRequired(io, "alpha", result.alpha);
        iot::mapRequired(io, "beta", result.beta);
        iot::mapRequired(io, "trans_A", result.transA);
        iot::mapRequired(io, "trans_B", result.transB);

        iot::mapRequired(io, "type_A", result.typeA);
        iot::mapRequired(io, "type_B", result.typeB);
        iot::mapRequired(io, "type_C", result.typeC);
        iot::mapRequired(io, "type_D", result.typeD);
        iot::mapRequired(io, "type_acc", result.typeAcc);

        iot::mapRequired(io, "mac_m", result.macM);
        iot::mapRequired(io, "mac_n", result.macN);
        iot::mapRequired(io, "mac_k", result.macK);
        iot::mapRequired(io, "workgroup_size_x", result.workgroupSizeX);
        iot::mapRequired(io, "workgroup_size_y", result.workgroupSizeY);
        iot::mapRequired(io, "unroll_x", result.unrollX);
        iot::mapRequired(io, "unroll_y", result.unrollY);
        iot::mapRequired(io, "loadLDS_A", result.loadLDSA);
        iot::mapRequired(io, "loadLDS_B", result.loadLDSB);
        iot::mapRequired(io, "storeLDS_D", result.storeLDSD);
        iot::mapRequired(io, "prefetch", result.prefetch);
        iot::mapRequired(io, "prefetchInFlight", result.prefetchInFlight);
        iot::mapRequired(io, "prefetchLDSFactor", result.prefetchLDSFactor);
        iot::mapRequired(io, "betaInFma", result.betaInFma);
        iot::mapRequired(io, "scheduler", result.scheduler);

        iot::mapRequired(io, "numWarmUp", result.numWarmUp);
        iot::mapRequired(io, "numOuter", result.numOuter);
        iot::mapRequired(io, "numInner", result.numInner);

        iot::mapRequired(io, "kernelGenerate", result.kernelGenerate);
        iot::mapRequired(io, "kernelAssemble", result.kernelAssemble);
        iot::mapRequired(io, "kernelExecute", result.kernelExecute);

        iot::mapRequired(io, "checked", result.checked);
        iot::mapRequired(io, "correct", result.correct);
    }

    static void mapping(IO& io, GEMMResult& arch, EmptyContext& ctx)
    {
        mapping(io, arch);
    }
};

std::string gemmKernelName(GEMMResult const& result, std::shared_ptr<KernelOptions> const& options)
{
    std::ostringstream rv;
    rv << "GEMM_" << result.transA << result.transB;

    rv << "_";
    for(auto const& t : {result.typeA, result.typeB, result.typeC, result.typeD, result.typeAcc})
        rv << t.substr(0, 1);

    rv << "_MT";
    streamJoin(rv, std::vector{result.macM, result.macN, result.macK}, "x");

    rv << "_WG";
    streamJoin(rv, std::vector{result.workgroupSizeX, result.workgroupSizeY}, "x");

    rv << "_LDS";
    streamJoin(rv, std::vector{result.loadLDSA, result.loadLDSB, result.storeLDSD}, "");

    rv << "_UNROLL";
    streamJoin(rv, std::vector{result.unrollX, result.unrollY}, "x");

    if(result.prefetch)
    {
        rv << "_PF";
        streamJoin(rv, std::vector{result.prefetchInFlight, result.prefetchLDSFactor}, "x");
    }

    rv << "_" << result.scheduler;

    return rv.str();
}

// D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
template <typename A, typename B, typename C, typename D>
GEMMResult GEMM(GEMMProblem prob, bool checkResult, bool doVisualize)
{
    GEMMResult result(prob);

    AssertFatal(result.m % result.macM == 0, "MacroTile size mismatch (M)");
    AssertFatal(result.n % result.macN == 0, "MacroTile size mismatch (N)");

    AssertFatal(result.workgroupSizeX % wavefront_size == 0,
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

    uint wavetile_per_wavefront_m = wavefront_size * result.macM / wave_m / result.workgroupSizeX;
    uint wavetile_per_wavefront_n = result.macN / wave_n / result.workgroupSizeY;

    AssertFatal(result.macM % (wave_m * wavetile_per_wavefront_m) == 0,
                "WaveTile size mismatch (M)",
                ShowValue(result.macM),
                ShowValue(wave_m),
                ShowValue(wavetile_per_wavefront_m));
    AssertFatal(result.macN % (wave_n * wavetile_per_wavefront_n) == 0,
                "WaveTile size mismatch (N)",
                ShowValue(result.macN),
                ShowValue(wave_n),
                ShowValue(wavetile_per_wavefront_n));

    uint workgroup_size_x = result.workgroupSizeX * result.workgroupSizeY;
    uint workgroup_size_y = 1;

    // one macro tile per workgroup
    uint num_workgroup_x = result.m / result.macM;
    uint num_workgroup_y = result.n / result.macN;

    auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
    auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
    auto NZ = std::make_shared<Expression::Expression>(1u);

    // Host data
    RandomGenerator random(31415u);
    std::vector<A>  h_A = random.vector<A>(result.m * result.k, -1.0, 1.0);
    std::vector<B>  h_B = random.vector<B>(result.k * result.n, -1.0, 1.0);
    std::vector<C>  h_C = random.vector<C>(result.m * result.n, -1.0, 1.0);

    std::shared_ptr<A> d_A = make_shared_device(h_A);
    std::shared_ptr<B> d_B = make_shared_device(h_B);
    std::shared_ptr<C> d_C = make_shared_device(h_C);
    std::shared_ptr<D> d_D = make_shared_device<D>(result.m * result.n, 0.0);

    auto command = std::make_shared<Command>();

    //TODO: Handle transposed matrices more elegantly
    if(result.transA == "T")
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
    if(result.transB == "T")
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
        rocRoller::Operations::E_Mul(6, 4, 2))); // beta * C
    execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
        rocRoller::Operations::E_Mul(7, 3, 5))); // alpha * (A * B)
    if(result.betaInFma)
    {
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(8, 6, 7))); // beta * C + alpha * (A * B)
    }
    else
    {
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(8, 7, 6))); // alpha * (A * B) + beta * C
    }
    command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

    command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
        rocRoller::Operations::T_Store_Tiled(TypeInfo<D>::Var.dataType, 2, 8))); // D

    bool            logArgs = Log::getLogger()->should_log(spdlog::level::debug);
    KernelArguments runtimeArgs(logArgs);

    runtimeArgs.append("A", d_A.get());
    runtimeArgs.append("d_a_limit", (size_t)result.m * result.k);

    runtimeArgs.append("d_a_size_0", (size_t)result.m);
    runtimeArgs.append("d_a_size_1", (size_t)result.k);

    //TODO: Handle transposed matrices more elegantly
    if(result.transA == "T")
    {
        runtimeArgs.append("d_a_stride_0", (size_t)result.k);
        runtimeArgs.append("d_a_stride_1", (size_t)1);
    }
    else
    {
        runtimeArgs.append("d_a_stride_0", (size_t)1);
        runtimeArgs.append("d_a_stride_1", (size_t)result.m);
    }

    runtimeArgs.append("B", d_B.get());
    runtimeArgs.append("d_b_limit", (size_t)result.k * result.n);

    runtimeArgs.append("d_b_size_0", (size_t)result.k);
    runtimeArgs.append("d_b_size_1", (size_t)result.n);

    //TODO: Handle transposed matrices more elegantly
    if(result.transB == "T")
    {
        runtimeArgs.append("d_b_stride_0", (size_t)result.n);
        runtimeArgs.append("d_b_stride_1", (size_t)1);
    }
    else
    {
        runtimeArgs.append("d_b_stride_0", (size_t)1);
        runtimeArgs.append("d_b_stride_1", (size_t)result.k);
    }

    runtimeArgs.append("C", d_C.get());
    runtimeArgs.append("d_c_limit", (size_t)result.m * result.n);
    runtimeArgs.append("d_c_size_0", (size_t)result.m);
    runtimeArgs.append("d_c_size_1", (size_t)result.n);
    runtimeArgs.append("d_c_stride_0", (size_t)1);
    runtimeArgs.append("d_c_stride_1", (size_t)result.m);

    runtimeArgs.append("alpha", result.alpha);

    runtimeArgs.append("beta", result.beta);

    runtimeArgs.append("D", d_D.get());
    runtimeArgs.append("d_d_limit", (size_t)result.m * result.n);
    runtimeArgs.append("d_d_stride_0", (size_t)1);
    runtimeArgs.append("d_d_stride_1", (size_t)result.m);

    if(logArgs)
        Log::getLogger()->debug(runtimeArgs.toString());

    auto params = std::make_shared<CommandParameters>();
    params->setManualKernelDimension(2);
    // TODO: Calculate these values internally based on workgroup sizes.
    params->setWaveTilesPerWavefront(wavetile_per_wavefront_m, wavetile_per_wavefront_n);

    auto mac_tile_A = KernelGraph::CoordinateGraph::MacroTile({result.macM, result.macK},
                                                              LayoutType::MATRIX_A,
                                                              {wave_m, wave_n, wave_k, wave_b},
                                                              result.loadLDSA ? MemoryType::LDS
                                                                              : MemoryType::WAVE);
    auto mac_tile_B = KernelGraph::CoordinateGraph::MacroTile({result.macK, result.macN},
                                                              LayoutType::MATRIX_B,
                                                              {wave_m, wave_n, wave_k, wave_b},
                                                              result.loadLDSB ? MemoryType::LDS
                                                                              : MemoryType::WAVE);
    auto mac_tile_C = KernelGraph::CoordinateGraph::MacroTile({result.macM, result.macN},
                                                              LayoutType::MATRIX_ACCUMULATOR,
                                                              {wave_m, wave_n, wave_k, wave_b});
    auto mac_tile_D = KernelGraph::CoordinateGraph::MacroTile({result.macM, result.macN},
                                                              LayoutType::MATRIX_ACCUMULATOR,
                                                              {wave_m, wave_n, wave_k, wave_b},
                                                              result.storeLDSD ? MemoryType::LDS
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
    postParams->setManualWavefrontCount(
        {static_cast<uint>(result.macM / wave_m / wavetile_per_wavefront_m),
         static_cast<uint>(result.macN / wave_n / wavetile_per_wavefront_n)});

    auto kernelOptions     = std::make_shared<KernelOptions>();
    kernelOptions->unrollX = result.unrollX;
    kernelOptions->unrollY = result.unrollY;

    if(prob.prefetch)
    {
        kernelOptions->unrollK           = 2;
        kernelOptions->prefetch          = true;
        kernelOptions->prefetchInFlight  = prob.prefetchInFlight;
        kernelOptions->prefetchLDSFactor = prob.prefetchLDSFactor;

        if(prob.prefetchLDSFactor != 0)
        {
            kernelOptions->prefetchMixMemOps = true;
        }
    }

    if(result.matchMemoryAccess)
    {
        kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_A] = result.transA == "T";
        kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_B] = result.transB == "T";
    }

    kernelOptions->setNextFreeVGPRToMax = false;

    auto kernelName = gemmKernelName(result, kernelOptions);

    // Build GEMM kernel
    CommandKernel commandKernel(command, kernelName, params, postParams, kernelOptions);

    // TODO: Make this optional.
    {
        std::ofstream outfile(kernelName + ".yaml");
        outfile << toYAML(commandKernel.getKernelGraph());
    }

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
        std::vector<D> h_D(result.m * result.n, 0.0);
        AssertFatal(
            hipMemcpy(h_D.data(), d_D.get(), result.m * result.n * sizeof(D), hipMemcpyDeviceToHost)
            == (hipError_t)HIP_SUCCESS);

        // Host result
        std::vector<D> h_result(result.m * result.n, 0.0);
        rocRoller::CPUMM(h_result,
                         h_C,
                         h_A,
                         h_B,
                         result.m,
                         result.n,
                         result.k,
                         result.alpha,
                         result.beta,
                         result.transA == "T",
                         result.transB == "T");

        double rnorm = relativeNorm(h_D, h_result);

        bool isCorrect = rnorm < 3e-5;
        std::cout << "Result: " << (isCorrect ? "Correct" : "Incorrect") << std::endl;
        std::cout << "RNorm: " << rnorm << std::endl;
        result.checked = true;
        if(!isCorrect)
        {
            std::cerr << "WARNING: Result incorrect. RNorm too large: " << rnorm << std::endl;
            result.correct = false;
        }
        else
        {
            result.correct = true;
        }
    }

    double totalTime = 0;
    for(auto ke : result.kernelExecute)
        totalTime += static_cast<double>(ke) / 1.e9;
    double averageTime = totalTime / (result.numInner * result.numOuter);

    std::cout << "Average runtime (s): " << averageTime << std::endl;
    std::cout << "Average GFLOPS: "
              << (double)result.m * result.n * result.k * 2.0 / averageTime * 1.e-9 << std::endl;

    result.kernelAssemble = TimerPool::nanoseconds("CommandKernel::assembleKernel");
    result.kernelGenerate = TimerPool::nanoseconds("CommandKernel::generateKernel");

    return result;
}

int main(int argc, const char* argv[])
{
    ParseOptions po("GEMM Driver: D (MxN) = alpha * A (MxK) * B (KxN) + beta * C (MxN)",
                    Settings::getInstance()->help());

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
    po.addArg("betaInFma", Arg({"betaInFma"}, "Use beta in fma instruction instead of alpha."));
    po.addArg("scheduler", Arg({"scheduler"}, "Which scheduler to use."));
    po.addArg("match_memory_access",
              Arg({"match_memory_access"},
                  "Match memory access to transpose. "
                  "Currently decreases performance."));
    po.addArg("prefetch", Arg({"prefetch"}, "Enable prefetching (UnrollK=2 implied)."));
    po.addArg("prefetchInFlight",
              Arg({"prefetchInFlight"}, "Number of prefetches in flight at the same time"));
    po.addArg("prefetchLDSFactor",
              Arg({"prefetchLDSFactor"}, "Prefetch 1/prefetchLDSFactor of MacroTile from LDS"));

    // Benchmarking options
    po.addArg("yaml", Arg({"o", "yaml"}, "Results"));
    po.addArg("num_warmup", Arg({"num_warmup"}, "Number of warm-up runs."));
    po.addArg("num_outer", Arg({"num_outer"}, "Number of outer runs."));
    po.addArg("num_inner", Arg({"num_inner"}, "Number of inner runs."));
    po.addArg("visualize",
              Arg({"visualize"}, "Dump out volumes describing memory access patterns."));

    po.addArg("device", Arg({"device"}, "GPU Device Ordinal"));

    po.parse_args(argc, argv);

    GEMMProblem prob;
    prob.name              = "GEMMv00";
    prob.m                 = po.get("M", 3072);
    prob.n                 = po.get("N", 4096);
    prob.k                 = po.get("K", 4096);
    prob.alpha             = po.get("alpha", 2.0f);
    prob.beta              = po.get("beta", 0.5f);
    prob.macM              = po.get("mac_m", 64);
    prob.macN              = po.get("mac_n", 64);
    prob.macK              = po.get("mac_k", 64);
    prob.workgroupSizeX    = po.get("workgroup_size_x", wavefront_size * 2);
    prob.workgroupSizeY    = po.get("workgroup_size_y", 2);
    prob.unrollX           = po.get("unroll_x", 0);
    prob.unrollY           = po.get("unroll_y", 0);
    prob.loadLDSA          = po.get("loadLDS_A", true);
    prob.loadLDSB          = po.get("loadLDS_B", true);
    prob.storeLDSD         = po.get("storeLDS_D", true);
    prob.betaInFma         = po.get("betaInFma", true);
    prob.typeA             = po.get("type_A", std::string("float"));
    prob.typeB             = po.get("type_B", std::string("float"));
    prob.typeC             = po.get("type_C", std::string("float"));
    prob.typeD             = po.get("type_D", std::string("float"));
    prob.typeAcc           = po.get("type_acc", std::string("float"));
    prob.transA            = po.get("trans_A", std::string("N"));
    prob.transB            = po.get("trans_B", std::string("N"));
    prob.scheduler         = po.get("scheduler", std::string("Priority"));
    prob.matchMemoryAccess = po.get("match_memory_access", true);
    prob.prefetch          = po.get("prefetch", false);
    prob.prefetchInFlight  = po.get("prefetchInFlight", 0);
    prob.prefetchLDSFactor = po.get("prefetchLDSFactor", 0);

    prob.numWarmUp = po.get("num_warmup", 3);
    prob.numOuter  = po.get("num_outer", 5);
    prob.numInner  = po.get("num_inner", 2);
    prob.device    = po.get("device", 0);

    HIP_CHECK(hipSetDevice(prob.device));

    bool doVisualize = po.get("visualize", false);

    AssertFatal(prob.typeAcc == "float");

    GEMMResult result(prob);

    if(prob.typeA == "float" && prob.typeB == "float" && prob.typeC == "float"
       && prob.typeD == "float")
    {
        result = GEMM<float, float, float, float>(prob, true, doVisualize);
    }
    else if(prob.typeA == "half" && prob.typeB == "half" && prob.typeC == "half"
            && prob.typeD == "half")
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
