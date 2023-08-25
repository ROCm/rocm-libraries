
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

#include "include/GEMMParameters.hpp"
#include "include/Parser.hpp"
#include "include/visualize.hpp"

using namespace rocRoller;

const int         wavefrontSize = 64;
const std::string clientName    = "GEMMv00";

template <typename IO>
struct rocRoller::Serialization::
    MappingTraits<GEMMClient::Result, IO, rocRoller::Serialization::EmptyContext>
{
    static const bool flow = false;
    using iot              = IOTraits<IO>;

    static void mapping(IO& io, GEMMClient::Result& result)
    {
        iot::mapRequired(io, "client", clientName);
        iot::mapRequired(io, "device", result.runParams.device);
        iot::mapRequired(io, "M", result.solutionParams.problemParams.m);
        iot::mapRequired(io, "N", result.solutionParams.problemParams.n);
        iot::mapRequired(io, "K", result.solutionParams.problemParams.k);
        iot::mapRequired(io, "alpha", result.solutionParams.problemParams.alpha);
        iot::mapRequired(io, "beta", result.solutionParams.problemParams.beta);
        iot::mapRequired(
            io, "trans_A", GEMMClient::toString(result.solutionParams.problemParams.transA));
        iot::mapRequired(
            io, "trans_B", GEMMClient::toString(result.solutionParams.problemParams.transB));

        iot::mapRequired(io, "type_A", result.solutionParams.problemParams.typeA);
        iot::mapRequired(io, "type_B", result.solutionParams.problemParams.typeB);
        iot::mapRequired(io, "type_C", result.solutionParams.problemParams.typeC);
        iot::mapRequired(io, "type_D", result.solutionParams.problemParams.typeD);
        iot::mapRequired(io, "type_acc", result.solutionParams.problemParams.typeAcc);

        iot::mapRequired(io, "mac_m", result.solutionParams.macM);
        iot::mapRequired(io, "mac_n", result.solutionParams.macN);
        iot::mapRequired(io, "mac_k", result.solutionParams.macK);
        iot::mapRequired(io, "workgroup_size_x", result.solutionParams.workgroupSizeX);
        iot::mapRequired(io, "workgroup_size_y", result.solutionParams.workgroupSizeY);
        iot::mapRequired(io, "unroll_x", result.solutionParams.unrollX);
        iot::mapRequired(io, "unroll_y", result.solutionParams.unrollY);
        iot::mapRequired(io, "loadLDS_A", result.solutionParams.loadLDSA);
        iot::mapRequired(io, "loadLDS_B", result.solutionParams.loadLDSB);
        iot::mapRequired(io, "storeLDS_D", result.solutionParams.storeLDSD);
        iot::mapRequired(io, "prefetch", result.solutionParams.prefetch);
        iot::mapRequired(io, "prefetchInFlight", result.solutionParams.prefetchInFlight);
        iot::mapRequired(io, "prefetchLDSFactor", result.solutionParams.prefetchLDSFactor);
        iot::mapRequired(io, "betaInFma", result.solutionParams.betaInFma);
        iot::mapRequired(io, "scheduler", result.solutionParams.scheduler);

        iot::mapRequired(io, "numWarmUp", result.runParams.numWarmUp);
        iot::mapRequired(io, "numOuter", result.runParams.numOuter);
        iot::mapRequired(io, "numInner", result.runParams.numInner);

        iot::mapRequired(io, "kernelGenerate", result.kernelGenerate);
        iot::mapRequired(io, "kernelAssemble", result.kernelAssemble);
        iot::mapRequired(io, "kernelExecute", result.kernelExecute);

        iot::mapRequired(io, "checked", result.checked);
        iot::mapRequired(io, "correct", result.correct);
    }

    static void mapping(IO& io, GEMMClient::Result& result, EmptyContext& ctx)
    {
        mapping(io, result);
    }
};

// D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
template <typename A, typename B, typename C, typename D>
GEMMClient::Result GEMM(GEMMClient::SolutionParameters const& solutionParams,
                        GEMMClient::RunParameters const&      runParams,
                        bool                                  checkResult,
                        bool                                  doVisualize)
{
    AssertFatal(solutionParams.problemParams.m % solutionParams.macM == 0,
                "MacroTile size mismatch (M)");
    AssertFatal(solutionParams.problemParams.n % solutionParams.macN == 0,
                "MacroTile size mismatch (N)");

    AssertFatal(solutionParams.workgroupSizeX % wavefrontSize == 0,
                "Workgroup Size X must be multiply of wave front size");

    GEMMClient::Result result;
    result.solutionParams = solutionParams;
    result.runParams      = runParams;

    int wave_m = 0, wave_n = 0, wave_k = 0, wave_b = 0;

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
        = wavefrontSize * solutionParams.macM / wave_m / solutionParams.workgroupSizeX;
    uint wavetile_per_wavefront_n = solutionParams.macN / wave_n / solutionParams.workgroupSizeY;

    AssertFatal(solutionParams.macM % (wave_m * wavetile_per_wavefront_m) == 0,
                "WaveTile size mismatch (M)",
                ShowValue(solutionParams.macM),
                ShowValue(wave_m),
                ShowValue(wavetile_per_wavefront_m));
    AssertFatal(solutionParams.macN % (wave_n * wavetile_per_wavefront_n) == 0,
                "WaveTile size mismatch (N)",
                ShowValue(solutionParams.macN),
                ShowValue(wave_n),
                ShowValue(wavetile_per_wavefront_n));

    uint workgroup_size_x = solutionParams.workgroupSizeX * solutionParams.workgroupSizeY;
    uint workgroup_size_y = 1;

    // one macro tile per workgroup
    uint num_workgroup_x = solutionParams.problemParams.m / solutionParams.macM;
    uint num_workgroup_y = solutionParams.problemParams.n / solutionParams.macN;

    auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
    auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
    auto NZ = std::make_shared<Expression::Expression>(1u);

    // Host data
    RandomGenerator random(31415u);
    std::vector<A>  h_A = random.vector<A>(
        solutionParams.problemParams.m * solutionParams.problemParams.k, -1.0, 1.0);
    std::vector<B> h_B = random.vector<B>(
        solutionParams.problemParams.k * solutionParams.problemParams.n, -1.0, 1.0);
    std::vector<C> h_C = random.vector<C>(
        solutionParams.problemParams.m * solutionParams.problemParams.n, -1.0, 1.0);

    std::shared_ptr<A> d_A = make_shared_device(h_A);
    std::shared_ptr<B> d_B = make_shared_device(h_B);
    std::shared_ptr<C> d_C = make_shared_device(h_C);
    std::shared_ptr<D> d_D = make_shared_device<D>(
        solutionParams.problemParams.m * solutionParams.problemParams.n, 0.0);

    auto command = std::make_shared<Command>();

    //TODO: Handle transposed matrices more elegantly
    switch(solutionParams.problemParams.transA)
    {
    case GEMMClient::TransposeType::T:
        command->addOperation(
            std::make_shared<rocRoller::Operations::Operation>(rocRoller::Operations::T_Load_Tiled(
                TypeInfo<A>::Var.dataType, 2, 0, {(size_t)0, (size_t)1}))); // AT
        break;
    case GEMMClient::TransposeType::N:
        command->addOperation(
            std::make_shared<rocRoller::Operations::Operation>(rocRoller::Operations::T_Load_Tiled(
                TypeInfo<A>::Var.dataType, 2, 0, {(size_t)1}))); // AN
        break;
    default:
        Throw<FatalError>("Bad transpose option");
    }

    //TODO: Handle transposed matrices more elegantly
    switch(solutionParams.problemParams.transB)
    {
    case GEMMClient::TransposeType::T:
        command->addOperation(
            std::make_shared<rocRoller::Operations::Operation>(rocRoller::Operations::T_Load_Tiled(
                TypeInfo<B>::Var.dataType, 2, 1, {(size_t)0, (size_t)1}))); // BT
        break;
    case GEMMClient::TransposeType::N:
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(TypeInfo<B>::Var.dataType,
                                                2,
                                                1,
                                                {
                                                    (size_t)1,
                                                }))); // BN
        break;
    default:
        Throw<FatalError>("Bad transpose option");
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
    if(solutionParams.betaInFma)
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
    runtimeArgs.append("d_a_limit",
                       (size_t)solutionParams.problemParams.m * solutionParams.problemParams.k);

    runtimeArgs.append("d_a_size_0", (size_t)solutionParams.problemParams.m);
    runtimeArgs.append("d_a_size_1", (size_t)solutionParams.problemParams.k);

    //TODO: Handle transposed matrices more elegantly
    if(solutionParams.problemParams.transA == GEMMClient::TransposeType::T)
    {
        runtimeArgs.append("d_a_stride_0", (size_t)solutionParams.problemParams.k);
        runtimeArgs.append("d_a_stride_1", (size_t)1);
    }
    else
    {
        runtimeArgs.append("d_a_stride_0", (size_t)1);
        runtimeArgs.append("d_a_stride_1", (size_t)solutionParams.problemParams.m);
    }

    runtimeArgs.append("B", d_B.get());
    runtimeArgs.append("d_b_limit",
                       (size_t)solutionParams.problemParams.k * solutionParams.problemParams.n);

    runtimeArgs.append("d_b_size_0", (size_t)solutionParams.problemParams.k);
    runtimeArgs.append("d_b_size_1", (size_t)solutionParams.problemParams.n);

    //TODO: Handle transposed matrices more elegantly
    if(solutionParams.problemParams.transB == GEMMClient::TransposeType::T)
    {
        runtimeArgs.append("d_b_stride_0", (size_t)solutionParams.problemParams.n);
        runtimeArgs.append("d_b_stride_1", (size_t)1);
    }
    else
    {
        runtimeArgs.append("d_b_stride_0", (size_t)1);
        runtimeArgs.append("d_b_stride_1", (size_t)solutionParams.problemParams.k);
    }

    runtimeArgs.append("C", d_C.get());
    runtimeArgs.append("d_c_limit",
                       (size_t)solutionParams.problemParams.m * solutionParams.problemParams.n);
    runtimeArgs.append("d_c_size_0", (size_t)solutionParams.problemParams.m);
    runtimeArgs.append("d_c_size_1", (size_t)solutionParams.problemParams.n);
    runtimeArgs.append("d_c_stride_0", (size_t)1);
    runtimeArgs.append("d_c_stride_1", (size_t)solutionParams.problemParams.m);

    runtimeArgs.append("alpha", solutionParams.problemParams.alpha);

    runtimeArgs.append("beta", solutionParams.problemParams.beta);

    runtimeArgs.append("D", d_D.get());
    runtimeArgs.append("d_d_limit",
                       (size_t)solutionParams.problemParams.m * solutionParams.problemParams.n);
    runtimeArgs.append("d_d_stride_0", (size_t)1);
    runtimeArgs.append("d_d_stride_1", (size_t)solutionParams.problemParams.m);

    if(logArgs)
        Log::getLogger()->debug(runtimeArgs.toString());

    auto params = std::make_shared<CommandParameters>();
    params->setManualKernelDimension(2);
    // TODO: Calculate these values internally based on workgroup sizes.
    params->setWaveTilesPerWavefront(wavetile_per_wavefront_m, wavetile_per_wavefront_n);

    auto mac_tile_A = KernelGraph::CoordinateGraph::MacroTile(
        {solutionParams.macM, solutionParams.macK},
        LayoutType::MATRIX_A,
        {wave_m, wave_n, wave_k, wave_b},
        solutionParams.loadLDSA ? MemoryType::LDS : MemoryType::WAVE);
    auto mac_tile_B = KernelGraph::CoordinateGraph::MacroTile(
        {solutionParams.macK, solutionParams.macN},
        LayoutType::MATRIX_B,
        {wave_m, wave_n, wave_k, wave_b},
        solutionParams.loadLDSB ? MemoryType::LDS : MemoryType::WAVE);
    auto mac_tile_C
        = KernelGraph::CoordinateGraph::MacroTile({solutionParams.macM, solutionParams.macN},
                                                  LayoutType::MATRIX_ACCUMULATOR,
                                                  {wave_m, wave_n, wave_k, wave_b});
    auto mac_tile_D = KernelGraph::CoordinateGraph::MacroTile(
        {solutionParams.macM, solutionParams.macN},
        LayoutType::MATRIX_ACCUMULATOR,
        {wave_m, wave_n, wave_k, wave_b},
        solutionParams.storeLDSD ? MemoryType::LDS : MemoryType::WAVE);

    params->setDimensionInfo(4, mac_tile_A);
    params->setDimensionInfo(11, mac_tile_B);
    params->setDimensionInfo(18, mac_tile_C);
    params->setDimensionInfo(30, mac_tile_C);
    params->setDimensionInfo(32, mac_tile_C);
    params->setDimensionInfo(34, mac_tile_D);

    params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
    params->setManualWorkitemCount({NX, NY, NZ});

    if(solutionParams.scheduler != "")
    {
        auto schedulerValue = fromString<Scheduling::SchedulerProcedure>(solutionParams.scheduler);
        Settings::getInstance()->set(Settings::Scheduler, schedulerValue);
    }

    auto postParams = std::make_shared<CommandParameters>();
    postParams->setManualWavefrontCount(
        {static_cast<uint>(solutionParams.macM / wave_m / wavetile_per_wavefront_m),
         static_cast<uint>(solutionParams.macN / wave_n / wavetile_per_wavefront_n)});

    auto kernelOptions     = std::make_shared<KernelOptions>();
    kernelOptions->unrollX = solutionParams.unrollX;
    kernelOptions->unrollY = solutionParams.unrollY;

    if(solutionParams.prefetch)
    {
        kernelOptions->unrollK           = 2;
        kernelOptions->prefetch          = true;
        kernelOptions->prefetchInFlight  = solutionParams.prefetchInFlight;
        kernelOptions->prefetchLDSFactor = solutionParams.prefetchLDSFactor;

        if(solutionParams.prefetchLDSFactor != 0)
        {
            kernelOptions->prefetchMixMemOps = true;
        }
    }
    else
    {
        kernelOptions->prefetch = false;
    }

    if(solutionParams.matchMemoryAccess)
    {
        kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_A]
            = solutionParams.problemParams.transA == GEMMClient::TransposeType::T;
        kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_B]
            = solutionParams.problemParams.transB == GEMMClient::TransposeType::T;
    }

    kernelOptions->setNextFreeVGPRToMax = false;

    auto kernelName = result.solutionParams.generateKernelName();

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
    for(int i = 0; i < runParams.numWarmUp; ++i)
    {
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());
    }

    // Benchmark runs
    for(int outer = 0; outer < runParams.numOuter; ++outer)
    {
        HIP_TIMER(t_kernel, "GEMM");
        HIP_TIC(t_kernel);
        for(int inner = 0; inner < runParams.numInner; ++inner)
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
        std::vector<D> h_D(solutionParams.problemParams.m * solutionParams.problemParams.n, 0.0);
        AssertFatal(
            hipMemcpy(h_D.data(),
                      d_D.get(),
                      solutionParams.problemParams.m * solutionParams.problemParams.n * sizeof(D),
                      hipMemcpyDeviceToHost)
            == (hipError_t)HIP_SUCCESS);

        // Host result
        std::vector<D> h_result(solutionParams.problemParams.m * solutionParams.problemParams.n,
                                0.0);
        rocRoller::CPUMM(h_result,
                         h_C,
                         h_A,
                         h_B,
                         solutionParams.problemParams.m,
                         solutionParams.problemParams.n,
                         solutionParams.problemParams.k,
                         solutionParams.problemParams.alpha,
                         solutionParams.problemParams.beta,
                         solutionParams.problemParams.transA == GEMMClient::TransposeType::T,
                         solutionParams.problemParams.transB == GEMMClient::TransposeType::T);

        double rnorm = relativeNorm(h_D, h_result);

        bool isCorrect = rnorm < 3e-5;
        std::cout << "Result: " << (isCorrect ? "Correct" : "Incorrect") << std::endl;
        std::cout << "RNorm: " << rnorm << std::endl;
        result.checked = true;
        result.correct = isCorrect;
        if(!isCorrect)
        {
            std::cerr << "WARNING: Result incorrect. RNorm too large: " << rnorm << std::endl;
        }
    }

    double totalTime = 0;
    for(auto ke : result.kernelExecute)
        totalTime += static_cast<double>(ke) / 1.e9;
    double averageTime = totalTime / (runParams.numInner * runParams.numOuter);

    std::cout << "Average runtime (s): " << averageTime << std::endl;
    std::cout << "Average GFLOPS: "
              << (double)solutionParams.problemParams.m * solutionParams.problemParams.n
                     * solutionParams.problemParams.k * 2.0 / averageTime * 1.e-9
              << std::endl;

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

    GEMMClient::ProblemParameters problem;
    problem.m       = po.get("M", 3072);
    problem.n       = po.get("N", 4096);
    problem.k       = po.get("K", 4096);
    problem.alpha   = po.get("alpha", 2.0f);
    problem.beta    = po.get("beta", 0.5f);
    problem.typeA   = po.get("type_A", std::string("float"));
    problem.typeB   = po.get("type_B", std::string("float"));
    problem.typeC   = po.get("type_C", std::string("float"));
    problem.typeD   = po.get("type_D", std::string("float"));
    problem.typeAcc = po.get("type_acc", std::string("float"));
    problem.transA  = fromString<GEMMClient::TransposeType>(po.get("trans_A", std::string("N")));
    problem.transB  = fromString<GEMMClient::TransposeType>(po.get("trans_B", std::string("N")));

    GEMMClient::SolutionParameters solution;
    solution.problemParams = problem;

    solution.macM              = po.get("mac_m", 64);
    solution.macN              = po.get("mac_n", 64);
    solution.macK              = po.get("mac_k", 64);
    solution.workgroupSizeX    = po.get("workgroup_size_x", wavefrontSize * 2);
    solution.workgroupSizeY    = po.get("workgroup_size_y", 2);
    solution.unrollX           = po.get("unroll_x", 0);
    solution.unrollY           = po.get("unroll_y", 0);
    solution.loadLDSA          = po.get("loadLDS_A", true);
    solution.loadLDSB          = po.get("loadLDS_B", true);
    solution.storeLDSD         = po.get("storeLDS_D", true);
    solution.betaInFma         = po.get("betaInFma", true);
    solution.scheduler         = po.get("scheduler", std::string("Priority"));
    solution.matchMemoryAccess = po.get("match_memory_access", true);
    solution.prefetch          = po.get("prefetch", false);
    solution.prefetchInFlight  = po.get("prefetchInFlight", 0);
    solution.prefetchLDSFactor = po.get("prefetchLDSFactor", 0);

    GEMMClient::RunParameters runParams;
    runParams.numWarmUp = po.get("num_warmup", 3);
    runParams.numOuter  = po.get("num_outer", 5);
    runParams.numInner  = po.get("num_inner", 2);
    runParams.device    = po.get("device", 0);

    bool doVisualize = po.get("visualize", false);
    bool checkResult = true;

    std::string filename = po.get("yaml", std::string());

    // Currently, we only support F32 accumulation
    AssertFatal(problem.typeAcc == "float");

    HIP_CHECK(hipSetDevice(runParams.device));

    GEMMClient::Result result;

    if(problem.typeA == "float" && problem.typeB == "float" && problem.typeC == "float"
       && problem.typeD == "float")
    {
        result = GEMM<float, float, float, float>(solution, runParams, checkResult, doVisualize);
    }
    else if(problem.typeA == "half" && problem.typeB == "half" && problem.typeC == "half"
            && problem.typeD == "half")
    {
        result = GEMM<Half, Half, Half, Half>(solution, runParams, checkResult, doVisualize);
    }
    else
    {
        Throw<FatalError>("Unsupported combination of datatypes for GEMM");
    }

    if(!filename.empty())
    {
        std::ofstream file(filename);
        Serialization::writeYAML(file, result);
    }

    return 0;
}
