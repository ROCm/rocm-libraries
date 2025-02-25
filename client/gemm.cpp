
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Serialization/YAML.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>

#include <common/Utilities.hpp>

#include "include/DataParallelGEMMSolution.hpp"
#include "include/GEMMParameters.hpp"
#include "include/Parser.hpp"
#include "include/StreamKGEMMSolution.hpp"
#include "include/TensileGEMMSolution.hpp"

using namespace rocRoller;

const std::string clientName = "GEMMv00";

template <typename IO>
struct rocRoller::Serialization::
    MappingTraits<Client::GEMMClient::Result, IO, rocRoller::Serialization::EmptyContext>
{
    static const bool flow = false;
    using iot              = IOTraits<IO>;

    static void mapping(IO& io, Client::GEMMClient::Result& result)
    {
        iot::mapRequired(io, "client", clientName);
        iot::mapRequired(io, "device", result.benchmarkResults.runParams.device);
        iot::mapRequired(io, "M", result.solutionParams.problemParams.m);
        iot::mapRequired(io, "N", result.solutionParams.problemParams.n);
        iot::mapRequired(io, "K", result.solutionParams.problemParams.k);
        iot::mapRequired(io, "alpha", result.solutionParams.problemParams.alpha);
        iot::mapRequired(io, "beta", result.solutionParams.problemParams.beta);
        iot::mapRequired(io,
                         "trans_A",
                         Client::GEMMClient::toString(result.solutionParams.problemParams.transA));
        iot::mapRequired(io,
                         "trans_B",
                         Client::GEMMClient::toString(result.solutionParams.problemParams.transB));

        iot::mapRequired(io, "type_A", result.solutionParams.problemParams.typeA);
        iot::mapRequired(io, "type_B", result.solutionParams.problemParams.typeB);
        iot::mapRequired(io, "type_C", result.solutionParams.problemParams.typeC);
        iot::mapRequired(io, "type_D", result.solutionParams.problemParams.typeD);
        iot::mapRequired(io, "type_acc", result.solutionParams.problemParams.typeAcc);

        iot::mapRequired(io, "mac_m", result.solutionParams.macM);
        iot::mapRequired(io, "mac_n", result.solutionParams.macN);
        iot::mapRequired(io, "mac_k", result.solutionParams.macK);
        iot::mapRequired(io, "wave_m", result.solutionParams.waveM);
        iot::mapRequired(io, "wave_n", result.solutionParams.waveN);
        iot::mapRequired(io, "wave_k", result.solutionParams.waveK);
        iot::mapRequired(io, "wave_b", result.solutionParams.waveB);
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

        iot::mapRequired(io, "streamK", result.solutionParams.streamK);
        iot::mapRequired(io, "numWGs", result.solutionParams.numWGs);
        iot::mapRequired(io, "streamKTwoTile", result.solutionParams.streamKTwoTile);

        iot::mapRequired(io, "numWarmUp", result.benchmarkResults.runParams.numWarmUp);
        iot::mapRequired(io, "numOuter", result.benchmarkResults.runParams.numOuter);
        iot::mapRequired(io, "numInner", result.benchmarkResults.runParams.numInner);

        iot::mapRequired(io, "kernelGenerate", result.benchmarkResults.kernelGenerate);
        iot::mapRequired(io, "kernelAssemble", result.benchmarkResults.kernelAssemble);
        iot::mapRequired(io, "kernelExecute", result.benchmarkResults.kernelExecute);

        iot::mapRequired(io, "checked", result.benchmarkResults.checked);
        iot::mapRequired(io, "correct", result.benchmarkResults.correct);
        iot::mapRequired(io, "rnorm", result.benchmarkResults.rnorm);
    }

    static void mapping(IO& io, Client::GEMMClient::Result& result, EmptyContext& ctx)
    {
        mapping(io, result);
    }
};

// D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
template <typename A, typename B, typename C, typename D>
Client::GEMMClient::Result GEMM(Client::GEMMClient::SolutionParameters const& solutionParams,
                                Client::RunParameters const&                  runParams,
                                bool                                          checkResult,
                                bool                                          doVisualize)
{
    // Host Data
    RandomGenerator random(31415u);
    auto            h_A = random.vector<A>(
        solutionParams.problemParams.m * solutionParams.problemParams.k, -1.0, 1.0);
    auto h_B = random.vector<B>(
        solutionParams.problemParams.k * solutionParams.problemParams.n, -1.0, 1.0);
    std::vector<C> h_C = random.vector<C>(
        solutionParams.problemParams.m * solutionParams.problemParams.n, -1.0, 1.0);
    std::vector<D> h_D(solutionParams.problemParams.m * solutionParams.problemParams.n, 0.0);

    using unsegTypeA = typename UnsegmentedTypeOf<A>::type;
    using unsegTypeB = typename UnsegmentedTypeOf<B>::type;

    if(solutionParams.scheduler == "TENSILE_ASM")
    {
        Client::GEMMClient::Result result;
        auto                       versionString = GPUArchitectureLibrary::getInstance()
                                 ->GetDefaultHipDeviceArch()
                                 .target()
                                 .getVersionString();
        if(versionString == "gfx90a")
        {
            Client::GEMMClient::TensileGEMMSolution<A, B, C, D> gemmKernel(solutionParams);
            result = gemmKernel.benchmark(runParams, checkResult, doVisualize, h_A, h_B, h_C, h_D);
        }
        else
        {
            std::cout << "Not running TENSILE_ASM for " << versionString << std::endl;
            result.solutionParams             = solutionParams;
            result.benchmarkResults.runParams = runParams;
        }

        return result;
    }
    else if(solutionParams.streamK)
    {
        Client::GEMMClient::Result result;
        auto defaultDevice = GPUArchitectureLibrary::getInstance()->GetDefaultHipDeviceArch();
        if(defaultDevice.HasCapability(GPUCapability::ArchAccUnifiedRegs))
        {
            Client::GEMMClient::StreamKGEMMSolution<A, B, C, D> gemmKernel(solutionParams);
            result = gemmKernel.benchmark(runParams, checkResult, doVisualize, h_A, h_B, h_C, h_D);
        }
        else
        {
            std::cout << "Not running StreamK for " << defaultDevice.target().getVersionString()
                      << std::endl;
            result.solutionParams             = solutionParams;
            result.benchmarkResults.runParams = runParams;
        }

        return result;
    }
    else
    {
        Client::GEMMClient::DataParallelGEMMSolution<A, B, C, D> gemmKernel(solutionParams);

        // TODO: Make this optional.
        {
            auto          kernelName = solutionParams.generateKernelName();
            std::ofstream outfile(kernelName + ".yaml");
            outfile << toYAML(gemmKernel.getKernel()->getKernelGraph());
        }

        auto result = gemmKernel.benchmark(runParams, checkResult, doVisualize, h_A, h_B, h_C, h_D);

        return result;
    }
}

// D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
template <typename A, typename C, typename D>
Client::GEMMClient::Result GEMMMixed(Client::GEMMClient::SolutionParameters const& solutionParams,
                                     Client::RunParameters const&                  runParams,
                                     bool                                          checkResult,
                                     bool                                          doVisualize,
                                     auto                                          typeB)
{
    if(typeB == "fp8")
    {
        return GEMM<A, FP8, C, D>(solutionParams, runParams, checkResult, doVisualize);
    }
    else if(typeB == "bf8")
    {
        return GEMM<A, BF8, C, D>(solutionParams, runParams, checkResult, doVisualize);
    }
    else if(typeB == "fp6")
    {
        return GEMM<A, FP6, C, D>(solutionParams, runParams, checkResult, doVisualize);
    }
    else if(typeB == "bf6")
    {
        return GEMM<A, BF6, C, D>(solutionParams, runParams, checkResult, doVisualize);
    }
    else if(typeB == "fp4")
    {
        return GEMM<A, FP4, C, D>(solutionParams, runParams, checkResult, doVisualize);
    }
    else
        Throw<FatalError>("Invalid type for Mixed GEMM.");
}

// D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
template <typename C, typename D>
Client::GEMMClient::Result GEMMMixed(Client::GEMMClient::SolutionParameters const& solutionParams,
                                     Client::RunParameters const&                  runParams,
                                     bool                                          checkResult,
                                     bool                                          doVisualize,
                                     auto                                          typeA,
                                     auto                                          typeB)
{
    if(typeA == "fp8")
    {
        return GEMMMixed<FP8, C, D>(solutionParams, runParams, checkResult, doVisualize, typeB);
    }
    else if(typeA == "bf8")
    {
        return GEMMMixed<BF8, C, D>(solutionParams, runParams, checkResult, doVisualize, typeB);
    }
    else if(typeA == "fp6")
    {
        return GEMMMixed<FP6, C, D>(solutionParams, runParams, checkResult, doVisualize, typeB);
    }
    else if(typeA == "bf6")
    {
        return GEMMMixed<BF6, C, D>(solutionParams, runParams, checkResult, doVisualize, typeB);
    }
    else if(typeA == "fp4")
    {
        return GEMMMixed<FP4, C, D>(solutionParams, runParams, checkResult, doVisualize, typeB);
    }
    else
        Throw<FatalError>("Invalid type for Mixed GEMM.");
}

int main(int argc, const char* argv[])
{
    ParseOptions po("GEMM Driver: D (MxN) = alpha * A (MxK) * B (KxN) + beta * C (MxN)",
                    Settings::getInstance()->help());

    // Problem definition
    po.addArg("M", Arg({"M"}, "Tensor size M."));
    po.addArg("N", Arg({"N"}, "Tensor size N."));
    po.addArg("K", Arg({"K"}, "Tensor size K."));
    po.addArg("trans_A",
              Arg({"trans_A"}, "N: A is not to be transposed.  T: A is to be transposed."));
    po.addArg("trans_B",
              Arg({"trans_B"}, "N: B is not to be transposed.  T: B is to be transposed."));
    po.addArg("alpha", Arg({"a", "alpha"}, "Alpha scalar."));
    po.addArg("beta", Arg({"b", "beta"}, "Beta scalar."));
    po.addArg(
        "type_A",
        Arg({"type_A"},
            "Datatype of A matrix [float | half | fp8 | bf8 | fp6 | bf6 | fp4].  Default: float."));
    po.addArg(
        "type_B",
        Arg({"type_B"},
            "Datatype of B matrix [float | half | fp8 | bf8 | fp6 | bf6 | fp4].  Default: float."));
    po.addArg("type_C", Arg({"type_C"}, "Datatype of C matrix [float | half].  Default: float."));
    po.addArg("type_D", Arg({"type_D"}, "Datatype of D matrix [float | half].  Default: float."));
    po.addArg("type_acc", Arg({"type_acc"}, "Datatype of accumulation [float]"));
    po.addArg("hgemm",
              Arg({"hgemm"},
                  "Enable (=1) to overwrite types to: --type_A=half --type_B=half --type_C=half "
                  "--type_D=half "
                  "--type_acc=float."));
    // Kernel options
    po.addArg("mac_m", Arg({"mac_m"}, "(Macro) Tile size M."));
    po.addArg("mac_n", Arg({"mac_n"}, "(Macro) Tile size N."));
    po.addArg("mac_k", Arg({"mac_k"}, "(Macro) Tile size K."));

    po.addArg("wave_m", Arg({"wave_m"}, "(MFMA) Tile size M."));
    po.addArg("wave_n", Arg({"wave_n"}, "(MFMA) Tile size N."));
    po.addArg("wave_k", Arg({"wave_k"}, "(MFMA) Tile size K."));
    po.addArg("wave_b", Arg({"wave_b"}, "(MFMA) Tile size K."));
    po.addArg("mfma",
              Arg({"mfma"},
                  "MFMA instruction to use.  Default 32x32x2x1 for floats, 32x32x8x1 for halfs."));
    po.addArg("workgroup_size_x", Arg({"workgroup_size_x"}, "Workgroup size in the x dimension."));
    po.addArg("workgroup_size_y", Arg({"workgroup_size_y"}, "Workgroup size in the y dimension."));
    po.addArg("unroll_x", Arg({"unroll_x"}, "Unroll size in X."));
    po.addArg("unroll_y", Arg({"unroll_y"}, "Unroll size in Y."));
    po.addArg("loadLDS_A", Arg({"loadLDS_A"}, "Use LDS when loading A."));
    po.addArg("loadLDS_B", Arg({"loadLDS_B"}, "Use LDS when loading B."));
    po.addArg("storeLDS_D", Arg({"storeLDS_D"}, "Use LDS when storing D."));
    po.addArg("betaInFma", Arg({"betaInFma"}, "Use beta in FMA instruction instead of alpha."));
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
    po.addArg("streamK", Arg({"streamK"}, "Enable StreamK algorithm."));
    po.addArg("numWGs", Arg({"numWGs"}, "Number of workgroups to use with StreamK algorithm."));
    po.addArg("streamKTwoTile", Arg({"streamKTwoTile"}, "Enable two-tile StreamK algorithm."));

    // Benchmarking options
    po.addArg("yaml", Arg({"o", "yaml"}, "Results"));
    po.addArg("num_warmup", Arg({"num_warmup"}, "Number of warm-up runs."));
    po.addArg("num_outer", Arg({"num_outer"}, "Number of outer runs."));
    po.addArg("num_inner", Arg({"num_inner"}, "Number of inner runs."));
    po.addArg("visualize",
              Arg({"visualize"}, "Dump out volumes describing memory access patterns."));
    po.addArg("save", Arg({"save"}, "Save assembly to file name."));

    po.addArg("device", Arg({"device"}, "GPU Device Ordinal"));

    po.parse_args(argc, argv);

    Client::GEMMClient::ProblemParameters problem;
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
    problem.transA
        = fromString<Client::GEMMClient::TransposeType>(po.get("trans_A", std::string("N")));
    problem.transB
        = fromString<Client::GEMMClient::TransposeType>(po.get("trans_B", std::string("N")));

    bool forceHGEMM = po.get("hgemm", false);
    if(forceHGEMM)
    {
        problem.typeA   = "half";
        problem.typeB   = "half";
        problem.typeC   = "half";
        problem.typeD   = "half";
        problem.typeAcc = "float";
    }

    Client::GEMMClient::SolutionParameters solution;
    solution.problemParams = problem;

    solution.macM              = po.get("mac_m", 64);
    solution.macN              = po.get("mac_n", 64);
    solution.macK              = po.get("mac_k", 64);
    solution.workgroupSizeX    = po.get("workgroup_size_x", Client::GEMMClient::wavefrontSize * 2);
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
    solution.streamK           = po.get("streamK", false);
    int defaultNumWGs          = 0;
    if(solution.streamK)
    {
        hipDeviceProp_t deviceProperties;
        AssertFatal(hipGetDeviceProperties(&deviceProperties, 0) == (hipError_t)HIP_SUCCESS);
        defaultNumWGs = deviceProperties.multiProcessorCount;
    }
    solution.numWGs = po.get("numWGs", defaultNumWGs);
    if(solution.numWGs == 0)
    {
        solution.numWGs = defaultNumWGs;
    }
    solution.streamKTwoTile = po.get("streamKTwoTile", false);
    AssertFatal(!solution.streamKTwoTile || solution.streamK);

    std::string mfma = po.get("mfma", std::string());
    if(!mfma.empty())
    {
        bool fail = false;
        try
        {
            std::istringstream iss(mfma);
            std::string        token;

            iss.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);
            std::getline(iss, token, 'x');
            solution.waveM = std::stoi(token);
            std::getline(iss, token, 'x');
            solution.waveN = std::stoi(token);
            std::getline(iss, token, 'x');
            solution.waveK = std::stoi(token);
            iss.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            std::getline(iss, token, 'x');
            solution.waveB = std::stoi(token);
        }
        catch(const std::invalid_argument&)
        {
            fail = true;
        }
        catch(const std::ios_base::failure&)
        {
            fail = true;
        }
        fail |= (solution.waveM < 1) || (solution.waveN < 1) || (solution.waveK < 1)
                || (solution.waveB < 1);
        if(fail)
        {
            std::cout << "Invalid format for MFMA instruction." << std::endl;
            std::cout << std::endl;
            std::cout << "The MFMA argument should be formatted like:" << std::endl;
            std::cout << std::endl;
            std::cout << "    --mfma=MxNxKxB" << std::endl;
            std::cout << std::endl;
            std::cout << "For example: --mfma=32x32x2x1" << std::endl;
            exit(1);
        }
    }
    solution.waveM = po.get("wave_m", solution.waveM);
    solution.waveN = po.get("wave_n", solution.waveN);
    solution.waveK = po.get("wave_k", solution.waveK);
    solution.waveB = po.get("wave_b", solution.waveB);

    Client::RunParameters runParams;
    runParams.numWarmUp = po.get("num_warmup", 3);
    runParams.numOuter  = po.get("num_outer", 5);
    runParams.numInner  = po.get("num_inner", 2);
    runParams.device    = po.get("device", 0);

    bool doVisualize = po.get("visualize", false);
    bool checkResult = true;

    std::string assembly = po.get("save", std::string());
    if(!assembly.empty())
    {
        Settings::getInstance()->set(Settings::SaveAssembly, true);
        Settings::getInstance()->set(Settings::AssemblyFile, assembly);
    }

    std::string filename = po.get("yaml", std::string());

    // Currently, we only support F32 accumulation
    AssertFatal(problem.typeAcc == "float");

    HIP_CHECK(hipSetDevice(runParams.device));

    Client::GEMMClient::Result result;

    auto isF8F6F4 = [](auto dtype) {
        return (dtype == "fp8" || dtype == "bf8" || dtype == "fp6" || dtype == "bf6"
                || dtype == "fp4");
    };

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
    else if(problem.typeA == "fp8" && problem.typeB == "fp8" && problem.typeC == "float"
            && problem.typeD == "float")
    {
        result = GEMM<FP8, FP8, float, float>(solution, runParams, checkResult, doVisualize);
    }
    else if(problem.typeA == "bf8" && problem.typeB == "bf8" && problem.typeC == "float"
            && problem.typeD == "float")
    {
        result = GEMM<BF8, BF8, float, float>(solution, runParams, checkResult, doVisualize);
    }
    else if(problem.typeA == "fp6" && problem.typeB == "fp6" && problem.typeC == "float"
            && problem.typeD == "float")
    {
        result = GEMM<FP6, FP6, float, float>(solution, runParams, checkResult, doVisualize);
    }
    else if(problem.typeA == "bf6" && problem.typeB == "bf6" && problem.typeC == "float"
            && problem.typeD == "float")
    {
        result = GEMM<BF6, BF6, float, float>(solution, runParams, checkResult, doVisualize);
    }
    else if(problem.typeA == "fp4" && problem.typeB == "fp4" && problem.typeC == "float"
            && problem.typeD == "float")
    {
        result = GEMM<FP4, FP4, float, float>(solution, runParams, checkResult, doVisualize);
    }
    else if((problem.typeA != problem.typeB) && isF8F6F4(problem.typeA) && isF8F6F4(problem.typeB))
    {
        result = GEMMMixed<float, float>(
            solution, runParams, checkResult, doVisualize, problem.typeA, problem.typeB);
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

    if(!result.benchmarkResults.correct)
        return 1;

    return 0;
}
