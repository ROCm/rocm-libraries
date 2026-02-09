/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

#include <Tensile/ArithmeticUnitTypes.hpp>
#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include "BenchmarkTimer.hpp"
#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "HardwareMonitorListener.hpp"
#include "MetaRunListener.hpp"
#include "ProgressListener.hpp"
#include "ReferenceValidator.hpp"
#include "SolutionIterator.hpp"
#include "TimingEvents.hpp"

#include "LibraryUpdateReporter.hpp"
#include "LogReporter.hpp"
#include "MetaResultReporter.hpp"
#include "PerformanceReporter.hpp"
#include "ResultFileReporter.hpp"
#include "ResultReporter.hpp"

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/program_options.hpp>

#include <cstddef>

namespace po = boost::program_options;

namespace Tensile
{
    namespace Client
    {

        template <typename T>
        po::typed_value<T>* value_default(std::string const& desc)
        {
            return po::value<T>()->default_value(T(), desc);
        }

        template <typename T>
        po::typed_value<T>* value_default()
        {
            return po::value<T>()->default_value(T());
        }

        template <typename T>
        po::typed_value<std::vector<T>>* vector_default_empty()
        {
            return value_default<std::vector<T>>("[]");
        }

        po::options_description all_options()
        {
            po::options_description options("Tensile client options");

            // clang-format off
            options.add_options()
                ("help,h", "Show help message.")

                ("config-file",              vector_default_empty<std::string>(), "INI config file(s) to read.")

                ("library-file,l",           po::value<std::string>(), "Load a (YAML) solution library.  If not specified, we will use "
                                                                       "the embedded library, if available.")
                ("code-object,c",            vector_default_empty<std::string>(), "Code object file with kernel(s).  If none are "
                                                                                  "specified, we will use the embedded code "
                                                                                  "object(s) if available.")

                ("performance-metric",       po::value<PerformanceMetric>()->default_value(PerformanceMetric::DeviceEfficiency), "Metric for benchmarking results")

                ("problem-identifier",       po::value<std::string>(), "Problem identifer (Einstein notation). Either "
                                                                       "this or free/batch/bound must be specified.")
                ("free",                     value_default<ContractionProblem::FreeIndices>("[]"),  "Free index. Order: a,b,ca,cb,da,db")
                ("batch",                    value_default<ContractionProblem::BatchIndices>("[]"), "Batch index. Order: a,b,c,d")
                ("bound",                    value_default<ContractionProblem::BoundIndices>("[]"), "Bound/summation index. Order: a,b")

                ("type",                     po::value<DataType>()->default_value(DataType::None), "Data type")
                ("a-type",                   po::value<DataType>()->default_value(DataType::None), "A data type")
                ("b-type",                   po::value<DataType>()->default_value(DataType::None), "B data type")
                ("c-type",                   po::value<DataType>()->default_value(DataType::None), "C data type")
                ("d-type",                   po::value<DataType>()->default_value(DataType::None), "D data type")
                ("alpha-type",               po::value<DataType>()->default_value(DataType::None), "alpha data type")
                ("beta-type",                po::value<DataType>()->default_value(DataType::None), "beta data type")
                ("high-precision-accumulate", po::value<bool>()->default_value(false), "Use high-precision accumulate.")
                ("stochastic-rounding",      po::value<bool>()->default_value(false), "Use stochastic rounding.")
                ("f32-xdl-math-op",          po::value<DataType>()->default_value(DataType::None), "Use xf32 compute for float input and output matrices.")
                ("strided-batched",          po::value<bool>()->default_value(true), "Use strided-batched or general batched")
                ("kernel-language",          po::value<KernelLanguage>()->default_value(KernelLanguage::Any), "Select kernel language.")
                ("deterministic-mode",       po::value<bool>()->default_value(false), "Enforce deterministic summation patterns"
                                                                                      "by not splitting U among workgroups")
                ("arithmetic-unit",          po::value<ArithmeticUnit>()->default_value(ArithmeticUnit::Any), "Select arithmetic unit.")

                ("init-a",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for A")
                ("init-b",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for B")
                ("init-c",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for C")
                ("init-d",                   po::value<InitMode>()->default_value(InitMode::Zero), "Initialization for D")
                ("init-alpha",               po::value<InitMode>()->default_value(InitMode::Two), "Initialization for alpha")
                ("init-beta",                po::value<InitMode>()->default_value(InitMode::Two), "Initialization for beta")
                ("pristine-on-gpu",          po::value<bool>()->default_value(true), "Keep a pristine copy of inputs on GPU for performance")
                ("c-equal-d",                po::value<bool>()->default_value(false), "C equals D")
                ("offset-a",                 po::value<size_t>()->default_value(0), "buffer a start offset")
                ("offset-b",                 po::value<size_t>()->default_value(0), "buffer b start offset")
                ("offset-c",                 po::value<size_t>()->default_value(0), "buffer c start offset")
                ("offset-d",                 po::value<size_t>()->default_value(0), "buffer d start offset")
                ("print-valids",             po::value<bool>()->default_value(false), "Print values that pass validation")
                ("print-max",                po::value<int>()->default_value(-1), "Max number of values to print")
                ("num-elements-to-validate", po::value<int>()->default_value(0), "Number of elements to validate")
                ("bounds-check",             po::value<BoundsCheckMode>()->default_value(BoundsCheckMode::Disable),
                "1:Use sentinel values to check memory boundaries."
                "2:Memory bound check by front guard page"
                "3:Memory bound check by back guard page"
                "4:Memory bound check by both side guard page")

                ("print-tensor-a",           po::value<bool>()->default_value(false), "Print tensor A.")
                ("print-tensor-b",           po::value<bool>()->default_value(false), "Print tensor B.")
                ("print-tensor-c",           po::value<bool>()->default_value(false), "Print tensor C.")
                ("print-tensor-d",           po::value<bool>()->default_value(false), "Print tensor D.")
                ("print-tensor-ref",         po::value<bool>()->default_value(false), "Print reference tensor D.")

                ("dump-tensors",             po::value<bool>()->default_value(false), "Binary dump tensors instead of printing.")

                ("convolution-identifier",   po::value<std::string>(), "Convolution problem identifer:  ConvolutionType_ActFormat_FilterFormat_Filter_Stride_Dilation_Groups.  Example: ConvolutionBackwardWeights_NCHW_filter:3x3_stride:1x1_dilation:1x1_groups:1.  Batch count, spacial dimensions (H,W,D), Cin and Cout filters are determined by the problem dimensions.")
                ("convolution-vs-contraction",  po::value<bool>()->default_value(false), "Compare reference convolution against contraction.")

                ("device-idx",               po::value<int>()->default_value(0), "Device index")
                ("use-default-stream",       po::value<bool>()->default_value(false), "Use default Hip stream to run kernels.")
                ("platform-idx",             po::value<int>()->default_value(0), "OpenCL Platform Index")

                ("num-warmups",              po::value<int>()->default_value(0), "Number of warmups to run")
                ("sync-after-warmups",       po::value<bool>()->default_value(true), "Synchronize GPU after warmup kernel runs")
                ("num-benchmarks",           po::value<int>()->default_value(1), "Number of benchmarks to run")
                ("num-enqueues-per-sync",    po::value<int>()->default_value(1), "Enqueues per sync, will affect by min-flops-per-sync")
                ("num-syncs-per-benchmark",  po::value<int>()->default_value(1), "Syncs per benchmark")
                ("min-flops-per-sync",       po::value<size_t>()->default_value(0), "Minimum number of flops per sync to increase stability for small problems.")
                ("use-gpu-timer",            po::value<bool>()->default_value(true), "Use GPU timer")
                ("sleep-percent",            po::value<int>()->default_value(0), "Sleep percentage")
                ("hardware-monitor",         po::value<bool>()->default_value(true), "Use hardware monitor.")
                ("flush-count",              po::value<size_t>()->default_value(1), "Number of copies of arrays to allocate for cache flushing in timing code."
                                                                                    " Functions are called iters times in a timing loop."
                                                                                    " If the problem memory footprint is small enough, then arrays will be cached."
                                                                                    " flush_count can be used to prevent caching."
                                                                                    " For example, for sgemm with transA=transB=N:"
                                                                                    " problem_memory_footprint = (m*k + k*n + m*n) * sizeof(float)."
                                                                                    " To flush arrays before reusing set:"
                                                                                    " flush_count >= 1 + cache_size / problem_memory_footprint"
                                                                                    " Note that in the calculation of flush_count any padding from leading"
                                                                                    " dimensions are not loaded to cache and not included in the problem_memory_footprint."
                                                                                    " If you specify flush_count you cannot also specify flush_memory_size")
                ("flush-memory-size",      po::value<size_t>()->default_value(0),   "Used only in timing code for cache flushing. Set to greater than"
                                                                                    " cache size, so that arrays are flushed from cache before they are reused. When the size of arrays (the problem_memory_footprint)"
                                                                                    " is smaller than flush_memory_size, then flush_count copies of arrays are allocated where:"
                                                                                    " flush_count = flush_memory_size / problem_memory_footprint."
                                                                                    " For sgemm with transA=transB=N"
                                                                                    " problem_memory_footprint = (m*k + k*n + m*n) * sizeof(float). Note that any padding from leading"
                                                                                    " dimensions are not loaded to cache and not included in the problem_memory_footprint."
                                                                                    " If you specify flush_memory_size you cannot also specify flush_count."
                                                                                    " Also note that Tensile allocates enough memory once at setup to accommodate"
                                                                                    " the largest problem. Similarly, the largest problem will be used to calculate flush_count."
                                                                                    " Configs with largely contrasting sizes may not guarantee cache eviction for the smaller problems")

                ("perf-l2-read-hits",        po::value<double>()->default_value(0.0), "L2 read hits")
                ("perf-l2-write-hits",       po::value<double>()->default_value(0.5), "L2 write hits")
                ("perf-l2-read-bw-mul",      po::value<double>()->default_value(2.0), "L2 read bandwidth multiplier")
                ("perf-read-efficiency",     po::value<double>()->default_value(0.85), "Read efficiency")
                ("perf-ops-per-cycle",       po::value<int>()->default_value(64), "Ops per cycle")
                ("csv-export-extra-cols",    po::value<bool>()->default_value(false), "CSV exports winner information")
                ("csv-merge-same-problems",  po::value<bool>()->default_value(false), "CSV merge rows of same problem id")

                ("problem-size,p",           vector_default_empty<std::string>(), "Specify a problem size.  Comma-separated list of "
                                                                                  "sizes, in the order of the Einstein notation.")

                ("a-strides",                vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("b-strides",                vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("c-strides",                vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("d-strides",                vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("convolution-problem",      vector_default_empty<std::string>(), "Specify a Convolution problem size. Comma-separated list of sizes:"
                                                                                  "Spatial(w,h,d),Filter(x,y,z),Stride(v,u,#),"
                                                                                  "Dilation(j,l,^),Pad start(q,p,$),Pad end(q_,p_,$_)")

                ("a-zero-pads",                vector_default_empty<std::string>(), "Comma-separated tuple(s) of anchor dim,"
                                                                                  "summation dim, leading pad, trailing pad."
                                                                                  "Each tuple must be separated with a semi-colon.")

                ("b-zero-pads",                vector_default_empty<std::string>(), "Comma-separated tuple(s) of anchor dim,"
                                                                                  "summation dim, leading pad, trailing pad."
                                                                                  "Each tuple must be separated with a semi-colon.")

                ("a-ops",                    vector_default_empty<TensorOp>(), "Operations applied to A.")
                ("b-ops",                    vector_default_empty<TensorOp>(), "Operations applied to B.")
                ("c-ops",                    vector_default_empty<TensorOp>(), "Operations applied to C.")
                ("d-ops",                    vector_default_empty<TensorOp>(), "Operations applied to D.")

                ("problem-start-idx",        po::value<int>()->default_value(0),  "First problem to run")
                ("num-problems",             po::value<int>()->default_value(-1), "Number of problems to run")

                ("solution-start-idx",       po::value<int>()->default_value(-1), "First solution to run")
                ("num-solutions",            po::value<int>()->default_value(-1), "Number of solutions to run")
                ("best-solution",            po::value<bool>()->default_value(false), "Best solution benchmark mode")

                ("results-file",             po::value<std::string>()->default_value("results.csv"), "File name to write results.")
                ("log-file",                 po::value<std::string>(),                               "File name for output log.")
                ("log-file-append",          po::value<bool>()->default_value(false),                "Append to log file.")
                ("log-level",                po::value<LogLevel>()->default_value(LogLevel::Debug),  "Log level")

                ("library-update-file",      po::value<std::string>()->default_value(""), "File name for writing indices "
                                                                                          "and speeds suitable for updating "
                                                                                          "an existing library logic file.")
                ("library-update-comment",   po::value<bool>()->default_value(false), "Include solution name as a "
                                                                                      "comment in library update "
                                                                                      "file.")


                ("exit-on-error",            po::value<bool>()->default_value(false), "Exit run early on failed kernels or other errors.")
                ("selection-only",           po::value<bool>()->default_value(false), "Don't run any solutions, only print kernel selections.")
                ("max-workspace-size",       po::value<size_t>()->default_value(32*1024*1024), "Max workspace for training")
                ("granularity-threshold",    po::value<double>()->default_value(0.0), "Don't run a solution if total granularity is below")
                ;
            // clang-format on

            return options;
        }

        std::shared_ptr<Hardware> GetHardware(po::variables_map const& args)
        {
            int deviceCount = 0;
            HIP_CHECK_EXC(hipGetDeviceCount(&deviceCount));

            int deviceIdx = args["device-idx"].as<int>();

            if(deviceIdx >= deviceCount)
                throw std::runtime_error(concatenate(
                    "Invalid device index ", deviceIdx, " (", deviceCount, " total found.)"));

            HIP_CHECK_EXC(hipSetDevice(deviceIdx));

            return hip::GetCurrentDevice();
        }

        hipStream_t GetStream(po::variables_map const& args)
        {
            if(args["use-default-stream"].as<bool>())
                return 0;

            hipStream_t stream;
            HIP_CHECK_EXC(hipStreamCreate(&stream));
            return stream;
        }

        std::shared_ptr<MasterSolutionLibrary<ContractionProblem>>
            LoadSolutionLibrary(po::variables_map const& args)
        {
            auto filename = args["library-file"];
            if(!filename.empty())
            {
                return std::dynamic_pointer_cast<MasterSolutionLibrary<ContractionProblem>>(
                    LoadLibraryFile<ContractionProblem>(filename.as<std::string>()));
            }

            auto embeddedLibrary
                = std::dynamic_pointer_cast<MasterSolutionLibrary<ContractionProblem>>(
                    EmbeddedLibrary<ContractionProblem>::Get());

            if(embeddedLibrary != nullptr)
                return embeddedLibrary;

            throw std::runtime_error("Client must be linked with an embedded library or "
                                     "a library must be specified at runtime.");
        }

        void LoadCodeObjects(po::variables_map const& args, hip::SolutionAdapter& adapter)
        {
            auto const& filenames = args["code-object"].as<std::vector<std::string>>();
            auto        logLevel  = args["log-level"].as<LogLevel>();

            if(filenames.empty())
            {
                adapter.loadEmbeddedCodeObjects();
            }
            else
            {
                //only trigger exception when failed to load all code objects.
                bool       loaded   = false;
                hipError_t retError = hipSuccess;

                for(auto const& filename : filenames)
                {
                    hipError_t ret;

                    if(logLevel >= LogLevel::Verbose)
                        std::cout << "Loading " << filename << std::endl;
                    ret = adapter.loadCodeObjectFile(filename);

                    if(ret == hipSuccess)
                        loaded = true;
                    else
                        retError = ret;
                }

                if(!loaded)
                    HIP_CHECK_EXC(retError);
            }
        }

        std::vector<size_t> split_ints(std::string const& value)
        {
            std::vector<std::string> parts;
            boost::split(parts, value, boost::algorithm::is_any_of(",;"));

            std::vector<size_t> rv;
            rv.reserve(parts.size());

            for(auto const& part : parts)
                if(part != "")
                    rv.push_back(boost::lexical_cast<size_t>(part));

            return rv;
        }

        void parse_arg_ints(po::variables_map& args, std::string const& name)
        {
            auto inValue = args[name].as<std::vector<std::string>>();

            std::vector<std::vector<size_t>> outValue;
            outValue.reserve(inValue.size());
            for(auto const& str : inValue)
                outValue.push_back(split_ints(str));

            boost::any v(outValue);

            args.at(name).value() = v;
        }

        void fix_data_types(po::variables_map& args)
        {
            auto type = args["type"].as<DataType>();

            // These types use the same data type for all inputs/outputs, so we allow
            // using the overarching 'type' parameter.
            if(type == DataType::Float || type == DataType::Double || type == DataType::ComplexFloat
               || type == DataType::ComplexDouble || type == DataType::Int32)
            {
                args.at("a-type").value()     = boost::any(type);
                args.at("b-type").value()     = boost::any(type);
                args.at("c-type").value()     = boost::any(type);
                args.at("d-type").value()     = boost::any(type);
                args.at("alpha-type").value() = boost::any(type);
                args.at("beta-type").value()  = boost::any(type);
            }
        }

        po::variables_map parse_args(int argc, const char* argv[])
        {
            auto options = all_options();

            po::variables_map args;
            po::store(po::parse_command_line(argc, argv, options), args);
            po::notify(args);

            if(args.count("help"))
            {
                std::cout << options << std::endl;
                exit(1);
            }

            if(args.count("config-file"))
            {
                auto configFiles = args["config-file"].as<std::vector<std::string>>();
                for(auto filename : configFiles)
                {
                    std::cout << "loading config file " << filename << std::endl;
                    std::ifstream file(filename.c_str());
                    if(file.bad())
                        throw std::runtime_error(concatenate("Could not open ", filename));
                    po::store(po::parse_config_file(file, options), args);
                }
            }

            fix_data_types(args);

            parse_arg_ints(args, "problem-size");
            parse_arg_ints(args, "a-strides");
            parse_arg_ints(args, "b-strides");
            parse_arg_ints(args, "c-strides");
            parse_arg_ints(args, "d-strides");
            parse_arg_ints(args, "a-zero-pads");
            parse_arg_ints(args, "b-zero-pads");

            if(args["convolution-vs-contraction"].as<bool>())
                parse_arg_ints(args, "convolution-problem");
            return args;
        }

        size_t getMaxWorkspace(std::shared_ptr<MasterSolutionLibrary<ContractionProblem>>& library,
                               std::shared_ptr<Hardware>&                                  hardware,
                               po::variables_map&                                          args,
                               std::vector<ContractionProblem>&                            problems,
                               int firstProblemIdx,
                               int lastProblemIdx)
        {
            // get max workspace size
            size_t maxWorkspaceSize = 0;

            auto            solutionIterator = SolutionIterator::Default(library, hardware, args);
            MetaRunListener listeners;
            listeners.addListener(solutionIterator);
            auto reporters = std::make_shared<MetaResultReporter>();
            listeners.setReporter(reporters);

            listeners.preBenchmarkRun();

            for(int problemIdx = firstProblemIdx; problemIdx <= lastProblemIdx; problemIdx++)
            {
                auto& problem = problems[problemIdx];

                problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

                listeners.preProblem(problem);

                while(solutionIterator->moreSolutionsInProblem())
                {
                    auto solution = solutionIterator->getSolution();
                    if(solution == nullptr)
                        throw std::runtime_error("Could not find a solution");

                    listeners.preSolution(*solution);

                    if(solutionIterator->runCurrentSolution())
                    {
                        maxWorkspaceSize = std::max(
                            maxWorkspaceSize,
                            solution->requiredWorkspaceSize(problems[problemIdx], *hardware));
                    }

                    listeners.postSolution();
                }

                listeners.postProblem();
            }

            listeners.postBenchmarkRun();

            return maxWorkspaceSize;
        }

    } // namespace Client
} // namespace Tensile

size_t calculate_flush_count(size_t                                       arg_flush_count,
                             size_t                                       arg_flush_memory_size,
                             Tensile::Client::ClientProblemFactory const& problemFactory)
{
    size_t default_arg_flush_count       = 1;
    size_t default_arg_flush_memory_size = 0;
    size_t flush_count                   = default_arg_flush_count;

    size_t cached_size = 0;

    for(auto const& problem : problemFactory.problems())
    {
        size_t aSize = problem.a().elementBytes() * problem.a().totalLogicalElements();
        size_t bSize = problem.b().elementBytes() * problem.b().totalLogicalElements();
        size_t cSize = problem.c().elementBytes() * problem.c().totalLogicalElements();

        cached_size = std::max(cached_size, aSize + bSize + cSize);
    }

    if(arg_flush_count != default_arg_flush_count
       && arg_flush_memory_size != default_arg_flush_memory_size)
    {
        std::cout << "Tensile WARNING: cannot set both flush_count and flush_memory_size"
                  << std::endl;
        std::cout << "Tensile WARNING: using flush_count = " << arg_flush_count << std::endl;
        flush_count = arg_flush_count;
    }
    else if(arg_flush_count != default_arg_flush_count)
    {
        flush_count = arg_flush_count;

        std::cout << "flush_memory_size = ";
        Tensile::print_memory_size(flush_count * cached_size);
        std::cout << std::endl;
    }
    else if(arg_flush_memory_size != default_arg_flush_memory_size)
    {
        flush_count = 1 + (arg_flush_memory_size - 1) / cached_size;

        std::cout << "flush_count = " << flush_count << std::endl;
    }
    return flush_count;
}

int main(int argc, const char* argv[])
{
    using namespace Tensile;
    using namespace Tensile::Client;

    auto args = parse_args(argc, argv);

    ClientProblemFactory problemFactory(args);

    auto        hardware = GetHardware(args);
    hipStream_t stream   = GetStream(args);

    auto                          library = LoadSolutionLibrary(args);
    Tensile::hip::SolutionAdapter adapter;
    LoadCodeObjects(args, adapter);

    auto filename = args["library-file"].as<std::string>();

    size_t      directoryPos     = filename.rfind('/');
    std::string libraryDirectory = filename;
    if(directoryPos != std::string::npos)
        libraryDirectory.resize(directoryPos + 1);
    else
        libraryDirectory = '.';

    adapter.initializeLazyLoading(hardware->archName(), libraryDirectory);

    auto problems        = problemFactory.problems();
    int  firstProblemIdx = args["problem-start-idx"].as<int>();
    int  numProblems     = args["num-problems"].as<int>();
    if(numProblems < 0)
        numProblems = problems.size();
    int lastProblemIdx = firstProblemIdx + numProblems - 1;

    int  firstSolutionIdx = args["solution-start-idx"].as<int>();
    int  numSolutions     = args["num-solutions"].as<int>();
    bool gpuTimer         = args["use-gpu-timer"].as<bool>();
    bool runKernels       = !args["selection-only"].as<bool>();
    bool exitOnError      = args["exit-on-error"].as<bool>();

    if(firstSolutionIdx < 0)
        firstSolutionIdx = library->solutions.begin()->first;

    if(numSolutions < 0)
    {
        auto iter = library->solutions.end();
        iter--;
    }

    size_t maxWorkspaceSizeLimit = args["max-workspace-size"].as<size_t>();
    size_t maxWorkspaceSize
        = getMaxWorkspace(library, hardware, args, problems, firstProblemIdx, lastProblemIdx);
    maxWorkspaceSize         = std::min(maxWorkspaceSize, maxWorkspaceSizeLimit);
    size_t flush_count       = args["flush-count"].as<size_t>();
    size_t flush_memory_size = args["flush-memory-size"].as<size_t>();

    std::vector<std::shared_ptr<DataInitialization>> dataInit;
    auto solutionIterator = SolutionIterator::Default(library, hardware, args);

    MetaRunListener listeners;

    flush_count = calculate_flush_count(flush_count, flush_memory_size, problemFactory);

    for(size_t i = 0; i < flush_count; i++)
    {
        dataInit.push_back(DataInitialization::Get(args, problemFactory, maxWorkspaceSize));
        listeners.addListener(dataInit[i]);
    }

    listeners.addListener(solutionIterator);
    listeners.addListener(std::make_shared<ProgressListener>(args));
    if(runKernels)
    {
        listeners.addListener(std::make_shared<ReferenceValidator>(args, dataInit[0], stream));
        listeners.addListener(std::make_shared<BenchmarkTimer>(args, *hardware));
        listeners.addListener(std::make_shared<HardwareMonitorListener>(args));
    }

    auto reporters = std::make_shared<MetaResultReporter>();
    reporters->addReporter(PerformanceReporter::Default(args));

    // PerformanceReporter needs to be called before these two, or else values
    // will be missing
    reporters->addReporter(LogReporter::Default(args));
    reporters->addReporter(ResultFileReporter::Default(args));
    reporters->addReporter(LibraryUpdateReporter::Default(args));

    if(args.count("log-file"))
    {
        std::string filename = args["log-file"].as<std::string>();
        auto        logFile  = std::make_shared<std::ofstream>(
            filename.c_str(), args["log-file-append"].as<bool>() ? std::ios::app : std::ios::out);

        reporters->addReporter(LogReporter::Default(args, logFile, LogLevel::Normal));
    }

    listeners.setReporter(reporters);

    // ReferenceValidator validator(args, dataInit);
    // BenchmarkTimer timer(args);

    reporters->report(ResultKey::ProblemCount, problemFactory.problems().size());

    while(listeners.needMoreBenchmarkRuns())
    {
        listeners.preBenchmarkRun();

        for(int problemIdx = firstProblemIdx; problemIdx <= lastProblemIdx; problemIdx++)
        {
            auto& problem = problems[problemIdx];
            problem.setWorkspaceSize(dataInit[0]->workspaceSize());

            reporters->report(ResultKey::ProblemIndex, problemIdx);
            reporters->report(ResultKey::ProblemProgress,
                              concatenate(problemIdx, "/", lastProblemIdx));

            // std::cout << "Problem: " << problem.operationDescription() <<
            // std::endl; std::cout << "a: " << problem.a() << std::endl; std::cout <<
            // "b: " << problem.b() << std::endl; std::cout << "c: " << problem.c() <<
            // std::endl; std::cout << "d: " << problem.d() << std::endl;

            listeners.preProblem(problem);

            while(solutionIterator->moreSolutionsInProblem())
            {
                auto solution = solutionIterator->getSolution();
                if(solution == nullptr)
                    throw std::runtime_error("Could not find a solution");

                listeners.preSolution(*solution);

                if(solutionIterator->runCurrentSolution() && runKernels)
                {
                    try
                    {
                        while(listeners.needMoreRunsInSolution())
                        {
                            std::vector<std::shared_ptr<ContractionInputs>> inputs;
                            std::vector<std::vector<KernelInvocation>>      kernels;
                            for(size_t i = 0; i < flush_count; i++)
                            {
                                inputs.push_back(dataInit[i]->prepareGPUInputs(problem));
                                kernels.push_back(
                                    solution->solve(problem, *(inputs[i]), *hardware));
                            }

                            size_t       warmupInvocations = listeners.numWarmupRuns();
                            size_t       eventCount        = gpuTimer ? kernels[0].size() : 0;
                            TimingEvents warmupStartEvents(warmupInvocations, eventCount);
                            TimingEvents warmupStopEvents(warmupInvocations, eventCount);

                            for(int i = 0; i < warmupInvocations; i++)
                            {
                                listeners.preWarmup();
                                if(gpuTimer)
                                    HIP_CHECK_EXC(adapter.launchKernels(kernels[0],
                                                                        stream,
                                                                        warmupStartEvents[i],
                                                                        warmupStopEvents[i]));
                                else
                                    HIP_CHECK_EXC(adapter.launchKernels(
                                        kernels[0], stream, nullptr, nullptr));
                                listeners.postWarmup();
                                // Do validation after first warmup
                                if(i == 0)
                                    listeners.validateWarmups(
                                        inputs[0], warmupStartEvents, warmupStopEvents);
                            }

                            size_t syncs = listeners.numSyncs();
                            size_t enq   = listeners.numEnqueuesPerSync();

                            listeners.preSyncs();

                            for(int i = 0; i < syncs; i++)
                            {
                                TimingEvents startEvents(enq, eventCount);
                                TimingEvents stopEvents(enq, eventCount);

                                listeners.preEnqueues();

                                for(int j = 0; j < enq; j++)
                                {
                                    int flush_index = (j + i * enq + 1) % flush_count;
                                    if(gpuTimer)
                                        HIP_CHECK_EXC(adapter.launchKernels(kernels[flush_index],
                                                                            stream,
                                                                            startEvents[j],
                                                                            stopEvents[j]));
                                    else
                                        HIP_CHECK_EXC(adapter.launchKernels(
                                            kernels[flush_index], stream, nullptr, nullptr));
                                }

                                listeners.postEnqueues(startEvents, stopEvents);
                                listeners.validateEnqueues(inputs[0], startEvents, stopEvents);
                            }

                            listeners.postSyncs();
                        }
                    }
                    catch(std::runtime_error const& err)
                    {
                        reporters->report(ResultKey::Validation, "INVALID");
                        reporters->log(LogLevel::Error,
                                       concatenate("Exception occurred: ", err.what(), "\n"));
                    }
                }

                listeners.postSolution();

                if(exitOnError && listeners.error() > 0)
                {
                    // error range in shell is [0-255]
                    return std::min(listeners.error(), 255);
                }
            }

            listeners.postProblem();
        }

        listeners.postBenchmarkRun();
    }

    listeners.finalizeReport();

    // error range in shell is [0-255]
    return std::min(listeners.error(), 255);
}
