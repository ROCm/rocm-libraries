/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/Contractions.hpp>
#include <Tensile/DataTypes.hpp>
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
#include "TimingInstrumentation.hpp"

#include "LibraryUpdateReporter.hpp"
#include "LogReporter.hpp"
#include "MetaResultReporter.hpp"
#include "PerformanceReporter.hpp"
#include "ResultFileReporter.hpp"
#include "ResultReporter.hpp"

#include "ProgramOptions.hpp"
#include "Utility.hpp"

#ifndef TENSILELITE_CLIENT_ENABLE_ROCPROFSDK
#define TENSILELITE_CLIENT_ENABLE_ROCPROFSDK 0
#endif
#if TENSILELITE_CLIENT_ENABLE_ROCPROFSDK
#include "Profiler.hpp"
#endif

#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>

namespace TensileLite
{
    namespace Client
    {
        __global__ void flush_icache()
        {
            asm __volatile__("s_icache_inv \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t" ::
                                 :);
        }

        uint32_t flush_grid_size()
        {
            hipDeviceProp_t deviceProps;
            HIP_CHECK_EXC(hipGetDeviceProperties(&deviceProps, 0));
            uint32_t numBlocks = deviceProps.multiProcessorCount * 60;
            return numBlocks;
        }

        float estimate_flush_kernel_time(hipStream_t stream, bool useGPUTimer)
        {
            const int                                                   flushIter = 100000;
            hipEvent_t                                                  start, stop;
            std::chrono::time_point<std::chrono::high_resolution_clock> begTime;

            if(useGPUTimer)
            {
                HIP_CHECK_EXC(hipEventCreate(&start));
                HIP_CHECK_EXC(hipEventCreate(&stop));
            }

            //warmup runs
            for(int i = 0; i < flushIter; i++)
            {
                hipLaunchKernelGGL(flush_icache, flush_grid_size(), 64, 0, stream);
            }

            if(useGPUTimer)
            {
                HIP_CHECK_EXC(hipEventRecord(start, stream));
            }
            else
            {
                HIP_CHECK_EXC(hipStreamSynchronize(stream));
                begTime = std::chrono::high_resolution_clock::now();
            }

            for(int i = 0; i < flushIter; i++)
            {
                hipLaunchKernelGGL(flush_icache, flush_grid_size(), 64, 0, stream);
            }

            float time{};

            if(useGPUTimer)
            {
                HIP_CHECK_EXC(hipEventRecord(stop, stream));
                HIP_CHECK_EXC(hipEventSynchronize(stop));
            }

            HIP_CHECK_EXC(hipStreamSynchronize(stream));

            if(useGPUTimer)
            {
                HIP_CHECK_EXC(hipEventElapsedTime(&time, start, stop));
                HIP_CHECK_EXC(hipEventDestroy(start));
                HIP_CHECK_EXC(hipEventDestroy(stop));
            }
            else
            {
                time = std::chrono::duration<float,
                                             std::milli>{std::chrono::high_resolution_clock::now()
                                                         - begTime}
                           .count();
            }
            return time / flushIter;
        }

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

                ("type",                     po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "Data type")
                ("a-type",                   po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "A data type")
                ("b-type",                   po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "B data type")
                ("c-type",                   po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "C data type")
                ("d-type",                   po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "D data type")
                ("e-type",                   po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "E data type")
                ("amaxD-type",               po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "amaxD data type")
                ("alpha-type",               po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "alpha data type")
                ("beta-type",                po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "beta data type")
                ("compute-input-type-A",     po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "compute input data type A")
                ("compute-input-type-B",     po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "compute input data type B")
                ("f32-xdl-math-op",          po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "Use xf32 compute for float input and output matrices.")
                ("mx-a-block",               po::value<int>()->default_value(0), "block of mx datatype input matrix A")
                ("mx-b-block",               po::value<int>()->default_value(0), "block of mx datatype input matrix B")
                ("mx-a-type",                po::value<rocisa::DataType>()->default_value(rocisa::DataType::E8), "type of mx datatype input matrix A")
                ("mx-b-type",                po::value<rocisa::DataType>()->default_value(rocisa::DataType::E8), "type of mx datatype input matrix B")
                ("swizzle-tensor-a",         po::value<bool>()->default_value(false), "Swizzle input tensor A.")
                ("swizzle-tensor-b",         po::value<bool>()->default_value(false), "Swizzle input tensor B.")
                ("activation-compute-type",  po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "Activation compute type.")
                ("high-precision-accumulate", po::value<bool>()->default_value(false), "Use high-precision accumulate.")
                ("sparse",                   po::value<int>()->default_value(0), "A or B matrix is sparse matrix.")
                ("strided-batched",          po::value<bool>()->default_value(true), "Use strided-batched or general batched")
                ("grouped-gemm",             po::value<bool>()->default_value(false), "Use grouped gemm")
                ("kernel-language",          po::value<KernelLanguage>()->default_value(KernelLanguage::Any), "Select kernel language.")
                ("deterministic-mode",       po::value<bool>()->default_value(false), "Enforce deterministic summation patterns"
                                                                                      "by not splitting U among workgroups")

                ("init-seed",                po::value<unsigned int>()->default_value(0), "The seed for srand")
                ("init-a",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for A")
                ("init-b",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for B")
                ("init-c",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for C")
                ("init-d",                   po::value<InitMode>()->default_value(InitMode::Zero), "Initialization for D")
                ("init-e",                   po::value<InitMode>()->default_value(InitMode::Zero), "Initialization for E")
                ("init-alpha",               po::value<InitMode>()->default_value(InitMode::Two), "Initialization for alpha")
                ("init-beta",                po::value<InitMode>()->default_value(InitMode::Two), "Initialization for beta")
                ("init-bias",                po::value<InitMode>()->default_value(InitMode::One), "Initialization for bias")
                ("init-scaleA",              po::value<InitMode>()->default_value(InitMode::Two), "Initialization for scaleA")
                ("init-scaleB",              po::value<InitMode>()->default_value(InitMode::Two), "Initialization for scaleB")
                ("init-scaleC",              po::value<InitMode>()->default_value(InitMode::Two), "Initialization for scaleC")
                ("init-scaleD",              po::value<InitMode>()->default_value(InitMode::Two), "Initialization for scaleD")
                ("init-scaleAlphaVec",       po::value<InitMode>()->default_value(InitMode::One), "Initialization for scaleAlphaVec")
                ("init-mx-a",                po::value<InitMode>()->default_value(InitMode::One), "Initialization for MX Scale for A")
                ("init-mx-b",                po::value<InitMode>()->default_value(InitMode::One), "Initialization for MX Scale for B")
                ("pristine-on-gpu",          po::value<bool>()->default_value(true), "Keep a pristine copy of inputs on GPU for performance")
                ("c-equal-d",                po::value<bool>()->default_value(false), "C equals D")
                ("offset-a",                 po::value<size_t>()->default_value(0), "buffer a start offset")
                ("offset-b",                 po::value<size_t>()->default_value(0), "buffer b start offset")
                ("offset-c",                 po::value<size_t>()->default_value(0), "buffer c start offset")
                ("offset-d",                 po::value<size_t>()->default_value(0), "buffer d start offset")
                ("offset-e",                 po::value<size_t>()->default_value(0), "buffer e start offset")
                ("print-valids",             po::value<bool>()->default_value(false), "Print values that pass validation")
                ("print-max",                po::value<int>()->default_value(-1), "Max number of values to print")
                ("num-elements-to-validate", po::value<int>()->default_value(0), "Number of elements to validate")
                ("bounds-check",             po::value<BoundsCheckMode>()->default_value(BoundsCheckMode::Disable),
                "1:Use sentinel values to check memory boundaries."
                "2:Memory bound check by front guard page"
                "3:Memory bound check by back guard page"
                "4:Memory bound check by both side guard page")
                ("prune-mode",               po::value<PruneSparseMode>()->default_value(PruneSparseMode::PruneRandom), "prune Sparse mode")

                ("print-tensor-a",                  po::value<bool>()->default_value(false), "Print tensor A.")
                ("print-tensor-b",                  po::value<bool>()->default_value(false), "Print tensor B.")
                ("print-tensor-c",                  po::value<bool>()->default_value(false), "Print tensor C.")
                ("print-tensor-d",                  po::value<bool>()->default_value(false), "Print tensor D.")
                ("print-tensor-ref",                po::value<bool>()->default_value(false), "Print reference tensor D.")
                ("print-tensor-bias",               po::value<bool>()->default_value(false), "Print tensor Bias.")
                ("print-tensor-scale-alpha-vec",    po::value<bool>()->default_value(false), "Print tensor ScaleAlphaVec.")
                ("print-tensor-amaxd",              po::value<bool>()->default_value(false), "Print tensor AmaxD value from both CPU and GPU.")

                ("dump-tensors",             po::value<bool>()->default_value(false), "Binary dump tensors instead of printing.")

                ("device-idx",               po::value<int>()->default_value(0), "Device index")
                ("use-default-stream",       po::value<bool>()->default_value(false), "Use default Hip stream to run kernels.")
                ("platform-idx",             po::value<int>()->default_value(0), "OpenCL Platform Index")

                ("num-warmups",              po::value<int>()->default_value(0), "Number of warmups to run")
                ("sync-after-warmups",       po::value<bool>()->default_value(true), "Synchronize GPU after warmup kernel runs")
                ("num-benchmarks",           po::value<int>()->default_value(1), "Number of benchmarks to run")
                ("num-enqueues-per-sync",    po::value<int>()->default_value(1), "Enqueues per sync, will affect by min-flops-per-sync")
                ("max-enqueues-per-sync",    po::value<int>()->default_value(-1), "Max Enqueues per sync, will affect by min-flops-per-sync")
                ("num-syncs-per-benchmark",  po::value<int>()->default_value(1), "Syncs per benchmark")
                ("skip-slow-solution-ratio", po::value<float>()->default_value(0.0), "ratio to skip slow solution during warm-up stage")
                ("min-flops-per-sync",       po::value<size_t>()->default_value(0), "Minimum number of flops per sync to increase stability for small problems.")
                ("use-gpu-timer",            po::value<bool>()->default_value(true), "Use GPU timer")
                ("sleep-percent",            po::value<int>()->default_value(0), "Sleep percentage")
                ("hardware-monitor",         po::value<bool>()->default_value(true), "Use hardware monitor.")

                ("perf-l2-read-hits",        po::value<double>()->default_value(0.0), "L2 read hits")
                ("perf-l2-write-hits",       po::value<double>()->default_value(0.5), "L2 write hits")
                ("perf-l2-read-bw-mul",      po::value<double>()->default_value(2.0), "L2 read bandwidth multiplier")
                ("perf-read-efficiency",     po::value<double>()->default_value(0.85), "Read efficiency")
                ("perf-ops-per-cycle",       po::value<int>()->default_value(64), "Ops per cycle")
                ("csv-export-extra-cols",    po::value<bool>()->default_value(false), "CSV exports winner information")
                ("csv-merge-same-problems",  po::value<bool>()->default_value(false), "CSV merge rows of same problem id")
                ("PrintWinnersOnly",         po::value<bool>()->default_value(false), "PrintWinnersOnly")

                ("problem-size,p",           vector_default_empty<std::string>(), "Specify a problem size.  Comma-separated list of "
                                                                                  "sizes, in the order of the Einstein notation.")

                ("original-problem-size",   vector_default_empty<std::string>(), "Original problem sizes before macrotile padding. "
                                                                                  "When present, validation uses these sizes to slice output.")

                ("prob-sol-map",             vector_default_empty<std::string>(), "[probIdx, solIdx]")

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

                ("e-strides",                vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")
                ("bias-strides",             vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")
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

                ("a-ops",                    vector_default_empty<TensorOp>(), "Operations applied to A.")
                ("b-ops",                    vector_default_empty<TensorOp>(), "Operations applied to B.")
                ("c-ops",                    vector_default_empty<TensorOp>(), "Operations applied to C.")
                ("d-ops",                    vector_default_empty<TensorOp>(), "Operations applied to D.")

                ("exit-on-error",            po::value<bool>()->default_value(false), "Exit run early on failed kernels or other errors.")
                ("selection-only",           po::value<bool>()->default_value(false), "Don't run any solutions, only print kernel selections.")
                ("max-workspace-size",       po::value<size_t>()->default_value(32*1024*1024), "Max workspace for training")
                ("granularity-threshold",    po::value<double>()->default_value(0.0), "Don't run a solution if total granularity is below")
                ("prediction-threshold",     po::value<double>()->default_value(2.0), "Don't run a solution if predicted performance is low")

                ("activation-type",           po::value<ActivationType>()->default_value(ActivationType::None), "An activation type")
                ("activation-hpa",            po::value<bool>()->default_value(false), "Use the same data type as high precision accumulate.")
                ("activation-no-guard",          po::value<bool>()->default_value(false), "Use activation guard to deall with nan outputs.")
                ("activation-additional-args",vector_default_empty<std::string>(), "Activation additional floating-point number arguments.")
                ("activation-enum-args",      po::value<std::vector<ActivationType>>()->default_value(std::vector<ActivationType>(1, ActivationType::None), "[]"), "Activation enum argument.")
                ("use-bias",                  po::value<int>()->default_value(0), "Use bias.")
                ("bias-source",               po::value<int>()->default_value(3), "Bias source.")
                ("use-scaleAB",               po::value<std::string>()->default_value(""), "Use scaleAB.")
                ("use-scaleCD",               po::value<bool>()->default_value(false), "Use scaleCD.")
                ("use-scaleAlphaVec",         po::value<int>()->default_value(0), "Use scaleAlphaVec.")
                ("bias-type-args",            po::value<std::vector<rocisa::DataType>>()->default_value(std::vector<rocisa::DataType>(1, rocisa::DataType::None), "[]"), "Bias data type args.")
                ("factor-dim-args",           po::value<std::vector<int>>()->default_value(std::vector<int>(1, 0), "[]"), "factor dimensions args.")
                ("icache-flush-args",         po::value<std::vector<bool>>()->default_value(std::vector<bool>(1, false), "[]"), "ICache flush args.")
                ("use-e",                     po::value<bool>()->default_value(false), "Use E.")
                ("use-gradient",              po::value<bool>()->default_value(false), "Use gradient.")
                ("use-user-args",             po::value<bool>()->default_value(false), "Use user argument structure as kernel input.")
                ("rotating-buffer-size",      po::value<int32_t>()->default_value(0), "Size of rotating buffer in the unit of MB.")
                ("rotating-buffer-mode",      po::value<int32_t>()->default_value(0), "Rotating mode.")
                ("output-amaxD",              po::value<bool>()->default_value(false), "Output AmaxD.")
                ("timing-instrumentation",    po::value<bool>()->default_value(false)->implicit_value(true), "Enable detailed timing instrumentation output to stderr.")
                ("rocprof-counter",           vector_default_empty<std::string>(), "Rocprof counters.")
                ("metadata-layout",           po::value<int32_t>()->default_value(0), "Sparse Metadata Layout")
                ;
            // clang-format on

            return options;
        }

        /** Dump parsed program options to a text file for comparison (Boost vs replacement).
         *  Set env TENSILE_DUMP_OPTIONS=1 to enable. Writes to /tmp/tensilelite_program_options_dump.txt */
        void DumpProgramOptionsToFile(po::variables_map const& args, char const* path)
        {
            std::ofstream out(path);
            if(!out)
            {
                std::cerr << "TENSILE_DUMP_OPTIONS: failed to open " << path << " for write\n";
                return;
            }
            out << "# program_options dump (one key=value per line; vectors as size then "
                   "elements)\n";
#define DUMP_OPT(K, T)                                          \
    do                                                          \
    {                                                           \
        if(args.count(K))                                       \
        {                                                       \
            try                                                 \
            {                                                   \
                out << (K) << "=" << args[(K)].as<T>() << "\n"; \
            }                                                   \
            catch(...)                                          \
            {                                                   \
                out << (K) << "=(as<" #T "> failed)\n";         \
            }                                                   \
        }                                                       \
    } while(0)
#define DUMP_VEC(K, T)                                          \
    do                                                          \
    {                                                           \
        if(args.count(K))                                       \
        {                                                       \
            try                                                 \
            {                                                   \
                auto const& v = args[(K)].as<std::vector<T>>(); \
                out << (K) << ".size=" << v.size();             \
                for(size_t i = 0; i < v.size(); ++i)            \
                    out << " " << v[i];                         \
                out << "\n";                                    \
            }                                                   \
            catch(...)                                          \
            {                                                   \
                out << (K) << "=(vector as failed)\n";          \
            }                                                   \
        }                                                       \
    } while(0)
#define DUMP_VECVEC(K)                                                            \
    do                                                                            \
    {                                                                             \
        if(args.count(K))                                                         \
        {                                                                         \
            try                                                                   \
            {                                                                     \
                auto const& v = args[(K)].as<std::vector<std::vector<size_t>>>(); \
                out << (K) << ".size=" << v.size() << "\n";                       \
                for(size_t i = 0; i < v.size(); ++i)                              \
                {                                                                 \
                    out << (K) << "[" << i << "]=";                               \
                    for(size_t j = 0; j < v[i].size(); ++j)                       \
                        out << (j ? " " : "") << v[i][j];                         \
                    out << "\n";                                                  \
                }                                                                 \
            }                                                                     \
            catch(...)                                                            \
            {                                                                     \
                out << (K) << "=(vecvec failed)\n";                               \
            }                                                                     \
        }                                                                         \
    } while(0)
            if(args.count("config-file"))
            {
                try
                {
                    auto const& v = args["config-file"].as<std::vector<std::string>>();
                    out << "config-file.size=" << v.size() << "\n";
                    for(size_t i = 0; i < v.size(); ++i)
                        out << "config-file[" << i << "]=" << v[i] << "\n";
                }
                catch(...)
                {
                    out << "config-file=(failed)\n";
                }
            }
            DUMP_OPT("library-file", std::string);
            if(args.count("code-object"))
            {
                try
                {
                    auto const& v = args["code-object"].as<std::vector<std::string>>();
                    out << "code-object.size=" << v.size() << "\n";
                    for(size_t i = 0; i < v.size(); ++i)
                        out << "code-object[" << i << "]=" << v[i] << "\n";
                }
                catch(...)
                {
                    out << "code-object=(failed)\n";
                }
            }
            DUMP_OPT("performance-metric", PerformanceMetric);
            DUMP_OPT("problem-identifier", std::string);
            DUMP_OPT("type", rocisa::DataType);
            DUMP_OPT("a-type", rocisa::DataType);
            DUMP_OPT("b-type", rocisa::DataType);
            DUMP_OPT("c-type", rocisa::DataType);
            DUMP_OPT("d-type", rocisa::DataType);
            DUMP_OPT("e-type", rocisa::DataType);
            DUMP_OPT("amaxD-type", rocisa::DataType);
            DUMP_OPT("alpha-type", rocisa::DataType);
            DUMP_OPT("beta-type", rocisa::DataType);
            DUMP_OPT("compute-input-type", rocisa::DataType);
            DUMP_OPT("f32-xdl-math-op", rocisa::DataType);
            DUMP_OPT("swizzle-tensor-a", bool);
            DUMP_OPT("swizzle-tensor-b", bool);
            DUMP_OPT("activation-compute-type", rocisa::DataType);
            DUMP_OPT("high-precision-accumulate", bool);
            DUMP_OPT("sparse", int);
            DUMP_OPT("strided-batched", bool);
            DUMP_OPT("grouped-gemm", bool);
            DUMP_OPT("kernel-language", KernelLanguage);
            DUMP_OPT("deterministic-mode", bool);
            DUMP_OPT("init-seed", unsigned int);
            DUMP_OPT("init-a", InitMode);
            DUMP_OPT("init-b", InitMode);
            DUMP_OPT("init-c", InitMode);
            DUMP_OPT("init-d", InitMode);
            DUMP_OPT("init-e", InitMode);
            DUMP_OPT("init-alpha", InitMode);
            DUMP_OPT("init-beta", InitMode);
            DUMP_OPT("init-bias", InitMode);
            DUMP_OPT("init-scaleA", InitMode);
            DUMP_OPT("init-scaleB", InitMode);
            DUMP_OPT("init-scaleC", InitMode);
            DUMP_OPT("init-scaleD", InitMode);
            DUMP_OPT("init-scaleAlphaVec", InitMode);
            DUMP_OPT("pristine-on-gpu", bool);
            DUMP_OPT("c-equal-d", bool);
            DUMP_OPT("num-elements-to-validate", int);
            DUMP_OPT("bounds-check", BoundsCheckMode);
            DUMP_OPT("prune-mode", PruneSparseMode);
            DUMP_OPT("device-idx", int);
            DUMP_OPT("use-default-stream", bool);
            DUMP_OPT("num-warmups", int);
            DUMP_OPT("sync-after-warmups", bool);
            DUMP_OPT("num-benchmarks", int);
            DUMP_OPT("num-enqueues-per-sync", int);
            DUMP_OPT("max-enqueues-per-sync", int);
            DUMP_OPT("num-syncs-per-benchmark", int);
            DUMP_OPT("skip-slow-solution-ratio", float);
            DUMP_OPT("min-flops-per-sync", size_t);
            DUMP_OPT("use-gpu-timer", bool);
            DUMP_OPT("sleep-percent", int);
            DUMP_OPT("hardware-monitor", bool);
            DUMP_OPT("perf-l2-read-hits", double);
            DUMP_OPT("perf-l2-write-hits", double);
            DUMP_OPT("perf-l2-read-bw-mul", double);
            DUMP_OPT("perf-read-efficiency", double);
            DUMP_OPT("csv-export-extra-cols", bool);
            DUMP_OPT("csv-merge-same-problems", bool);
            DUMP_OPT("PrintWinnersOnly", bool);
            DUMP_VECVEC("problem-size");
            if(args.count("prob-sol-map"))
            {
                try
                {
                    auto const& m = args["prob-sol-map"].as<std::map<int, int>>();
                    out << "prob-sol-map.size=" << m.size() << "\n";
                    for(auto const& p : m)
                        out << "prob-sol-map " << p.first << "=" << p.second << "\n";
                }
                catch(...)
                {
                    out << "prob-sol-map=(failed)\n";
                }
            }
            DUMP_VECVEC("a-strides");
            DUMP_VECVEC("b-strides");
            DUMP_VECVEC("c-strides");
            DUMP_VECVEC("d-strides");
            DUMP_VECVEC("e-strides");
            DUMP_VECVEC("bias-strides");
            DUMP_OPT("problem-start-idx", int);
            DUMP_OPT("num-problems", int);
            DUMP_OPT("solution-start-idx", int);
            DUMP_OPT("num-solutions", int);
            DUMP_OPT("best-solution", bool);
            DUMP_OPT("results-file", std::string);
            DUMP_OPT("log-file", std::string);
            DUMP_OPT("log-file-append", bool);
            DUMP_OPT("log-level", LogLevel);
            DUMP_OPT("library-update-file", std::string);
            DUMP_OPT("library-update-comment", bool);
            DUMP_OPT("exit-on-error", bool);
            DUMP_OPT("selection-only", bool);
            DUMP_OPT("max-workspace-size", size_t);
            DUMP_OPT("granularity-threshold", double);
            DUMP_OPT("activation-type", ActivationType);
            DUMP_OPT("activation-no-guard", bool);
            if(args.count("activation-additional-args"))
            {
                try
                {
                    auto const& v
                        = args["activation-additional-args"].as<std::vector<std::vector<double>>>();
                    out << "activation-additional-args.size=" << v.size() << "\n";
                    for(size_t i = 0; i < v.size(); ++i)
                    {
                        out << "activation-additional-args[" << i << "]=";
                        for(size_t j = 0; j < v[i].size(); ++j)
                            out << (j ? "," : "") << v[i][j];
                        out << "\n";
                    }
                }
                catch(...)
                {
                    out << "activation-additional-args=(failed)\n";
                }
            }
            DUMP_VEC("activation-enum-args", ActivationType);
            DUMP_OPT("use-bias", int);
            DUMP_OPT("bias-source", int);
            DUMP_OPT("use-scaleAB", std::string);
            DUMP_OPT("use-scaleCD", bool);
            DUMP_OPT("use-scaleAlphaVec", int);
            DUMP_VEC("bias-type-args", rocisa::DataType);
            DUMP_VEC("factor-dim-args", int);
            DUMP_VEC("icache-flush-args", bool);
            DUMP_OPT("use-e", bool);
            DUMP_OPT("use-gradient", bool);
            DUMP_OPT("use-user-args", bool);
            DUMP_OPT("rotating-buffer-size", int32_t);
            DUMP_OPT("rotating-buffer-mode", int32_t);
            DUMP_OPT("output-amaxD", bool);
            DUMP_OPT("timing-instrumentation", bool);
#undef DUMP_OPT
#undef DUMP_VEC
#undef DUMP_VECVEC
            std::cerr << "TENSILE_DUMP_OPTIONS: wrote " << path << "\n";
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

        std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>>
            LoadSolutionLibrary(po::variables_map const& args)
        {
            auto filename = args["library-file"];
            if(!filename.empty())
            {
                return std::dynamic_pointer_cast<MasterSolutionLibrary<ContractionProblemGemm>>(
                    LoadLibraryFile<ContractionProblemGemm>(filename.as<std::string>()));
            }

            auto embeddedLibrary
                = std::dynamic_pointer_cast<MasterSolutionLibrary<ContractionProblemGemm>>(
                    EmbeddedLibrary<ContractionProblemGemm>::Get());

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

        template <typename T>
        T parse_num(std::string const& s)
        {
            T                  t{};
            std::istringstream ss(s);
            ss >> t;
            if(!ss && !ss.eof())
                throw std::runtime_error("Failed to parse number: " + s);
            return t;
        }

        template <typename T>
        std::vector<T> split_nums(std::string const& value)
        {
            std::vector<std::string> parts = po::split_string(value);
            std::vector<T>           rv;
            rv.reserve(parts.size());
            for(auto const& part : parts)
                if(!part.empty())
                    rv.push_back(parse_num<T>(part));
            return rv;
        }

        template <typename T>
        void parse_arg_nums(po::variables_map& args, std::string const& name)
        {
            auto inValue = args[name].as<std::vector<std::string>>();

            std::vector<std::vector<T>> outValue;
            outValue.reserve(inValue.size());
            for(auto const& str : inValue)
                outValue.push_back(split_nums<T>(str));

            args.at(name).value() = std::any(outValue);
        }

        void parse_arg_bools(po::variables_map& args, std::string const& name)
        {
            auto opts             = args[name].as<std::vector<bool>>();
            args.at(name).value() = std::any(opts);
        }

        void parse_arg_ints(po::variables_map& args, std::string const& name)
        {
            parse_arg_nums<size_t>(args, name);
        }

        void parse_arg_double(po::variables_map& args, std::string const& name)
        {
            parse_arg_nums<double>(args, name);
        }

        void parse_bias_type_args(po::variables_map& args, std::string const& name)
        {
            auto type             = args[name].as<std::vector<rocisa::DataType>>();
            args.at(name).value() = std::any(type);
        }

        void parse_activation_enum_args(po::variables_map& args, std::string const& name)
        {
            auto type             = args[name].as<std::vector<ActivationType>>();
            args.at(name).value() = std::any(type);
        }

        void parse_activation_int(po::variables_map& args, std::string const& name)
        {
            auto type             = args[name].as<ActivationType>();
            args.at(name).value() = std::any(type);
        }

        template <typename T>
        void parse_arg_nums_map(po::variables_map& args, std::string const& name)
        {
            auto inValue = args[name].as<std::vector<std::string>>();

            std::map<int, int> outValue;
            for(auto const& str : inValue)
            {
                auto vec         = split_nums<T>(str);
                outValue[vec[0]] = vec[1];
            }

            args.at(name).value() = std::any(outValue);
        }

        void parse_arg_ints_map(po::variables_map& args, std::string const& name)
        {
            parse_arg_nums_map<int>(args, name);
        }

        void fix_data_types(po::variables_map& args)
        {
            auto type = args["type"].as<rocisa::DataType>();

            // These types use the same data type for all inputs/outputs, so we allow
            // using the overarching 'type' parameter.
            if(type == rocisa::DataType::Float || type == rocisa::DataType::Double
               || type == rocisa::DataType::ComplexFloat || type == rocisa::DataType::ComplexDouble
               || type == rocisa::DataType::Int32)
            {
                args.at("a-type").value()     = std::any(type);
                args.at("b-type").value()     = std::any(type);
                args.at("c-type").value()     = std::any(type);
                args.at("d-type").value()     = std::any(type);
                args.at("alpha-type").value() = std::any(type);
                args.at("beta-type").value()  = std::any(type);
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

#if !(TENSILELITE_CLIENT_ENABLE_ROCPROFSDK)
            if(args["rocprof-counter"].as<std::vector<std::string>>().size())
            {
                throw std::runtime_error("rocprof-counter is provided but client is not built with -DTENSILELITE_CLIENT_ENABLE_ROCPROFSDK.");
            }
#endif
            fix_data_types(args);

            parse_arg_ints(args, "problem-size");
            parse_arg_ints(args, "original-problem-size");
            parse_arg_ints(args, "a-strides");
            parse_arg_ints(args, "b-strides");
            parse_arg_ints(args, "c-strides");
            parse_arg_ints(args, "d-strides");
            parse_arg_ints(args, "e-strides");
            parse_arg_ints(args, "bias-strides");
            parse_bias_type_args(args, "bias-type-args");
            parse_activation_int(args, "activation-type");
            parse_activation_enum_args(args, "activation-enum-args");
            parse_arg_double(args, "activation-additional-args");
            parse_arg_bools(args, "icache-flush-args");
            // std::cout << "Pasring parse_arg_ints_map()" << std::endl;
            parse_arg_ints_map(args, "prob-sol-map");
            return args;
        }

    } // namespace Client
} // namespace TensileLite

int main(int argc, const char* argv[])
{
    using namespace TensileLite;
    using namespace TensileLite::Client;

    auto args = parse_args(argc, argv);

    // Enable timing instrumentation if requested
    g_timingInstrumentationEnabled = args["timing-instrumentation"].as<bool>();

    // Set srand
    unsigned int seed = args["init-seed"].as<unsigned int>();
    if(seed == 0)
    {
        seed = time(NULL);
    }
    std::cout << std::endl << "srand seed is set to " << seed << std::endl << std::endl;
    srand(seed);

    ClientProblemFactory problemFactory(args);

    std::shared_ptr<Hardware> hardware;
    hipStream_t              stream;
    {
        ScopedTimer timer("hip_initialization");
        hardware = GetHardware(args);
        stream   = GetStream(args);
    }

    std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library;
    {
        ScopedTimer timer("library_loading");
        library = LoadSolutionLibrary(args);
        if(!library)
            throw std::runtime_error("Failed to load solution library");
    }

    TensileLite::hip::SolutionAdapter adapter;
#if TENSILELITE_CLIENT_ENABLE_ROCPROFSDK
    RocProfiler::getInstance().start();
#endif
    {
        ScopedTimer timer("code_object_loading");
        LoadCodeObjects(args, adapter);
    }
#if TENSILELITE_CLIENT_ENABLE_ROCPROFSDK
    RocProfiler::getInstance().stop();
#endif

    auto filename = args["library-file"].as<std::string>();

    size_t      directoryPos     = filename.rfind('/');
    std::string libraryDirectory = filename;
    if(directoryPos != std::string::npos)
        libraryDirectory.resize(directoryPos + 1);
    else
        libraryDirectory = '.';

    {
        ScopedTimer timer("lazy_loading_init");
        auto result = adapter.initializeLazyLoading(hardware->archName(), libraryDirectory);
        if(result != hipSuccess)
        {
            std::string str = "Lazy loading failed. (" + std::to_string(int(result)) + ").";
            std::runtime_error(str.c_str());
        }
    }

    auto problems        = problemFactory.problems();
    int  firstProblemIdx = args["problem-start-idx"].as<int>();
    int  numProblems     = args["num-problems"].as<int>();
    if(numProblems < 0)
        numProblems = problems.size();
    int lastProblemIdx = firstProblemIdx + numProblems - 1;

    int         firstSolutionIdx = args["solution-start-idx"].as<int>();
    int         numSolutions     = args["num-solutions"].as<int>();
    bool        gpuTimer         = args["use-gpu-timer"].as<bool>();
    bool        runKernels       = !args["selection-only"].as<bool>();
    bool        exitOnError      = args["exit-on-error"].as<bool>();
    bool        groupedGemm      = args["grouped-gemm"].as<bool>();
    const auto& icacheFlushArgs  = args["icache-flush-args"].as<std::vector<bool>>();

    float skip_slow_solution_ratio = args["skip-slow-solution-ratio"].as<float>();
    if(skip_slow_solution_ratio > 1.0 || skip_slow_solution_ratio < 0.0)
    {
        std::cout << "Invalid Skip Slow Solution Ratio: " << skip_slow_solution_ratio << std::endl;
        std::cout << "Please Set Valid Ratio : (0.0 ~ 1.0)." << std::endl;
        exit(1);
    }

    if(firstSolutionIdx < 0)
        firstSolutionIdx = library->solutions.begin()->first;

    if(numSolutions < 0)
    {
        auto iter = library->solutions.end();
        iter--;
    }

    std::shared_ptr<DataInitialization> dataInit;
    {
        // Re-seed before data init: HIP runtime init above may consume rand() non-deterministically
        srand(seed);
        ScopedTimer timer("data_init_setup");
        dataInit = std::make_shared<DataInitialization>(args, problemFactory);
    }

    std::shared_ptr<SolutionIterator> solutionIterator;
    {
        ScopedTimer timer("solution_iterator_setup");
        solutionIterator = SolutionIterator::Default(library, hardware, args);
    }

    MetaRunListener listeners;
    std::shared_ptr<BenchmarkTimer> benchmarkTimer;
    float                           flushTimeMs{};

    {
        ScopedTimer timer("listener_setup");
        listeners.addListener(dataInit);
        listeners.addListener(solutionIterator);
        listeners.addListener(std::make_shared<ProgressListener>(args));

        if(runKernels)
        {
            bool hasIcacheFlush
                = std::any_of(begin(icacheFlushArgs), end(icacheFlushArgs), [](auto i) { return i; });
            flushTimeMs = hasIcacheFlush ? estimate_flush_kernel_time(stream, gpuTimer) : 0.f;
            listeners.addListener(std::make_shared<ReferenceValidator>(args, dataInit));
            benchmarkTimer = std::make_shared<BenchmarkTimer>(args, *hardware, flushTimeMs * 1000);
            listeners.addListener(benchmarkTimer);
            listeners.addListener(std::make_shared<HardwareMonitorListener>(args));
#if TENSILELITE_CLIENT_ENABLE_ROCPROFSDK
            if (!args["rocprof-counter"].as<std::vector<std::string>>().empty())
                listeners.addListener(Profiler::Default(args));
#endif
        }
    }

    std::shared_ptr<MetaResultReporter> reporters;
    {
        ScopedTimer timer("reporter_setup");
        reporters = std::make_shared<MetaResultReporter>();
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
    }

    // ReferenceValidator validator(args, dataInit);
    // BenchmarkTimer timer(args);

    reporters->report(ResultKey::ProblemCount, problemFactory.problems().size());

    bool  useUserArgs = args["use-user-args"].as<bool>();
    void* dUA         = nullptr;
    void* dUAHost     = nullptr;

    if(Debug::Instance().getBenchmark())
    {
        std::stringstream ss;
        static_cast<void>(benchmarkAllocation(ss));
        std::cout << ss.str();
    }

    while(listeners.needMoreBenchmarkRuns())
    {
        listeners.preBenchmarkRun();
        const auto flushGridSize = flush_grid_size();
        for(auto icacheFlush : icacheFlushArgs)
        {
            benchmarkTimer->setIFlushTimeUs(icacheFlush ? flushTimeMs * 1000 : 0.f);

            for(int problemIdx = firstProblemIdx; problemIdx <= lastProblemIdx; problemIdx++)
            {
                auto problem = problems[problemIdx].get();

                reporters->report(ResultKey::ProblemIndex, problemIdx);
                reporters->report(ResultKey::ProblemProgress,
                                  concatenate(problemIdx, "/", lastProblemIdx));

                {
                    ScopedTimer timer("pre_problem");
                    listeners.preProblem(problem);
                }
                std::shared_ptr<ProblemInputs> inputs;
                {
                    ScopedTimer timer("gpu_input_preparation");
                    inputs = dataInit->prepareGPUInputs(problem);
                }

                size_t warmupInvocations    = listeners.numWarmupRuns();
                size_t syncs                = listeners.numSyncs();
                size_t enq                  = listeners.numEnqueuesPerSync();
                size_t maxRotatingBufferNum = max(warmupInvocations, syncs * enq);

                std::vector<std::shared_ptr<ProblemInputs>> inputArr;
                {
                    ScopedTimer timer("rotating_buffer_preparation");
                    inputArr = dataInit->prepareRotatingGPUOutput(
                        maxRotatingBufferNum, problem, inputs, stream);
                    static_cast<void>(hipDeviceSynchronize());
                }
                bool resetInput = false;
                while(solutionIterator->moreSolutionsInProblem())
                {
                    std::shared_ptr<ContractionSolution> solution;
                    {
                        ScopedTimer timer("solution_selection");
                        solution = solutionIterator->getSolution();
                    }
                    if(solution == nullptr)
                        throw std::runtime_error("Could not find a solution");

                    {
                        ScopedTimer timer("pre_solution");
                        listeners.preSolution(solution.get());
                    }
                    if(solutionIterator->runCurrentSolution() && runKernels)
                    {
                        try
                        {
                            while(listeners.needMoreRunsInSolution())
                            {
                                if(resetInput)
                                {
                                    ScopedTimer timer("gpu_input_reset");
                                    auto inputs = dataInit->prepareGPUInputs(problem);
                                    inputArr[0] = inputs;
                                }
                                resetInput = true;

                                std::vector<std::vector<KernelInvocation>> kernels;
                                {
                                    ScopedTimer timer("kernel_solving");
                                    for(size_t r = 0; r < inputArr.size(); r++)
                                    {
                                        auto kernel = useUserArgs
                                                          ? solution->solveTensileGPU((*problem),
                                                                                      *inputArr[r],
                                                                                      *hardware,
                                                                                      &dUA,
                                                                                      &dUAHost,
                                                                                      nullptr,
                                                                                      0,
                                                                                      stream)
                                                          : solution->solve((*problem),
                                                                            *inputArr[r],
                                                                            *hardware,
                                                                            nullptr,
                                                                            0,
                                                                            stream);
                                        kernels.push_back(kernel);
                                    }
                                }

                                size_t       warmupInvocations = listeners.numWarmupRuns();
                                size_t       warmupEventCount  = kernels[0].size();
                                TimingEvents warmupStartEvents(warmupInvocations, warmupEventCount);
                                TimingEvents warmupStopEvents(warmupInvocations, warmupEventCount);

                                if(warmupInvocations > 0)
                                {
                                    {
                                        ScopedTimer timer("warmup_runs");
                                        listeners.preWarmup();
                                        HIP_CHECK_EXC(adapter.launchKernels(kernels[0],
                                                                            stream,
                                                                            warmupStartEvents[0],
                                                                            warmupStopEvents[0]));
                                    }

                                    {
                                        ScopedTimer timer("validate_warmups");
                                        listeners.validateWarmups(
                                            inputs, warmupStartEvents, warmupStopEvents);
                                    }

                                    {
                                        ScopedTimer timer("warmup_runs");
                                        for(int i = 1; i < warmupInvocations; i++)
                                        {
                                            size_t kIdx = i % kernels.size();
                                            HIP_CHECK_EXC(adapter.launchKernels(kernels[kIdx],
                                                                                stream,
                                                                                warmupStartEvents[i],
                                                                                warmupStopEvents[i]));
                                        }
                                        listeners.postWarmup(
                                            warmupStartEvents, warmupStopEvents, stream);
                                    }
                                }

#if TENSILELITE_CLIENT_ENABLE_ROCPROFSDK
                                TimingEvents ProfilerStartEvents(1, warmupEventCount);
                                TimingEvents ProfilerStopEvents(1, warmupEventCount);
                                listeners.preProfiler();
                                HIP_CHECK_EXC(adapter.launchKernels(kernels[warmupInvocations % kernels.size()],
                                                                    stream,
                                                                    ProfilerStartEvents[0],
                                                                    ProfilerStopEvents[0]));
                                listeners.postProfiler();
#endif

                                size_t syncs      = listeners.numSyncs();
                                size_t enq        = listeners.numEnqueuesPerSync();
                                size_t eventCount = gpuTimer ? kernels[0].size() : 0;

                                {
                                    ScopedTimer timer("benchmark_runs");
                                    listeners.preSyncs();
                                    if(enq)
                                        for(int i = 0; i < syncs; i++)
                                        {
                                            TimingEvents startEvents(enq, eventCount);
                                            TimingEvents stopEvents(enq, eventCount);

                                            listeners.preEnqueues(stream);

                                            for(int j = 0; j < enq; j++)
                                            {
                                                size_t kIdx = ((i * enq) + j) % kernels.size();
                                                HIP_CHECK_EXC(adapter.launchKernels(
                                                    kernels[kIdx], stream, nullptr, nullptr));

                                                if(icacheFlush)
                                                {
                                                    hipLaunchKernelGGL(
                                                        flush_icache, flushGridSize, 64, 0, stream);
                                                }
                                            }

                                            listeners.postEnqueues(startEvents, stopEvents, stream);
                                            listeners.validateEnqueues(inputs, startEvents, stopEvents);
                                        }

                                    listeners.postSyncs();
                                }

                                if(useUserArgs)
                                {
                                    solution->relaseDeviceUserArgs(dUA, dUAHost);
                                }
                            }
                        }
                        catch(std::runtime_error const& err)
                        {
                            reporters->report(ResultKey::Validation, "INVALID");
                            reporters->log(LogLevel::Error,
                                           concatenate("Exception occurred: ", err.what(), "\n"));
                        }
                    }

                    {
                        ScopedTimer timer("post_solution");
                        listeners.postSolution();
                    }

                    if(exitOnError && listeners.error() > 0)
                    {
                        flushTimingBuffer();
                        // error range in shell is [0-255]
                        return std::min(listeners.error(), 255);
                    }
                }

                {
                    ScopedTimer timer("post_problem");
                    listeners.postProblem();
                }
            }
        }

        listeners.postBenchmarkRun();
    }

    {
        ScopedTimer timer("finalize_report");
        listeners.finalizeReport();
    }

    // Flush all buffered timing records to stderr
    flushTimingBuffer();

    // error range in shell is [0-255]
    return std::min(listeners.error(), 255);
}
