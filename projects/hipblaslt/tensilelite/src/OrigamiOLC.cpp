/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/OrigamiOLC.hpp>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>

#include <Tensile/PredictionLibrary.hpp>

#include <origami/nn/olc/loaded_model.hpp>
#include <origami/nn/nn.hpp>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <map>
#include <mutex>
#include <thread>

namespace TensileLite
{
    namespace OrigamiOLC
    {
        namespace
        {
            std::mutex& loadMutex()
            {
                static std::mutex m;
                return m;
            }

            // Model path -> handle (invalid_handle when refused): one load per model file.
            std::map<std::string, origami::nn::model_handle_t>& loaded()
            {
                static std::map<std::string, origami::nn::model_handle_t> l;
                return l;
            }

            Logger& loggerSlot()
            {
                static Logger l;
                return l;
            }

            Logger& acceptedSlot()
            {
                static Logger l;
                return l;
            }

            Logger& refusedSlot()
            {
                static Logger l;
                return l;
            }

            // A/B type code of the file names holding each class's prediction rows.
            char const* fileToken(std::string const& dtype)
            {
                return dtype == "bf16" ? "BB" : dtype == "tf32" ? "SS" : nullptr;
            }

            bool inClass(ContractionSolution::ProblemType const& t, std::string const& dtype)
            {
                using rocisa::DataType;
                if(dtype == "bf16")
                    return t.aType == DataType::BFloat16 && t.bType == DataType::BFloat16;
                return t.aType == DataType::Float && t.bType == DataType::Float
                       && t.cType == DataType::Float && t.dType == DataType::Float
                       && t.f32XdlMathOp == DataType::XFloat32;
            }
        } // namespace

        bool enabled()
        {
            static const bool on = [] {
                const char* v = std::getenv("HIPBLASLT_ORIGAMI_OLC");
                return v != nullptr && std::strcmp(v, "1") == 0;
            }();
            return on;
        }

        std::vector<std::string> const& dtypes()
        {
            static const std::vector<std::string> d{"bf16", "tf32"};
            return d;
        }

        char const* logicClass(std::string const& logicStem)
        {
            const std::string prefix = "TensileLibrary_";
            if(logicStem.rfind(prefix, 0) != 0)
                return nullptr;
            const auto        start = prefix.size();
            const std::string token = logicStem.substr(start, logicStem.find('_', start) - start);
            for(auto const& dtype : dtypes())
                if(token == fileToken(dtype))
                    return dtype.c_str();
            return nullptr;
        }

        char const* problemClass(ContractionProblemGemm const& problem)
        {
            using rocisa::DataType;
            if(problem.a().dataType() == DataType::BFloat16
               && problem.b().dataType() == DataType::BFloat16)
                return "bf16";
            if(problem.a().dataType() == DataType::Float && problem.b().dataType() == DataType::Float
               && problem.c().dataType() == DataType::Float
               && problem.d().dataType() == DataType::Float
               && problem.f32XdlMathOp() == DataType::XFloat32)
                return "tf32";
            return nullptr;
        }

        bool libraryManifest(std::string const&                                 libdir,
                             std::string const&                                 arch,
                             std::string const&                                 dtype,
                             std::vector<origami::nn::olc::manifest_entry_t>& out,
                             std::string&                                       error)
        {
            namespace olc = origami::nn::olc;
            namespace fs = std::filesystem;
            char const* token = fileToken(dtype);
            if(token == nullptr)
            {
                error = "unknown dtype class " + dtype;
                return false;
            }
            std::vector<std::string> files;
            std::error_code          ec;
            for(fs::directory_iterator it(libdir, ec), end; !ec && it != end; it.increment(ec))
            {
                if(!it->is_regular_file(ec))
                    continue;
                const std::string name = it->path().filename().string();
                const bool dat = name.size() > 4
                                 && (name.compare(name.size() - 4, 4, ".dat") == 0
                                     || (name.size() > 9
                                         && name.compare(name.size() - 9, 9, ".dat.zlib") == 0));
                if(name.rfind("TensileLibrary_", 0) != 0 || name.find(arch) == std::string::npos
                   || name.find("_lazy_") != std::string::npos || !dat)
                    continue;
                const auto start = std::string("TensileLibrary_").size();
                if(name.substr(start, name.find('_', start) - start) == token)
                    files.push_back(name);
            }
            if(ec)
            {
                error = "cannot list " + libdir + ": " + ec.message();
                return false;
            }
            std::sort(files.begin(), files.end());

            // Files load on a few threads; each observes the rows it deserializes on its own
            // thread (the observer is thread-local). Every loaded file is alive until its rows
            // are collected, so the thread count also bounds the extra memory.
            using Row     = ProblemPredictionLibrary<ContractionProblemGemm, ContractionSolution>;
            using Kernels = std::vector<olc::library_kernel_t>;
            auto loadOne  = [&](std::string const& path, Kernels& found) {
                const std::function<void(Row const&)> collect = [&](Row const& row) {
                    for(std::size_t i = 0; i < row.solution_list.size(); ++i)
                    {
                        auto const& solution = row.solution_list[i].second;
                        if(!inClass(solution->problemType, dtype))
                            continue;
                        olc::library_kernel_t k;
                        k.name   = solution->name();
                        k.index  = row.solution_list[i].first;
                        k.config = row.origami_config_list[i];
                        k.fields.num_loads_coalesced_a = solution->sizeMapping.NumLoadsCoalescedA;
                        k.fields.num_loads_coalesced_b = solution->sizeMapping.NumLoadsCoalescedB;
                        found.push_back(std::move(k));
                    }
                };
                Row::deserializeObserver = &collect;
                bool ok                  = false;
                try
                {
                    ok = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(path)
                         != nullptr;
                }
                catch(std::exception const&)
                {
                }
                Row::deserializeObserver = nullptr;
                return ok;
            };
            constexpr std::size_t    kMaxLoadThreads = 8;
            const std::size_t        threads         = std::min<std::size_t>(
                {files.size(),
                 std::max<std::size_t>(1, std::thread::hardware_concurrency()),
                 kMaxLoadThreads});
            std::vector<Kernels>     found(files.size());
            std::vector<char>        loadedOk(files.size(), 0);
            std::atomic<std::size_t> next{0};
            auto                     work = [&] {
                for(std::size_t f; (f = next.fetch_add(1)) < files.size();)
                    loadedOk[f] = loadOne((fs::path(libdir) / files[f]).string(), found[f]);
            };
            std::vector<std::thread> pool;
            for(std::size_t t = 1; t < threads; ++t)
                pool.emplace_back(work);
            work();
            for(auto& t : pool)
                t.join();
            std::vector<olc::library_kernel_t> kernels;
            for(std::size_t f = 0; f < files.size(); ++f)
            {
                if(!loadedOk[f])
                {
                    error = "cannot load " + (fs::path(libdir) / files[f]).string();
                    return false;
                }
                kernels.insert(kernels.end(), found[f].begin(), found[f].end());
            }
            if(kernels.empty())
            {
                error = "no " + dtype + " prediction-row kernels in " + libdir;
                return false;
            }
            out = olc::canonical_manifest(kernels);
            return true;
        }

        bool checkLibrary(std::string const&                   libdir,
                          std::string const&                   arch,
                          std::string const&                   modelPath,
                          origami::nn::olc::library_diff_t& diff,
                          std::string&                         error)
        {
            namespace olc = origami::nn::olc;
            olc::status_t status;
            auto         model = olc::ensemble_t::load_file(modelPath, status, error);
            if(!model)
                return false;
            std::vector<olc::manifest_entry_t> library;
            if(!libraryManifest(libdir, arch, model->head_string("dtype"), library, error))
                return false;
            diff = olc::diff_library(*model, library);
            return true;
        }

        origami::nn::model_handle_t loadForLogic(std::string const& logicStem,
                                                 std::string const& libdir)
        {
            namespace nn = origami::nn;
            namespace olc = origami::nn::olc;
            const std::string path
                = nn::index_weights_path(logicStem, nn::backend_id_t::olc_v1, libdir);
            if(path.empty())
                return nn::invalid_handle;

            std::lock_guard<std::mutex> lock(loadMutex());
            if(auto it = loaded().find(path); it != loaded().end())
                return it->second;

            // The stem's arch is its last field (..._gfx950), its dtype class its type token.
            const char*       dtype  = logicClass(logicStem);
            const std::string arch   = logicStem.substr(logicStem.rfind('_') + 1);
            auto              status = olc::status_t::incompatible;
            std::string       detail;
            auto              handle = nn::invalid_handle;
            if(dtype == nullptr)
                detail = "logic file " + logicStem + " is in no dtype class";
            else
            {
                std::string                       error;
                std::vector<olc::manifest_entry_t> library;
                if(!libraryManifest(libdir, arch, dtype, library, error))
                    detail = "cannot read the kernel library: " + error;
                else
                    handle = nn::load_olc_model(
                        path, olc::deployment_t{arch, dtype, &library}, &status, &detail);
            }
            const std::string cls = dtype ? dtype : "?";
            if(handle >= 0)
            {
                if(acceptedSlot())
                    acceptedSlot()("Origami OLC " + cls + " model " + path + ": " + detail);
            }
            else if(refusedSlot())
                refusedSlot()("Origami OLC " + cls + " model rejected (" + olc::to_string(status)
                              + "): " + path + ": " + detail + "; using stock Origami ranking for "
                              + cls);
            loaded()[path] = handle;
            return handle;
        }

        origami::nn::olc::streamk_launch_t streamKLaunch(ContractionSolution const&    solution,
                                                           ContractionProblemGemm const& problem,
                                                           Hardware const&               hardware)
        {
            origami::nn::olc::streamk_launch_t sk;
            AMDGPU const* gpu = dynamic_cast<AMDGPU const*>(&hardware);
            if(gpu == nullptr)
                return sk;
            auto const& sm               = solution.sizeMapping;
            sk.is_streamk                = sm.isStreamK();
            sk.mt_x                      = sm.macroTile.x;
            sk.mt_y                      = sm.macroTile.y;
            sk.workspace_size_per_elem_c = sm.workspaceSizePerElemC;
            sk.cu_occupancy              = sm.CUOccupancy;
            sk.cu_count                  = static_cast<std::uint64_t>(gpu->computeUnitCount);
            if(sk.is_streamk)
            {
                try
                {
                    const auto d      = solution.computeStreamKDecisions(problem, hardware);
                    sk.final_grid     = d.finalGrid;
                    sk.tiles          = d.tiles;
                    sk.sk_tiles       = d.skTiles;
                    sk.sk_split       = d.skSplit;
                    sk.reduction      = d.reduction;
                    sk.is_dynamic     = d.isDynamic;
                    sk.dp_only        = d.dpOnly;
                    sk.iters_per_tile = std::max<size_t>(1, problem.getItersPerTile(sm));
                }
                catch(std::exception const&)
                {
                    return sk;
                }
            }
            sk.valid = true;
            return sk;
        }

        void setLoadLoggers(Logger accepted, Logger refused)
        {
            acceptedSlot() = std::move(accepted);
            refusedSlot()  = std::move(refused);
        }

        void setCallLogger(Logger logger)
        {
            loggerSlot() = std::move(logger);
        }

        Logger const& callLogger()
        {
            return loggerSlot();
        }
    } // namespace OrigamiOLC
} // namespace TensileLite
