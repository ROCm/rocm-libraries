/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/
#include "hipblaslt_bench_options.hpp"
#ifdef HIPBLASLT_ENABLE_JIT
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#endif

namespace hipblaslt_bench_options
{
    bool& jit_gemm()
    {
        static bool value = false;
        return value;
    }

    std::string& jit_output_dir()
    {
        static std::string value;
        return value;
    }

#ifdef HIPBLASLT_ENABLE_JIT
    hipblaslt_ext::experimental::jit::tensilelite::Options jit_generate_options()
    {
        namespace fs    = std::filesystem;
        auto configured = [](const char* name, const char* fallback) {
            const char* value = std::getenv(name);
            return std::string(value && *value ? value : fallback);
        };
        hipblaslt_ext::experimental::jit::tensilelite::Options options;
        options.pythonExecutable = configured("HIPBLASLT_JIT_PYTHON", HIPBLASLT_JIT_PYTHON);
        options.tensileSourceDirectory
            = configured("HIPBLASLT_JIT_TENSILE_SOURCE", HIPBLASLT_JIT_TENSILE_SOURCE);
        options.pythonPath     = configured("HIPBLASLT_JIT_PYTHONPATH", HIPBLASLT_JIT_PYTHONPATH);
        options.cxxCompiler    = configured("HIPBLASLT_JIT_CXX", HIPBLASLT_JIT_CXX);
        options.offloadBundler = configured("HIPBLASLT_JIT_OFFLOAD_BUNDLER", HIPBLASLT_JIT_BUNDLER);
        // Empty configPath requests Origami prediction; the runtime supplies the device ISA.
        try
        {
            const fs::path root = jit_output_dir().empty()
                                      ? fs::temp_directory_path()
                                      : fs::absolute(fs::u8path(jit_output_dir()));
            fs::create_directories(root);
            std::random_device random;
            bool               created = false;
            for(int attempt = 0; attempt < 128 && !created; ++attempt)
            {
                const auto directory = root
                                       / ("hipblaslt-jit-" + std::to_string(random()) + "-"
                                          + std::to_string(random()));
                if(fs::create_directory(directory))
                {
                    const auto path = (directory / "solution").u8string();
                    options.outputPath.assign(path.begin(), path.end());
                    created = true;
                }
            }
            if(!created)
                throw std::invalid_argument("Cannot create a unique JIT artifact directory");
        }
        catch(const fs::filesystem_error& error)
        {
            throw std::invalid_argument(std::string("Cannot create JIT artifact directory: ")
                                        + error.what());
        }
        return options;
    }
#endif

    int32_t& sm_count_target()
    {
        static int32_t v = 0;
        return v;
    }

    int32_t& streamk_tile_scheduling_mode()
    {
        static int32_t v = -1;
        return v;
    }

    std::string& streamk_tile_scheduling_mode_str()
    {
        static std::string v;
        return v;
    }

    int32_t& uniform_summation_order()
    {
        static int32_t v = -1;
        return v;
    }

    std::string& uniform_summation_order_str()
    {
        static std::string v;
        return v;
    }
}
