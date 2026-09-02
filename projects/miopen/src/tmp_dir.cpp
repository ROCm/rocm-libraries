// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/tmp_dir.hpp>
#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/process.hpp>
#include <miopen/unique_path.hpp>

#include <thread>
#include <string_view>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_SAVE_TEMP_DIR)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_EXIT_STATUS_TEMP_DIR)

namespace miopen {

TmpDir::TmpDir(std::string_view prefix) : path{fs::temp_directory_path()}
{
    std::string p{prefix.empty() ? "" : (prefix[0] == '-' ? "" : "-")};

    path /= miopen::unique_path("miopen" + p.append(prefix) + "-%%%%-%%%%-%%%%-%%%%");

    fs::create_directories(path);
}

int TmpDir::Execute(std::string_view cmd, std::string_view args, bool allowChangingCwd) const
{
    if(env::enabled(MIOPEN_DEBUG_SAVE_TEMP_DIR))
    {
        MIOPEN_LOG_I2(path);
    }
    const auto status = Process{cmd}(args, allowChangingCwd ? path : "");
    if(env::enabled(MIOPEN_DEBUG_EXIT_STATUS_TEMP_DIR))
    {
        MIOPEN_LOG_I2(status);
    }
    return status;
}

TmpDir::~TmpDir()
{
    if(!env::enabled(MIOPEN_DEBUG_SAVE_TEMP_DIR))
    {
#ifdef _WIN32
        const constexpr int remove_max_retries{5};
        int count{0};
        while(count < remove_max_retries)
        {
            try
            {
                fs::remove_all(path);
                break;
            }
            catch(const fs::filesystem_error& err)
            {
                MIOPEN_LOG_W(err.what());
                std::this_thread::sleep_for(std::chrono::milliseconds{125});
            }
            ++count;
        }
#else
        if(!this->path.empty())
            fs::remove_all(this->path);
#endif
    }
}

} // namespace miopen
