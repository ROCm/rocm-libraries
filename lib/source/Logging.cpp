/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
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

#include <iostream>

#include "rocRoller/Utilities/Error.hpp"
#include "rocRoller/Utilities/Logging.hpp"

#include <spdlog/cfg/helpers.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace rocRoller
{
    namespace Log
    {
        const std::string ENV_LOG_CONSOLE = "ROCROLLER_LOG_CONSOLE";
        const std::string ENV_LOG_FILE    = "ROCROLLER_LOG_FILE";
        const std::string ENV_LOG_LEVEL   = "ROCROLLER_LOG_LEVEL";

        bool initLogger()
        {
            char* envLogConsole = getenv(ENV_LOG_CONSOLE.c_str());
            char* envLogFile    = getenv(ENV_LOG_FILE.c_str());
            char* envLogLevel   = getenv(ENV_LOG_LEVEL.c_str());

            std::vector<spdlog::sink_ptr> sinks;

            if(!envLogConsole || std::string(envLogConsole) != "0")
            {
                auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
                sinks.push_back(consoleSink);
            }

            if(envLogFile)
            {
                auto fileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                    envLogFile, 5 * 1024 * 1024, 5);
                sinks.push_back(fileSink);
            }

            auto defaultLog = std::make_shared<spdlog::logger>(
                "rocRollerLog", std::begin(sinks), std::end(sinks));

            spdlog::set_default_logger(defaultLog);

            if(envLogLevel)
            {
                spdlog::cfg::helpers::load_levels(envLogLevel);
            }

            return true;
        }

        std::shared_ptr<spdlog::logger> getLogger()
        {
            bool doneInit = initLogger();
            AssertFatal(doneInit, "Logger failed to initialize");

            auto defaultLog = spdlog::get("rocRollerLog");
            return defaultLog;
        }
    }
}
