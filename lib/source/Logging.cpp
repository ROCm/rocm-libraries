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
#include "rocRoller/Utilities/Settings_fwd.hpp"

#include <spdlog/cfg/helpers.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace rocRoller
{
    namespace Log
    {
        bool initLogger()
        {
            auto settings = Settings::getInstance();

            std::vector<spdlog::sink_ptr> sinks;

            if(settings->get(Settings::LogConsole))
            {
                auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
                sinks.push_back(consoleSink);
            }

            std::string logFile = settings->get(Settings::LogFile);
            if(!logFile.empty())
            {
                auto fileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                    logFile, 5 * 1024 * 1024, 5);
                sinks.push_back(fileSink);
            }

            auto defaultLog = std::make_shared<spdlog::logger>(
                "rocRollerLog", std::begin(sinks), std::end(sinks));

            spdlog::set_default_logger(defaultLog);

            LogLevel    logLevel   = settings->get(Settings::LogLvl);
            std::string s_logLevel = settings->toString(logLevel);
            if(s_logLevel != "None")
            {
                spdlog::cfg::helpers::load_levels(s_logLevel);
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
