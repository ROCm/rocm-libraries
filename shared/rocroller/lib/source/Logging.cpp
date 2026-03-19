// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <spdlog/cfg/helpers.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace rocRoller
{
    namespace Log
    {
        spdlog::level::level_enum convertLogLevel(LogLevel level)
        {
            switch(level)
            {
            case LogLevel::Trace:
                return spdlog::level::level_enum::trace;
            case LogLevel::Debug:
                return spdlog::level::level_enum::debug;
            case LogLevel::Verbose:
            case LogLevel::Info:
                return spdlog::level::level_enum::info;
            case LogLevel::Terse:
                // does not correspond to a spdlog log level
                // fall through to less verbose option
            case LogLevel::Warning:
                return spdlog::level::level_enum::warn;
            case LogLevel::Error:
                return spdlog::level::level_enum::err;
            case LogLevel::Critical:
                return spdlog::level::level_enum::critical;
            case LogLevel::None:
            default:
                return spdlog::level::level_enum::off;
            }
        }

        bool initLogger()
        {
            auto settings = Settings::getInstance();

            std::vector<spdlog::sink_ptr> sinks;

            if(settings->get(Settings::LogConsole))
            {
                auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_st>();
                auto level       = settings->get(Settings::LogConsoleLvl);
                if(level != LogLevel::None)
                {
                    consoleSink->set_level(convertLogLevel(level));
                }
                sinks.push_back(consoleSink);
            }

            std::string logFile = settings->get(Settings::LogFile);
            if(!logFile.empty())
            {
                auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_st>(logFile, true);
                auto level    = settings->get(Settings::LogFileLvl);
                if(level != LogLevel::None)
                {
                    fileSink->set_level(convertLogLevel(level));
                }
                sinks.push_back(fileSink);
            }

            auto defaultLog = std::make_shared<spdlog::logger>(
                "rocRollerLog", std::begin(sinks), std::end(sinks));
            defaultLog->set_pattern("[rr/%l] %v");

            spdlog::set_default_logger(defaultLog);

            LogLevel logLevel = settings->get(Settings::LogLvl);
            if(logLevel != LogLevel::None)
            {
                defaultLog->set_level(convertLogLevel(logLevel));
            }

            return true;
        }

        std::shared_ptr<spdlog::logger> getSpdLogger()
        {
            static bool doneInit = initLogger();
            AssertFatal(doneInit, "Logger failed to initialize");

            auto defaultLog = spdlog::get("rocRollerLog");
            return defaultLog;
        }

        LoggerPtr getLogger()
        {
            return std::make_shared<logger>();
        }

        void logger::log(LogLevel level, const std::string& str)
        {
            auto spdlog_level = convertLogLevel(level);
            getSpdLogger()->log(spdlog_level, str);
        }

        bool logger::should_log(LogLevel level) const
        {
            auto spdlog_level = convertLogLevel(level);
            return getSpdLogger()->should_log(spdlog_level);
        }
    }
}
