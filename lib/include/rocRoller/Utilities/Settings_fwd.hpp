#pragma once

#include <string>

namespace rocRoller
{
    enum class LogLevel
    {
        None = 0,
        Error,
        Warning,
        Terse,
        Verbose,
        Debug,
        Count //Count is a special Enum entry acting as the "size" of enum LogLevel
    };

    enum class F8Mode
    {
        NaNoo,
        OCP,
        Count
    };

    std::string toString(F8Mode);

    F8Mode getDefaultF8ModeForCurrentHipDevice();

    const char* getDefaultArchitectureFilePath();

    class Settings;
}
