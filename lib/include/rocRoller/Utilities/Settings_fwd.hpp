#pragma once

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

    class Settings;
}
