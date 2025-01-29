#pragma once

namespace rocRoller
{
    enum class LogLevel
    {
        None = 0,
        Critical,
        Error,
        Warning,
        Terse,
        Info,
        Verbose,
        Debug,
        Trace,
        Count //Count is a special Enum entry acting as the "size" of enum LogLevel
    };

    class Settings;
}
