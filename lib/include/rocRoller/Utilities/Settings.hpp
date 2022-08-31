#pragma once

#include <any>
#include <bitset>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <rocRoller/Utilities/LazySingleton.hpp>

namespace rocRoller
{
    template <typename T>
    struct SettingsOption
    {
        using Type = T;

        std::string name        = "";
        std::string description = "";
        T           defaultValue;
        int         bit = -1;
    };

    class Settings : public LazySingleton<Settings>
    {
    public:
        using bitFieldType = std::bitset<32>;

        static inline SettingsOption<bool> LogConsole{
            "ROCROLLER_LOG_CONSOLE", "Log debug info to stdout/stderr", true, 0};

        static inline SettingsOption<bool> SaveAssembly{
            "ROCROLLER_SAVE_ASSEMBLY",
            "Assembly code written to text file in the current working directory as it is "
            "generated",
            true,
            1};

        static inline SettingsOption<bool> BreakOnThrow{
            "ROCROLLER_BREAK_ON_THROW", "Cause exceptions thrown to cause a segfault", false, 2};

        static inline SettingsOption<std::string> ArchitectureFile{
            "ROCROLLER_ARCHITECTURE_FILE",
            "GPU Architecture file",
            "source/rocRoller/GPUArchitecture_def.msgpack",
            -1};

        static inline SettingsOption<std::string> AssemblyFile{
            "ROCROLLER_ASSEMBLY_FILE", "File name to write assembly", "", -1};

        static inline SettingsOption<std::string> LogFile{
            "ROCROLLER_LOG_FILE", "Log file name", "", -1};

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

        static inline SettingsOption<LogLevel> LogLvl{
            "ROCROLLER_LOG_LEVEL", "Log level", LogLevel::None, -1};

        static inline SettingsOption<int> RandomSeed{"ROCROLLER_RANDOM_SEED",
                                                     "Seed used with RandomGenerator class",
                                                     std::numeric_limits<int>::min(),
                                                     -1};

        /**
         * @brief Generate defaultValue for SettingsBitField based on other defaultValues.
         *
         * Called in SettingsBitField struct to be set as its defaultValue.
         */
        static inline bitFieldType constructDefaultBitField()
        {
            bitFieldType bitField(SettingsBitField.defaultValue);

            bitField.set(LogConsole.bit, LogConsole.defaultValue);
            bitField.set(SaveAssembly.bit, SaveAssembly.defaultValue);
            bitField.set(BreakOnThrow.bit, BreakOnThrow.defaultValue);

            return bitField;
        }

        static inline SettingsOption<bitFieldType> SettingsBitField{
            "ROCROLLER_DEBUG",
            "Bitfield mask to enable/disable other debug otpions",
            constructDefaultBitField(),
            -1};

        Settings();

        /**
         * @brief Gets the value of a SettingsOption.
         *
         * Set values for SettingsOption are stored in m_values map,
         * otherwise call getValue().
         *
         * @tparam Option A SettingsOption templated by its defaultValue type.
         * @param opt The SettingsOption we are getting.
         * @return A value of type Option::Type that opt is set to.
         */
        template <typename Option>
        typename Option::Type get(Option const& opt);

        /**
         * @brief Set the value of a SettingsOption.
         *
         * Set the value of a SettingsOption.
         * Mismatching Option::Type and value type throws an error.
         *
         * @tparam Option A SettingsOption templated by its defaultValue type.
         * @tparam T The type of val.
         * @param opt The SettingsOption we are assigning val to.
         * @param val The value we are trying to assign to opt.
         */
        template <typename Option, typename T>
        void set(Option const& opt, T const& val);

        /**
         * @brief Stringify Settings variables
         *
         * @tparam T Settings variable type to stringify. Supported: LogLevel.
         * @param var The Settings varaible being stringified.
         * @return std::string The string conversion of var.
         */
        template <typename T>
        std::string toString(T const& var) const;

    private:
        friend rocRoller::LazySingleton<Settings>;

        /**
         * @brief Get the value of a SettingsOption.
         *
         * The value we get for a SettingsOption follows the precedence:
         *   - explicit call to set() will set the value, if not
         *   - corresponding env var will be used, if no env var then
         *   - coresponding bit in SettingsBitField will be used, if bit < 0 then
         *   - default value set in struct is used.
         *
         * @tparam Option SettingsOption templated by its defaultValue type.
         * @param opt The Settingsption whose value is being fetched.
         * @return A value of type Option::Type that opt will be set to.
         */
        template <typename Option>
        typename Option::Type getValue(Option const& opt);

        /**
         * @brief Extract the value of an Option from the bit field.
         *
         * SettingsOption with bit >= 0 will probe SettingsBitField for their value.
         *
         * @tparam Option SettingsOption templated by its value type.
         * @param opt SettingsOption whose value we are extracting from SettingsBitField.
         * @return Corresponding bool value of bit or defaultValue if not in SettingsBitField.
         */
        template <typename Option>
        typename Option::Type extractBitValue(Option const& opt);

        /**
         * @brief Get correct corresponding typed value of a string.
         *
         * @tparam T The type the string will be converted to.
         * @param var The string whose value will be converted.
         * @return T The typed conversion of the string.
         */
        template <typename T>
        T getTypeValue(std::string const& var) const;

        std::map<std::string, std::any> m_values;
        std::vector<std::string>        setBitOptions;
    };
}

#include "Settings_impl.hpp"
