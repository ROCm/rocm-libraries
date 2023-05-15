#pragma once

#include <any>
#include <bitset>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <rocRoller/Scheduling/Costs/Cost_fwd.hpp>
#include <rocRoller/Scheduling/Scheduler_fwd.hpp>

#include <rocRoller/Utilities/LazySingleton.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>

namespace rocRoller
{
    /**
     * @brief Options are represented by the SettingsOption struct.
     *
     * @tparam T type of underlying option.
     */
    template <typename T>
    struct SettingsOption
    {
        using Type = T;

        std::string name        = "";
        std::string description = "";
        T           defaultValue;
        int         bit = -1;
    };

    std::string ToString(LogLevel level);
    std::string toString(LogLevel level);

    /**
     * @brief Settings class is derived from lazy singleton class and handles options
     * that are defined through environment variables or developer defined options.
     *
     * Getting a value requires a call to get(SettingsOption opt). When get() is called,
     * we probe m_values which maps an option name to its corresponding value. If opt does
     * not exist in m_values then we assign the value based on the following precedence order:
     *     1. the corresponding env variable will be used. If no env variable is set then
     *     2. set(opt, val) will always set, or overwrite, the value of opt in m_values. Otherwise
     *     3. the corresponding bit field value in SettingsBitField is used. Lastly
     *     4. the default value is used if the value is not otherwise obtained.
     */
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
            false,
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

        static inline SettingsOption<Scheduling::SchedulerProcedure> Scheduler{
            "ROCROLLER_SCHEDULER",
            "Scheduler used when scheduling independent instruction streams.",
            Scheduling::SchedulerProcedure::Sequential,
            -1};

        static inline SettingsOption<Scheduling::CostFunction> SchedulerCost{
            "ROCROLLER_SCHEDULER_COST",
            "Default cost function for schedulers.",
            Scheduling::CostFunction::LinearWeighted,
            -1};

        static inline SettingsOption<std::string> SchedulerWeights{
            "ROCROLLER_SCHEDULER_WEIGHTS",
            "Scheduler Weight (YAML) File for LinearWeighted cost function.",
            "",
            -1};

        static inline SettingsOption<LogLevel> LogLvl{
            "ROCROLLER_LOG_LEVEL", "Log level", LogLevel::None, -1};

        static inline SettingsOption<int> RandomSeed{"ROCROLLER_RANDOM_SEED",
                                                     "Seed used with RandomGenerator class",
                                                     std::numeric_limits<int>::min(),
                                                     -1};

        static inline SettingsOption<bool> KernelAnalysis{
            "ROCROLLER_KERNEL_ANALYSIS",
            "Whether to track and report register liveness.",
            false,
            -1};

        static inline SettingsOption<bool> AllowUnkownInstructions{
            "ROCROLLER_ALLOW_UNKNOWN_INSTRUCTIONS",
            "Whether to allow arbitrary instructions.",
            false,
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
            "Bitfield mask to enable/disable other debug options",
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
         * @brief Alias function to convert char* to std::string
         *
         * @tparam Option
         * @param opt
         * @param val
         */
        template <typename Option>
        inline void set(Option const& opt, char const* val);

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
