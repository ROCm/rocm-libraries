/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#pragma once

#include <any>
#include <bitset>
#include <functional>
#include <limits>
#include <map>
#include <shared_mutex>
#include <string>
#include <vector>

#include <rocRoller/Assemblers/Assembler_fwd.hpp>
#include <rocRoller/Scheduling/Costs/Cost_fwd.hpp>
#include <rocRoller/Scheduling/Scheduler_fwd.hpp>
#include <rocRoller/Utilities/Concepts.hpp>
#include <rocRoller/Utilities/LazySingleton.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    /**
     * @brief Base class for SettingsOption.
     *
     * Keeps track of its instances, so do not create temporaries.
    */
    struct SettingsOptionBase
    {
        std::string name;
        std::string description;

        SettingsOptionBase(std::string name, std::string description);
        virtual std::string             help() const;
        virtual std::optional<std::any> getFromEnv() const  = 0;
        virtual std::any                getDefault() const  = 0;
        virtual int                     getBitIndex() const = 0;

        /**
         * @brief Getter for instances.
        */
        static std::vector<SettingsOptionBase const*> const& instances();

    private:
        inline static std::vector<SettingsOptionBase const*> m_instances;
    };

    /**
     * @brief Options are represented by the SettingsOption struct.
     *
     * @tparam T type of underlying option.
     */
    template <typename T, bool LazyValue = false>
    struct SettingsOption : public SettingsOptionBase
    {
        using Type = T;
        using DefaultValueType =
            typename std::conditional<LazyValue, std::function<T(void)>, T>::type;
        DefaultValueType defaultValue;
        int              bit;

        SettingsOption(std::string      name,
                       std::string      description,
                       DefaultValueType defaultValue,
                       int              bit);

        std::string             help() const;
        std::optional<std::any> getFromEnv() const;
        std::any                getDefault() const;
        T                       getTypeValue(std::string const&) const;
        T                       getValue() const;
        int                     getBitIndex() const;
    };

    std::string   toString(LogLevel level);
    std::ostream& operator<<(std::ostream&, LogLevel const&);

    /**
     * @brief Settings class is derived from lazy singleton class and handles options
     * that are defined through environment variables or developer defined options.
     *
     * Getting a value requires a call to get(SettingsOption opt). When get() is called,
     * we probe m_values which maps an option name to its corresponding value. If opt does
     * not exist in m_values then we assign the value based on the following precedence order:
     *     1. the corresponding env variable will be used. If no env variable is set then
     *     2. the default value is used if the value is not otherwise obtained.
     *
     * A call to set(SettingsOption opt) sets (or overwrites) the corresponding value in m_values.
     */
    class Settings : public LazySingleton<Settings>
    {
    public:
        using bitFieldType = std::bitset<32>;

        static inline const SettingsOption<bool> LogConsole{
            "ROCROLLER_LOG_CONSOLE", "Log debug info to stdout/stderr", true, 0};

        static inline const SettingsOption<bool> SaveAssembly{
            "ROCROLLER_SAVE_ASSEMBLY",
            "Assembly code written to text file in the current working directory as it is "
            "generated",
            false,
            1};

        static inline const SettingsOption<bool> BreakOnThrow{
            "ROCROLLER_BREAK_ON_THROW", "Cause exceptions thrown to cause a segfault", false, 2};

        static inline const SettingsOption<std::string> ROCMPath{
            "ROCM_PATH", "Path where ROCM is installed", "/opt/rocm", -1};

        static inline const SettingsOption<std::string> ArchitectureFile{
            "ROCROLLER_ARCHITECTURE_FILE", "GPU Architecture file", "", -1};

        static inline const SettingsOption<std::string> AssemblyFile{
            "ROCROLLER_ASSEMBLY_FILE", "File name to write assembly", "", -1};

        static inline const SettingsOption<std::string> LogFile{
            "ROCROLLER_LOG_FILE", "Log file name", "", -1};

        static inline const SettingsOption<Scheduling::SchedulerProcedure> Scheduler{
            "ROCROLLER_SCHEDULER",
            "Scheduler used when scheduling independent instruction streams.",
            Scheduling::SchedulerProcedure::Sequential,
            -1};

        static inline const SettingsOption<Scheduling::CostFunction> SchedulerCost{
            "ROCROLLER_SCHEDULER_COST",
            "Default cost function for schedulers.",
            Scheduling::CostFunction::LinearWeighted,
            -1};

        static inline const SettingsOption<std::string> SchedulerWeights{
            "ROCROLLER_SCHEDULER_WEIGHTS",
            "Scheduler Weight (YAML) File for LinearWeighted cost function.",
            "",
            -1};

        static inline const SettingsOption<LogLevel> LogLvl{
            "ROCROLLER_LOG_LEVEL", "Log level", LogLevel::None, -1};

        static inline const SettingsOption<LogLevel> LogFileLvl{
            "ROCROLLER_LOG_FILE_LEVEL", "Log level for output to the log file", LogLevel::None, -1};

        static inline const SettingsOption<LogLevel> LogConsoleLvl{
            "ROCROLLER_LOG_CONSOLE_LEVEL",
            "Log level for output to the console",
            LogLevel::None,
            -1};

        static inline const SettingsOption<int> RandomSeed{"ROCROLLER_RANDOM_SEED",
                                                           "Seed used with RandomGenerator class",
                                                           std::numeric_limits<int>::min(),
                                                           -1};

        static inline const SettingsOption<bool> KernelAnalysis{
            "ROCROLLER_KERNEL_ANALYSIS",
            "Whether to track and report register liveness.",
            false,
            -1};

        static inline const SettingsOption<bool> AllowUnkownInstructions{
            "ROCROLLER_ALLOW_UNKNOWN_INSTRUCTIONS",
            "Whether to allow arbitrary instructions.",
            false,
            -1};

        static inline const SettingsOption<bool> EnforceGraphConstraints{
            "ROCROLLER_ENFORCE_GRAPH_CONSTRAINTS",
            "Whether to enforce kernel graph constraints. (Could negatively impact code gen "
            "performance)",
            false,
            -1};

        static inline const SettingsOption<bool> LogGraphs{
            "ROCROLLER_LOG_GRAPHS", "Whether to log graphs after each lowering stage.", true, -1};

        static inline const SettingsOption<AssemblerType> KernelAssembler{
            "ROCROLLER_ASSEMBLER", "Which assembler method to use", AssemblerType::Subprocess, -1};

        static inline const SettingsOption<std::string> SubprocessAssemblerPath{
            "ROCROLLER_SUBPROCESS_ASSEMBLER_PATH",
            "Path to subprocess assembler",
            "/opt/rocm/llvm/bin/clang++",
            -1};

        static inline const std::string BitfieldName = "ROCROLLER_DEBUG";

        static inline const SettingsOption<F8Mode, /*LazyValue*/ true> F8ModeOption{
            "ROCROLLER_SET_F8MODE",
            "F8 mode that can be either NaNoo or OCP",
            getDefaultF8ModeForCurrentHipDevice,
            -1};

        /**
         * @brief Creates a help dialog for the environment variables with
         * their names, default values and bit-offset (if it has one).
         */
        std::string help() const;

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
         * @brief Gets the default value of a SettingsOption.
         *
         * @tparam Option A SettingsOption templated by its defaultValue type.
         * @param opt The SettingsOption we are getting.
         * @return A value of type Option::Type is defaulted to.
         */
        template <typename Option>
        static typename Option::Type getDefault(Option const& opt);

        /**
         * Static convenience function to get a settings value.
        */
        template <typename Option>
        static typename Option::Type Get(Option const& opt);

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

        using MapReaderLock = std::shared_lock<std::shared_mutex>;
        using MapWriterLock = std::unique_lock<std::shared_mutex>;

        std::shared_mutex               m_mapLock;
        std::map<std::string, std::any> m_values;
        std::vector<std::string>        m_setBitOptions;
    };
}

#include <rocRoller/Utilities/Settings_impl.hpp>
