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

#include <concepts>
#include <optional>

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>
#include <rocRoller/Scheduling/WaitStateHazardCounter.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        using RegisterHazardMap
            = std::unordered_map<Register::RegisterId,
                                 std::vector<Scheduling::WaitStateHazardCounter>,
                                 Register::RegisterIdHash>;

        template <class T>
        concept CWaitStateObserver = requires(T obs, Instruction const& inst)
        {
            requires CObserverConst<T>;

            {
                /*
                 * Get the maximum number of NOPs for the worst case.
                 *
                 * This amount is used to populate the RegisterHazardMap.
                 *
                 * @param inst The instruction that is analyzed for hazard potential.
                 * @return Worst case NOPs required if the hazard is discovered.
                 */
                obs.getMaxNops(inst)
                } -> std::same_as<int>;

            {
                /*
                 * The condition that could trigger the hazard for this rule.
                 *
                 * @param instRef The instruction that is analyzed for hazard potential.
                 * @return True if the instruction could cause a hazard according to this rule.
                 */
                obs.trigger(inst)
                } -> std::same_as<bool>;

            {
                /*
                 * Is this hazard caused by writing a register?
                 *
                 * Note: Create seperate rules if reading and writing both cause hazards.
                 *
                 * True if the hazard is caused by writing registers, False if by reading.
                 */
                obs.writeTrigger()
                } -> std::same_as<bool>;

            {
                /*
                 * Get the number of NOPs for the instruction according to this hazard rule.
                 *
                 * Holds the logic for determining whether an instruction causes a hazard according
                 * to this hazard rule. Should check the RegisterHazardMap to determine if the conditions
                 * for the rule are met. If so, a number of NOPs are returned according to the number in
                 * the RegisterHazardMap. If not, return 0 NOPs.
                 *
                 * @param inst The instruction to be analyzed
                 * @return The number of NOPs this instruction should have according to this rule.
                 */
                obs.getNops(inst)
                } -> std::same_as<int>;

            {
                /*
                * Get a descriptive comment string
                */
                obs.getComment()
                } -> std::same_as<std::string>;
        };

        template <class DerivedObserver>
        class WaitStateObserver
        {
        public:
            WaitStateObserver() {}

            WaitStateObserver(ContextPtr context)
                : m_context(context)
            {
                m_hazardMap = std::make_shared<RegisterHazardMap>();
            }

            /**
             * @brief Peeks at instruction and returns required NOPs
             *
             * @param inst Instruction to be peeked
             * @return InstructionStatus with NOP count that will occur
             */
            InstructionStatus peek(Instruction const& inst) const;

            /**
             * @brief Modify an instruction. Will modify NOPs if there is a wait state hazard.
             *
             * @param inst Instruction to be modified
             */
            void modify(Instruction& inst) const;

            void observe(Instruction const& inst);

        protected:
            std::weak_ptr<Context> m_context;

            // Represents registers and their associated wait state hazards
            std::shared_ptr<RegisterHazardMap> m_hazardMap;

            /**
             * @brief Common observer function.
             * Creates hazards for all src or dst registers (depending on read/write trigger).
             *
             * @param inst Instruction to be observed for wait state hazards
             */
            virtual void observeHazard(Instruction const& inst);

            /**
             * @brief Observe function to decrement the Hazard Counters being tracked
             *
             * @param inst Instruction to be observed
            */
            void decrementHazardCounters(Instruction const& inst);

            /**
             * @brief Get the Nops for an OpCode from a map
             *
             * @param opCode OpCode to be analyzed
             * @param latencyAndNops Map of latency (passes) and required NOPs
             * @return Number of NOPs
             */
            int getNopFromLatency(std::string const&                  opCode,
                                  std::unordered_map<int, int> const& latencyAndNops) const;

            /**
             * @brief Check register for hazards and get number of required NOPs if so.
             * Checks all sub-registers.
             *
             * @param reg Register to be analyzed
             * @return The number of required NOPs. nullopt if there is no hazard found.
             */
            std::optional<int> checkRegister(Register::ValuePtr const& reg) const;

            /**
             * @brief Checks all source registers for hazards and get required NOPs.
             *
             * @param inst Instruction to be analyzed
             * @return The number of required NOPs. nullopt if there is no hazard found.
             */
            std::optional<int> checkSrcs(Instruction const& inst) const;

            /**
             * @brief Checks all destination registers for hazards and get required NOPs.
             *
             * @param inst Instruction to be analyzed
             * @return The number of required NOPs. nullopt if there is no hazard found.
             */
            std::optional<int> checkDsts(Instruction const& inst) const;
        };
    }
}

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver_impl.hpp>
