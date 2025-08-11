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

#include <memory>
#include <string>

namespace rocRoller
{
    namespace Scheduling
    {
        enum class SchedulerProcedure : int
        {
            Sequential = 0,
            RoundRobin,
            Random,
            Cooperative,
            Priority,
            Count
        };

        enum class Dependency : int
        {
            None = 0, //< Temporary. Should only be used for unlocking.
            Branch, //< Non-preemptible: Loops and ConditionalOp
            M0, //< Preemptible: The M0 special-purpose register
            VCC, //< Preemptible: The VCC special-purpose register
            SCC, //< Non-preemptible: The SCC special-purpose register, which is
            //  implicitly written by many instructions.
            Count
        };

        enum class LockOperation : int
        {
            None = 0,
            Lock,
            Unlock,
            Count
        };

        class Scheduler;
        class LockState;

        using SchedulerPtr = std::shared_ptr<Scheduler>;

        std::string toString(SchedulerProcedure);
        std::string toString(Dependency);
        std::string toString(LockOperation);
    }
}
