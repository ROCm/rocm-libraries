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

namespace rocRoller
{
    namespace Scheduling
    {
        class WaitStateHazardCounter
        {
        public:
            WaitStateHazardCounter() {}
            WaitStateHazardCounter(int maxRequiredNops, bool written)
                : m_counter(maxRequiredNops)
                , m_maxCounter(maxRequiredNops)
                , m_written(written)
            {
            }

            /**
             * @brief Decrements the counter
             *
             * @param nops Number of NOPs associated with the decrement
             */
            void decrement(int nops)
            {
                m_counter--;
                m_counter -= nops;
            }

            /**
             * @brief Determine if the counter has reach the end of life
             *
             * @return true if this still represents a hazard
             * @return false if the counter can be safely removed
             */
            bool stillAlive() const
            {
                return m_counter > 0;
            }

            int getRequiredNops() const
            {
                return m_counter;
            }

            int getMaxNops() const
            {
                return m_maxCounter;
            }

            bool regWasWritten() const
            {
                return m_written;
            }

            bool regWasRead() const
            {
                return !m_written;
            }

        private:
            int  m_counter    = 0;
            int  m_maxCounter = 0;
            bool m_written    = false;
        };
    }
}
