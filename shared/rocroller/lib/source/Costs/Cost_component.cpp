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

#include <rocRoller/Scheduling/Costs/Cost.hpp>
#include <rocRoller/Scheduling/Costs/LinearWeightedCost.hpp>
#include <rocRoller/Scheduling/Costs/MinNopsCost.hpp>
#include <rocRoller/Scheduling/Costs/NoneCost.hpp>
#include <rocRoller/Scheduling/Costs/UniformCost.hpp>
#include <rocRoller/Scheduling/Costs/WaitCntNopCost.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    template <>
    void Component::ComponentFactory<Scheduling::Cost>::registerImplementations()
    {
        registerComponent<Scheduling::LinearWeightedCost>();
        registerComponent<Scheduling::MinNopsCost>();
        registerComponent<Scheduling::NoneCost>();
        registerComponent<Scheduling::UniformCost>();
        registerComponent<Scheduling::WaitCntNopCost>();
    }
}
