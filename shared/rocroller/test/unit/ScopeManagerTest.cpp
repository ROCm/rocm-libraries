/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/ScopeManager.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include "GenericContextFixture.hpp"

namespace ScopeManagerTest
{
    using namespace rocRoller;
    namespace CF = rocRoller::KernelGraph::ControlGraph;
    namespace CT = rocRoller::KernelGraph::CoordinateGraph;

    class ScopeManagerTest : public GenericContextFixture
    {
    };

    TEST_F(ScopeManagerTest, Deallocation)
    {
        // Test a register would not be deleted if it has an associated
        // Deallocate op (see `ScopeManager::popAndReleaseScope`)

        auto                                 kg = std::make_shared<KernelGraph::KernelGraph>();
        rocRoller::KernelGraph::ScopeManager scope(m_context, kg);
        scope.pushNewScope();

        auto dim = kg->coordinates.addElement(KernelGraph::CoordinateGraph::Dimension());
        scope.addRegister(dim);

        // Associate a register with the Dimension
        Register::ValuePtr reg;
        m_context->registerTagManager()->addRegister(dim, reg);

        // Create a Deallocate op associated with the Dimension
        auto deallocate
            = kg->control.addElement(rocRoller::KernelGraph::ControlGraph::Deallocate());
        kg->mapper.connect<KernelGraph::CoordinateGraph::Dimension>(deallocate, dim);

        EXPECT_TRUE(m_context->registerTagManager()->hasRegister(dim));

        scope.popAndReleaseScope();

        // The register should still exist even after the scope is released
        EXPECT_TRUE(m_context->registerTagManager()->hasRegister(dim));
    }
}
