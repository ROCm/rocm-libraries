// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

    TEST_F(ScopeManagerTest, ScopeDeallocates)
    {
        // Test a register would be deleted if it does not have an associated
        // Deallocate op (see `ScopeManager::popAndReleaseScope`)

        auto                                 kg = std::make_shared<KernelGraph::KernelGraph>();
        rocRoller::KernelGraph::ScopeManager scope(m_context, kg);
        scope.pushNewScope();

        auto dim = kg->coordinates.addElement(KernelGraph::CoordinateGraph::Dimension());
        scope.addRegister(dim);

        // Associate a register with the Dimension
        Register::ValuePtr reg
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);
        m_context->registerTagManager()->addRegister(dim, reg);

        EXPECT_TRUE(m_context->registerTagManager()->hasRegister(dim));

        scope.popAndReleaseScope();

        // The register should not exist after the scope is released
        EXPECT_FALSE(m_context->registerTagManager()->hasRegister(dim));
    }

    TEST_F(ScopeManagerTest, ScopeDoesntDeallocateIfDeallocateNodeExists)
    {
        // Test a register would not be deleted if it has an associated
        // Deallocate op (see `ScopeManager::popAndReleaseScope`)

        auto                                 kg = std::make_shared<KernelGraph::KernelGraph>();
        rocRoller::KernelGraph::ScopeManager scope(m_context, kg);
        scope.pushNewScope();

        auto dim = kg->coordinates.addElement(KernelGraph::CoordinateGraph::Dimension());
        scope.addRegister(dim);

        // Associate a register with the Dimension
        Register::ValuePtr reg
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);
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
