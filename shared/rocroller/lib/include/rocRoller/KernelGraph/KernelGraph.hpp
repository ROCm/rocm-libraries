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

#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/CommandSolution_fwd.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/KernelGraph/Constraints.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform_fwd.hpp>

namespace rocRoller
{

    /**
     * @ingroup rocRoller
     * @defgroup KernelGraph Kernel Graph
     *
     * @brief Graph (both coordinate and control) representation of a GPU kernel.
     */

    namespace KernelGraph
    {
        /**
         * @brief Kernel graph container: control and coordinate graphs, control-to-coordinate mapper.
         * @ingroup KernelGraph
         */
        class KernelGraph
        {
            std::vector<GraphConstraint> m_constraints{
                &NoDanglingMappings, &SingleControlRoot, &NoRedundantSetCoordinates};
            std::vector<std::string> m_transforms;

        public:
            ControlGraph::ControlGraph       control;
            CoordinateGraph::CoordinateGraph coordinates;
            ControlToCoordinateMapper        mapper;

            std::string toDOT(bool drawMappings = false, std::string title = "") const;

            template <typename T>
            std::pair<int, T> getDimension(int                         controlIndex,
                                           Connections::ConnectionSpec conn) const;

            template <typename T>
            std::pair<int, T> getDimension(int controlIndex, NaryArgument arg) const;

            template <typename T>
            std::pair<int, T> getDimension(int controlIndex, int subDimension = 0) const;

            /**
             * @brief Check the input constraints against the KernelGraph's current state.
             *
             * @param constraints
             * @return ConstraintStatus
             */
            ConstraintStatus
                checkConstraints(const std::vector<GraphConstraint>& constraints) const;

            /**
             * @brief Check the KernelGraph's global constraints against its current state.
             *
             * @return ConstraintStatus
             */
            ConstraintStatus checkConstraints() const;

            /**
             * @brief Add the given constraints to the KernelGraph's global constraints.
             *
             * @param constraints
             */
            void addConstraints(const std::vector<GraphConstraint>& constraints);

            std::vector<GraphConstraint> getConstraints() const;

            /**
             * @brief Returns new KernelGraph given a particular transformation
             *
             * @param GraphTransform
            */
            KernelGraph transform(GraphTransformPtr const& transform);

            std::vector<std::string> const& appliedTransforms() const;
            void addAppliedTransforms(std::vector<std::string> const& transforms);
        };

        /**
         * Translate from Command to (initial) KernelGraph.
         *
         * Resulting KernelGraph matches the Command operations
         * closely.
         *
         * @ingroup KernelGraph
         */
        KernelGraph translate(CommandPtr);

        /**
         * Generate assembly from a KernelGraph.
         *
         * @ingroup KernelGraph
         */
        Generator<Instruction> generate(KernelGraph, AssemblyKernelPtr);

        std::string toYAML(KernelGraph const& g);
        KernelGraph fromYAML(std::string const& str);

    }
}

#include <rocRoller/KernelGraph/KernelGraph_impl.hpp>
