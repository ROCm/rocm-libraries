// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/CommandSolution_fwd.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/KernelGraph/Constraints.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowArgumentTracer_fwd.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
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
            std::vector<GraphConstraint> m_constraints{&NoDanglingMappings,
                                                       &SingleControlRoot,
                                                       &NoRedundantSetCoordinates,
                                                       &WalkableControlGraph};
            std::vector<std::string>     m_transforms;

            std::unordered_map<int, rocRoller::KernelGraph::CoordinateGraph::Transformer>
                m_transformers;

            /**
             * @brief Set expression using a ForLoop op in a given transformer
             *
             */
            void setTransformerByForLoopOp(CoordinateGraph::Transformer& transformer,
                                           int                           forLoopOp);

            /**
             * @brief Set expression using a SetCoordinate op in a given transformer
             *
             */
            void setTransformerBySetCoordinate(CoordinateGraph::Transformer& transformer,
                                               int                           setCoordinateOp);

        public:
            ControlGraph::ControlGraph       control;
            CoordinateGraph::CoordinateGraph coordinates;
            ControlToCoordinateMapper        mapper;

            /**
            * Modelled addresses for ops with memory access.
            * Set by @ref rocRoller::KernelGraph::ModelAddresses transform.
            */
            std::unordered_map<int, std::vector<size_t>> modelledAddresses;

            /**
            * Set up the coordinate graph and transducer for existing transformers.
            */
            void initializeTransformersForCodeGen(rocRoller::Expression::ExpressionTransducer);

            /**
            *  Build a transformer for a given operation in control graph if the transformer
            *  does not exist, otherwise return existing transformer.
            */
            CoordinateGraph::Transformer buildTransformer(int op);

            /**
            *  Build a transformer for a given operation in control graph. If the transformer
            *  exists, it will be re-built.
            */
            CoordinateGraph::Transformer buildTransformer(int op, IgnoreCachePolicy const);

            /**
            *  Build transformers for all operations in control graph. Rebuild transformers
            *  if they already exist.
            */
            void buildAllTransformers();

            /**
            *  Set expression of an op's transformer using a given coordinate and expression.
            */
            void updateTransformer(int op, int coord, Expression::ExpressionPtr expr);

            std::unordered_map<int, CoordinateGraph::Transformer> const& getAllTransformers() const
            {
                return m_transformers;
            }

            /**
            *  Set both control and coordinate graphs to be restricted mode.
            */
            void setRestricted()
            {
                control.setRestricted();
                coordinates.setRestricted();
            }

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
        KernelGraph translate(CommandPtr, CommandParametersPtr params = nullptr);

        /**
         * Generate assembly from a KernelGraph.
         *
         * @ingroup KernelGraph
         */
        Generator<Instruction> generate(KernelGraph graph, AssemblyKernelPtr kernel);

        /**
         * Testing: Supply a specific ControlFlowArgumentTracer.
         *
         * @ingroup KernelGraph
         * @ingroup Testing
         */
        Generator<Instruction> generate(KernelGraph                 graph,
                                        AssemblyKernelPtr           kernel,
                                        ControlFlowArgumentTracer&& argTracer);

        std::string toYAML(KernelGraph const& g);
        KernelGraph fromYAML(std::string const& str);

    }
}

#include <rocRoller/KernelGraph/KernelGraph_impl.hpp>
