#pragma once

#include "KernelGraph_fwd.hpp"

#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/CommandSolution_fwd.hpp>
#include <rocRoller/Context.hpp>

#include <rocRoller/KernelGraph/Constraints.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /*
         * Kernel graph containers
         */

        class KernelGraph
        {
            std::vector<GraphConstraint> m_constraints{&NoDanglingMappings};

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

            /**
             * @brief Returns new KernelGraph given a particular transformation
             *
             * @param GraphTransform
            */
            KernelGraph transform(std::shared_ptr<GraphTransform> const&);
        };

        /**
         * Translate from Command to (initial) KernelGraph.
         *
         * Resulting KernelGraph matches the Command operations
         * closely.
         */
        KernelGraph translate(CommandPtr);

        /*
         * Code generation
         */
        Generator<Instruction> generate(KernelGraph, std::shared_ptr<AssemblyKernel>);

        std::string toYAML(KernelGraph const& g);
        KernelGraph fromYAML(std::string const& str);

    }
}

#include "KernelGraph_impl.hpp"
