#pragma once

#include <sstream>
#include <vector>

#include "CodeGen/Instruction_fwd.hpp"
#include "Context_fwd.hpp"
#include "ExecutableKernel.hpp"

namespace rocRoller
{
    class ScheduledInstructions
    {
    public:
        explicit ScheduledInstructions(std::shared_ptr<Context> ctx);
        ~ScheduledInstructions() = default;

        void schedule(const Instruction& instruction);

        std::string toString() const;
        /**
         * @brief Get the Executable Kernel object for the currently scheduled instructions.
         *
         * @return std::shared_ptr<ExecutableKernel> Returns a pointer to the Executable Kernel
         */
        std::shared_ptr<ExecutableKernel> getExecutableKernel();

        /**
         * @brief Assemble the currently scheduled instructions.
         *
         * @return std::vector<char> The binary representation of the assembled instructions.
         */
        std::vector<char> assemble() const;
        void              clear();

    private:
        std::ostringstream     m_instructionstream;
        std::weak_ptr<Context> m_context;
    };

}
