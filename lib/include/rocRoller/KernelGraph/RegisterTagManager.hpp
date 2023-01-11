#pragma once

#include <map>
#include <memory>

#include "RegisterTagManager_fwd.hpp"

#include "Context_fwd.hpp"
#include "DataTypes/DataTypes.hpp"
#include "InstructionValues/Register_fwd.hpp"
#include "Operations/Command_fwd.hpp"

namespace rocRoller
{
    class RegisterTagManager
    {
    public:
        RegisterTagManager(ContextPtr context);
        ~RegisterTagManager();

        std::shared_ptr<Register::Value> getRegister(int tag);

        std::shared_ptr<Register::Value> getRegister(int            tag,
                                                     Register::Type regType,
                                                     VariableType   varType,
                                                     size_t         ValueCount = 1);

        std::shared_ptr<Register::Value> getRegister(int tag, Register::ValuePtr tmpl);

        /**
         * @brief Add a register to the RegisterTagManager with the provided tag.
         *
         * @param tag The tag the of the register
         * @param value The register value to be added
         */
        void addRegister(int tag, Register::ValuePtr value);

        void deleteRegister(int tag);

        /**
         * @brief Returns whether or not a register has already been added to the
         *        Register Manager.
         *
         * @param tag
         * @return true
         * @return false
         */
        bool hasRegister(int tag) const;

    private:
        std::weak_ptr<Context>                          m_context;
        std::map<int, std::shared_ptr<Register::Value>> m_registers;
    };
}

#include "RegisterTagManager_impl.hpp"
