/**
 * @copyright Copyright 2022 Advanced Micro Devices, Inc.
 */

#pragma once

#include "CopyGenerator.hpp"
#include "Instruction.hpp"

#include "../Context.hpp"
#include "../Utilities/Generator.hpp"

namespace rocRoller
{

    class MemoryInstructions
    {
    public:
        MemoryInstructions(std::shared_ptr<Context> context);

        MemoryInstructions(MemoryInstructions const& rhs) = default;
        MemoryInstructions(MemoryInstructions&& rhs)      = default;
        ~MemoryInstructions()                             = default;

        enum MemoryKind
        {
            Flat,
            Scalar,
            Local
        };

        /**
         * @brief Generate the instructions required to perform a load.
         *
         *
         * @param kind The kind of memory operation to perform.
         * @param dest The register to store the loaded data in.
         * @param addr  The register containing the address to load the data from.
         * @param offset String containing an offset to be added to addr.
         * @param numBytes The number of bytes to load.
         */
        Generator<Instruction> load(MemoryKind                       kind,
                                    std::shared_ptr<Register::Value> dest,
                                    std::shared_ptr<Register::Value> addr,
                                    std::shared_ptr<Register::Value> offset,
                                    int                              numBytes,
                                    std::string                      comment = "");

        /**
         * @brief Generate the instructions required to perform a store.
         *
         *
         * @param kind The kind of memory operation to perform.
         * @param addr The register containing the address to store the data.
         * @param data  The register containing the data to store.
         * @param offset String containing an offset to be added to addr.
         * @param numBytes The number of bytes to load.
         */
        Generator<Instruction> store(MemoryKind                       kind,
                                     std::shared_ptr<Register::Value> addr,
                                     std::shared_ptr<Register::Value> data,
                                     std::shared_ptr<Register::Value> offset,
                                     int                              numBytes,
                                     std::string                      comment = "");

        /**
         * @brief Generate the instructions required to perform a flat load.
         *
         *
         * @param dest The register to store the loaded data in.
         * @param addr  The register containing the address to load the data from.
         * @param offset String containing an offset to be added to addr.
         * @param numBytes The number of bytes to load.
         */
        Generator<Instruction> loadFlat(std::shared_ptr<Register::Value> dest,
                                        std::shared_ptr<Register::Value> addr,
                                        std::string                      offset,
                                        int                              numBytes);

        /**
         * @brief Generate the instructions required to perform a flat store.
         *
         *
         * @param addr The register containing the address to store the data.
         * @param data  The register containing the data to store.
         * @param offset String containing an offset to be added to addr.
         * @param numBytes The number of bytes to load.
         */
        Generator<Instruction> storeFlat(std::shared_ptr<Register::Value> addr,
                                         std::shared_ptr<Register::Value> data,
                                         std::string                      offset,
                                         int                              numBytes);

        /**
         * @brief Generate the instructions required to load scalar data into a register from memory.
         *
         *
         * @param dest The register to store the loaded data in.
         * @param base  The register containing the address to load the data from.
         * @param offset The value containing an offset to be added to the address in base.
         * @param numBytes The total number of bytes to load.
         */
        Generator<Instruction> loadScalar(std::shared_ptr<Register::Value> dest,
                                          std::shared_ptr<Register::Value> base,
                                          std::shared_ptr<Register::Value> offset,
                                          int                              numBytes);

        /**
         * @brief Generate the instructions required to perform an LDS load.
         *
         *
         * @param dest The register to store the loaded data in.
         * @param addr  The register containing the address to load the data from.
         * @param offset Offset to be added to addr.
         * @param numBytes The number of bytes to load.
         */
        Generator<Instruction> loadLocal(std::shared_ptr<Register::Value> dest,
                                         std::shared_ptr<Register::Value> addr,
                                         std::string                      offset,
                                         int                              numBytes,
                                         std::string                      comment = "");

        /**
         * @brief Generate the instructions required to perform an LDS store.
         *
         *
         * @param addr The register containing the address to store the data.
         * @param data  The register containing the data to store.
         * @param offset Offset to be added to addr.
         * @param numBytes The number of bytes to load.
         */
        Generator<Instruction> storeLocal(std::shared_ptr<Register::Value> addr,
                                          std::shared_ptr<Register::Value> data,
                                          std::string                      offset,
                                          int                              numBytes,
                                          std::string                      comment = "");

        /**
         * @brief Generate the instructions required to add a memory barrier.
         *
         * @return Generator<Instruction>
         */
        Generator<Instruction> barrier();

    private:
        std::weak_ptr<Context> m_context;

        std::string            genOffsetModifier(std::string offset) const;
        Generator<Instruction> genLocalAddr(std::shared_ptr<Register::Value>& addr) const;
    };
}

#include "MemoryInstructions_impl.hpp"
