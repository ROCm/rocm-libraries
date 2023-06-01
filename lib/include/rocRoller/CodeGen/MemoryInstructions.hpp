/**
 * @copyright Copyright 2022 Advanced Micro Devices, Inc.
 */

#pragma once

#include "Buffer.hpp"
#include "BufferInstructionOptions.hpp"
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
            Local,
            Buffer
        };

        enum MemoryDirection : int
        {
            Load = 0,
            Store
        };

        /**
         * @brief Generate the instructions required to perform a load.
         *
         *
         * @param kind The kind of memory operation to perform.
         * @param dest The register to store the loaded data in.
         * @param addr  The register containing the address to load the data from.
         * @param offset Register containing an offset to be added to addr.
         * @param numBytes The number of bytes to load.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         * @param buffDesc Buffer descriptor to use when `kind` is `Buffer`. (Default = nullptr)
         * @param buffOpts Buffer options. (Default = BufferInstructionOptions())
         */
        Generator<Instruction> load(MemoryKind                        kind,
                                    std::shared_ptr<Register::Value>  dest,
                                    std::shared_ptr<Register::Value>  addr,
                                    std::shared_ptr<Register::Value>  offset,
                                    int                               numBytes,
                                    std::string                       comment  = "",
                                    bool                              high     = false,
                                    std::shared_ptr<BufferDescriptor> buffDesc = nullptr,
                                    BufferInstructionOptions buffOpts = BufferInstructionOptions());

        /**
         * @brief Generate the instructions required to perform a store.
         *
         *
         * @param kind The kind of memory operation to perform.
         * @param addr The register containing the address to store the data.
         * @param data  The register containing the data to store.
         * @param offset Register containing an offset to be added to addr.
         * @param numBytes The number of bytes to store.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         * @param buffDesc Buffer descriptor to use when `kind` is `Buffer`. (Default = nullptr)
         * @param buffOpts Buffer options. (Default = BufferInstructionOptions())
         */
        Generator<Instruction> store(MemoryKind                        kind,
                                     std::shared_ptr<Register::Value>  addr,
                                     std::shared_ptr<Register::Value>  data,
                                     std::shared_ptr<Register::Value>  offset,
                                     int                               numBytes,
                                     std::string const                 comment  = "",
                                     bool                              high     = false,
                                     std::shared_ptr<BufferDescriptor> buffDesc = nullptr,
                                     BufferInstructionOptions          buffOpts
                                     = BufferInstructionOptions());

        /**
         * @brief Generate the instructions required to perform a load or store.
         *
         * @tparam Dir Whether to load or store data
         * @param kind The kind of memory operation to perform.
         * @param addr The register containing the address.
         * @param data  The register containing the data.
         * @param offset Register containing an offset to be added to addr.
         * @param numBytes The number of bytes.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         * @param high Whether the value will be loaded or stored into the high bits of the register. (Default=false)
         * @param buffDesc Buffer descriptor to use when `kind` is `Buffer`. (Default = nullptr)
         * @param buffOpts Buffer options. (Default = BufferInstructionOptions())
         */
        template <MemoryDirection Dir>
        Generator<Instruction> moveData(MemoryKind                        kind,
                                        std::shared_ptr<Register::Value>  addr,
                                        std::shared_ptr<Register::Value>  data,
                                        std::shared_ptr<Register::Value>  offset,
                                        int                               numBytes,
                                        std::string const                 comment  = "",
                                        bool                              high     = false,
                                        std::shared_ptr<BufferDescriptor> buffDesc = nullptr,
                                        BufferInstructionOptions          buffOpts
                                        = BufferInstructionOptions());

        /**
         * @brief Generate instructions that will load two 16bit values and pack them into
         *        a single register.
         *
         * @param kind The kind of memory operation to perform.
         * @param dest The register to store the loaded data in.
         * @param addr1 The register containing the address of the first 16bit value to load the data from.
         * @param offset1 Register containing an offset to be added to addr1.
         * @param addr2 The register containing the address of the second 16bit value to load the data from.
         * @param offset2 Register containing an offset to be added to addr2.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         * @param buffDesc Buffer descriptor to use when `kind` is `Buffer`. (Default = nullptr)
         * @param buffOpts Buffer options. (Default = BufferInstructionOptions())
         */
        Generator<Instruction> loadAndPack(MemoryKind                        kind,
                                           std::shared_ptr<Register::Value>  dest,
                                           std::shared_ptr<Register::Value>  addr1,
                                           std::shared_ptr<Register::Value>  offset1,
                                           std::shared_ptr<Register::Value>  addr2,
                                           std::shared_ptr<Register::Value>  offset2,
                                           std::string const                 comment  = "",
                                           std::shared_ptr<BufferDescriptor> buffDesc = nullptr,
                                           BufferInstructionOptions          buffOpts
                                           = BufferInstructionOptions());

        /**
         * @brief Generate instructions that will pack 2 16bit values into a single 32bit register and store the value
         *        to memmory.
         *
         * @param kind The kind of memory operation to perform.
         * @param addr The register containing the address to store the data.
         * @param data1 The register containing the first 16bit value to store.
         * @param data2 The register containing the second 16bit value to store.
         * @param offset Register containing an offset to be added to addr.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         */
        Generator<Instruction> packAndStore(MemoryKind                       kind,
                                            std::shared_ptr<Register::Value> addr,
                                            std::shared_ptr<Register::Value> data1,
                                            std::shared_ptr<Register::Value> data2,
                                            std::shared_ptr<Register::Value> offset,
                                            std::string const                comment = "");

        /**
         * @brief Generate the instructions required to perform a flat load.
         *
         *
         * @param dest The register to store the loaded data in.
         * @param addr  The register containing the address to load the data from.
         * @param offset Offset to be added to addr.
         * @param numBytes The number of bytes to load.
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         */
        Generator<Instruction> loadFlat(std::shared_ptr<Register::Value> dest,
                                        std::shared_ptr<Register::Value> addr,
                                        int                              offset,
                                        int                              numBytes,
                                        bool                             high = false);

        /**
         * @brief Generate the instructions required to perform a flat store.
         *
         *
         * @param addr The register containing the address to store the data.
         * @param data  The register containing the data to store.
         * @param offset Offset to be added to addr.
         * @param numBytes The number of bytes to load.
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         */
        Generator<Instruction> storeFlat(std::shared_ptr<Register::Value> addr,
                                         std::shared_ptr<Register::Value> data,
                                         int                              offset,
                                         int                              numBytes,
                                         bool                             high = false);

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
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         */
        Generator<Instruction> loadLocal(std::shared_ptr<Register::Value> dest,
                                         std::shared_ptr<Register::Value> addr,
                                         int                              offset,
                                         int                              numBytes,
                                         std::string const                comment = "",
                                         bool                             high    = false);

        /**
         * @brief Generate the instructions required to perform an LDS store.
         *
         *
         * @param addr The register containing the address to store the data.
         * @param data  The register containing the data to store.
         * @param offset Offset to be added to addr.
         * @param numBytes The number of bytes to load.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         */
        Generator<Instruction> storeLocal(std::shared_ptr<Register::Value> addr,
                                          std::shared_ptr<Register::Value> data,
                                          int                              offset,
                                          int                              numBytes,
                                          std::string const                comment = "",
                                          bool                             high    = false);

        /**
         * @brief Generate the instructions required to perform a buffer load.
         *
         *
         * @param dest The register to store the loaded data in.
         * @param addr  The register containing the address to load the data from.
         * @param offset Offset to be added to addr.
         * @param buffDesc Buffer descriptor to use.
         * @param buffOpts Buffer options
         * @param numBytes The number of bytes to load.
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         */
        Generator<Instruction> loadBuffer(std::shared_ptr<Register::Value>  dest,
                                          std::shared_ptr<Register::Value>  addr,
                                          int                               offset,
                                          std::shared_ptr<BufferDescriptor> buffDesc,
                                          BufferInstructionOptions          buffOpts,
                                          int                               numBytes,
                                          bool                              high = false);

        /**
         * @brief Generate the instructions required to perform a buffer store.
         *
         *
         * @param data  The register containing the data to store.
         * @param addr The register containing the address to store the data.
         * @param offset Offset to be added to addr.
         * @param buffDesc Buffer descriptor to use
         * @param buffOpts Buffer options
         * @param numBytes The number of bytes to load.
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         */
        Generator<Instruction> storeBuffer(std::shared_ptr<Register::Value>  data,
                                           std::shared_ptr<Register::Value>  addr,
                                           int                               offset,
                                           std::shared_ptr<BufferDescriptor> buffDesc,
                                           BufferInstructionOptions          buffOpts,
                                           int                               numBytes,
                                           bool                              high = false);

        /**
         * @brief Generate the instructions required to add a memory barrier.
         *
         * @return Generator<Instruction>
         */
        Generator<Instruction> barrier();

    private:
        const int m_wordSize = 4;

        std::weak_ptr<Context> m_context;

        int chooseWidth(int numWords, const std::vector<int>& potentialWidths, int maxWidth) const;
        std::string            genOffsetModifier(int) const;
        Generator<Instruction> genLocalAddr(std::shared_ptr<Register::Value>& addr) const;
        Generator<Instruction> packForStore(Register::ValuePtr& result,
                                            Register::ValuePtr  toPack) const;
    };

    std::string   toString(MemoryInstructions::MemoryDirection const& d);
    std::ostream& operator<<(std::ostream& stream, MemoryInstructions::MemoryDirection n);
}

#include "MemoryInstructions_impl.hpp"
