// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <deque>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>

namespace rocRoller::Scheduling::LDSModel
{
    enum class LdsDirection
    {
        Read,
        Write
    };

    struct MemoryOpLDS
    {
        LdsDirection direction;
    };

    using MemoryOp = std::variant<MemoryOpLDS>;

    struct Summary
    {
        struct Banks
        {
            uint   bankIndex;
            size_t workitemsAccessed;
            bool   imbalanced;
        };
        struct Access
        {
            int                           ldsTag;
            std::vector<Banks>            accessedBanks;
            std::vector<std::vector<int>> banksToWorkitems;
        };

        std::map<int, Access> tagToAccess;
        std::set<int>         imbalancedTags;

        std::string toString() const;
    };

    std::ostream& operator<<(std::ostream& stream, Summary const& summary);

    struct RuntimeLDSInstruction
    {
        MemoryOpLDS         memoryOp;
        int                 dwords;
        std::vector<size_t> baseAddresses;

        std::string toString() const;
    };

    std::ostream& operator<<(std::ostream& stream, RuntimeLDSInstruction const& instruction);

    struct OperationAccesses
    {
        int                                operationTag;
        int                                ldsTag;
        std::vector<RuntimeLDSInstruction> instructions;
    };

    struct OperationsAnalysis
    {
        std::map<int, OperationAccesses> tagToAccess;
        GPUArchitectureGFX               gfx;

        std::string toString() const;
    };

    std::ostream& operator<<(std::ostream& stream, OperationsAnalysis const& operationAnalysis);

    struct LDSBankAccess
    {
        int          operationTag;
        int          ldsTag;
        LdsDirection direction;
        uint         workitem;
        uint         bankIndex;
    };

    /**
     * @brief Calculate the data cycles for an LDS instruction
     *
     * I.e. how long does the instruction take once it is at the front of the LDS queue?
     *
     * @param instr The LDS instruction
     * @param gfx The GPU architecture
     */
    uint getInstructionDataCycles(const RuntimeLDSInstruction& instr, GPUArchitectureGFX gfx);

    /**
     * @brief Get the number of cycles to issue an instruction (stalls excluded)
     *
     * I.e. how long does instruction take to enter the queue (issue) when there's sufficient space in the LDS queue?
     *
     * @param memoryOp The memory operation (read or write)
     * @param dwords Number of dwords being accessed (1 for b32, 2 for b64, 3 for b96, 4 for b128)
     * @return Number of cycles to issue the instruction
     */
    uint getInstructionIssueCycles(const MemoryOpLDS& memoryOp, uint dwords);

    /**
     * @brief Get the equilibrium cycles when continuously issuing the same instruction
     *
     * @param instr The LDS instruction with memory operation details and addresses
     * @param gfx The GPU architecture
     * @return Total number of cycles for the instruction
     */
    uint getInstructionCycles(const RuntimeLDSInstruction& instr, GPUArchitectureGFX gfx);

    /**
     * @brief Get the number of LDS banks for a given GPU architecture and memory operation
     *
     * @param gfx The GPU architecture
     * @param memoryOp The memory operation (read or write)
     * @param dwords Number of dwords (1 for b32, 2 for b64, 3 for b96, 4 for b128)
     * @return Number of LDS banks (64 for ds_read_b64/b128 on GFX950, otherwise 32)
     */
    uint getNumLDSBanks(GPUArchitectureGFX gfx, const MemoryOpLDS& memoryOp, uint dwords);

    /**
     * @brief Returns the number of threads that can operate per clock for a given memory operation
     *
     * @param memoryOp Read/write
     * @param dwords Number of dwords (1 for b32, 2 for b64, 3 for b96, 4 for b128)
     * @param gfx The GPU architecture
     */
    uint getThreadsPerClock(const MemoryOpLDS& memoryOp, uint dwords, GPUArchitectureGFX gfx);

    /**
     * @brief Divide addresses into thread groups based on threadsPerClock limit
     *
     * @param addresses Vector of base addresses to divide into groups
     * @param threadsPerClock Maximum number of threads that can operate per clock
     * @return Vector of thread groups, each containing addresses for that group
     */
    std::vector<std::vector<size_t>> divideIntoThreadGroups(const std::vector<size_t>& addresses,
                                                            uint threadsPerClock);

    /**
     * @brief Determines how many addresses accesses each bank
     *
     * For multi-dword accesses, includes addresses at starting at the baseAddress and
     * extending for dwords number of dwords.
     *
     * @param baseAddresses Vector of base addresses.
     * @param dwords Number of dwords accessed starting from each base address
     * @param gfx The GPU architecture
     * @param memoryOp The memory operation (read or write)
     * @return A map, where the key is the bank index and the value is the count of addresses accessing that bank
     */
    std::map<uint, uint> createBankToAddressCounts(const std::vector<size_t>& baseAddresses,
                                                   uint                       dwords,
                                                   GPUArchitectureGFX         gfx,
                                                   const MemoryOpLDS&         memoryOp);

    /**
     * @brief Calculate the number of clock cycles needed to resolve bank conflicts
     *
     * Simulates the bank conflict resolution process where only one address per bank
     * can be serviced per clock cycle.
     *
     * @param bankToAddressCounts Map from bank index to count of addresses accessing that bank
     * @return Number of clock cycles
     */
    uint calculateBankConflictCycles(const std::map<uint, uint>& bankToAddressCounts);

    /**
     * @brief Compute bank-to-address-count mappings for each thread group
     *
     * This function divides the instruction's base addresses into thread groups
     * and computes the bank-to-address-count mapping for each group.
     *
     * @param instr The LDS instruction containing memory operation details and addresses
     * @param gfx The GPU architecture
     * @return Vector of bank-to-address count mappings, one per thread group
     */
    std::vector<std::map<uint, uint>>
        computeThreadGroupBankMappings(const RuntimeLDSInstruction& instr, GPUArchitectureGFX gfx);

    /**
     * @brief Calculate total cycles from bank mappings
     *
     * Takes the per-thread-group bank mappings and calculates the total
     * number of cycles needed, including bank conflict resolution
     *
     * @param threadGroupBankMappings Vector of bank to address count mappings
     * @return Total number of clock cycles
     */
    uint calculateTotalCyclesFromBankMappings(
        const std::vector<std::map<uint, uint>>& threadGroupBankMappings);

    /**
     * @brief Check if an instruction's wave accesses non-overlapping banks across cycles
     *
     * The hardware will optimize this by having the waves access out-of-phase,
     * resulting in no penalty from bank conflicts in cases with more than 1 active SIMD.
     *
     * @param instr The LDS instruction containing memory operation details and addresses
     * @param gfx The GPU architecture
     * @return true if each cycle accesses different banks (no overlap), false otherwise
     */
    bool hasNonOverlappingBankAccess(const RuntimeLDSInstruction& instr, GPUArchitectureGFX gfx);

    /**
     * @brief Parse LDS instruction information from opcode
     *
     * @param opCode The opcode string (e.g., "ds_read_b64", "ds_write_b128")
     * @return Optional pair of (LdsDirection, dwords) if opcode is supported, nullopt otherwise
     */
    std::optional<std::pair<LdsDirection, int>>
        getLdsInfoFromOpcodeIfSupported(const std::string& opCode);

    /**
     * @brief Calculate queue slots required for an LDS operation
     *
     * @param direction Read or Write operation
     * @param dwords Number of dwords (1-4)
     * @return Number of queue slots required
     */
    int getQueueSlotsRequired(LdsDirection direction, int dwords);

    /**
     * @brief LDS Scheduler for managing LDS instruction scheduling and queue state
     *
     * This class encapsulates the scheduling logic for LDS instructions,
     * managing command and data queues, tracking conflicts, and predicting stalls.
     * It can be used independently of the observer system.
     *
     * Note: This scheduler works in hardware cycles, not quadcycles.
     */
    class LDSModule
    {
    public:
        LDSModule(GPUArchitectureGFX gfx, int waveCount);

        void incrementProgramCycleBy(int cycles);
        int  getProgramCycle() const
        {
            return m_programCycle;
        }
        void reset();

        // Returns tuple of (stall_cycles, additional_cycles)
        std::tuple<int, int> predictCycles(const RuntimeLDSInstruction& instr) const;
        void                 scheduleInstruction(const RuntimeLDSInstruction& instr);

        int predictWaitcntStallCycles(int waitcnt) const;

        void updateQueues();

    private:
        // gfx942 to gfx950
        // actually 40 deep 16 entries wide -> simplify to 10 deep 64 entries wide
        static constexpr int dataQueueSize    = 10;
        static constexpr int commandQueueSize = 8;

        // Before gfx942, data queue should be 64 deep 16 entries wide, command queue remains same

        GPUArchitectureGFX m_gfx;
        int                m_programCycle;
        int                m_multiplierQueueSlots;
        int                m_multiplierWaveCount;

        std::deque<int> m_commandQueue;
        std::deque<int> m_waitcntQueue; // Includes round-trip delay
        std::deque<int> m_dataQueue;

        int getRemainingDataSlots() const;
    };
}
