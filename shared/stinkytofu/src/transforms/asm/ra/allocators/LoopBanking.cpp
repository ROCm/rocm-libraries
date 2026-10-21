// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The banks greedy-compact-loopbank offers the values a loop names.
//
// A loop pays an s_set_vgpr_msb whenever a slot's bank changes between the
// instructions that use it. ProducerPinMatrix fixes the matrix operands, so
// what is left to choose is the bank of everything else the loop names: the
// addresses, the counters, the data staged through LDS. Each such block gets
// the bank the loop's own count (LoopMsbCost) likes best.
//
// One bank per block, not per group of blocks that meet in a slot: in a real
// loop every slot sees most of the loop's values in turn, so the transitive
// closure of "met in a slot" is one or two groups holding nearly everything,
// which can neither fit in a bank nor follow the pinned operands they sit
// next to.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "GreedyPlacement.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/transforms/asm/ra/LoopMsbCost.hpp"

namespace stinkytofu {
namespace {

constexpr size_t kNoBlock = static_cast<size_t>(-1);

/// Rounds of the bank search. Each keeps only moves that lower the count, so
/// the search ends; the bound only caps how long a long tail of small gains
/// may take.
constexpr int kSearchRounds = 4;

/// Most blocks one search tries. Every try walks the whole loop again, so the
/// search grows with blocks times loop length. A loop that names thousands of
/// movable blocks is an FMA loop with no pinned operand to follow, which weight
/// order already keeps in few banks; on the corpus none of those searches
/// found anything.
constexpr size_t kMaxSearchedBlocks = 512;

}  // namespace

std::vector<LoopBanking> loopBankings(const AllocationContext& context,
                                      const std::vector<Block>& blocks,
                                      const std::vector<size_t>& blockOf) {
    const LoopMsbCost* cost = context.loopMsb;
    if (cost == nullptr || !cost->hasLoops()) return {};

    // Only the banks a compact colouring needs anyway, and those pinned values
    // hold: past both, a moved value raises the highest register, which can
    // cost a wave.
    const uint32_t peak = context.intervals.peakPressure(RegType::V);
    uint32_t needed = std::max<uint32_t>(1, (peak + kVgprBankSize - 1) / kVgprBankSize);
    for (const Block& block : blocks) {
        if (block.regClass == RegType::V && block.pinReason != nullptr &&
            block.hintBase.has_value())
            needed = std::max(needed, (*block.hintBase + block.width - 1) / kVgprBankSize + 1);
    }
    const uint32_t banks = std::min(context.target.indexCount(RegType::V) / kVgprBankSize, needed);
    if (banks == 0) return {};

    const auto blockOfValue = [&](SSAValueID value) {
        return value < blockOf.size() ? blockOf[value] : kNoBlock;
    };
    const auto movable = [&](size_t index) {
        if (index == kNoBlock) return false;
        const Block& block = blocks[index];
        return block.regClass == RegType::V && block.pinReason == nullptr &&
               !block.maxIndex.has_value();
    };

    // The movable blocks the loops name, in block order so the same function
    // always gets the same answer.
    std::vector<bool> named(blocks.size(), false);
    cost->forEachSlotValue([&](SSAValueID value) {
        const size_t index = blockOfValue(value);
        if (movable(index)) named[index] = true;
    });
    std::vector<size_t> searched;
    for (size_t index = 0; index < blocks.size(); ++index) {
        if (named[index]) searched.push_back(index);
    }
    if (searched.empty() || searched.size() > kMaxSearchedBlocks) return {};

    // Registers in use per bank over time from what cannot move: pinned values
    // where they were lifted, capped ones in the first bank. Each member is one
    // register over its own live segments, so values that take turns in one
    // register are not counted twice.
    SlotIndex slots = 0;
    for (const Block& block : blocks) {
        for (const Member& member : block.members) {
            for (const LiveSegment& segment : context.intervals.rangeOf(member.value).segments())
                slots = std::max(slots, segment.end + 1);
        }
    }
    std::vector<std::vector<int32_t>> fixed(banks, std::vector<int32_t>(slots + 1, 0));
    for (const Block& block : blocks) {
        if (block.regClass != RegType::V) continue;
        const bool pinned = block.pinReason != nullptr && block.hintBase.has_value();
        if (!pinned && !block.maxIndex.has_value()) continue;
        for (const Member& member : block.members) {
            const uint32_t bank = pinned ? (*block.hintBase + member.offset) / kVgprBankSize : 0;
            if (bank >= banks) continue;
            for (const LiveSegment& segment : context.intervals.rangeOf(member.value).segments()) {
                ++fixed[bank][segment.start];
                --fixed[bank][segment.end];
            }
        }
    }
    for (std::vector<int32_t>& profile : fixed) {
        for (size_t slot = 1; slot < profile.size(); ++slot) profile[slot] += profile[slot - 1];
    }

    // One bank for each searched block, and the room that leaves.
    struct Trial {
        std::vector<std::vector<int32_t>> live;
        std::vector<uint32_t> bankOf;
    };
    // A member's register, plus the one alignment skips after an odd tuple,
    // carried by its first member.
    const auto unitsOf = [](const Block& block, size_t member) {
        return member == 0 && block.width > 1 && block.width % 2 != 0 ? 2 : 1;
    };
    const auto fits = [&](const Trial& trial, size_t index, uint32_t bank) {
        const Block& block = blocks[index];
        for (size_t member = 0; member < block.members.size(); ++member) {
            const int32_t units = unitsOf(block, member);
            for (const LiveSegment& segment :
                 context.intervals.rangeOf(block.members[member].value).segments()) {
                for (SlotIndex slot = segment.start; slot < segment.end; ++slot) {
                    if (trial.live[bank][slot] + units > static_cast<int32_t>(kVgprBankSize))
                        return false;
                }
            }
        }
        return true;
    };
    const auto occupy = [&](Trial& trial, size_t index, uint32_t bank, int32_t sign) {
        const Block& block = blocks[index];
        for (size_t member = 0; member < block.members.size(); ++member) {
            const int32_t units = sign * unitsOf(block, member);
            for (const LiveSegment& segment :
                 context.intervals.rangeOf(block.members[member].value).segments()) {
                for (SlotIndex slot = segment.start; slot < segment.end; ++slot)
                    trial.live[bank][slot] += units;
            }
        }
        trial.bankOf[index] = bank;
    };

    // The count with each searched value somewhere in its bank, and anything
    // else where it was lifted from, which is where pinned and held values stay.
    std::vector<bool> isSearched(blocks.size(), false);
    for (size_t index : searched) isSearched[index] = true;
    const auto countOf = [&](const Trial& trial) {
        return cost->mainLoopFirstCount(
            [&](SSAValueID value, const StinkyRegister& reg) -> uint32_t {
                const size_t index = blockOfValue(value);
                if (index != kNoBlock && isSearched[index])
                    return trial.bankOf[index] * kVgprBankSize;
                const std::optional<RegKey> hint = context.constraints.hintFor(value);
                return hint.has_value() ? hint->idx : reg.reg.idx;
            });
    };

    // Block by block, the bank that lowers the count, among those it fits.
    // Every move is kept only if it helps, so the search never ends above where
    // it started.
    const auto search = [&](Trial& trial) {
        uint64_t best = countOf(trial);
        for (int round = 0; round < kSearchRounds; ++round) {
            bool moved = false;
            for (size_t index : searched) {
                const uint32_t from = trial.bankOf[index];
                uint32_t keep = from;
                occupy(trial, index, from, -1);
                for (uint32_t bank = 0; bank < banks; ++bank) {
                    if (bank == from || !fits(trial, index, bank)) continue;
                    trial.bankOf[index] = bank;
                    const uint64_t count = countOf(trial);
                    if (count < best) {
                        best = count;
                        keep = bank;
                        moved = true;
                    }
                }
                occupy(trial, index, keep, +1);
            }
            if (!moved) break;
        }
    };

    // Two starts, since the search only goes downhill from wherever it
    // begins. The producer's banks, which are legal by construction, so only a
    // move has to find room; a block put past the banks in use starts in the
    // highest one it fits. And the lowest bank each block fits, taken in the
    // order blocks start, which is where compacting would put the loop's
    // values: a loop whose producer spread them over two banks can be cheaper
    // there than anywhere a search from the producer's banks reaches, though
    // a bank packed that full is also the likeliest to have no register free
    // when placement gets there. A start with a block that has no producer
    // register, or no room in any bank, is not tried.
    const auto producerStart = [&]() -> std::optional<Trial> {
        Trial trial{fixed, std::vector<uint32_t>(blocks.size(), 0)};
        for (size_t index : searched) {
            const Block& block = blocks[index];
            if (!block.hintBase.has_value()) return std::nullopt;
            uint32_t bank = *block.hintBase / kVgprBankSize;
            if (bank >= banks) {
                bank = banks;
                while (bank > 0 && !fits(trial, index, bank - 1)) --bank;
                if (bank == 0) return std::nullopt;
                --bank;
            }
            occupy(trial, index, bank, +1);
        }
        return trial;
    };
    const auto lowestStart = [&]() -> std::optional<Trial> {
        Trial trial{fixed, std::vector<uint32_t>(blocks.size(), 0)};
        std::vector<size_t> order = searched;
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return blocks[a].start < blocks[b].start; });
        for (size_t index : order) {
            uint32_t bank = 0;
            while (bank < banks && !fits(trial, index, bank)) ++bank;
            if (bank == banks) return std::nullopt;
            occupy(trial, index, bank, +1);
        }
        return trial;
    };

    // Both, the producer's first, and each even when the search beat nothing:
    // its banks are then the best it knows. Compacting with no banks at all is
    // greedy-compact's colouring, which a chain can offer beside this one.
    std::vector<LoopBanking> bankings;
    const auto offer = [&](std::optional<Trial> trial) {
        if (!trial.has_value()) return;
        search(*trial);
        LoopBanking banking(blocks.size());
        for (size_t index : searched) banking[index] = trial->bankOf[index];
        if (std::find(bankings.begin(), bankings.end(), banking) == bankings.end())
            bankings.push_back(std::move(banking));
    };
    offer(producerStart());
    offer(lowestStart());
    return bankings;
}

}  // namespace stinkytofu
