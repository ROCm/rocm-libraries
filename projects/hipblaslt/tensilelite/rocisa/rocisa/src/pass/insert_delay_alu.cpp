/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "code.hpp"
#include "enum.hpp"
#include "instruction/common.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocisa
{
    static const std::unordered_map<DelayALUType, unsigned> aluDepMax = {
        // the maximum number of VALU instructions that can be encoded in a delay ALU instruction
        {DelayALUType::VALU, 4},
        // the maximum number of SALU instructions that can be counted backwards from the current instruction
        {DelayALUType::SALU, 1},
        // the maximum number of TRANS instructions that can encode in an s_delay_alu instruction
        {DelayALUType::TRANS, 3},
        // default case for other types of instructions
        {DelayALUType::OTHER, 0},
    };

    static const unsigned SKIP_MAX = 5;

    using _Hash  = decltype([](std::shared_ptr<RegisterContainer> ptr) { return ptr->hash(); });
    using _Equal = decltype([](const std::shared_ptr<RegisterContainer>& a,
                               const std::shared_ptr<RegisterContainer>& b) { return *a == *b; });

    // TODO: If there are enough cycles between dependent instructions, inserting a delay ALU may not be necessary.
    struct DelayInfo
    {
        DelayALUType type;
        unsigned     type_count;
        unsigned     total_count;
    };

    static DelayALUType _getDelayAluType(const std::shared_ptr<Instruction>& inst)
    {
        auto preStr = inst->preStr();
        if(!preStr.compare(0, 4, "v_s_"))
        {
            return DelayALUType::TRANS;
        }
        else if(!preStr.compare(0, 2, "v_"))
        {
            return DelayALUType::VALU;
        }
        else if(!preStr.compare(0, 2, "s_"))
        {
            return DelayALUType::SALU;
        }

        return DelayALUType::OTHER;
    }

    auto _getDstSrcRegs(const std::shared_ptr<Instruction>& inst)
    {
        std::unordered_set<std::shared_ptr<RegisterContainer>, _Hash, _Equal> dsts;
        std::unordered_set<std::shared_ptr<RegisterContainer>, _Hash, _Equal> srcs;
        const std::vector<InstructionInput>*                                  instInputs = nullptr;

        auto commonInst = std::dynamic_pointer_cast<CommonInstruction>(inst);
        if(commonInst)
        {
            if(auto dst = std::dynamic_pointer_cast<RegisterContainer>(commonInst->dst))
            {
                dsts.insert(dst);
            }

            if(auto dst1 = std::dynamic_pointer_cast<RegisterContainer>(commonInst->dst1))
            {
                dsts.insert(dst1);
            }

            instInputs = &commonInst->srcs;
        }

        auto compositeInst = std::dynamic_pointer_cast<CompositeInstruction>(inst);
        if(compositeInst)
        {
            if(auto dst = std::dynamic_pointer_cast<RegisterContainer>(compositeInst->dst))
            {
                dsts.insert(dst);
            }

            instInputs = &compositeInst->srcs;
        }

        if(!instInputs)
        {
            return std::make_pair(dsts, srcs);
        }

        for(const auto& src : *instInputs)
        {
            if(auto ptr = std::get_if<std::shared_ptr<Container>>(&src))
            {
                if(auto reg = std::dynamic_pointer_cast<RegisterContainer>(*ptr))
                {
                    srcs.insert(reg);
                }
            }
        }

        return std::make_pair(dsts, srcs);
    }

    static void _insertDelayAlu(std::shared_ptr<Module> module)
    {
        // recursively process each submodule to insert delay ALU instructions
        for(const auto& item : module->items())
        {
            if(auto submodule = std::dynamic_pointer_cast<Module>(item))
            {
                _insertDelayAlu(submodule);
            }
        }

        auto& items = module->items();
        if(items.size() < 2)
        {
            return; // no need to insert delay ALU
        }

        std::unordered_map<std::shared_ptr<RegisterContainer>, size_t, _Hash, _Equal>
                                                     last_dst_inst_idx;
        std::unordered_map<size_t, DelayInfo>        inst_idx_delay_info;
        std::unordered_map<DelayALUType, DelayInfo>  delay_info;
        std::map<size_t, std::shared_ptr<SDelayAlu>> dep_idxs;
        unsigned                                     total_count = 0;
        inst_idx_delay_info.reserve(items.size());
        for(size_t i = 0; i < items.size(); ++i)
        {
            auto& item = items[i];

            auto inst = std::dynamic_pointer_cast<Instruction>(item);
            if(!inst)
            {
                continue;
            }

            DelayALUType alu_type = _getDelayAluType(inst);
            ++delay_info[alu_type].type_count;
            ++total_count;
            inst_idx_delay_info[i]
                = DelayInfo{alu_type, delay_info[alu_type].type_count, total_count};

            auto [dsts, srcs] = _getDstSrcRegs(inst);

            // TODO: confirm this is best strategy
            // Prefer registers present in last_dst_inst_idx; if both are present, select the one with the higher index.
            auto src_it = std::max_element(
                srcs.begin(), srcs.end(), [&last_dst_inst_idx](const auto& a, const auto& b) {
                    return last_dst_inst_idx.find(a) != last_dst_inst_idx.end()
                           && (last_dst_inst_idx.find(b) == last_dst_inst_idx.end()
                               || last_dst_inst_idx[a] < last_dst_inst_idx[b]);
                });

            do
            {
                if(src_it == srcs.end())
                {
                    break;
                }

                auto src = *src_it;
                if(last_dst_inst_idx.find(src) == last_dst_inst_idx.end())
                {
                    break;
                }

                size_t last_idx     = last_dst_inst_idx[src];
                auto   dep_alu_type = inst_idx_delay_info[last_idx].type;
                auto   inst_cnt     = delay_info[dep_alu_type].type_count
                                - inst_idx_delay_info[last_idx].type_count;
                if(inst_cnt > aluDepMax.at(dep_alu_type))
                {
                    break;
                }

                int skip_cnt = -1;
                // if(!dep_idxs.empty())
                // {
                //     skip_cnt = total_count
                //                - inst_idx_delay_info[dep_idxs.rbegin()->first].total_count - 1;
                // }

                // If there are no dependencies or the last dependency is already set, we can insert a new delay ALU
                if(skip_cnt < 0 || skip_cnt >= SKIP_MAX || dep_idxs.rbegin()->second->hasInstID1())
                {
                    dep_idxs[i] = std::make_shared<SDelayAlu>(dep_alu_type, inst_cnt);
                    break;
                }

                // // pack the current instruction into last delay ALU
                // dep_idxs.rbegin()->second->setInstID1(skip_cnt, dep_alu_type, inst_cnt);
            } while(false);

            if(!dsts.empty())
            {
                for(const auto& dst : dsts)
                {
                    last_dst_inst_idx[dst] = i;
                }
            }
        }

        for(auto it = dep_idxs.rbegin(); it != dep_idxs.rend(); ++it)
        {
            auto [idx, inst] = *it;
            module->add(inst, idx);
        }
    }

    void insertDelayAlu(std::shared_ptr<Module> module)
    {
        if(!rocIsa::getInstance().getAsmCaps()["s_delay_alu"])
        {
            return;
        }

        _insertDelayAlu(module);
    }
} // namespace rocisa
