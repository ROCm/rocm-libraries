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
#pragma once
#include "base.hpp"
#include "container.hpp"
#include "enum.hpp"
#include "format.hpp"
#include "helper.hpp"

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

using InstructionInput = std::variant<std::shared_ptr<rocisa::Container>, int, double, std::string>;

namespace rocisa
{
    inline std::string InstructionInputToString(const InstructionInput& input)
    {
        return std::visit(
            [](auto&& arg) -> std::string {
                using T = std::decay_t<decltype(arg)>;
                if constexpr(std::is_same_v<T, std::shared_ptr<rocisa::Container>>)
                {
                    return arg->toString();
                }
                else if constexpr(std::is_same_v<T, int>)
                {
                    return std::to_string(arg);
                }
                else if constexpr(std::is_same_v<T, double>)
                {
                    char buffer[32];
                    snprintf(buffer, sizeof(buffer), "%.17g", arg);
                    auto s = std::string(buffer);
                    if(s.find('.') == std::string::npos && s.find('e') == std::string::npos)
                    {
                        s += ".0";
                    }
                    return s;
                }
                else if constexpr(std::is_same_v<T, std::string>)
                {
                    return arg;
                }
            },
            input);
    }

    inline InstructionInput copyInstructionInput(const InstructionInput& src)
    {
        InstructionInput dst;
        std::visit(
            [&dst](auto&& arg) -> void {
                using T = std::decay_t<decltype(arg)>;
                if constexpr(std::is_same_v<T, std::shared_ptr<rocisa::Container>>)
                {
                    dst = arg->clone();
                }
                else
                {
                    dst = arg;
                }
            },
            src);
        return std::move(dst);
    }

    inline void splitSrcs(const std::vector<InstructionInput>& srcs,
                          std::vector<InstructionInput>&       srcs1,
                          std::vector<InstructionInput>&       srcs2)
    {
        for(const auto& s : srcs)
        {
            std::visit(
                [&srcs1, &srcs2](auto&& arg) -> void {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr(std::is_same_v<T, std::shared_ptr<rocisa::Container>>)
                    {
                        auto regContainer = std::dynamic_pointer_cast<RegisterContainer>(arg);

                        auto [r1, r2] = regContainer->splitRegContainer();
                        srcs1.push_back(r1);
                        srcs2.push_back(r2);
                    }
                    else
                    {
                        srcs1.push_back(arg);
                        srcs2.push_back(arg);
                    }
                },
                s);
        }
    }

    struct Instruction : public Item
    {
        InstType    instType;
        std::string comment;
        std::string instStr;
        bool        outputInlineAsm;

        Instruction(InstType instType, const std::string& comment = "")
            : instType(instType)
            , comment(comment)
            , instStr("")
            , outputInlineAsm(false)
        {
        }

        Instruction(const Instruction& other)
            : Item(other)
            , instType(other.instType)
            , comment(other.comment)
            , instStr(other.instStr)
            , outputInlineAsm(other.outputInlineAsm)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            std::cout << typeid(*this).name() << std::endl;
            throw std::runtime_error("You should override clone function in derived class");
            return std::make_shared<Item>("");
        }

        void setInlineAsm(bool isTrue)
        {
            outputInlineAsm = isTrue;
        }

        std::string formatOnly(const std::string& instStr, const std::string& comment) const
        {
            return formatStr(outputInlineAsm,
                             instStr,
                             comment,
                             rocIsa::getInstance().getOutputOptions().outputNoComment);
        }

        std::string formatWithComment(const std::string& instStr) const
        {
            return formatStr(outputInlineAsm,
                             instStr,
                             comment,
                             rocIsa::getInstance().getOutputOptions().outputNoComment);
        }

        std::string formatWithExtraComment(const std::string& instStr,
                                           const std::string& extraComment) const
        {
            std::string combinedComment = comment + extraComment;
            return formatStr(outputInlineAsm,
                             instStr,
                             combinedComment,
                             rocIsa::getInstance().getOutputOptions().outputNoComment);
        }

        void setInst(const std::string& instStr)
        {
            this->instStr = instStr;
        }

        virtual std::string preStr() const
        {
            return instStr;
        }

        std::string toString() const override
        {
            throw std::runtime_error("You should override toString function in derived class");
            return "";
        }

        virtual std::vector<InstructionInput> getParams() const = 0;

        virtual int getIssueLatency() const
        {
            return 1; // Default issue latency is 1, should be overridden in derived classes
        }

        virtual int getIssueCycles() const
        {
            return 1; // Default issue cycles is 1, should be overridden in derived classes
        }
    };

    struct CompositeInstruction : public Instruction
    {
        std::shared_ptr<Container>    dst;
        std::vector<InstructionInput> srcs;

        CompositeInstruction(InstType                             instType,
                             const std::shared_ptr<Container>&    dst,
                             const std::vector<InstructionInput>& srcs,
                             const std::string&                   comment = "")
            : Instruction(instType, comment)
            , dst(dst)
            , srcs(srcs)
        {
        }

        CompositeInstruction(const CompositeInstruction& other)
            : Instruction(other)
            , dst(other.dst ? other.dst->clone() : nullptr)
        {
            for(auto& src : other.srcs)
            {
                srcs.push_back([](const InstructionInput& input) -> InstructionInput {
                    return std::visit(
                        [](auto&& arg) -> InstructionInput {
                            using T = std::decay_t<decltype(arg)>;
                            if constexpr(std::is_same_v<T, std::shared_ptr<rocisa::Container>>)
                            {
                                return arg->clone();
                            }
                            else
                            {
                                return arg;
                            }
                        },
                        input);
                }(src));
            }
        }

        virtual std::vector<std::shared_ptr<Instruction>> setupInstructions() const
        {
            return {};
        };

        std::vector<std::shared_ptr<Instruction>> getInstructions()
        {
            return setupInstructions();
        }

        std::vector<InstructionInput> getParams() const override
        {
            std::vector<InstructionInput> plist;
            if(dst)
            {
                plist.push_back(dst);
            }
            if(!srcs.empty())
            {
                plist.insert(plist.end(), srcs.begin(), srcs.end());
            }
            return plist;
        }

        std::string toString() const override
        {
            auto        trueInsts = setupInstructions();
            std::string result;
            for(const auto& inst : trueInsts)
            {
                if(!result.empty())
                {
                    result += "\n";
                }
                result += inst->toString();
            }
            return result;
        }
    };

    struct CommonInstruction : public Instruction
    {
        std::shared_ptr<Container>    dst;
        std::shared_ptr<Container>    dst1; // Usually we don't need this
        std::vector<InstructionInput> srcs;
        std::optional<DPPModifiers>   dpp;
        std::optional<SDWAModifiers>  sdwa;
        std::optional<VOP3PModifiers> vop3;

        CommonInstruction(InstType                             instType,
                          const std::shared_ptr<Container>&    dst,
                          const std::vector<InstructionInput>& srcs,
                          std::optional<DPPModifiers>          dpp     = std::nullopt,
                          std::optional<SDWAModifiers>         sdwa    = std::nullopt,
                          std::optional<VOP3PModifiers>        vop3    = std::nullopt,
                          const std::string&                   comment = "")
            : Instruction(instType, comment)
            , dst(dst)
            , dst1(nullptr)
            , srcs(srcs)
            , dpp(dpp)
            , sdwa(sdwa)
            , vop3(vop3)
        {
        }

        CommonInstruction(const CommonInstruction& other)
            : Instruction(other)
            , dst(other.dst ? other.dst->clone() : nullptr)
            , dst1(other.dst1 ? other.dst1->clone() : nullptr)
            , dpp(other.dpp)
            , sdwa(other.sdwa)
            , vop3(other.vop3)
        {
            for(auto& src : other.srcs)
            {
                srcs.push_back([](const InstructionInput& input) -> InstructionInput {
                    return std::visit(
                        [](auto&& arg) -> InstructionInput {
                            using T = std::decay_t<decltype(arg)>;
                            if constexpr(std::is_same_v<T, std::shared_ptr<rocisa::Container>>)
                            {
                                return arg->clone();
                            }
                            else
                            {
                                return arg;
                            }
                        },
                        input);
                }(src));
            }
        }

        virtual std::string getArgStr() const
        {
            std::string kStr;
            if(dst && !dst->toString().empty())
            {
                kStr += dst->toString();
            }
            if(dst1 && !dst1->toString().empty())
            {
                if(!kStr.empty())
                {
                    kStr += ", ";
                }
                kStr += dst1->toString();
            }
            if(!srcs.empty())
            {
                if(!kStr.empty())
                {
                    kStr += ", ";
                }
                kStr += InstructionInputToString(srcs[0]);
            }
            for(size_t i = 1; i < srcs.size(); ++i)
            {
                kStr += ", " + InstructionInputToString(srcs[i]);
            }
            return kStr;
        }

        std::vector<InstructionInput> getParams() const override
        {
            std::vector<InstructionInput> l;
            if(dst)
            {
                l.push_back(dst);
            }
            if(dst1)
            {
                l.push_back(dst1);
            }
            if(!srcs.empty())
            {
                for(const auto& src : srcs)
                {
                    l.push_back(src);
                }
            }
            return l;
        }

        std::string toString() const override
        {
            auto        newInstStr = preStr();
            std::string kStr       = newInstStr + " " + getArgStr();
            if(dpp)
            {
                kStr += dpp->toString();
            }
            if(sdwa)
            {
                kStr += sdwa->toString();
            }
            if(vop3)
            {
                kStr += vop3->toString();
            }
            return formatWithComment(kStr);
        }
    };

    struct MacroInstruction : public Instruction
    {
        std::string                   name;
        std::vector<InstructionInput> args;

        MacroInstruction(const std::string&                   name,
                         const std::vector<InstructionInput>& args,
                         const std::string&                   comment = "")
            : Instruction(InstType::INST_MACRO, comment)
            , name(name)
            , args(args)
        {
        }

        MacroInstruction(const MacroInstruction& other)
            : Instruction(other)
            , name(other.name)
        {
            for(auto& arg : other.args)
            {
                args.push_back([](const InstructionInput& input) -> InstructionInput {
                    return std::visit(
                        [](auto&& arg) -> InstructionInput {
                            using T = std::decay_t<decltype(arg)>;
                            if constexpr(std::is_same_v<T, std::shared_ptr<rocisa::Container>>)
                            {
                                return arg->clone();
                            }
                            else
                            {
                                return arg;
                            }
                        },
                        input);
                }(arg));
            }
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<MacroInstruction>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return args;
        }

        std::string getArgStr() const
        {
            std::string kStr;
            if(!args.empty())
            {
                kStr += " " + InstructionInputToString(args[0]);
                for(size_t i = 1; i < args.size(); ++i)
                {
                    kStr += ", " + InstructionInputToString(args[i]);
                }
            }
            return kStr;
        }

        std::string toString() const override
        {
            return formatWithComment(name + getArgStr());
        }
    };
} // namespace rocisa
