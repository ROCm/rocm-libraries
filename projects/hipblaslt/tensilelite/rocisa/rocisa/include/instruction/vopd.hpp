// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// VOPD (dual issue VALU) pairing rules for VDualOp, per ISA family.
//
// Sources: RDNA3.5 ISA 7.6 with tables 91/92 (gfx11), RDNA4 ISA 7.8 with
// tables 100/101 (gfx120x). Family differences are arch caps (see
// initArchCaps): HasVOPD, VOPDSharedSrcVgpr, VOPDXMayWriteYSrc, VOPDMaxMinNum.
//
// The assembler rejects most violations, but not a dependency between the two
// halves, which RDNA4 forbids in one direction only. Checking every rule here
// lets the kernel writer pair only what the target family allows.
#pragma once
#include "container.hpp"
#include "instruction.hpp"

#include <cstdlib>
#include <set>
#include <string>

namespace rocisa
{
    namespace vopd
    {
        // VDualOp encodes two-source ops only: no v_dual_mov_b32 (one source) and
        // no fmaak/fmamk (literal operand). With two sources, the SRC2 parity rule
        // (fmac/dot2acc read their dst) follows from the dst parity rule.
        inline bool isTwoSourceOp(const std::string& op, bool maxMinNum)
        {
            static const std::set<std::string> common = {"fmac_f32",
                                                         "mul_f32",
                                                         "add_f32",
                                                         "sub_f32",
                                                         "subrev_f32",
                                                         "mul_dx9_zero_f32",
                                                         "cndmask_b32",
                                                         "dot2acc_f32_f16",
                                                         "dot2acc_f32_bf16"};
            if(common.count(op))
                return true;
            if(maxMinNum)
                return op == "max_num_f32" || op == "min_num_f32";
            return op == "max_f32" || op == "min_f32";
        }

        inline bool isYOnlyOp(const std::string& op)
        {
            return op == "add_nc_u32" || op == "lshlrev_b32" || op == "and_b32";
        }

        // Ops that read their destination through the SRC2 port.
        inline bool accumulates(const std::string& op)
        {
            return op == "fmac_f32" || op == "dot2acc_f32_f16" || op == "dot2acc_f32_bf16";
        }

        enum class OperandKind
        {
            Vgpr,
            Scalar, // SGPR, VCC, EXEC, M0, SCC: one constant-bus read each
            Inline,
            Literal,
            Invalid
        };

        struct Operand
        {
            OperandKind kind = OperandKind::Invalid;
            int         vgpr = -1; // VGPR index, -1 if symbolic
            std::string key; // identity of a scalar or literal
            std::string error;
        };

        inline bool isInlineInt(long long v)
        {
            return v >= -16 && v <= 64;
        }

        inline bool isInlineFloat(double v)
        {
            static const double values[]
                = {0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 0.15915494309189535};
            for(double c : values)
                if(v == c)
                    return true;
            return false;
        }

        inline Operand classify(const InstructionInput& input)
        {
            Operand o;
            if(auto p = std::get_if<int>(&input))
            {
                o.kind = isInlineInt(*p) ? OperandKind::Inline : OperandKind::Literal;
                o.key  = std::to_string(*p);
                return o;
            }
            if(auto p = std::get_if<double>(&input))
            {
                o.kind = isInlineFloat(*p) ? OperandKind::Inline : OperandKind::Literal;
                o.key  = InstructionInputToString(input);
                return o;
            }
            if(auto p = std::get_if<std::string>(&input))
            {
                static const std::set<std::string> scalars = {"vcc",
                                                              "vcc_lo",
                                                              "vcc_hi",
                                                              "exec",
                                                              "exec_lo",
                                                              "exec_hi",
                                                              "m0",
                                                              "scc",
                                                              "src_scc"};
                o.key                                      = *p;
                if(scalars.count(*p))
                {
                    o.kind = OperandKind::Scalar;
                    return o;
                }
                char*     end = nullptr;
                long long i   = std::strtoll(p->c_str(), &end, 0);
                if(!p->empty() && *end == '\0')
                {
                    o.kind = isInlineInt(i) ? OperandKind::Inline : OperandKind::Literal;
                    return o;
                }
                double d = std::strtod(p->c_str(), &end);
                if(!p->empty() && *end == '\0')
                {
                    o.kind = isInlineFloat(d) ? OperandKind::Inline : OperandKind::Literal;
                    return o;
                }
                o.kind = OperandKind::Literal; // a symbol
                return o;
            }
            auto c = std::get<std::shared_ptr<Container>>(input);
            if(!c)
            {
                o.error = "missing operand";
                return o;
            }
            if(std::dynamic_pointer_cast<VCC>(c) || std::dynamic_pointer_cast<EXEC>(c)
               || std::dynamic_pointer_cast<EXECLO>(c) || std::dynamic_pointer_cast<EXECHI>(c))
            {
                o.kind = OperandKind::Scalar;
                o.key  = c->toString();
                return o;
            }
            auto r = std::dynamic_pointer_cast<RegisterContainer>(c);
            if(!r)
            {
                o.error = "operand " + c->toString() + " is not encodable in VOPD";
                return o;
            }
            if(r->isMinus || r->isAbs || r->halfSelect)
            {
                o.error = "VOPD has no operand modifiers (neg, abs, true16 half): " + r->toString();
                return o;
            }
            if(r->regNum != 1)
            {
                o.error = "VOPD operands are 32-bit: " + r->toString();
                return o;
            }
            if(r->regType == "s")
            {
                o.kind = OperandKind::Scalar;
                o.key  = r->toString();
                return o;
            }
            if(r->regType != "v")
            {
                o.error = "operand " + r->toString() + " is not encodable in VOPD";
                return o;
            }
            o.kind = OperandKind::Vgpr;
            if(!r->regName && !r->isMacro && !r->isInlineAsm)
                o.vgpr = r->regIdx;
            o.key = r->toString();
            return o;
        }

        // Why the pair cannot be a VOPD on the current ISA, or "" if it can.
        inline std::string check(const std::string&                opX,
                                 const std::shared_ptr<Container>& dstX,
                                 const InstructionInput&           src0X,
                                 const InstructionInput&           src1X,
                                 const std::string&                opY,
                                 const std::shared_ptr<Container>& dstY,
                                 const InstructionInput&           src0Y,
                                 const InstructionInput&           src1Y)
        {
            auto& isa  = rocIsa::getInstance();
            auto& caps = isa.getArchCaps();
            if(!capOrDefault(caps, "HasVOPD"))
                return "VOPD is not modelled for this ISA";
            if(isa.getKernel().wavefront != 32)
                return "VOPD requires wave32";

            bool maxMinNum = capOrDefault(caps, "VOPDMaxMinNum");
            if(!isTwoSourceOp(opX, maxMinNum))
                return "v_dual_" + opX + " cannot be the X op";
            if(!isTwoSourceOp(opY, maxMinNum) && !isYOnlyOp(opY))
                return "v_dual_" + opY + " cannot be the Y op";

            // Destinations: VGPRs of opposite parity.
            Operand dX = classify(dstX), dY = classify(dstY);
            for(auto* d : {&dX, &dY})
            {
                if(!d->error.empty())
                    return d->error;
                if(d->kind != OperandKind::Vgpr)
                    return "VOPD destination must be a VGPR";
                if(d->vgpr < 0)
                    return "symbolic VGPR " + d->key + ": parity cannot be checked";
            }
            if((dX.vgpr & 1) == (dY.vgpr & 1))
                return "VOPD destinations must be one even and one odd VGPR";

            Operand s0X = classify(src0X), s1X = classify(src1X);
            Operand s0Y = classify(src0Y), s1Y = classify(src1Y);
            for(auto* s : {&s0X, &s1X, &s0Y, &s1Y})
            {
                if(!s->error.empty())
                    return s->error;
                if(s->kind == OperandKind::Vgpr && s->vgpr < 0)
                    return "symbolic VGPR " + s->key + ": bank cannot be checked";
            }
            if(s1X.kind != OperandKind::Vgpr || s1Y.kind != OperandKind::Vgpr)
                return "VOPD VSRC1 must be a VGPR";

            // VGPR banks (index % 4): one cache read port per source slot.
            bool shared    = capOrDefault(caps, "VOPDSharedSrcVgpr");
            auto bankClash = [shared](const Operand& a, const Operand& b) {
                if(a.kind != OperandKind::Vgpr || b.kind != OperandKind::Vgpr)
                    return false;
                if(a.vgpr % 4 != b.vgpr % 4)
                    return false;
                return !(shared && a.vgpr == b.vgpr);
            };
            if(bankClash(s0X, s0Y))
                return "VOPD SRC0 operands must use different VGPR banks";
            if(bankClash(s1X, s1Y))
                return "VOPD VSRC1 operands must use different VGPR banks";

            // Constant bus: at most 2 scalars, or 1 scalar + 1 literal, or one
            // shared literal. v_dual_cndmask_b32 reads VCC.
            std::set<std::string> scalars, literals;
            for(auto* s : {&s0X, &s0Y})
            {
                if(s->kind == OperandKind::Scalar)
                    scalars.insert(s->key);
                else if(s->kind == OperandKind::Literal)
                    literals.insert(s->key);
            }
            for(const auto* op : {&opX, &opY})
                if(*op == "cndmask_b32")
                    scalars.insert(VCC().toString());
            if(literals.size() > 1)
                return "VOPD halves may share one literal at most";
            if(scalars.size() + literals.size() > 2)
                return "VOPD reads at most 2 scalars, or 1 scalar and 1 literal";

            // Both halves read old values. RDNA4 forbids X writing a Y source
            // (dstY, read by an accumulating Y, never equals dstX: parity).
            if(!capOrDefault(caps, "VOPDXMayWriteYSrc"))
            {
                for(auto* s : {&s0Y, &s1Y})
                    if(s->kind == OperandKind::Vgpr && s->vgpr == dX.vgpr)
                        return "VOPD X must not write a source of Y on this ISA";
            }
            return "";
        }
    } // namespace vopd
} // namespace rocisa
