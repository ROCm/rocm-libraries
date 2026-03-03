// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/CodeGen/Annotate.hpp>

namespace rocRoller
{
    inline AddComment::AddComment(std::string comment, bool skipCommentOnly)
        : m_comment(std::move(comment))
        , m_skipCommentOnly(skipCommentOnly)
    {
    }

    inline Instruction AddComment::operator()(Instruction inst)
    {
        if(!m_skipCommentOnly || !inst.isCommentOnly())
            inst.addComment(m_comment);
        return inst;
    }

    inline AddControlOp::AddControlOp(int op)
        : m_op(op)
    {
    }

    inline Instruction AddControlOp::operator()(Instruction inst)
    {
        inst.addControlOp(m_op);
        return inst;
    }

    inline std::string toString(SourceLocationPart part)
    {
        switch(part)
        {
        case SourceLocationPart::Function:
            return "Function";
        case SourceLocationPart::File:
            return "File";
        case SourceLocationPart::Line:
            return "Line";
        case SourceLocationPart::Column:
            return "Column";
        case SourceLocationPart::Count:
            break;
        }

        Throw<FatalError>("Invalid SouceLocationPart");
        return "";
    }

    inline AddLocation::AddLocation(EnumBitset<SourceLocationPart> parts, std::source_location loc)
        : m_parts(std::move(parts))
        , m_location(std::move(loc))
    {
    }

    inline AddLocation::AddLocation(std::source_location loc)
        : AddLocation({SourceLocationPart::Line}, std::move(loc))
    {
    }

    inline AddLocation AddLocation::onlyFirst()
    {
        AddLocation rv = *this;

        rv.m_onlyFirst = true;
        rv.m_expired   = false;

        return rv;
    }

    inline std::string AddLocation::comment() const
    {
        std::string rv;
        auto        remainingParts = m_parts.count();

        if(m_parts[SourceLocationPart::Function])
        {
            rv += m_location.function_name();
            remainingParts--;
            if(remainingParts > 0)
                rv += "\n";
        }

        if(m_parts[SourceLocationPart::File])
        {
            rv += m_location.file_name();
            remainingParts--;
            if(remainingParts > 0)
                rv += ":";
        }

        if(m_parts[SourceLocationPart::Line])
        {
            rv += std::to_string(m_location.line());
            remainingParts--;
            if(remainingParts > 0)
                rv += ":";
        }

        if(m_parts[SourceLocationPart::Column])
        {
            rv += std::to_string(m_location.column());
            remainingParts--;
        }

        AssertFatal(remainingParts == 0);

        return rv;
    }

    inline Instruction AddLocation::operator()(Instruction inst)
    {
        if(m_expired)
            return inst;

        if(m_onlyFirst && inst.isCommentOnly())
            return inst;

        if(m_onlyFirst)
            m_expired = true;

        inst.addComment(comment());

        return inst;
    }
}
