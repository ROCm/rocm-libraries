/**
 * @copyright Copyright 2022 Advanced Micro Devices, Inc.
 */

#pragma once

#include "BufferInstructionOptions.hpp"
namespace rocRoller
{
    BufferInstructionOptions::BufferInstructionOptions()
    {
        m_offen = false;
        m_glc   = false;
        m_slc   = false;
        m_lds   = false;
    }

    void BufferInstructionOptions::setOffen(bool on)
    {
        m_offen = on;
    }

    void BufferInstructionOptions::setGlc(bool glc)
    {
        m_glc = glc;
    }

    void BufferInstructionOptions::setSlc(bool slc)
    {
        m_slc = slc;
    }

    void BufferInstructionOptions::setLds(bool lds)
    {
        m_lds = lds;
    }

    bool BufferInstructionOptions::getOffen()
    {
        return m_offen;
    }

    bool BufferInstructionOptions::getGlc()
    {
        return m_glc;
    }

    bool BufferInstructionOptions::getSlc()
    {
        return m_slc;
    }

    bool BufferInstructionOptions::getLds()
    {
        return m_lds;
    }
}
