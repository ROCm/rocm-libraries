/**
 * @copyright Copyright 2022 Advanced Micro Devices, Inc.
 */

#pragma once

#include "BufferInstructionOptions.hpp"
namespace rocRoller
{
    inline BufferInstructionOptions::BufferInstructionOptions()
    {
        m_offen = false;
        m_glc   = false;
        m_slc   = false;
        m_lds   = false;
    }

    inline void BufferInstructionOptions::setOffen(bool on)
    {
        m_offen = on;
    }

    inline void BufferInstructionOptions::setGlc(bool glc)
    {
        m_glc = glc;
    }

    inline void BufferInstructionOptions::setSlc(bool slc)
    {
        m_slc = slc;
    }

    inline void BufferInstructionOptions::setLds(bool lds)
    {
        m_lds = lds;
    }

    inline bool BufferInstructionOptions::getOffen()
    {
        return m_offen;
    }

    inline bool BufferInstructionOptions::getGlc()
    {
        return m_glc;
    }

    inline bool BufferInstructionOptions::getSlc()
    {
        return m_slc;
    }

    inline bool BufferInstructionOptions::getLds()
    {
        return m_lds;
    }
}
