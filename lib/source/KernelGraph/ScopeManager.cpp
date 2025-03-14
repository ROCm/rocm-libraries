/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/Context.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/ScopeManager.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{

    void ScopeManager::pushNewScope()
    {
        m_tags.push_back({});
    }

    void ScopeManager::addRegister(int tag)
    {
        // Do not add if the tag is already in scope.
        for(auto s : m_tags)
        {
            if(s.count(tag) > 0)
                return;
        }
        m_tags.back().insert(tag);
    }

    void ScopeManager::popAndReleaseScope()
    {
        for(auto tag : m_tags.back())
        {
            if(m_context->registerTagManager()->hasRegister(tag))
            {
                if(!hasDeallocate(*m_graph, tag))
                    m_context->registerTagManager()->deleteTag(tag);
            }
        }
        m_tags.pop_back();
    }

}
