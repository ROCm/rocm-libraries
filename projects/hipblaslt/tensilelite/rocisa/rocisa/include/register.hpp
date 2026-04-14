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
#include "code.hpp"
#include "container.hpp"
#include "enum.hpp"
#include "instruction/common.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocisa
{
    class RegisterPool
    {
    public:
        enum class Status
        {
            Unavailable = 0,
            Available   = 1,
            InUse       = 2
        };

        struct Register
        {
            Status      status;
            std::string tag;

            Register(Status s, const std::string& t)
                : status(s)
                , tag(t)
            {
            }
        };

        // defaultPreventOverflow: control behavior of checkout and checkoutAligned when preventOverflow is not explicitly specificed.
        RegisterPool(size_t             size,
                     const RegisterType type,
                     bool               defaultPreventOverflow,
                     bool               printRP = false)
            : m_printRP(printRP)
            , m_type(type)
            , m_defaultPreventOverflow(defaultPreventOverflow)
        {
            m_pool.resize(size, Register(Status::Unavailable, "init"));
            m_occupancyLimitSize    = 0;
            m_occupancyLimitMaxSize = 0;
        }

        RegisterPool(const RegisterPool& other)
            : m_printRP(other.m_printRP)
            , m_type(other.m_type)
            , m_defaultPreventOverflow(other.m_defaultPreventOverflow)
            , m_pool(other.m_pool)
            , m_checkOutSize(other.m_checkOutSize)
            , m_checkOutSizeTemp(other.m_checkOutSizeTemp)
            , m_occupancyLimitSize(other.m_occupancyLimitSize)
            , m_occupancyLimitMaxSize(other.m_occupancyLimitMaxSize)
        {
        }

        void setOccupancyLimit(size_t maxSize, size_t size)
        {
            m_occupancyLimitSize    = size;
            m_occupancyLimitMaxSize = maxSize;
        }

        void resetOccupancyLimit()
        {
            m_occupancyLimitSize    = 0;
            m_occupancyLimitMaxSize = 0;
        }

        const std::vector<Register> getPool() const
        {
            return m_pool;
        }

        // Adds registers to the pool so they can be used as temps
        // Convenience function that takes a range and returns it in string form
        std::string addRange(size_t start, size_t stop, const std::string& tag = "")
        {
            add(start, stop - start + 1, tag);
            if(start == stop)
            {
                return std::to_string(start);
            }
            return std::to_string(start) + "-" + std::to_string(stop);
        }

        // Adds registers to the pool so they can be used as temps
        void add(size_t start, size_t size, const std::string& tag = "")
        {
            if(m_printRP)
            {
                printf("RP::add(%zu..%zu for '%s')\n", start, start + size - 1, tag.c_str());
            }
            size_t newSize = start + size;
            size_t oldSize = m_pool.size();
            if(newSize > oldSize)
            {
                m_pool.resize(newSize, {Status::Unavailable, tag});
            }
            for(size_t i = start; i < start + size; ++i)
            {
                if(m_pool[i].status == Status::Unavailable)
                {
                    m_pool[i].status = Status::Available;
                    m_pool[i].tag    = tag;
                }
                else if(m_pool[i].status == Status::Available)
                {
                    printf("Warning: RegisterPool::add(%zu,%zu) pool[%zu](%s) already available\n",
                           start,
                           start + size - 1,
                           i,
                           m_pool[i].tag.c_str());
                }
                else if(m_pool[i].status == Status::InUse)
                {
                    printf("Warning: RegisterPool::add(%zu,%zu) pool[%zu](%s) already in use\n",
                           start,
                           start + size - 1,
                           i,
                           m_pool[i].tag.c_str());
                }
                else
                {
                    throw std::runtime_error("RegisterPool::add() invalid status");
                }
            }
            if(m_printRP)
            {
                printf("%s\n", state().c_str());
            }
        }

        void addFromCheckOut(size_t start)
        {
            if(m_checkOutSize.find(start) != m_checkOutSize.end())
            {
                size_t size = m_checkOutSize[start];
                for(size_t i = start; i < start + size; ++i)
                {
                    if(m_pool[i].status != Status::InUse)
                    {
                        throw std::runtime_error(
                            "RegisterPool::addFromCheckOut() is not in InUse state");
                    }
                    m_pool[i].status = Status::Available;
                }
                m_checkOutSizeTemp[start] = {size, m_pool[start].tag};
                m_checkOutSize.erase(start);
                if(m_printRP)
                {
                    printf("RP::addFromCheckOut('%s') @ %zu +%zu\n",
                           m_pool[start].tag.c_str(),
                           start,
                           size);
                }
            }
            else
            {
                throw std::runtime_error(
                    "RegisterPool::addFromCheckOut() but it was never checked out");
            }
        }

        void remove(size_t start, size_t size, const std::string& tag = "")
        {
            if(m_printRP)
            {
                printf("RP::remove(%zu..%zu) for %s\n", start, start + size - 1, tag.c_str());
            }
            size_t newSize = start + size;
            size_t oldSize = m_pool.size();
            if(newSize > oldSize)
            {
                printf("Warning: RegisterPool::remove(%zu,%zu) but poolSize=%zu\n",
                       start,
                       start + size - 1,
                       oldSize);
            }
            for(size_t i = start; i < start + size; ++i)
            {
                if(m_pool[i].status == Status::Available)
                {
                    m_pool[i].status = Status::Unavailable;
                }
                else if(m_pool[i].status == Status::Unavailable)
                {
                    printf("Warning: RegisterPool::remove(%zu,%zu) pool[%zu](%s) already "
                           "unavailable\n",
                           start,
                           start + size - 1,
                           i,
                           m_pool[i].tag.c_str());
                }
                else if(m_pool[i].status == Status::InUse)
                {
                    printf("Warning: RegisterPool::remove(%zu,%zu) pool[%zu](%s) still in use\n",
                           start,
                           start + size - 1,
                           i,
                           m_pool[i].tag.c_str());
                }
                else
                {
                    throw std::runtime_error("RegisterPool::remove() invalid status");
                }
            }
        }

        void removeFromCheckOut(size_t start)
        {
            if(m_checkOutSizeTemp.find(start) != m_checkOutSizeTemp.end())
            {
                auto [size, tag] = m_checkOutSizeTemp[start];
                for(size_t i = start; i < start + size; ++i)
                {
                    if(m_pool[i].status != Status::Available)
                    {
                        throw std::runtime_error(
                            "RegisterPool::removeFromCheckOut() is not in Available state");
                    }
                    m_pool[i].status = Status::InUse;
                    m_pool[i].tag    = tag;
                }
                m_checkOutSize[start] = size;
                m_checkOutSizeTemp.erase(start);
                if(m_printRP)
                {
                    printf("RegisterPool::removeFromCheckOut('%s') @ %zu +%zu\n",
                           m_pool[start].tag.c_str(),
                           start,
                           size);
                }
            }
            else
            {
                throw std::runtime_error(
                    "RegisterPool::removeFromCheckOut() but it was never checked out");
            }
        }

        size_t
            checkOut(size_t size, const std::string& tag = "_untagged_", int preventOverflow = -1)
        {
            return checkOutAligned(size, 1, tag, preventOverflow);
        }

        size_t checkOutAligned(size_t             size,
                               size_t             alignment,
                               const std::string& tag             = "_untagged_aligned_",
                               int                preventOverflow = -1)
        {
            if(preventOverflow == -1)
            {
                preventOverflow = int(m_defaultPreventOverflow);
            }
            if(size == 0)
            {
                throw std::invalid_argument("Size must be greater than 0");
            }

            size_t found = -1;
            for(size_t i = 0; i < m_pool.size(); ++i)
            {
                // alignment
                if(i % alignment != 0)
                {
                    continue;
                }
                // enough space
                if(i + size > m_pool.size())
                {
                    continue;
                }
                // all available
                bool allAvailable = true;
                for(size_t j = 0; j < size; ++j)
                {
                    if(m_pool[i + j].status != Status::Available)
                    {
                        allAvailable = false;
                        i += j; // move to next position
                        break;
                    }
                }
                if(allAvailable)
                {
                    found = i;
                    break;
                }
                else
                {
                    continue;
                }
            }

            // success without overflowing
            if(found != -1)
            {
                for(size_t i = found; i < found + size; ++i)
                {
                    m_pool[i].status = Status::InUse;
                    m_pool[i].tag    = tag;
                }
                m_checkOutSize[found] = size;
                if(m_printRP)
                {
                    printf("RP::checkOut '%s' (%zu,%zu) @ %zu avail=%zu\n",
                           tag.c_str(),
                           size,
                           alignment,
                           found,
                           available());
                }
                return found;
            }
            // need overflow
            else
            {
                // where does tail sequence of available registers begin
                if(preventOverflow)
                {
                    throw std::runtime_error("RegisterPool::checkOutAligned: register overflow "
                                             "prevented by preventOverflow flag");
                }
                size_t start = m_pool.size();
                if(start)
                {
                    for(size_t i = m_pool.size() - 1; i > 0; --i)
                    {
                        if(m_pool[i].status == Status::Available)
                        {
                            m_pool[i].tag = tag;
                            start         = i;
                            continue;
                        }
                        else
                        {
                            break;
                        }
                    }

                    start = ((start + alignment - 1) / alignment) * alignment; // align start
                }

                // new checkout can begin at start
                size_t newSize = start + size;
                size_t oldSize = m_pool.size();
                if(m_occupancyLimitSize > 0)
                {
                    if(newSize > m_occupancyLimitSize && newSize <= m_occupancyLimitMaxSize)
                    {
                        printf("newSize %zu OldSize %zu Limit %zu\n",
                               newSize,
                               oldSize,
                               m_occupancyLimitSize);
                        if(m_occupancyLimitSize < newSize)
                        {
                            throw std::runtime_error(
                                "RegisterPool::checkOutAligned: occupancy limit exceeded");
                        }
                    }
                }
                size_t overflow = (newSize > oldSize) ? (newSize - oldSize) : 0;
                for(size_t i = start; i < m_pool.size(); ++i)
                {
                    m_pool[i].status = Status::InUse;
                    m_pool[i].tag    = tag;
                }
                for(size_t i = 0; i < overflow; ++i)
                {
                    if(m_pool.size() < start)
                    {
                        // this is padding to meet alignment requirements
                        m_pool.push_back(Register(Status::Available, tag));
                    }
                    else
                    {
                        m_pool.push_back(Register(Status::InUse, tag));
                    }
                }
                m_checkOutSize[start] = size;
                if(m_printRP)
                {
                    printf("RP::checkOut '%s' (%zu,%zu) @ %zu (overflow)\n",
                           tag.c_str(),
                           size,
                           alignment,
                           start);
                }
                return start;
            }
        }

        std::vector<size_t> checkOutMulti(const std::vector<size_t>&      sizes,
                                          size_t                          alignment,
                                          const std::vector<std::string>& tags)
        {
            if(sizes.size() != tags.size())
            {
                throw std::invalid_argument("Sizes and tags must have the same length");
            }
            size_t totalSize = 0;
            for(const auto& s : sizes)
            {
                totalSize += s;
            }
            size_t idx = checkOutAligned(totalSize, alignment, "", false);
            // Overwrite the checkOutSize information
            m_checkOutSize.erase(idx);
            std::vector<size_t> idxVec;
            for(size_t sIdx = 0; sIdx < sizes.size(); ++sIdx)
            {
                idxVec.push_back(idx);
                m_checkOutSize[idx] = sizes[sIdx];
                for(size_t i = idx; i < idx + sizes[sIdx]; ++i)
                {
                    m_pool[i].tag = tags[sIdx];
                }
                idx += sizes[sIdx];
            }
            return idxVec;
        }

        // Initializes temporary registers with a given value
        std::shared_ptr<Item> initTmps(int initValue, size_t start = 0, size_t stop = -1)
        {
            std::shared_ptr<Module> module = std::make_shared<Module>("initTmps from RegisterPool");
            stop = (stop == -1 || stop > m_pool.size()) ? m_pool.size() : stop + 1;
            for(size_t i = start; i < stop; ++i)
            {
                if(m_pool[i].status == Status::Available)
                {
                    switch(m_type)
                    {
                    case RegisterType::Sgpr:
                        module->addT<SMovB32>(sgpr(i), initValue, "init tmp in pool");
                        break;
                    case RegisterType::Vgpr:
                        module->addT<VMovB32>(vgpr(i), initValue, std::nullopt, "init tmp in pool");
                        break;
                    default:
                        throw std::runtime_error("Invalid register pool type");
                    }
                }
            }
            return module;
        }

        void checkIn(size_t start)
        {
            if(m_checkOutSize.find(start) != m_checkOutSize.end())
            {
                size_t size = m_checkOutSize[start];
                for(size_t i = start; i < start + size; ++i)
                {
                    m_pool[i].status = Status::Available;
                }
                m_checkOutSize.erase(start);
                if(m_printRP)
                {
                    printf(
                        "RP::checkIn('%s') @ %zu +%zu\n", m_pool[start].tag.c_str(), start, size);
                }
            }
            else
            {
                printf("Warning: RegisterPool::checkIn('%s',%zu) but it was never checked out\n",
                       m_pool[start].tag.c_str(),
                       start);
            }
        }

        size_t size() const
        {
            return m_pool.size();
        }

        // Number of available registers
        size_t available() const
        {
            size_t numAvailable = 0;
            for(const auto& reg : m_pool)
            {
                if(reg.status == Status::Available)
                {
                    numAvailable++;
                }
            }
            return numAvailable;
        }
        // Size of registers of at least specified blockSize
        size_t availableBlock(size_t blockSize, size_t align) const
        {
            if(blockSize == 0)
            {
                blockSize = 1;
            }
            size_t blocksAvail     = 0;
            size_t consecAvailable = 0;
            for(size_t i = 0; i < m_pool.size(); ++i)
            {
                const auto& reg = m_pool[i];
                if(reg.status == Status::Available)
                {
                    if(!(consecAvailable == 0 && i % align != 0))
                    {
                        // do not increment if the first item is not aligned
                        consecAvailable++;
                    }
                }
                else
                {
                    blocksAvail += consecAvailable / blockSize;
                    consecAvailable = 0;
                }
            }
            blocksAvail += consecAvailable / blockSize;
            return blocksAvail * blockSize;
        }

        // Size of registers of at least specified blockSize
        size_t availableBlockMaxVgpr(size_t maxVgpr, size_t blockSize, size_t align) const
        {
            if(blockSize == 0)
            {
                blockSize = 1;
            }
            size_t blocksAvail     = 0;
            size_t consecAvailable = 0;
            for(size_t i = 0; i < maxVgpr; ++i)
            {
                if(i >= m_pool.size())
                {
                    if(!(consecAvailable == 0 && i % align != 0))
                    {
                        consecAvailable++;
                    }
                }
                else
                {
                    const auto& reg = m_pool[i];
                    if(reg.status == Status::Available)
                    {
                        if(!(consecAvailable == 0 && i % align != 0))
                        {
                            // do not increment if the first item is not aligned
                            consecAvailable++;
                        }
                    }
                    else
                    {
                        blocksAvail += consecAvailable / blockSize;
                        consecAvailable = 0;
                    }
                }
            }
            blocksAvail += consecAvailable / blockSize;
            return blocksAvail * blockSize;
        }

        size_t availableBlockAtEnd() const
        {
            size_t availCnt = 0;
            for(auto it = m_pool.rbegin(); it != m_pool.rend(); ++it)
            {
                if(it->status == Status::Available)
                {
                    availCnt++;
                }
                else
                {
                    break;
                }
            }
            return availCnt;
        }

        void checkFinalState() const
        {
            for(size_t si = 0; si < m_pool.size(); ++si)
            {
                if(m_pool[si].status == Status::InUse)
                {
                    if(m_printRP)
                    {
                        printf("%s\n", state().c_str());
                    }
                    throw std::runtime_error("RegisterPool::checkFinalState: temp ("
                                             + std::to_string(si) + ", '" + m_pool[si].tag
                                             + "') was never checked in.");
                }
            }
            // Should we need this? It'll print a lot of information
            if(m_printRP)
            {
                printf("total vgpr count: %zu\n", size());
            }
        }

        std::string state() const
        {
            std::string      stateStr;
            std::vector<int> placeValues = {1000, 100, 10, 1};
            for(size_t placeValueIdx = 1; placeValueIdx < placeValues.size(); ++placeValueIdx)
            {
                int placeValue      = placeValues[placeValueIdx];
                int priorPlaceValue = placeValues[placeValueIdx - 1];
                if(static_cast<int>(m_pool.size()) >= placeValue)
                {
                    std::string pvs;
                    for(size_t i = 0; i < m_pool.size(); ++i)
                    {
                        if(i % placeValue == 0)
                        {
                            pvs += std::to_string((i % priorPlaceValue) / placeValue);
                        }
                        else
                        {
                            pvs += " ";
                        }
                    }
                    stateStr += pvs + "\n";
                }
            }
            for(auto it = m_pool.begin(); it != m_pool.end(); ++it)
            {
                switch(it->status)
                {
                case Status::Unavailable:
                    stateStr += ".";
                    break;
                case Status::Available:
                    stateStr += "|";
                    break;
                case Status::InUse:
                    stateStr += "#";
                    break;
                }
            }
            return stateStr;
        }

        void growPool(size_t             rangeStart,
                      size_t             rangeEnd,
                      size_t             checkOutSize,
                      const std::string& comment = "")
        {
            std::vector<size_t> tl;
            if(checkOutSize == 1)
            {
                size_t              continuous = 0;
                std::vector<size_t> availList;
                for(size_t i = 0; i < m_pool.size(); ++i)
                {
                    const auto& s = m_pool[i];
                    if(s.status != Status::Available)
                    {
                        if(continuous > 0)
                        {
                            availList.push_back(continuous);
                            continuous = 0;
                        }
                    }
                    else
                    {
                        continuous++;
                    }
                }
                int rangeTotal = rangeEnd - rangeStart;
                for(const auto& numGpr : availList)
                {
                    size_t rangeTurn = numGpr / checkOutSize;
                    if(rangeTurn > 0)
                    {
                        tl.push_back(checkOut(checkOutSize * rangeTurn, comment));
                        rangeTotal -= rangeTurn;
                    }
                }
                if(rangeTotal > 0)
                {
                    tl.push_back(checkOut(checkOutSize * rangeTotal, comment));
                }
            }
            else
            {
                for(size_t i = rangeStart; i < rangeEnd; ++i)
                {
                    tl.push_back(checkOut(checkOutSize, comment));
                }
            }
            for(const auto& t : tl)
            {
                checkIn(t);
            }
        }

        void appendPool(size_t newSize)
        {
            size_t oldSize = m_pool.size();
            if(newSize > oldSize)
            {
                m_pool.resize(newSize, {Status::Available, "append pool"});
            }
        }

    private:
        bool                                                       m_printRP;
        RegisterType                                               m_type;
        bool                                                       m_defaultPreventOverflow;
        std::vector<Register>                                      m_pool;
        std::unordered_map<size_t, size_t>                         m_checkOutSize;
        std::unordered_map<size_t, std::pair<size_t, std::string>> m_checkOutSizeTemp;
        size_t                                                     m_occupancyLimitSize;
        size_t                                                     m_occupancyLimitMaxSize;
    };
} // namespace rocisa
