// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/ProviderRows.hpp>

namespace TensileLite
{
    namespace
    {
        thread_local ProviderRows threadRows = ProviderRows::All;
    }

    ProviderRows currentProviderRows()
    {
        return threadRows;
    }

    bool skipsProviderRow(ProviderRows rows, std::string const& predicateType)
    {
        switch(rows)
        {
        case ProviderRows::EqualityOnly:
            return predicateType == "RangeMatching" || predicateType == "PredictionMatching"
                   || predicateType == "GridBasedMatching" || predicateType == "FreeSizeMatching"
                   || predicateType == "ExperimentalMLP";
        case ProviderRows::ExceptEquality:
            return predicateType == "EqualityMatching";
        case ProviderRows::All:
            break;
        }
        return false;
    }

    ProviderRowsScope::ProviderRowsScope(ProviderRows rows)
        : m_previous(threadRows)
    {
        threadRows = rows;
    }

    ProviderRowsScope::~ProviderRowsScope()
    {
        threadRows = m_previous;
    }
} // namespace TensileLite
