// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /**
     * \addtogroup SolutionLibrary
     * @{
     */

    /**
     * The provider rows ExactLogicLibrary::findTopSolutions walks on the calling
     * thread. Providers are the rows tagged EqualityMatching, RangeMatching,
     * PredictionMatching, GridBasedMatching and FreeSizeMatching, and the
     * ExperimentalMLP rows, which hold MLP classification only; every other row
     * leads to providers and is always walked. Querying the Equality rows and the
     * other providers separately lets a caller place solutions from another
     * source between them.
     */
    enum class ProviderRows
    {
        All,
        EqualityOnly,
        ExceptEquality
    };

    TENSILELITEHOST_EXPORT ProviderRows currentProviderRows();

    /// Whether a walk under rows skips a row whose predicate has this type.
    TENSILELITEHOST_EXPORT bool skipsProviderRow(ProviderRows       rows,
                                                 std::string const& predicateType);

    /// Sets the calling thread's ProviderRows until destroyed.
    class TENSILELITEHOST_EXPORT ProviderRowsScope
    {
    public:
        explicit ProviderRowsScope(ProviderRows rows);
        ~ProviderRowsScope();

        ProviderRowsScope(ProviderRowsScope const&)            = delete;
        ProviderRowsScope& operator=(ProviderRowsScope const&) = delete;

    private:
        ProviderRows m_previous;
    };

    /**
     * @}
     */
} // namespace TensileLite
