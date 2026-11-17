// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

// How rankJitHeuristic combines two provider walks. A walk that ruled the
// layout out stays ruled out when the other walk is empty, and a duplicate
// prefix does not consume the remaining slots. The merged selection is ruled
// out only when it is empty, so an accepted algorithm does not skip the rest.
namespace hipblaslt_jit
{
    template <typename T>
    struct ProviderPage
    {
        std::vector<T> results;
        bool           layoutRuledOut = false;
    };

    // Either walk ruling the layout out keeps the merged walk ruled out.
    inline bool combineLayoutRuledOut(bool left, bool right)
    {
        return left || right;
    }

    // getAllSolutions runs when this is false. A rejection counts only when the
    // merged prebuilt selection is empty.
    inline bool layoutRuledOutForQuery(bool layoutRuledOut, bool selectionEmpty)
    {
        return selectionEmpty && layoutRuledOut;
    }

    // JIT rows are not prebuilt rows. equalityCount and otherPrebuiltCount are
    // the two prebuilt walks. jitCount is ignored: a JIT hit must not clear a
    // rejection the prebuilt walks reported.
    inline bool layoutRuledOutForQuery(bool   layoutRuledOut,
                                       size_t equalityCount,
                                       size_t otherPrebuiltCount,
                                       size_t jitCount)
    {
        (void)jitCount;
        return layoutRuledOutForQuery(layoutRuledOut,
                                      equalityCount == 0 && otherPrebuiltCount == 0);
    }

    template <typename T>
    struct UniqueFill
    {
        std::vector<T> kept;
        bool           layoutRuledOut = false;
    };

    // fetch(room) returns that walk's prefix. duplicate(candidate, kept) is true
    // when candidate is already selected or already kept. Asks for a wider prefix
    // until `need` new candidates are kept or a call returns fewer than requested.
    template <typename T, typename Fetch, typename Duplicate>
    UniqueFill<T>
        fillUniqueSlots(Fetch&& fetch, Duplicate&& duplicate, int need, bool layoutRuledOut)
    {
        UniqueFill<T> filled;
        filled.layoutRuledOut = layoutRuledOut;
        if(need <= 0)
            return filled;
        int room = need;
        for(;;)
        {
            ProviderPage<T> page = fetch(room);
            filled.layoutRuledOut
                = combineLayoutRuledOut(filled.layoutRuledOut, page.layoutRuledOut);
            std::vector<T> unique;
            for(const auto& candidate : page.results)
            {
                if(duplicate(candidate, unique))
                    continue;
                unique.push_back(candidate);
                if(static_cast<int>(unique.size()) == need)
                    break;
            }
            filled.kept = std::move(unique);
            if(static_cast<int>(filled.kept.size()) >= need
               || static_cast<int>(page.results.size()) < room)
                break;
            const int next = static_cast<int>(page.results.size())
                             + (need - static_cast<int>(filled.kept.size()));
            if(next <= room)
                break;
            room = next;
        }
        return filled;
    }
}
