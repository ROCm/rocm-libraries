/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_TEST_LENGTH_GENERATION_HPP
#define HIPTENSOR_TEST_LENGTH_GENERATION_HPP

#include "hash.hpp"

namespace hiptensor
{
    std::vector<size_t> generateRange(size_t lower, size_t upper, size_t step)
    {
        std::vector<size_t> sizes = {};

        for(auto i = lower; i <= upper; i *= step)
        {
            sizes.push_back(i);
        }

        return sizes;
    }

    std::vector<std::vector<size_t>> combine(std::vector<size_t> const& sizes1,
                                             std::vector<size_t> const& sizes2)
    {
        std::vector<std::vector<size_t>> combinedSizes = {};

        for(auto i = 0; i < sizes1.size(); i++)
        {
            for(auto j = 0; j < sizes2.size(); j++)
            {
                combinedSizes.push_back({sizes1[i], sizes2[j]});
            }
        }

        return combinedSizes;
    }

    std::vector<std::vector<size_t>> combine(std::vector<std::vector<size_t>> const& sizes1,
                                             std::vector<size_t> const&              sizes2)
    {
        std::vector<std::vector<size_t>> combinedSizes = {};

        for(auto i = 0; i < sizes1.size(); i++)
        {
            for(auto j = 0; j < sizes2.size(); j++)
            {
                std::vector<size_t> tempVector = sizes1[i];
                tempVector.push_back(sizes2[j]);
                combinedSizes.push_back(tempVector);
            }
        }

        return combinedSizes;
    }

    bool areValidLengths(std::vector<size_t> mLengths,
                         std::vector<size_t> nLengths,
                         std::vector<size_t> kLengths,
                         size_t              rank,
                         size_t              dims,
                         size_t              maxElems)
    {
        size_t elements = 0;

        if(dims == 2)
        {
            elements
                = std::accumulate(mLengths.begin(), mLengths.end(), size_t{1}, std::multiplies());
        }
        else if(dims == 3)
        {
            size_t elementsA
                = std::accumulate(mLengths.begin(), mLengths.end(), size_t{1}, std::multiplies())
                  * std::accumulate(kLengths.begin(), kLengths.end(), size_t{1}, std::multiplies());

            size_t elementsB
                = std::accumulate(nLengths.begin(), nLengths.end(), size_t{1}, std::multiplies())
                  * std::accumulate(kLengths.begin(), kLengths.end(), size_t{1}, std::multiplies());

            size_t elementsD
                = std::accumulate(mLengths.begin(), mLengths.end(), size_t{1}, std::multiplies())
                  * std::accumulate(nLengths.begin(), nLengths.end(), size_t{1}, std::multiplies());

            elements = elementsA + elementsB + elementsD + elementsD;
        }

        return elements <= maxElems;
    }

    // Convert vector of individual dimension lengths to correct tensor format for contraction
    // i.e. rank 2: {{m0, m1}, {n0, n1}, {k0, k1}}
    //           -> {{m0, m1, k0, k1}, {n0, n1, k0, k1}, {m0, m1, n0, n1}}
    std::vector<std::vector<size_t>> convertLengthsToTensorFormat(std::vector<size_t> mLengths,
                                                                  std::vector<size_t> nLengths,
                                                                  std::vector<size_t> kLengths)
    {
        int                              rank = mLengths.size();
        std::vector<std::vector<size_t>> tempLengths(3, std::vector<size_t>(rank * 2));

        for(int i = 0; i < rank; i++)
        {
            // Construct A tensor
            tempLengths[0][i]        = mLengths[i];
            tempLengths[0][i + rank] = kLengths[i];

            // Construct B tensor
            tempLengths[1][i]        = nLengths[i];
            tempLengths[1][i + rank] = kLengths[i];

            // Construct D/E tensor
            tempLengths[2][i]        = mLengths[i];
            tempLengths[2][i + rank] = nLengths[i];
        }

        return tempLengths;
    }

    std::vector<std::vector<size_t>> generateRandom2DLengths(size_t lower,
                                                             size_t upper,
                                                             size_t step,
                                                             size_t rank,
                                                             size_t maxElems,
                                                             size_t totalSizes,
                                                             bool   randomizeFromRange)
    {
        std::vector<std::vector<size_t>> outputLengths;
        int                              numSizes = 0;

        std::vector<size_t> sizes       = generateRange(lower, upper, step);
        int                 uniqueSizes = sizes.size();
        size_t              range       = upper - lower;

        // Seed the randomization
        std::srand(0);

        // Create a hash map for storing previously utilized size indices to prevent duplicates
        std::unordered_map<std::vector<size_t>, int, Hash> map;

        // Determine if the possible number of size combinations exceeds the given totalSizes and update accordingly
        size_t possibleSizes = randomizeFromRange ? pow(range, rank) : pow(uniqueSizes, rank);
        totalSizes           = std::min(totalSizes, possibleSizes);

        while(numSizes < totalSizes)
        {
            std::vector<size_t> tempLengths(rank);

            for(int i = 0; i < rank; i++)
            {
                if(randomizeFromRange)
                {
                    size_t val     = lower + std::rand() % range;
                    tempLengths[i] = val;
                }
                else
                {
                    int idx        = std::rand() % uniqueSizes;
                    tempLengths[i] = sizes[idx];
                }
            }

            // Determine if the generated lengths exceeds maxElems
            if(!areValidLengths(tempLengths, {}, {}, rank, 2, maxElems))
            {
                continue;
            }

            // Add the hashed index to the hash map
            map[tempLengths]++;

            // Only add the generated lengths to outputLengths if unique
            if(map[tempLengths] == 1)
            {
                outputLengths.push_back(tempLengths);
                numSizes++;
            }
        }

        return outputLengths;
    }

    std::vector<std::vector<size_t>> cull2DLengths(std::vector<std::vector<size_t>> lengths,
                                                   size_t                           rank,
                                                   size_t                           maxElems,
                                                   size_t                           totalSizes)
    {
        std::vector<std::vector<size_t>> finalLengths;

        for(auto i = 0; i < lengths.size(); i++)
        {
            if(areValidLengths(lengths[i], {}, {}, rank, 2, maxElems))
            {
                finalLengths.push_back(lengths[i]);
            }
        }

        return finalLengths;
    }

    void generate2DLengths(std::vector<std::vector<size_t>>& outputLengths,
                           size_t                            lower,
                           size_t                            upper,
                           size_t                            step,
                           size_t                            rank,
                           size_t                            maxElems,
                           size_t                            totalSizes         = 0,
                           bool                              randomizeFromRange = false)
    {
        // Generate sizes based on lower/upper bounds and step size
        std::vector<size_t> sizes = generateRange(lower, upper, step);

        // If totalSizes is given, randomly generate N total lengths based on lower/upper bounds and step size
        if(totalSizes != 0)
        {
            outputLengths = generateRandom2DLengths(
                lower, upper, step, rank, maxElems, totalSizes, randomizeFromRange);
            std::sort(outputLengths.begin(), outputLengths.end());
            return;
        }

        // If rank == 1 simply return sizes as a 2D vector
        if(rank == 1)
        {
            for(int i = 0; i < sizes.size(); i++)
            {
                outputLengths.push_back({sizes[i]});
            }
            return;
        }

        // Create the first 2 dimensions for lengths for rank >= 2
        // i.e. m0, m1
        std::vector<std::vector<size_t>> tempLengths;
        tempLengths = combine(sizes, sizes);

        for(int i = 2; i < rank; i++)
        {
            tempLengths = combine(tempLengths, sizes);
        }

        // Cull the generated sizes that exceed maxElems
        outputLengths = cull2DLengths(tempLengths, rank, maxElems, totalSizes);
    }

    std::vector<std::vector<std::vector<size_t>>> generateRandom3DLengths(size_t lower,
                                                                          size_t upper,
                                                                          size_t step,
                                                                          size_t rank,
                                                                          size_t maxElems,
                                                                          size_t totalSizes,
                                                                          bool   randomizeFromRange)
    {
        std::vector<std::vector<std::vector<size_t>>> outputLengths;
        size_t                                        numSizes = 0;

        std::vector<size_t> sizes       = generateRange(lower, upper, step);
        int                 uniqueSizes = sizes.size();
        size_t              range       = upper - lower;

        // Seed the randomization
        std::srand(0);

        // Create a hash map for storing previously utilized size indices to prevent duplicates
        std::unordered_map<std::vector<std::vector<size_t>>, int, Hash> map;

        // Determine if the possible number of size combinations exceeds the given totalSizes and update accordingly
        size_t possibleSizes
            = randomizeFromRange ? pow(range, rank * 3) : pow(uniqueSizes, rank * 3);
        totalSizes = std::min(totalSizes, possibleSizes);

        // Randomized lengths are generated as {{m0, m1, ...}, {n0, n1, ...}, {k0, k1, ...}}
        while(numSizes < totalSizes)
        {
            std::vector<std::vector<size_t>> tempLengths(3, std::vector<size_t>(rank));

            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < rank; j++)
                {
                    if(randomizeFromRange)
                    {
                        size_t val        = lower + std::rand() % range;
                        tempLengths[i][j] = val;
                    }
                    else
                    {
                        int idx           = std::rand() % uniqueSizes;
                        tempLengths[i][j] = sizes[idx];
                    }
                }
            }

            // Determine if the generated lengths exceeds maxElems
            std::vector<size_t> mLengths = tempLengths[0];
            std::vector<size_t> nLengths = tempLengths[1];
            std::vector<size_t> kLengths = tempLengths[2];

            if(!areValidLengths(mLengths, nLengths, kLengths, rank, 3, maxElems))
            {
                continue;
            }

            // Add the hashed index to the hash map
            map[tempLengths]++;

            // Only add the generated lengths to outputLengths if unique
            if(map[tempLengths] == 1)
            {
                // Convert the generated lengths to tensor format
                tempLengths = convertLengthsToTensorFormat(mLengths, nLengths, kLengths);
                outputLengths.push_back(tempLengths);
                numSizes++;
            }
        }

        return outputLengths;
    }

    std::vector<std::vector<std::vector<size_t>>> cull3DLengths(
        std::vector<std::vector<size_t>>& lengths, size_t rank, size_t maxElems, size_t totalSizes)
    {
        std::vector<std::vector<std::vector<size_t>>> finalLengths;
        std::vector<size_t>                           mLengths, nLengths, kLengths;

        for(auto i = 0; i < lengths.size(); i++)
        {
            std::vector<std::vector<size_t>> tempLengths;
            bool                             isValid = false;

            mLengths = std::vector<size_t>(lengths[i].begin(), lengths[i].begin() + rank);
            nLengths
                = std::vector<size_t>(lengths[i].begin() + rank, lengths[i].begin() + (rank * 2));
            kLengths = std::vector<size_t>(lengths[i].begin() + (rank * 2), lengths[i].end());

            if(areValidLengths(mLengths, nLengths, kLengths, rank, 3, maxElems))
            {
                finalLengths.push_back(convertLengthsToTensorFormat(mLengths, nLengths, kLengths));
            }
        }

        return finalLengths;
    }

    void generate3DLengths(std::vector<std::vector<std::vector<size_t>>>& outputLengths,
                           size_t                                         lower,
                           size_t                                         upper,
                           size_t                                         step,
                           size_t                                         rank,
                           size_t                                         maxElems,
                           size_t                                         totalSizes = 0,
                           bool randomizeFromRange                                   = false)
    {
        // If totalSizes is given, randomly generate N total lengths based on lower/upper bounds and step size
        if(totalSizes != 0)
        {
            outputLengths = generateRandom3DLengths(
                lower, upper, step, rank, maxElems, totalSizes, randomizeFromRange);
            std::sort(outputLengths.begin(), outputLengths.end());
            return;
        }

        // Generate sizes based on lower/upper bounds and step size
        std::vector<size_t> sizes = generateRange(lower, upper, step);

        // Create the first 3 dimensions for lengths
        // i.e. m0, n0, k0
        std::vector<std::vector<size_t>> tempLengths;
        tempLengths = combine(sizes, sizes);
        tempLengths = combine(tempLengths, sizes);

        for(int i = 3; i < rank * 3; i++)
        {
            tempLengths = combine(tempLengths, sizes);
        }

        // Cull the generated sizes that exceed maxElems
        outputLengths = cull3DLengths(tempLengths, rank, maxElems, totalSizes);
    }

} // namespace hiptensor

#endif // HIPTENSOR_TEST_LENGTH_GENERATION_HPP
