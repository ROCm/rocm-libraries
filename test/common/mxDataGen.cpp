#include "common/mxDataGen.hpp"

namespace rocRoller
{
    DGen::DataGeneratorOptions
        setOptions(const float min, const float max, int blockScaling, const DataPattern pattern)
    {
        DataGeneratorOptions opts;
        opts.pattern      = pattern;
        opts.min          = min;
        opts.max          = max;
        opts.blockScaling = blockScaling;
        return opts;
    }
}
