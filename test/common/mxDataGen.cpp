#include "common/mxDataGen.hpp"

namespace rocRoller
{
    void setOptions(DataGeneratorOptions& opts,
                    const float           min,
                    const float           max,
                    int                   blockScaling,
                    const DataPattern     pattern)
    {
        opts.pattern      = pattern;
        opts.min          = min;
        opts.max          = max;
        opts.blockScaling = blockScaling;
    }
}
