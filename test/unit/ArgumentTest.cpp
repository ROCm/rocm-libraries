
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/CodeGen/WaitCount.hpp>
#include <rocRoller/Coordinate.hpp>

using namespace rocRoller;

TEST(ArgumentsTest, Basic)
{
    size_t len void* args;
    KernelArguments  ks(args, len);

    T_LOAD_LINEAR load(...);

    ks.register(load)

        // -->

        load.sizes[0]
            ->value();
    load.strides[0]->value();

    if(load.addPredicateIfApplicable<Predicates::UnitStride>())
    {
    }

    if(load.addPredicateIfApplicable <)

        if(load.hasPredicate<Predicates::UnitStride>())
        {
        }
}

void FindExecutionDimensions_BottomUp(T_EXECUTE* node)
{
    auto inputData  = getInputNodes(node);
    auto outputData = getOutputNodes(node);

    // Asserts that all input and output dimensions have the same sizes or
    // have appropriate reduction/broadcast applied.  This might be fused
    // into `findBestDimensionOrdering()`.
    assertCompatibleDimensions(*inputData, *outputData);

    // Sorts the dimensions to the most compatible ordering to each other
    // e.g. could swap two dimensions if both out of coalescing order.
    // Will have to use heuristics to determine relative importance of
    // coalescing of different dimensions.
    auto dataDimensions = findBestDimensionOrdering<T_EXECUTE>(inputData, outputData);

    auto dimIter = dataDimensions.begin();

    assert(dimIter != dataDimensions.end());

    auto curDim = *dimIter;
    dimIter++;

    auto wavefrontDim = curDim;
    auto workgroupDim = Coordinate::Unity();
    auto gridDim = std::make_array(Coordinate::Unity(), Coordinate::Unity(), Coordinate::Unity());

    auto fitsIntoWavefront
        = Predicates::Dimensions::LessThanOrEqual(m_context->hw->wavefrontSize());

    if(fitsIntoWavefront(curDim))
    {
        //curDim->addPredicate(fitsIntoWavefront);

        if(dimIter == dataDimensions.end())
        {
            curDim->addPredicate(fitsIntoWavefront);
            wavefrontDim = curDim;
            // Single dimension with size < 64 (or 32). Not much we can do to optimize.
            node->setExecutionDimensions(wavefrontDim, workgroupDim, gridDim);
            return;
        }

        auto nextDim = *dimIter;
        dimIter++;
        auto proposedDims = Coordinate::Merge({curDim, nextDim})

            while(fitsIntoWavefront(proposedDims) && dimIter != dataDimensions.end())
        {
            curDim  = proposedDims;
            nextDim = *dimIter;
            dimIter++;
            proposedDims = Coordinate::Merge({curDim, nextDim});
        }

        wavefrontDim = curDim;
        curDim->addPredicate(fitsIntoWavefront);
    }

    if(curDim->applyPredicateIfApplicable(fitsIntoWavefront))
    {
        if(the next dimension has)
    }
}

Generator<Instruction> calculateIndex(Coordinate data, Coordinate Execution) {}

struct LowerIndex_GCN
{

    void visit(Regular const& value) {}
};

void NormalGEMM() {}
