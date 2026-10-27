// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// Decides which benchmark candidates' outputs are trustworthy: groups the candidates whose
// outputs agree into cohorts and elects the passing cohort. Pure host code; it makes no
// device call and knows nothing of kernels, only candidate indexes.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/OutputAgreement.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkValidation.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// One benchmark pass's vote. submit() each candidate whose output was captured, then
/// decide(). A candidate is classified at submit time:
///  - untouched: some output with a NaN encoding still holds the fill pattern; it joins no
///    cohort and fails.
///  - otherwise it joins the first cohort whose founder it agrees with on every comparable
///    output, or founds a new one. Only founders keep their images, so peak retention is the
///    cohort count plus the image being submitted.
class OutputCohorts
{
public:
    enum class Verdict
    {
        /// No candidate computed.
        NONE_PASS,
        /// Computing candidates exist but no output can be value-compared; all of them pass.
        INCONCLUSIVE,
        /// One computing candidate; nothing to compare it with.
        UNVERIFIED,
        /// The largest cohort holds more than half of the computing candidates.
        MAJORITY,
        /// No strict majority; the cohort of the lowest-index computing candidate passes.
        HEURISTIC_TIEBREAK
    };

    struct Failure
    {
        size_t index = 0;
        bool untouched = false;
        /// The first untouched output, or the first disagreement between this candidate's
        /// cohort founder and the passing cohort's founder.
        std::string reason;
        /// Lowest-index member of this candidate's cohort; equals `index` for untouched ones.
        size_t cohortFounder = 0;
    };

    struct Decision
    {
        Verdict verdict = Verdict::NONE_PASS;
        /// Ascending candidate indexes.
        std::vector<size_t> passing;
        /// Ascending candidate indexes: computing candidates outside the passing cohort, and
        /// every untouched candidate.
        std::vector<Failure> failed;
        /// Candidates that wrote their outputs.
        size_t computing = 0;
        size_t cohorts = 0;
        /// Meaningful for MAJORITY and HEURISTIC_TIEBREAK.
        size_t passingFounder = 0;
    };

    explicit OutputCohorts(std::vector<OutputBufferSpec> specs)
        : _specs(std::move(specs))
    {
        _names.reserve(_specs.size());
        for(const auto& spec : _specs)
        {
            _names.push_back("output " + std::to_string(spec.uid));
            _comparable = _comparable || output_agreement::elementBytes(spec.dataType) > 0;
        }
    }

    /// @p images holds one image per spec, in spec order; empty when there are no specs.
    void submit(size_t index, OutputImages&& images)
    {
        for(size_t i = 0; i < _specs.size() && i < images.size(); ++i)
        {
            if(output_agreement::imageUntouched(_specs[i].dataType, images[i]))
            {
                _untouched.push_back(
                    {index, true, _names[i] + " left untouched (still holds the NaN fill)", index});
                return;
            }
        }

        _touched.push_back(index);
        if(!_comparable)
        {
            return;
        }

        for(auto& cohort : _cohorts)
        {
            if(agree(cohort.images, images))
            {
                cohort.members.push_back(index);
                OutputImages().swap(images);
                return;
            }
        }

        // A founder keeps only the images that will be compared.
        for(size_t i = 0; i < _specs.size() && i < images.size(); ++i)
        {
            if(output_agreement::elementBytes(_specs[i].dataType) == 0)
            {
                std::vector<uint8_t>().swap(images[i]);
            }
        }
        _cohorts.push_back({std::move(images), {index}});
    }

    Decision decide() const
    {
        Decision decision;
        auto touched = _touched;
        std::sort(touched.begin(), touched.end());
        decision.computing = touched.size();
        decision.cohorts = _cohorts.size();
        decision.failed = _untouched;

        if(touched.empty())
        {
            decision.verdict = Verdict::NONE_PASS;
        }
        else if(!_comparable)
        {
            decision.verdict = Verdict::INCONCLUSIVE;
            decision.passing = touched;
        }
        else if(touched.size() == 1)
        {
            decision.verdict = Verdict::UNVERIFIED;
            decision.passing = touched;
        }
        else
        {
            decideByCohorts(touched, decision);
        }

        std::sort(decision.failed.begin(),
                  decision.failed.end(),
                  [](const Failure& lhs, const Failure& rhs) { return lhs.index < rhs.index; });
        return decision;
    }

    size_t cohortCount() const
    {
        return _cohorts.size();
    }

    /// Bytes of the images kept for comparison.
    size_t retainedBytes() const
    {
        size_t total = 0;
        for(const auto& cohort : _cohorts)
        {
            for(const auto& image : cohort.images)
            {
                total += image.size();
            }
        }
        return total;
    }

private:
    struct Cohort
    {
        OutputImages images;
        /// Ascending; the first is the founder.
        std::vector<size_t> members;
    };

    /// True when every comparable output of @p candidate agrees with @p founder.
    bool agree(const OutputImages& founder, const OutputImages& candidate) const
    {
        return firstDisagreement(founder, candidate).empty();
    }

    /// The first disagreement between two image sets, in spec order; empty when they agree.
    std::string firstDisagreement(const OutputImages& left, const OutputImages& right) const
    {
        for(size_t i = 0; i < _specs.size() && i < left.size() && i < right.size(); ++i)
        {
            if(output_agreement::elementBytes(_specs[i].dataType) == 0)
            {
                continue;
            }
            auto text = output_agreement::firstDisagreement(
                _names[i], _specs[i].dataType, left[i], right[i]);
            if(!text.empty())
            {
                return text;
            }
        }
        return {};
    }

    void decideByCohorts(const std::vector<size_t>& touched, Decision& decision) const
    {
        size_t passing = 0;
        for(size_t i = 1; i < _cohorts.size(); ++i)
        {
            // Strictly larger only, so the first largest cohort wins a tie.
            if(_cohorts[i].members.size() > _cohorts[passing].members.size())
            {
                passing = i;
            }
        }

        if(_cohorts[passing].members.size() * 2 > touched.size())
        {
            decision.verdict = Verdict::MAJORITY;
        }
        else
        {
            decision.verdict = Verdict::HEURISTIC_TIEBREAK;
            for(size_t i = 0; i < _cohorts.size(); ++i)
            {
                const auto& members = _cohorts[i].members;
                if(std::find(members.begin(), members.end(), touched.front()) != members.end())
                {
                    passing = i;
                    break;
                }
            }
        }

        const auto& winners = _cohorts[passing];
        decision.passing = winners.members;
        std::sort(decision.passing.begin(), decision.passing.end());
        decision.passingFounder = winners.members.front();

        for(size_t i = 0; i < _cohorts.size(); ++i)
        {
            if(i == passing)
            {
                continue;
            }
            auto reason = firstDisagreement(_cohorts[i].images, winners.images);
            if(reason.empty())
            {
                reason = "output disagrees with the passing cohort";
            }
            for(const size_t member : _cohorts[i].members)
            {
                decision.failed.push_back({member, false, reason, _cohorts[i].members.front()});
            }
        }
    }

    std::vector<OutputBufferSpec> _specs;
    std::vector<std::string> _names;
    bool _comparable = false;
    std::vector<size_t> _touched;
    std::vector<Failure> _untouched;
    std::vector<Cohort> _cohorts;
};

inline const char* verdictName(OutputCohorts::Verdict verdict)
{
    switch(verdict)
    {
    case OutputCohorts::Verdict::NONE_PASS:
        return "none-pass";
    case OutputCohorts::Verdict::INCONCLUSIVE:
        return "inconclusive";
    case OutputCohorts::Verdict::UNVERIFIED:
        return "unverified";
    case OutputCohorts::Verdict::MAJORITY:
        return "majority";
    case OutputCohorts::Verdict::HEURISTIC_TIEBREAK:
        return "heuristic-tiebreak";
    default:
        break;
    }
    return "unknown";
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
