/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#ifndef GUARD_TARGET_PROPERTIES_HPP
#define GUARD_TARGET_PROPERTIES_HPP

#include <miopen/config.hpp>
#include <string>
#include <tuple>
#include <stdexcept>
#include <miopen/stringutils.hpp>

#define WORKAROUND_ISSUE_1204 1 // ROCm may incorrectly report "sramecc-" for gfx900.
#define WORKAROUND_ISSUE_3001 1

namespace miopen {

static inline bool IsTF32Supported(const std::string& device_name)
{
    return device_name == "gfx942" || StartsWith(device_name, "gfx95");
}

struct Handle;

class TargetProperties
{
    struct xnack_t
    {
        std::string tag;

        xnack_t(const std::string& tag_) : tag(tag_) {}

        auto init(const std::string& raw_name, const std::string&) const
        {
            bool initialized = false;
            bool reported    = false;
            bool enabled     = false;

            auto tag_pos = raw_name.find(tag);
            if(tag_pos != std::string::npos)
            {
                tag_pos += tag.length();
                if(raw_name.length() > tag_pos)
                {
                    reported = true;
                    enabled  = (raw_name[tag_pos] == '+');
                }
            }

            initialized = true;
            return std::make_tuple(initialized, reported, enabled);
        }
    };

    struct sramecc_t
    {
        std::string tag;

        sramecc_t(const std::string& tag_) : tag(tag_) {}

        auto init(const std::string& raw_name, const std::string& dev_name) const
        {
            bool initialized = false;
            bool reported    = false;
            bool enabled     = (dev_name == "gfx906" || dev_name == "gfx908");

#if WORKAROUND_ISSUE_1204
            if(dev_name == "gfx900")
            {
                reported = false;
            }
            else
#endif
            {
                auto tag_pos = raw_name.find(tag);
                if(tag_pos != std::string::npos)
                {
                    tag_pos += tag.length();
                    if(raw_name.length() > tag_pos)
                    {
                        reported = (raw_name[tag_pos] == '+');
                    }
                }
                else
                {
                    reported = enabled;
                }
            }

            initialized = true;
            return std::make_tuple(initialized, reported, enabled);
        }
    };

    template <typename T>
    struct TargetProperty : public T
    {
        bool initialized = false;
        bool reported    = false;
        bool enabled     = false;

        TargetProperty() = default;
        TargetProperty(const std::string& tag_) : T(tag_) {}

        void CheckInit() const
        {
            if(!initialized)
                throw std::runtime_error("Error: not initialized targetProperty " + this->tag);
        }
        bool isReported() const
        {
            CheckInit();
            return reported;
        }
        bool isEnabled() const
        {
            CheckInit();
            return reported && enabled;
        }
        bool isDisabled() const
        {
            CheckInit();
            return !(reported && enabled);
        }

        void init(const std::string& raw_name, const std::string& dev_name)
        {
            std::tie(initialized, reported, enabled) = T::init(raw_name, dev_name);
        }
    };

    struct TargetPropertyXnack : public TargetProperty<xnack_t>
    {
        TargetPropertyXnack() : TargetProperty<xnack_t>(":xnack") {}
    };

    struct TargetPropertySramecc : public TargetProperty<sramecc_t>
    {
        TargetPropertySramecc() : TargetProperty<sramecc_t>(":sramecc") {}
    };

    void InitDbId();

    std::string name;
    std::string dbId;
    static const std::size_t MaxWaveScratchSize;
    static const std::size_t MaxLocalMemorySize;

public:
    virtual ~TargetProperties() = default;

    TargetPropertyXnack xnack;
    TargetPropertySramecc sramecc;

    virtual const std::string& Name() const { return name; }
    const std::string& DbId() const { return dbId; }

    virtual bool isXnackEnabled() const { return xnack.isEnabled(); }

    static std::size_t GetMaxWaveScratchSize() { return MaxWaveScratchSize; }
    static std::size_t GetMaxLocalMemorySize() { return MaxLocalMemorySize; }

    MIOPEN_INTERNALS_EXPORT void Init(const Handle*);
};

} // namespace miopen

#endif // GUARD_TARGET_PROPERTIES_HPP
