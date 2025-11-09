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

#include <string>

#define WORKAROUND_ISSUE_1204 1 // ROCm may incorrectly report "sramecc-" for gfx900.
#define WORKAROUND_ISSUE_3001 1

namespace miopen {

struct Handle;

class TargetProperties
{
    struct xnack_t
    {
        const std::string tag{":xnack"};
        virtual ~xnack_t() = default;
    };

    struct sramecc_t
    {
        const std::string tag{":sramecc"};
        virtual ~sramecc_t() = default;
    };

    template <typename T>
    struct TargetProperty : public T
    {
        bool initialized{false};
        bool reported{false};
        bool enabled{false};

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
            return !reported && enabled;
        }

        void Init(const std::string& raw_name, const std::string& dev_name)
        {
#if WORKAROUND_ISSUE_1204
            if(std::is_same_v<T, sramecc_t> && dev_name == "gfx900")
            {
                initialized = true;
                return; // reported == false, enabled == false;
            }
#endif

            // DKMS driver older than 5.9 may report incorrect state of SRAMECC feature.
            // Therefore we compute default SRAMECC and rely on it for now.
            if(std::is_same_v<T, sramecc_t> && (dev_name == "gfx906" || dev_name == "gfx908"))
            {
                reported    = true;
                enabled     = true;
                initialized = true;
            }

            auto tag_pos = raw_name.find(this->tag);
            if(tag_pos != std::string::npos)
            {
                tag_pos += this->tag.length();
                if(raw_name.length() > tag_pos)
                {
                    if(raw_name[tag_pos] == '+')
                    {
                        reported = true;
                        enabled  = true;
                    }
                    if(raw_name[tag_pos] == '-')
                    {
                        reported = true;
                    }
                }
            }

            initialized = true;
        }
    };

    void InitDbId();
    std::string name;
    std::string dbId;
    static const std::size_t MaxWaveScratchSize;
    static const std::size_t MaxLocalMemorySize;

public:
    virtual ~TargetProperties() = default;

    TargetProperty<xnack_t> xnack;
    TargetProperty<sramecc_t> sramecc;

    virtual const std::string& Name() const { return name; }
    const std::string& DbId() const { return dbId; }

    virtual bool isXnackEnabled() const { return xnack.isEnabled(); }

    // bool Sramecc() const { return sramecc.isEnabled(); }
    // bool SrameccReported() const { return sramecc.isReported(); }

    static std::size_t GetMaxWaveScratchSize() { return MaxWaveScratchSize; }
    static std::size_t GetMaxLocalMemorySize() { return MaxLocalMemorySize; }

    void Init(const Handle*);
};

} // namespace miopen

#endif // GUARD_TARGET_PROPERTIES_HPP
