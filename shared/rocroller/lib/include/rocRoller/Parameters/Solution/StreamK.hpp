/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#pragma once

#include <string>

namespace rocRoller
{
    enum class StreamKMode
    {
        None,
        Standard,
        TwoTile,
        TwoTileDPFirst,
        Count,
    };

    struct StreamKConfig
    {
        StreamKMode mode = StreamKMode::None;

        auto operator<=>(const StreamKConfig&) const = default;

        explicit operator bool() const
        {
            return mode == StreamKMode::Standard or mode == StreamKMode::TwoTile
                   or mode == StreamKMode::TwoTileDPFirst;
        }

        bool operator==(StreamKMode rhs) const
        {
            return mode == rhs;
        }

        StreamKConfig& operator=(StreamKMode newMode)
        {
            mode = newMode;
            return *this;
        }

        bool isTwoTileMode() const
        {
            return mode == StreamKMode::TwoTile or mode == StreamKMode::TwoTileDPFirst;
        }
    };

    std::string   toString(StreamKMode mode);
    std::ostream& operator<<(std::ostream& stream, StreamKMode const& mode);

    std::string   toString(StreamKConfig const& config);
    std::ostream& operator<<(std::ostream& stream, StreamKConfig const& config);
} // namespace rocRoller
