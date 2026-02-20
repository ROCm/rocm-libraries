/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2021 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_DATA_HPP
#define ROCSPARSE_DATA_HPP

#include "rocsparse_arguments.hpp"
#include "test_cleanup.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

// Class used to read Arguments data into the tests
class RocSPARSE_TestData
{
    // data filename
    static auto& filename()
    {
        static std::string filename
            = "(Uninitialized data. RocSPARSE_TestData::set_filename needs to be called first.)";
        return filename;
    }

    // filter iterator
    class iterator : public std::istream_iterator<Arguments>
    {
        bool (*const filter)(const Arguments&) = nullptr;

        // Skip entries for which filter is false
        void skip_filter()
        {
            if(filter)
                while(*this != std::istream_iterator<Arguments>{} && !filter(**this))
                    ++*static_cast<std::istream_iterator<Arguments>*>(this);
        }

    public:
        // Constructor takes a filter and iterator
        explicit iterator(bool filter_(const Arguments&), std::istream_iterator<Arguments> iter)
            : std::istream_iterator<Arguments>(iter)
            , filter(filter_)
        {
            skip_filter();
        }

        // Default end iterator and nullptr filter
        iterator() = default;

        // Preincrement iterator operator with filtering
        iterator& operator++()
        {
            ++*static_cast<std::istream_iterator<Arguments>*>(this);
            skip_filter();
            return *this;
        }

        // We do not need a postincrement iterator operator
        // We delete it here so that the base class's isn't silently called
        // To implement it, use "auto old = *this; ++*this; return old;"
        iterator operator++(int) = delete;
    };

public:
    // Initialize filename, optionally removing it at exit
    static void set_filename(std::string name, bool remove_atexit = false)
    {
        filename() = std::move(name);
        if(remove_atexit)
        {
            auto cleanup = [] { remove(filename().c_str()); };
            atexit(cleanup);
            auto err = at_quick_exit(cleanup);
            if(err)
            {
            }
        }
    }

    // begin() iterator which accepts an optional filter.
    static iterator begin(bool filter(const Arguments&) = nullptr)
    {
        static std::ifstream* ifs;

        // If this is the first time, or after test_cleanup::cleanup() has been called
        if(!ifs)
        {
            // Allocate a std::ifstream and register it to be deleted during cleanup
            ifs = test_cleanup::allocate(&ifs, filename(), std::ifstream::binary);
            if(!ifs || ifs->fail())
            {
                std::cerr << "Cannot open " << filename() << ": " << strerror(errno) << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        // We re-seek the file back to position 0
        ifs->clear();
        ifs->seekg(0);

        // Validate the data file format
        Arguments::validate(*ifs);

        // We create a filter iterator which will choose only the test cases we want right now.
        // This is to preserve Gtest structure while not creating no-op tests which "always pass".
        return iterator(filter, std::istream_iterator<Arguments>(*ifs));
    }

    // end() iterator
    static iterator end()
    {
        return {};
    }
};

#endif // ROCSPARSE_DATA_HPP
