/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

template <typename T>
class Parameter {
   public:
    virtual T default_value() const = 0;
    ///
    /// \return returns the updated value of the parameter
    virtual T get() = 0;

    /// used to internally renew state of the parameter if needed (for random parameters)
    virtual void renew(){};

    /// allocates memory for the array with specified size
    virtual void create_array(unsigned size){};

    /// used to fetch the updated param values
    virtual std::vector<T> get_array() { return {}; };

    virtual ~Parameter() {}
    ///
    /// \return returns if this parameter takes a single value (vs a range of values or many values)
    virtual bool single_value() const = 0;
};
