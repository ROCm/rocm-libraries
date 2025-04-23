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
#include <algorithm>  // std::remove_if
#include <cstdlib>
#include <ctime>
#include <memory>
#include <numeric>  // std::inner_product, std::accumulate
#include <random>
#include <stdexcept>
#include <thread>
#include <variant>
#include <vector>

#include "pipeline/log.h"
#include "parameters/parameter.h"
template <typename T>
class UniformRand : public Parameter<T> {
   public:
    UniformRand(T start, T end, unsigned seed = 0) : _generator(seed) {
        update(start, end);
        renew();
    }

    explicit UniformRand(T start, unsigned seed = 0) : UniformRand(start, start, seed) {}

    T default_value() const override {
        return static_cast<T>((_start + _end) / static_cast<T>(2));
    }

    T get() override {
        return _updated_val;
    };

    std::vector<T> get_array() override {
        return _param_values;
    }

    void renew_value() {
        std::unique_lock<std::mutex> lock(_lock);
        auto val = _generator();

        if (single_value()) {
            // If there is only a single value possible for the random variable
            // don't waste time on calling the rand function , just return it.
            _updated_val = _start;
        } else {
            _updated_val = static_cast<T>(
                ((double)val / (double)_generator.max()) * ((double)_end - (double)_start) + (double)_start);
        }
    }

    void renew_array() {
        for (uint i = 0; i < _size; i++) {
            renew_value();
            _param_values[i] = _updated_val;
        }
    }

    void renew() override {
        if (_param_values.size()) {
            renew_array();
        } else {
            renew_value();
        }
    }
    int update(T start, T end) {
        std::unique_lock<std::mutex> lock(_lock);
        if (end < start)
            end = start;

        _start = start;
        _end = end;
        return 0;
    }

    void create_array(unsigned vector_size) override {
        if (_param_values.size() == 0) {
            _param_values.resize(vector_size);
            _size = vector_size;
        }
    }

    bool single_value() const override {
        return (_start == _end);
    }

   private:
    T _start;
    T _end;
    T _updated_val;
    std::vector<T> _param_values;
    std::mt19937 _generator;
    std::mutex _lock;
    unsigned _size;
};

template <typename T>
struct CustomRand : public Parameter<T> {
    CustomRand(
        const T values[],
        const double frequencies[],
        size_t size, unsigned seed = 0) : _generator(seed) {
        update(values, frequencies, size);
        renew();
    }
    int update(
        const T values[],
        const double frequencies[],
        size_t size) {
        std::unique_lock<std::mutex> lock(_lock);

        if (size == 0)
            return -1;

        _values.assign(values, values + size);
        _frequencies.assign(frequencies, frequencies + size);
        _comltv_dist.resize(size, 0);
        double sum = 0;
        // filter out negative values if any, and sum it up
        std::copy_if(
            _frequencies.begin(),
            _frequencies.end(),
            _frequencies.begin(),
            [&](double in) -> double { if(in >= 0) { sum += in; return in;} return 0; });

        // NOTE: If there remains values with probability zero it may cause issues with sampling
        // TODO: Remove values associated with probabilities equal to 0 from the _frequencies and _values

        // Normalize the frequencies , so that the sum is equal to 1.0
        std::transform(
            _frequencies.begin(),
            _frequencies.end(),
            _frequencies.begin(),
            [&](double in) { return (double)in / sum; });

        // Compute the expected value by performing inner product of probs and values
        _mean = std::inner_product(
            _values.begin(),
            _values.end(),
            _frequencies.begin(), 0);

        // Create the partial sum of the probability distribution function (PDF), is used for random generation
        std::partial_sum(
            _frequencies.begin(),
            _frequencies.end(),
            _comltv_dist.begin(),
            std::plus<double>());

        return 0;
    }
    T default_value() const override {
        return static_cast<T>(_mean);
    }

    void renew_value() {
        std::unique_lock<std::mutex> lock(_lock);
        if (single_value()) {
            // If there is only a single value possible for the random variable
            // don't waste time on calling the rand function , just return it.
            _updated_val = _values[0];
        } else {
            // Generate a value between [0 1]
            double rand_val = (double)_generator() / (double)_generator.max();

            // Find the iterators pointing to the first element bigger than idx
            auto it = std::upper_bound(_comltv_dist.begin(), _comltv_dist.end(), rand_val);

            // Get the index and return the associated value
            unsigned idx = std::distance(_comltv_dist.begin(), it);

            _updated_val = _values[idx];
        }
    }

    void renew_array() {
        for (uint i = 0; i < _size; i++) {
            renew_value();
            _param_values[i] = _updated_val;
        }
    }

    void renew() override {
        if (_param_values.size()) {
            renew_array();
        } else {
            renew_value();
        }
    }
    T get() override {
        return _updated_val;
    };

    std::vector<T> get_array() override {
        return _param_values;
    }

    void create_array(unsigned vector_size) override {
        if (_param_values.size() == 0) {
            _param_values.resize(vector_size);
            _size = vector_size;
        }
    }

    bool single_value() const override {
        return (_values.size() == 1);
    }

   private:
    std::vector<T> _values;            //!< Values
    std::vector<double> _frequencies;  //!< Probabilities
    std::vector<double> _comltv_dist;  //!< commulative probabilities
    double _mean;
    T _updated_val;
    std::vector<T> _param_values;  //!< The values will be used in parameter_vx.h file after renewing
    std::mt19937 _generator;
    std::mutex _lock;
    unsigned _size;
};