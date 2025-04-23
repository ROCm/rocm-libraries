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
#include <memory>
#include <mutex>
#include <set>
#include <thread>

#include "parameters/parameter_random.h"
#include "parameters/parameter_simple.h"

const int MAX_SEEDS = 1024;

enum class RocalParameterType {
    DETERMINISTIC = 0,
    RANDOM_UNIFORM,
    RANDOM_CUSTOM
};

struct IntParam {
    IntParam(
        Parameter<int>* core,
        RocalParameterType type) : core(core),
                                   type(type) {}
    Parameter<int>* core;
    const RocalParameterType type;
};

struct FloatParam {
    FloatParam(
        Parameter<float>* core,
        RocalParameterType type) : core(core),
                                   type(type) {}
    Parameter<float>* core;
    const RocalParameterType type;
};

Parameter<int>* core(IntParam* arg);

Parameter<float>* core(FloatParam* arg);

using pParam = std::variant<IntParam*, FloatParam*>;
using pParamCore = std::variant<Parameter<int>*, Parameter<float>*>;

bool validate_custom_rand_param(pParam arg);
bool validate_uniform_rand_param(pParam rand_obj);
bool validate_simple_rand_param(pParam arg);

class ParameterFactory {
   public:
    static ParameterFactory* instance();
    ~ParameterFactory();
    void renew_parameters();
    void set_seed(unsigned seed);
    unsigned get_seed();
    void generate_seed();
    int64_t get_seed_from_seedsequence();

    template <typename T>
    Parameter<T>* create_uniform_rand_param(T start, T end) {
        auto gen = new UniformRand<T>(start, end, _seed);
        _parameters.insert(gen);
        return gen;
    }
    template <typename T>
    Parameter<T>* create_single_value_param(T value) {
        auto gen = new SimpleParameter<T>(value);
        _parameters.insert(gen);
        return gen;
    }
    template <typename T>
    void destroy_param(Parameter<T>* param) {
        if (_parameters.find(param) != _parameters.end())
            _parameters.erase(param);
        delete param;
    }
    IntParam* create_uniform_int_rand_param(int start, int end);
    FloatParam* create_uniform_float_rand_param(float start, float end);
    IntParam* create_custom_int_rand_param(const int* value, const double* frequencies, size_t size);
    FloatParam* create_custom_float_rand_param(const float* value, const double* frequencies, size_t size);
    IntParam* create_single_value_int_param(int value);
    FloatParam* create_single_value_float_param(float value);

   private:
    long long unsigned _seed;
    std::set<pParamCore> _parameters;  //<! Keeps the random generators used to randomized the augmentation parameters
    static ParameterFactory* _instance;
    static std::mutex _mutex;
    ParameterFactory();
    std::vector<int64_t> _seed_vector;
    int _seed_sequence_idx = 0;
};
