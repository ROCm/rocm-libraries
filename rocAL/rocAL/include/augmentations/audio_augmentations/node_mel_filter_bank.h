/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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
#include "pipeline/graph.h"
#include "pipeline/node.h"
#include "rocal_api_types.h"

class MelFilterBankNode : public Node {
   public:
     MelFilterBankNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
     MelFilterBankNode() = delete;
     void init(float freq_high, float freq_low, RocalMelScaleFormula mel_formula, int nfilter, bool normalize, float sample_rate);

   protected:
     void create_node() override;
     void update_node() override;

   private:
     float _freq_high = 0;
     float _freq_low = 0;
     int _mel_formula = 0;
     int _nfilter = 128;
     float _sample_rate = 44100;
     bool _normalize = true;
};
