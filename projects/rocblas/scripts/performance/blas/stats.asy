//   Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
//   ies of the Software, and to permit persons to whom the Software is furnished
//   to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
//   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
//   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
//   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
//   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Compute the ceiling of a / b.
int ceilquot(int a, int b)
{
  return (a + b - 1) # b;
}

// Compute the median of a sample.
real getmedian(real[] vals)
{
   vals = sort(vals);
   int half = quotient(vals.length, 2);
   return (vals.length % 2 == 0)
      ? 0.5 * ( vals[half - 1] + vals[half] ) : vals[half];
   return 0;
}

real getmean(real[] vals)
{
   return sum(vals) / vals.length;
}

// Bootstrap resampling method for computing the 95% band for the
// median.
real[] mediandev(real[] vals)
{
    real[] medlh = new real[3];
   int nsample = vals.length;
   real resample[] = new real[nsample];
   for(int i = 0; i < nsample; ++i) {
      resample[i] = vals[i];
   }
   real median = getmedian(resample);
   medlh[0] = median;

   // Number of resamples to perform:
   int nperm = 2000;
   real medians[] = new real[nperm];
   for(int i = 0; i < nperm; ++i) {
      for(int j = 0; j < nsample; ++j) {
	 resample[j] = vals[rand() % nsample];
      }
      medians[i] = getmedian(resample);
   }
   medians = sort(medians);
   real low = medians[(int)floor(nperm * 0.025)];
   real high = medians[(int)ceil(nperm * 0.975)];

   medlh[1] = low;
   medlh[2] = high;

   return medlh;
}

real[] ratiodev(real[] vA, real[] vB) {
    real[] medlh = new real[2];


    real ratio = getmedian(vA) / getmedian(vB);

    int nboot = 2000;
    real[] ratios =new real[nboot];
    ratios[0] = ratio;
    for(int n = 1; n < nboot; ++n) {
        real valA = vA[rand() % vA.length];
        real valB = vB[rand() % vB.length];
        ratios[n] = valA / valB;
    }
    ratios = sort(ratios);
    real low = ratios[(int)floor(nboot * 0.025)];
    real high = ratios[(int)ceil(nboot * 0.975)];

   medlh[0] = low;
   medlh[1] = high;

   return medlh;
}
