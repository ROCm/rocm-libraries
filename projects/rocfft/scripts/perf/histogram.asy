// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import graph;
import stats;

size(300, 200, IgnoreAspect);

scale(Linear, Linear);

string filename;

int nbinmult = 2;

usersetting();

if(filename == "") {
  filename = getstring("histogram data file");
}

file fin = input(filename).line();
real[] b = fin;

real bounds = infinity; //40000;

real[] a;
for(int i = 0; i < b.length; ++i) {
  if(abs(b[i]) < bounds) {
    a.push(b[i]);
  }
}

int N = nbinmult * bins(a);

real lower = min(0, min(a));
real upper = max(0, max(a));

real dx=(upper-lower)/N;
real[] freq=frequency(a,lower,upper,N);
write(freq);

if(a.length == 1) {
  real[] bin = {lower, upper};
  real[] count = {1};
  histogram(bins = bin,
	    count = count,
	    low=0.0,
	    fillpen=lightred,
	    drawpen=black,
	    bars=false,
	    legend="",
	    markersize=legendmarkersize);
} else {
  histogram(a,
	    lower,
	    upper,
	    N,
	    normalize=false,
	    low=0,
	    lightred,
	    black,
	    bars=true);
}

xequals(0.0);

//label((min(a), 0), string(min(a), 3), 1.5S);
//label((max(a), 0), string(max(a), 3), 1.5S);

if(a.length > 1) {
  
  real Step = 0.0;
  if(max(a) - min(a) < 4) {
    real order = ceil(log(max(a) - min(a))/log(10));
    Step = 0.5 * 10**(order-1);
  }
  xaxis("Speedup \%", BottomTop, LeftTicks(Step=Step));
} else {
  xaxis("Speedup \%", BottomTop, LeftTicks);
}
yaxis("Number of Transforms", LeftRight, RightTicks);


//add(legend(),point(E),20E);
