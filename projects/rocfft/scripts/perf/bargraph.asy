// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
import utils;

void drawbargraph(datapoint[][] data,
                  string[] legs,
                  string[] otherlegs) {
  // Let's go to the bar, eh?

  // Assumption: same number of data points.

  int nbars = data.length;

  real width = 1.0 / nbars;
  real skip = 0.5;
  
  // Loop through all the data sets.
  for(int n = 0; n < nbars; ++n) {
    pen p = Pen(n); // + opacity(0.5);

    int len = data[n].length;

    // Set up the left and right sides of the bars.
    real[] left = new real[len];
    real[] right = new real[len];
    left[0] = n * width;
    for(int i = 1; i < len; ++i) {
      left[i] = left[i - 1] + nbars * width + skip;
    }
    for(int i = 0; i < len; ++i) {
      right[i] = left[i] + width;
    }
    
    // Draw an invisible graph to set up the axes.
    real[] fakex = new real[len];
    fakex[0] = left[0];
    for(int i = 1; i < fakex.length; ++i) {
      fakex[i] = right[i];
    }
    real[] yvals = new real[len];
    for(int i = 0; i < len; ++i) {
      yvals[i] = data[n][i].y;
    }
    draw(graph(left, yvals), invisible, legend = Label(otherlegs[n], p)); 


    // TOTO: in log plots, compute a better bottom.
    real bottom = 0.0;
    
    // Draw the bars
    for(int i = 0; i < data[n].length; ++i) {
      pair p0 = Scale((left[i], data[n][i].y));
      pair p1 = Scale((right[i], data[n][i].y));
      pair p2 = Scale((right[i], bottom));
      pair p3 = Scale((left[i], bottom));
      filldraw(p0--p1--p2--p3--cycle, p, black);
    }

   
    // Draw the bounds:
    for(int i = 0; i < data[n].length; ++i) {
      real xval = 0.5 * (left[i] + right[i]);
      pair plow = (xval, data[n][i].ylow);
      dot(plow);
      pair phigh = (xval, data[n][i].yhigh);
      dot(phigh);
      draw(plow--phigh);
      draw(plow-(0.25*width)--plow+(0.25*width));
      draw(phigh-(0.25*width)--phigh+(0.25*width));
    }

    
    // This is there the legends go
    if(n == nbars - 1) {
      for(int i = 0; i <  data[n].length; ++i) {
	pair p = (0.5 * nbars * width + i * (skip + nbars * width), 0);
	//label(rotate(90) * Label(xleg[i]), align=S, p);
	label(rotate(90) * Label(data[n][i].label), align=S, p);
      }
    }
    
  }
}

texpreamble("\usepackage{bm}");

size(600, 400, IgnoreAspect);

// Input data:
string filenames = "";
string secondary_filenames = "";
string legendlist = "";

// Graph formatting
string xlabel = "Problem size type";
string ylabel = "Time [s]";

string primaryaxis = "time";
string secondaryaxis = "speedup";

string ivariable = "lengths";
//ivariable = "batch";
//ivariable = "placeness";

string scaling = "";

usersetting();

if(primaryaxis == "gflops") {
    ylabel = "GFLOP/s";
}

//write("filenames:\"", filenames+"\"");
if(filenames == "") {
    filenames = getstring("filenames");
}
    
if (legendlist == "") {
    legendlist = filenames;
}

bool myleg = ((legendlist == "") ? false : true);
string[] legends = set_legends(legendlist);
for (int i = 0; i < legends.length; ++i) {
  legends[i] = texify(legends[i]);
}

// TODO: the first column will eventually be text.
string[] testlist = listfromcsv(filenames);

// Data containers:
pair[][] xyval = new real[testlist.length][];
pair[][] ylowhigh = new real[testlist.length][];




// Data containers:
datapoint[][] datapoints = new datapoint[testlist.length][];
readfiles(testlist, datapoints);
for(int ridx = 0; ridx < datapoints.length; ++ridx) {
  for(int idx = 0; idx < datapoints[ridx].length; ++idx) {
    datapoints[ridx][idx].mklabel(ivariable);
  }
}

//readbarfiles(testlist, data);

// for(int n = 0; n < data.length; ++n) {
//   for(int i = 0; i < data[n].length; ++i) {
//     write(data[n][i].label, data[n][i].y, data[n][i].ylow, data[n][i].yhigh);
//   }
// }

//write(ylowhigh);

//write(xyval);

// Generate bar legends.
string[] legs = {};
for(int i = 0; i < xyval[0].length; ++i) {
  legs.push(string(xyval[0][i].x));
}

drawbargraph(datapoints, legs, legends);

xaxis(BottomTop);
yaxis("Time (ms)", LeftRight, RightTicks);

attach(legend(),point(plain.E),  20*plain.E);

