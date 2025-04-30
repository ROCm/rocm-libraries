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

texpreamble("\usepackage{bm}");

size(400, 300, IgnoreAspect);

// Input data:
string filenames = "";
string secondary_filenames = "";
string legendlist = "";

// Graph formatting
string xlabel = "Problem size $N$";
string ylabel = "Time (ms)";

string primaryaxis = "time";
string secondaryaxis = "speedup";

bool dobars = true;
bool dolegend = true;
real Ncut = inf;

int ngroup = 2;

string ivariable = "lengths";
//ivariable = "ndev";
string scaling = "";
//scaling = "strong";

usersetting();

if(ivariable == "ndev") {
    xlabel = "Number of devices";
}

if(ivariable == "batch") {
    xlabel = "Batch size";
}

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

string[] testlist = listfromcsv(filenames);

// Data containers:
datapoint[][] datapoints = new datapoint[testlist.length][];


readfiles(testlist, datapoints);

bool datapointless(datapoint a, datapoint b)
{
    return a.x < b.x;
}

pair[][] xyval = new real[testlist.length][];
pair[][] ylowhigh = new real[testlist.length][];

for(int ridx = 0; ridx < datapoints.length; ++ridx) {
  for(int idx = 0; idx < datapoints[ridx].length; ++idx) {
    datapoints[ridx][idx].mklabel(ivariable);
  }
}

for(int n = 0; n < datapoints.length; ++n) {
    datapoints[n] = sort(datapoints[n], datapointless);
    datapoints_to_xyvallowhigh(datapoints[n], xyval[n], ylowhigh[n]);
}


if(Ncut < inf) {
    for(int n = 0; n < datapoints.length; ++n) {
        while(xyval[n][xyval[n].length - 1].x > Ncut) {
            xyval[n].pop();
            ylowhigh[n].pop();
        }
    }
}

//write(xyval);
//write(ylowhigh);

// Find the bounds on the data to determine if the scales should be
// logarithmic.
real[] bounds = xyminmax( xyval );
bool xlog = true;
if(bounds[1] / bounds[0] < 10) {
    xlog = false;
}
bool ylog = true;
if(bounds[3] / bounds[2] < 10) {
    ylog = false;
}
scale(xlog ? Log : Linear, ylog ? Log : Linear);

// if(scaling == "strong") {
//     scale(Log , Log);
// }

// Plot the primary graph:
for(int n = 0; n < xyval.length; ++n)
{
    pen graphpen = Pen(n);
    if(n == 2) {
        graphpen = darkgreen;
    }
    string legend = myleg ? legends[n] : texify(testlist[n]);
    marker mark = marker(scale(0.5mm) * unitcircle, Draw(graphpen + solid));

    
    if(dobars) {
        // Compute the error bars:
        pair[] dp; // high
        pair[] dm; // low
        for(int i = 0; i < xyval[n].length; ++i) {
            dp.push((0, -xyval[n][i].y + ylowhigh[n][i].y));
            dm.push((0, -xyval[n][i].y + ylowhigh[n][i].x));
        }
        //write(dp);
        //write(dm);
        errorbars(xyval[n], dp, dm, graphpen);
    }
    
    // Actualy plot things:
    draw(graph(xyval[n]), graphpen, legend, mark);
    
    if(scaling == "strong") {
        real[] ndevs = new real[];
        real[] tscale = new real[];
        for(int idx = 0; idx < xyval[n].length; ++idx) {
            ndevs.push( xyval[n][idx].x );
            tscale.push( xyval[n][0].y / ndevs[idx] );
        }
        
        draw(graph(ndevs, tscale), graphpen+dashed);
    }
    if(scaling == "weak") {
        real[] ndevs = new real[];
        real[] tscale = new real[];
        for(int idx = 0; idx < xyval[n].length; ++idx) {
            ndevs.push( xyval[n][idx].x );
            tscale.push( xyval[n][0].y );
        }
	draw(graph(ndevs, tscale), graphpen+dashed);
    }
}


xaxis(xlabel, BottomTop, LeftTicks);

yaxis(ylabel, (secondary_filenames != "") ? Left : LeftRight,RightTicks);

// attach(legend(),point(plain.E),(((secondary_filenames != ""))
//                                ? 60*plain.E + 40 *plain.N
//                                 : 20*plain.E)  );
//attach(legend(),point(plain.S), N);
if(dolegend) {
    attach(legend(), point(S), 50*S);
}
    
if(secondary_filenames != "")
{
  
  write("secondary_filenames: ", secondary_filenames);
  string[] second_list = listfromcsv(secondary_filenames);
  for(int idx = 0; idx < second_list.length; ++idx) {
    write(second_list[idx]);
  }

  
    datapoint[][] datapoints = new datapoint[second_list.length][];
    
    readfiles(second_list, datapoints);


    for(int ridx = 0; ridx < datapoints.length; ++ridx) {
      for(int idx = 0; idx < datapoints[ridx].length; ++idx) {
	datapoints[ridx][idx].mklabel(ivariable);
      }
    }

    
    pair[][] xyval = new real[second_list.length][];
    pair[][] ylowhigh = new real[second_list.length][];
    for(int n = 0; n < datapoints.length; ++n) {
        datapoints[n] = sort(datapoints[n], datapointless);
        datapoints_to_xyvallowhigh(datapoints[n], xyval[n], ylowhigh[n]);
    }
    
    bool interval = true;
    
    // // FIXME: speedup has error bounds, so we should read it, but
    // // p-vals does not.
    // readfiles(second_list, xyval, ylowhigh, true);

    picture secondarypic = secondaryY(new void(picture pic) {
	int penidx = testlist.length; // initialize to end of previous pen.

            scale(pic, xlog ? Log : Linear, Linear);
            
            for(int n = 0; n < xyval.length; ++n)
            {
                pen graphpen = Pen(penidx + n);
                if(penidx + n == 2) {
                    graphpen = darkgreen;
                }
                graphpen += dashed;
                
                guide g = scale(0.5mm) * unitcircle;
                marker mark = marker(g, Draw(graphpen + solid));
                        
                if(interval)
                {
                    // Compute the error bars:
                    pair[] dp;
                    pair[] dm;
                    for(int i = 0; i < xyval[n].length; ++i) {
                        dp.push((0, xyval[n][i].y - ylowhigh[n][i].x));
                        dm.push((0, xyval[n][i].y - ylowhigh[n][i].y));
                    }

                    errorbars(pic, xyval[n], dp, dm, graphpen);
		    
                }
                int nbase = ngroup * (n # (ngroup - 1));
                int ncomp = nbase + (n % (ngroup - 1)) + 1;
                draw(pic,graph(pic, xyval[n]), graphpen,
                     legends[ncomp] + " over " + legends[nbase],mark);
		//write(xyval[n]);
		
		yequals(pic, 1.0, lightgrey);
		
            }

	    
            yaxis(pic, secondaryaxis, Right, black, LeftTicks);
            if(dolegend) {
	      attach(legend(pic), point(plain.E), 60*plain.E - 40 *plain.N  );
              //attach(legend(pic), point(plain.S), 120*S);
            }
        });
    add(secondarypic);
}
