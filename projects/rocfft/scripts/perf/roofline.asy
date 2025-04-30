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
import stats;

texpreamble("\usepackage{bm}");

size(400, 300, IgnoreAspect);

//scale(Linear,Linear);
scale(Log, Log);
//scale(Log,Linear);
//scale(Linear,Log);

// TODO:
// ai for real/complex transforms
// ai for 2D, 3D transforms
// get list of devices and set hw specs automatically


string[] listfromcsv(string input)
{
    string list[] = new string[];
    int n = -1;
    bool flag = true;
    int lastpos;
    while(flag) {
        ++n;
        int pos = find(input, ",", lastpos);
        string found;
        if(lastpos == -1) {
            flag = false;
            found = "";
        }
        found = substr(input, lastpos, pos - lastpos);
        if(flag) {
            list.push(found);
            lastpos = pos > 0 ? pos + 1 : -1;
        }
    }
    return list;
}


bool dolegend = true;

string filenames = "";
string legendlist = "";
real[] floatsizes = new real[];

real xmin = -inf;
real xmax = inf;

bool doxticks = true;
bool doyticks = true;
string xlabel = "Arithmetic intensity";
string ylabel = "GFLOPS, [s${}^{-1}$]";

bool normalize = false;

real bw = 1024;
real floatgflops = 13571.47;
real doublegflops = 3426.89;

usersetting();

if(filenames == "")
    filenames = getstring("filenames");
else
    write("filenames:\"", filenames+"\"");

string[] testlist = listfromcsv(filenames);

if(floatsizes.length == 0) {
    for(int i = 0; i < testlist.length; ++i) {
        floatsizes.push(getreal("floatsize"));
    }
} else {
    write("floatsizes: ", floatsizes);
}

bool myleg = ((legendlist == "") ? false: true);
string[] legends=set_legends(legendlist);

bool plotxval(real x) {
   return x >= xmin && x <= xmax;
}

real intensity(real n, real floatsize) {
    // assumes double-precision, ie 8 bytes per values.
    return 5.0 * n * log(n) / (floatsize * n);
}



real[][] x = new real[testlist.length][];
real[][] y = new real[testlist.length][];
real[][] ly = new real[testlist.length][];
real[][] hy = new real[testlist.length][];
real[][][] data = new real[testlist.length][][];
real xmax = 0.0;
real xmin = inf;

real minai = inf;
real maxai = -inf;

for(int n = 0; n < testlist.length; ++n)
{
    string filename = testlist[n];

    data[n] = new real[][];
    write(filename);

    int dataidx = 0;

    real[] nvals = new real[];
    
    bool moretoread = true;
    file fin = input(filename);
    while(moretoread) {
        int a = fin;
        if(a == 0) {
            moretoread = false;
            break;
        }
        
        int N = fin;
        if (N > 0) {
            xmax = max(a,xmax);
            xmin = min(a,xmin);

            real ai = intensity(a, floatsizes[n]);
            minai = min(ai, minai);
            maxai = max(ai, maxai);
            x[n].push(ai);
            nvals.push(a);
            
            data[n][dataidx] = new real[N];
            
            real vals[] = new real[N];
            for(int i = 0; i < N; ++i) {
                vals[i] = fin;
            }

            for(int i = 0; i < N; ++i) {
                data[n][dataidx][i] = vals[i];
            }

	    //if(a >= xmin && a <= xmax) {
            real[] medlh = mediandev(vals);
            y[n].push(medlh[0]);
            ly[n].push(medlh[1]);
            hy[n].push(medlh[2]);
            //}
            ++dataidx;
        }
    }
   
    pen p = Pen(n);
    if(n == 2)
        p = darkgreen;

    pair[] z;
    pair[] dp;
    pair[] dm;
    for(int i = 0; i < x[n].length; ++i) {
        if(plotxval(x[n][i])) {
            z.push((x[n][i] , y[n][i]));
            dp.push((0 , y[n][i] - hy[n][i]));
            dm.push((0 , y[n][i] - ly[n][i]));
        }
    }
    errorbars(z, dp, dm, p);

    if(n == 1) 
        p += dashed;
    if(n == 2) 
        p += Dotted;
    
    guide g = scale(0.5mm) * unitcircle;
    marker mark = marker(g, Draw(p + solid));

    bool drawme[] = new bool[x[n].length];
    for(int i = 0; i < drawme.length; ++i) {
        drawme[i] = true;
        if(!plotxval(x[n][i]))
	    drawme[i] = false;
        if(y[n][i] <= 0.0)
	    drawme[i] = false;
    }
    
    draw(graph(x[n], y[n], drawme), p,  
         myleg ? legends[n] : texify(filename), mark);
    for(int i = 0; i < nvals.length; ++i) {
        if (i%4 == 0)
            label((string)nvals[i], Scale((x[n][i], y[n][i])),NE);
    }
}





real floatai = floatgflops/bw;
pair pf0 = (floatai, floatgflops);
pair pf1 = (minai, floatgflops * minai / floatai);
pair pf2 = (maxai, floatgflops);
dot(Scale(pf0));
// dot(Scale(pf1));
// dot(Scale(pf2));
//yequals(floatgflops);
draw(Scale(pf0)--Scale(pf1));
draw(Scale(pf0)--Scale(pf2));


real doubleai = doublegflops/bw;
//dot(Scale((doubleai,doublegflops)));
pair pd0 = (doubleai, doublegflops);
pair pd2 = (maxai, doublegflops);
dot(Scale(pd0));
// dot(Scale(pd2));
//yequals(floatgflops);
draw(Scale(pd0)--Scale(pf1));
draw(Scale(pd0)--Scale(pd2));

//yequals(doublegflops);



if(doxticks)
   xaxis(xlabel,BottomTop,LeftTicks);
else
   xaxis(xlabel);

yaxis(ylabel,LeftRight, LeftTicks, ymax = max(floatgflops,doublegflops));

if(dolegend)
    attach(legend(),point(plain.E),(20*plain.E)  );



