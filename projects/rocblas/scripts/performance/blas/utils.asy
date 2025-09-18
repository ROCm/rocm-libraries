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

// Find the comma-separated strings to use in the legend
string[] set_legends(string runlegs)
{
   string[] legends;
   bool myleg=((runlegs== "") ? false: true);
   bool flag=true;
   int n=-1;
   int lastpos=0;
   string legends[];
   if(myleg) {
      string runleg;
      while(flag) {
	 ++n;
	 int pos=find(runlegs,",",lastpos);
	 if(lastpos == -1) {runleg=""; flag=false;}

	 runleg=substr(runlegs,lastpos,pos-lastpos);

	 lastpos=pos > 0 ? pos+1 : -1;
	 if(flag) legends.push(runleg);
      }
   }
   return legends;
}
