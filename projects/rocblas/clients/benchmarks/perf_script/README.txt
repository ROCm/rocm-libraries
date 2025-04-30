
Instruction to run the python script to collect performance data

1) copy rocblas-bench to this folder
   For example, "cp ../../../build/release/clients/staging/rocblas-bench ./"

2) "python measurePerformance.py --help" to see how to run the command line mode
    For example, "python measurePerformance.py -s 64-5760:64 -f gemm -r s --transa none --transb transpose > sgemm.txt"
    will collect performance of sgemm NT of square matrices [64:5760] with step 64.
    All the output will be dumped into sgemm.txt

3)  Plot the figure by refactoring with plotPerformance.py provided.
    "python plotPerformance.py" by default
    will read step (2) output sgemm.txt and plot a performance figure in sgemm.pdf

4) You can ignore the other automatic generated .txt files and folders during the run.


plotPerformance.py has the following dependencies:
python-pip
matplotlib
python-tk

These can be installed with:

sudo apt install python-pip

sudo pip install matplotlib

sudo apt install python-tk
