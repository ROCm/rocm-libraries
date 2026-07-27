# BRIEF
This python script is designed to allow automatic tuning of large sets of parameters. It is based off the examples implementations
The script works by generating instances through a combination of monte carlo and random walk, as well as sharing known good parameters between configurations.

# START YOUR FIRST TUNING
To set up a tuning, follow the following steps:
- create a new .gentune file for your use-case in the gentune_files/generation folder. You can base this off gemm_universal.gentune to have a starting point.
- create a standalone compileable program for the use case you would like to tune. Typically, this is best achieved by basing off the examples. (If you do not base your standalone on the examples, make sure benchmarker.py can properly interpret the output of your executable)
 Write the path to this file (relative to the base gentune directory) to the TEST_INSTANCE_TEMPLATE_PATH: of your gentune file 
- extract the compile commands neccessary for compiling your instance. Ensure all paths are either absolute or work from the base gentune directory. Write these commands to the COMPILE_CMD: part of your gentune file
- identify all relevant parameters that you would like to generate and tune. Specify them as GEN or TUNE parameters in the PARAMS section of your gentune file. (See PARAMETER_SPECIFICATION for mor details on this). You can include files from the layouts or data_formats section, or define your own include structure. Use the replace string (i.e. the name of the parameters) in your template at the locations you would like the parameter to be inserted. Tuning script will run a string replacement, replacing all occurrances of the parameter's name with one of it's possible values. Make sure this string is unique.
- Define a set of shapes in the BENCH_ARGS and VERIFY_ARGS. These are effectively the parameters that the tuning will pass to your example executable. Every shape will be optimized for individually. In BENCH_ARGS, specify a command that will run a benchmark as quickly as possible. The commands listed in VERIFY_ARGS should correspond 1:1 with the commands in BENCH_ARGS, but should run a verification of the results. This can be slower, as these will only be called when the optimizer has found a new, faster solution.
- You are now ready to run the optimizer for the first time! Refer to the USAGE section on how to call. Use verbose mode 2 (-v 2) to get the output of the compiler - this will help in debugging if you've set up the COMPILE_CMD and templates correctly. When run in verbose mode, failing instances will not be automatically deleted, and you can check these (in test_instances/) too. 
Note that not every combination of parameters may compile or pass verification. This is ok to a degree - the script will simply ignore these cases - but it does slow down the tuning process. If possible, avoid bad parameter combinations by specifying CONSTRAINTs (see the CONSTRAINT section for more info) 
- You are now tuning your parameters! Take some time to identify which parameters are relevant, include these in your tuning, add as many CONSTRAINTSs as you can identify to speed up the tuning process. Tuning could take a while, I often run it overnight/over the weekend with nohup. For help on interpreting the script's output, refer to the OUTPUT section.

# USAGE
call tuner_main with python3. It is recommended to pipe output to a file or to run with nohup.
example: python3 -u tuner_main.py -i generation/gemm_universal.gentune
# ARGUMENTS
-i: Specify the path of the gentune file you would like to tune, relative to the gentune base directory
-g: Base gentune directory, where all the gentune files live. Default gentune_files/
-t: number of threads you would like to use. Default 16
-p: print verbose output of current status every n seconds. Default 60
-v: verbosity level, for debugging purposes.

# PARAMETER SPECIFICATION
Parameters are specified through gentune files. These are files adhering to a specific format, the gentune format.
The gentune specification differenciates between GEN and TUNE parameters. 
For GEN parameters, the script will produce all posible combinations of these parameters and tune each combination seperatly. 
For TUNE parameters, the script will use it's optimization algorithm to find the optimal configuration of these parameters.

In their simplest form, parameters are specified by specifying the name of the parameter followed by the keyword GEN or TUNE (in the same line) followed by the list of possible values of this parameter, seperated by ; 
For example
    BlockSize TUNE 64; 128; 256
specifies the parameter with the name BlockSize and the possible values 64, 128 and 256.
it is also possible to define multiple parameters at the same time. For example:
    ADataType_; BDataType_; CDataType_; CShuffleDataType_ GEN half_t
will define all all 4 data types in the ;-separated list to half_t.
You could then define another parameter using the BlockSize parameter like this:
TransferSize TUNE REPLACE BlockSize*2; BlockSize*4
Due to the usage of the RELPACE keyword, the parameter BlockSize will be swapped for it's previously determined value. This way, the parameter size can be reduced 

In addition, parameter specifications can have CONDITION and PREPROCESSOR keywords. 
These keywords are only relevant when you use the option to outut an hpp file directly. 

With CONDITION, you can specify a constexpr condition that the line to add your instances in the hpp file will be conditioned by
With PREPROCESSOR, you can specify a preprocessor ifdef that the line to add your instances in the hpp file will be conditioned

Names of other parameters occurring in the body of parameters can optionally be replaced by using the REPLACE keyword after the GEN or TUNE keyword. Note that only parameters specified before the current parameter will be replaced (see parameter order)

# CONSTRAINTS
For every parameter, it is possible to specify CONSTRAINTs. These are useful for reducing the size of the search space and thus reducing the number of failed build attempts or verification failures. CONSTRAINTs can be used on both GEN and TUNE parameters. 
To define a constraint, add the keyword CONSTRAINT after the list of possible parameters. After the CONSTRAINT keyword, you can specify a string that will be evaluated by the python eval function. You can refer to other GEN or TUNE parameter names here - these will be replaced by their respective values before evaluation. 
Note however that it is only possible to refer to parameters defined before the current parameter. Parameters are generally ordered top-to bottom, include before body. For more details, see the section PARAMETER ORDER. 
The script will only generate a parameter combination when the evaluation of the python string defined here returns 1.

# INCLUDE HIRARCHY
To avoid duplication of parameters that are commonly used across multiple tunings, it is possible to include specifications from other files. 
There are two ways specify include structures and two fundamental concepts to understand
The concepts for combining multiple files are:
    -Addition: This will add multiple configurations to the list of configurations, but maintain them sepeartly. The tuning will independently tune both configurations.
        Example: If you have different parameters for RCR and RRR, you would like to have two seperate configurations, each with the parameters relevant to each layout.
    -Combination: this will generate one parameter set, with the parameters from both files combined. 

Using #INCLUDE - recommended for simpler use-cases:
specify #INCLUDE followed by the path to the file you would like to include - relative to the base_gentune_dir. All parameters 
Using IMPORT_STRUCTURE:
this is the recommended method for more complex use-cases. This structure allows you to explicitly state the combinations of included files you would like to generate. State the names of the files you would like to import within parenthesis ("), use the operators * and + to specify in which way you would like the configurations specified in these files to be combined.
The operators are:
(+) Operator: adds configurations defined by lval and rval to a list. The individual parameter sets remain seperate, and will be tuned independently.
(*) Operator: combines the parameters defined by lval and rval to one common configuration. Lval is defined first (See PARAMETER ORDER), parameters in rval can overwrite and refer to parameters defined in lval in their CONSTRAINT section. If lval or rval is itself a list, the combination will be applied to each element of the list as a cartesian product. 
The * operator has precedence over the + operator. You can change precedence by using brackets.
An example of usage is:
("AColMajor.gentune" + "ARowMajor.gentune") * ("BColMajor.gentune" + "BRowMajor.gentune")
would lead to four configurations being generated with each combination of the Row and Col parameters for A and B

# CACHE_BEST_RESULTS

Often, the ideal parameters for different sets of GEN parameters will be similar. To avoid re-optimizing every set of GEN parameters from scratch, you can define cache points by adding CACHE_BEST_RESULTS to the gentune files. When defined, all subsets of configurations and GEN parameters that share the same included parameters will exchange ideal solutions at these points.

# PARAMETER ORDER
The order in which parameters are defined is relevant for two reasons: First, because only parameters defined before the current parameter are substituted into CONSTRAINT statements. Second, repeated definitions of the same parameter will be overwriten by the newer definition. The following defined the order of parameters specified:
- By definition GEN parameters are always defined before TUNE parameters. You may reference all GEN parameters in TUNE parameteres' CONSTRAINT section, even if these are defined in a file not yet included. A GEN parameter may never be overwriten by a TUNE parameter of the same name.
- Parameters from included gentune files are always defined before parameters in the gentune file body, regardless of the position of the #INCLUDE command.
- within included files, #INCLUDE will be processed before #IMPORT_STRUCTURE. 
- within import structures, parameters are defined left-to-right. Note that only multiplication operators lead to parameter combination. This is also true throughout bracets.
- within files, and GEN or TUNE categories, parameters are ordered in the order they appear in the PARAMETERS section, top-to-bottom.

# BENCH_ARGS
specify a list of shapes relavant to your use case. These are the arguments which will be passed to your example instances. The separator is /n

# VERIFICATION
It is possible and has been practically encountered that a configuration might compile without errors, but return erronous results when run. To catch these configurations, a seperate set of VERIFY_CMD should be specified in the gentune file. Please provide an identical list of shapes as with BENCH_CMDS, but with verify mode turned on. As verification is a time-intense process, the verification step is only run when a configuration has been discovered that is faster than what is previously known. The separator is /n

# OUTPUT
By default, the gentune implementation will regularly output it's known best configurations. You can set the time interval these prints will happen with the -p option. 
In addition, there is the option to write the ideal instances directly to a hpp file.
To use this option, do the following:
    - specify a template for the .hpp output file, and mark the position where the instances should be added with #ADD_INSTANCES_HERE
    - write the path to this file (relative to the directory you are calling the script from) to the OUTPUT_HPP_TEMPLATE_PATH: of the gentune file.
    - write a template for the generation of an instance to the OUTPUT_HPP_TEMPLATE_CODE_LINE: section of the gentune file. Within this template, all occurrances of parameter names will be replaced by their values
    - call the python script with -o [path and name of desired output file]
Now, every time the python script prints it's results, your hpp file should also be updated.
