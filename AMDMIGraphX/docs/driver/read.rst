.. option::  <input file>

File to load

.. option::  --test 

Test MIGraphX with single layer GEMM model

.. option::  --onnx

Load as onnx

.. option::  --tf

Load as tensorflow

.. option::  --migraphx

Load as MIGraphX

.. option::  --migraphx-json

Load as MIGraphX JSON

.. option::  --batch [unsigned int] (Default: 1)

For a static model, set batch size. For a dynamic batch model, sets the batch size at runtime.

.. option::  --nhwc

Treat tensorflow format as nhwc

.. option::  --skip-unknown-operators

Skip unknown operators when parsing and continue to parse.

.. option::  --nchw

Treat tensorflow format as nchw

.. option::  --trim, -t [unsigned int]

Trim instructions from the end (Default: 0)

.. option::  --input-dim [std::vector<std::string>]

Dim of a parameter (format: "@name d1 d2 dn")

.. option:: --dyn-input-dim [std::vector<std::string>]

Set dynamic dimensions of a parameter using JSON formatting (format "@name" "dynamic_dimension_json")

.. option:: --default-dyn-dim

Set the default dynamic dimension (format {min:x, max:y, optimals:[o1,o2,...]})

.. option::  --optimize, -O

Optimize when reading

.. option::  --apply-pass, -p

Passes to apply to model

.. option::  --graphviz, -g

Print out a graphviz representation.

.. option::  --brief

Make the output brief.

.. option::  --cpp

Print out the program as cpp program.

.. option::  --json

Print out program as json.

.. option::  --netron

Print out program as a Netron viewable json file.

.. option::  --text

Print out program in text format.

.. option::  --binary

Print out program in binary format.

.. option::  --py

Print out program using python API.

.. option::  --output, -o [std::string]

Output to file.

