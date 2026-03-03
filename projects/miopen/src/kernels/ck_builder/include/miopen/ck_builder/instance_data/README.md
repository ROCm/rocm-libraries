# CK Builder Kernel Instance Data

The files in this directory are intended to be used to create kernel instances using CK Builder, which can be found at the path `experimental/builder` within the `composablekernel` project in `rocm-libraries`. The builder has its own documentation that can be found within `README.md` files within that directory.

In brief, the builder uses structural non-type template parameters in combination with C++ concepts and constraints to validate and resolve which specific instance should be returned by the builder for a given structure and its values. It's the responsibility of the project using the builder to provide concrete implementations of these structs that adhere to the concepts in the builder. This directory contains those structs for the MIOpen integration of CK Builder.
