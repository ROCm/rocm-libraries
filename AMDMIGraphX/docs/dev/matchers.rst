.. meta::
  :description: MIGraphX internal matchers
  :keywords: MIGraphX, code base, contribution, developing, matchers

Matchers
========

Introduction
------------

The matchers provide a way to compose several predicates together. A matcher such as ``m(m1, m2)`` first checks a match for ``m`` followed by a match for ``m1`` and ``m2`` subsequently.

The most commonly used matcher is the ``name`` matcher. It matches the instruction with the operator equal to the name specified::

    auto match_sum = name("sum");

The above matcher finds ``sum`` operators. To find ``sum`` operators with the output ``standard_shape``, use:

    auto match_sum = name("sum")(standard_shape());

Arguments
---------

To match arguments in the instructions, match each argument using the ``arg`` matcher::

    auto match_sum = name("sum")(arg(0)(name("@literal"), arg(1)(name("@literal"))));

The above matcher matches a ``sum`` operator with two arguments that are literals. Note that the ``args`` matcher eliminates the need to write ``arg(0)`` and ``arg(1)`` everytime::

    auto match_sum = name("sum")(args(name("@literal"), name("@literal")));


Binding
-------

To reference other instructions encountered while traversing through the instructions, use ``.bind``::

    auto match_sum = name("sum")(args(
                                    name("@literal").bind("one"), 
                                    name("@literal").bind("two")
                                )).bind("sum");


This associates the instruction to a name that can be read from the ``matcher_result`` when it matches.

Finding matches
---------------

To use the matchers to find instructions, write a callback object that contains the matcher and an ``apply`` function that takes the ``matcher_result`` when the match is found::

    struct match_find_sum
    {
        auto matcher() const { return name("sum"); }

        void apply(program& p, matcher_result r) const 
        { 
            // Do something with the result
        }
    };

    find_matches(prog, match_find_sum{});


Creating matchers
-----------------

The macros ``MIGRAPH_BASIC_MATCHER`` and ``MIGRAPH_PRED_MATCHER`` help in the creation of the matchers. Here is how you can create a matcher for shapes that are broadcasted::

    MIGRAPH_PRED_MATCHER(broadcasted_shape, instruction_ref ins) 
    { 
        return ins->get_shape().broadcasted(); 
    }

For parameters to the predicate, use ``make_basic_pred_matcher`` to create the matcher. Here is how you can create a matcher to check the number of dimensions of the shape::

    inline auto number_of_dims(std::size_t n)
    {
        return make_basic_pred_matcher([=](instruction_ref ins) { 
            return ins->get_shape().lens().size() == n; 
        });
    }
