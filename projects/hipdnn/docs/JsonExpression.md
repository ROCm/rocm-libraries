# JSON Expression Language

A **header-only expression compiler**
([`JsonExpression.hpp`](../plugin_sdk/include/hipdnn_plugin_sdk/ingestor/JsonExpression.hpp))
for boolean and arithmetic expressions written as JSON. A rule written as an
`nlohmann::json` value is compiled once into a reusable `Expression<Data>`,
then evaluated many times against different data sources. The runtime value
type (`Value`) is a small standalone variant that does **not** depend on
nlohmann/json — nlohmann is used only to express the rule being compiled.

The syntax is that of [JsonLogic](https://jsonlogic.com/), with the extensions
and restrictions described below; the operator set here is neither a superset
nor a subset of it, so treat this document, not jsonlogic.com, as the contract.

All names live in namespace `hipdnn_plugin_sdk::ingestor::jsonexpr`; the examples
below assume `namespace jexpr = hipdnn_plugin_sdk::ingestor::jsonexpr;`. The
headers are part of the Plugin SDK's kernel-ingestor subtree, so they compile and
install only when `HIPDNN_ENABLE_KERNEL_INGESTOR` is set.

`JsonExpression.hpp` is the entry point and the only header to include; it pulls
in the implementation, which is split by layer under
[`ingestor/jsonexpr/`](../plugin_sdk/include/hipdnn_plugin_sdk/ingestor/jsonexpr):

| Header | Contents |
| --- | --- |
| `Error.hpp` | `JsonExpressionCompileError` |
| `Value.hpp` | the runtime value type |
| `DataSource.hpp` | the type-erased data-source contract |
| `Node.hpp` | compiled tree nodes |
| `Operators.hpp` | one function per operator |
| `OperatorTable.hpp` | the operator table, and `OpNode` |
| `LayoutAliases.hpp` | the `stride_order` layout-name pre-pass |
| `Compiler.hpp` | json → node tree |
| `VarIterator.hpp` | iteration over referenced variables |

```cpp
#include <hipdnn_plugin_sdk/ingestor/JsonExpression.hpp>

namespace jexpr = hipdnn_plugin_sdk::ingestor::jsonexpr;

struct MyData {                                  // your data source
    jexpr::Value getData(const std::string& path) const;
};

nlohmann::json rule = {{"+", {"$x", "$y"}}};
auto expr = jexpr::compile<MyData>(rule);       // parse + build tree once
jexpr::Value a = expr(dataA);                    // evaluate - no re-parse
jexpr::Value b = expr(dataB);                    // reuse for other data
```

## Data source

The evaluator is templated on your data type, which must expose:

```cpp
jexpr::Value getData(const std::string& path) const;
```

The compiled expression passes the variable path (e.g. `"a.b.c"`) straight to
this accessor; your type owns path resolution. Conventions:

- the path is always non-empty: whole-document references are rejected at
  compile time, so a data source never receives `""`.
- returning `Value()` (null) means "not found"; `value_or_default` is how a rule
  supplies a fallback for such a path.

Any type satisfying this one-function contract is a valid data source;
`JsonDataSource` below is one implementation, not a privileged one.

### Sample: `JsonDataSource`

[`JsonDataSource.hpp`](../plugin_sdk/include/hipdnn_plugin_sdk/ingestor/JsonDataSource.hpp)
provides a ready-made data source backed by an `nlohmann::json` document, so you
can evaluate rules against JSON without writing an accessor:

```cpp
#include <hipdnn_plugin_sdk/ingestor/JsonDataSource.hpp>

jexpr::JsonDataSource src(nlohmann::json{{"q", {{"dims", {8, 16}}}}});
auto expr = jexpr::compile<jexpr::JsonDataSource>(rule);
jexpr::Value r = expr(src);
```

It resolves dotted keys (`a.b.c`), `[N]` array subscripts (`arr[0]`,
`rows[2].name`, `grid[0][1]`), and dot-form indices (`arr.1`). A leading
variable sigil is stripped, so `"$q.dims[0]"` and `"q.dims[0]"` address the same
location. It also offers the inverse, `setData`, which writes a `Value` back into
the document and creates intermediate objects and arrays on demand:

```cpp
src.setData("$q.dims[0]", 2);   // -> {"q":{"dims":[2, 16]}}
```

A `[N]` subscript grows an array (filling gaps with null); any other key creates
or descends into an object; the empty path replaces the whole document.
`setData` throws `std::invalid_argument` on a malformed path, a non-numeric
index applied to an array, or an index at or above `MAX_ARRAY_INDEX` — a `[N]`
subscript grows the array to `N`, so an unbounded index would turn a typo into
an allocation of arbitrary size. Objects and null in the document read back as
`Value` null, matching `Value`'s scalar/array-only model.

## `Value`

A json-like tagged value with no external dependency. Alternatives: null, bool,
`int64_t`, `double`, `std::string`, and `Array` (`std::vector<Value>`). Numeric
results are stored as integers when exactly integral (so `1 + 1` is `2`, not
`2.0`). Key members: the `is*()` / `as*()` inspectors, `truthy()`, `toNumber()`,
`dump()`, strict `operator==`, and the static `compare`, which returns a
`Value::Ordering` (`LESS` / `EQUAL` / `GREATER` / `UNORDERED`, the last being the
NaN case that makes every ordering test false).

There is intentionally no object alternative — nested structure is reached
through the data source's path accessor, not carried in a `Value`.

## Variables

A variable reference is a string prefixed with a sigil (`$` by default) — that
is the *only* way to read data. There is no `var` operator; writing one is a
compile-time error rather than a second spelling of the same thing. Strings
without the sigil are literals, so a variable can appear anywhere a literal
can:

| Reference    | Meaning                                    |
| ------------ | ------------------------------------------ |
| `"$x"`       | top-level key                              |
| `"$a.b.c"`   | nested path                                |
| `"$arr[0]"`  | array index (subscript)                    |
| `"$arr.1"`   | dot-form array index                       |
| `"$"`        | whole document — rejected at compile time  |
| `"$$text"`   | escaped string literal `"$text"`           |

Pass a different sigil as the second argument to `compile` if your keys begin
with `$`.

## Layout aliases

A tensor's layout is carried as a `stride_order`: an integer array giving, for
each logical dimension `d`, that dimension's stride rank, `0` being the
fastest-varying. The common layouts also have names, and a name **expands to
its array at compile time**, so the array stays the single canonical form and
evaluation only ever compares arrays.

| Alias   | Array         | | Alias   | Array           |
| ------- | ------------- |-| ------- | --------------- |
| `nchw`  | `[3,2,1,0]`   | | `ndhwc` | `[4,0,3,2,1]`   |
| `nhwc`  | `[3,0,2,1]`   | | `bhsd`  | `[3,2,1,0]`     |
| `ncdhw` | `[4,3,2,1,0]` | |         |                 |

```jsonc
{"==": ["$x.stride_order", "nhwc"]}                  // same as [3, 0, 2, 1]
{"in": ["$q.stride_order", ["bhsd", [3, 1, 2, 0]]]}  // a set of accepted layouts
```

A name is read as an alias **only where a `stride_order` reference gives it
that meaning** — opposite one in an `==` / `!=`, or as an element of the array
an `in` searches. Anywhere else `"nhwc"` is an ordinary string literal, so a
data field that happens to hold layout names is untouched.

A name is a *plain* string. A sigil-prefixed string in an alias position is a
variable reference (or, doubled, an escaped literal) and is left alone, so one
tensor's layout can be compared against another's:

```jsonc
{"==": ["$q.stride_order", "$k.stride_order"]}       // do q and k share a layout?
{"in": ["$q.stride_order", ["$k.stride_order", "nhwc"]]}
```

Note also that an alias names an *array*, not a distinct layout: `bhsd` and
`nchw` both expand to `[3,2,1,0]`, so either name accepts a tensor carrying
that stride order. Aliases are spelling conveniences, not narrowing checks.

In those positions a `stride_order` is an integer array, so a string can only
be an alias. Two mistakes are therefore compile-time errors rather than
expressions that quietly never match:

- an **unrecognized name** (`"nhcw"`), which would otherwise compare unequal
  against every array forever;
- an alias whose **rank contradicts a rank pin** in the same expression, e.g.
  `{"and": [{"==": ["$x.rank", 4]}, {"==": ["$x.stride_order", "ndhwc"]}]}` —
  every alias is fixed-rank, so a rank-5 alias on a tensor pinned to rank 4
  can never hold. Only pins reachable through `and` are considered; a pin
  inside an `or` / `if` arm is conditional and cannot contradict the alias.

## Supported operators

"Operands" is the argument count enforced at compile time; a mismatch raises
`JsonExpressionCompileError`. "Operand types" describes what each operator does
with the values it gets, not a static type system — the language is dynamically
typed, and *number* means the operand is put through `Number()`-style coercion
(`toNumber()`), so a numeric string works where a number is wanted. Every
operator below except the last three yields `null` when any operand is `null`
(see [Null is unknown](#null-is-unknown-and-it-propagates)).

### Data access

Reading a variable is not an operator — it is the sigil-prefixed string form
above, so `"$q.dims[0]"` resolves the path `q.dims[0]` against the data source
and yields `null` when it does not resolve. One operator supplies a fallback
for that case:

| Operator | Operands | Operand types | Description |
| --- | --- | --- | --- |
| `value_or_default` | 2 | value: any expression; default: any expression | Returns the first operand when it resolves, the second otherwise. Keys on *existence*, not truthiness, so a present `0`, `""`, or `false` is returned rather than the fallback. The default is evaluated lazily, so `{"value_or_default": ["$a", "$b"]}` reads "this field, else that one". |

### Logic and control

| Operator | Operands | Operand types | Description |
| --- | --- | --- | --- |
| `if` / `?:` | 2+ | condition, result pairs, optional trailing else: any | Evaluates condition/result pairs left to right and returns the first result whose condition is truthy; a lone trailing operand is the else branch. Branches are lazy — only the taken one is evaluated. An unresolved condition picks no branch and yields `null`. |
| `and` | 1+ | any | Truthy-fold. Returns the first falsy operand, else the last. Short-circuits. Three-valued: a definite falsy operand decides the result even beside an unresolved one; otherwise a `null` operand makes the whole conjunction `null`. |
| `or` | 1+ | any | Truthy-fold. Returns the first truthy operand, else the last. Short-circuits. Three-valued in the same way: a definite truthy operand decides the result even beside an unresolved one. |
| `!` | 1 | any | Logical negation of truthiness. |
| `!!` | 1 | any | Cast to boolean truthiness. |

### Comparison

| Operator | Operands | Operand types | Description |
| --- | --- | --- | --- |
| `==` | 2 | any | Strict equality — no type coercion, so `1 == "1"` is false. An integer and a double of equal value still compare equal. |
| `!=` | 2 | any | Strict inequality, the negation of `==`. |
| `<` | 2–3 | number (coerced); strings compare lexicographically | Less-than. The 3-operand form is the between-chain `a < b < c`. |
| `<=` | 2–3 | number (coerced); strings compare lexicographically | Less-than-or-equal, with the same between-chain form. |
| `>` | 2 | number (coerced); strings compare lexicographically | Greater-than. |
| `>=` | 2 | number (coerced); strings compare lexicographically | Greater-than-or-equal. |

### Arithmetic

| Operator | Operands | Operand types | Description |
| --- | --- | --- | --- |
| `+` | any | number (coerced) | Sum of all operands; `0` with no operands. |
| `-` | 1–2 | number (coerced) | Unary negation with one operand, subtraction with two. |
| `*` | any | number (coerced) | Product of all operands; `1` with no operands. |
| `/` | 2 | number (coerced) | Division. Declines (yields `null`) on a zero divisor rather than producing `inf`/`NaN`. |
| `%` | 2 | number (coerced) | Remainder (`fmod`). Declines on a zero divisor. |
| `min` | 1+ | number (coerced) | Smallest operand. |
| `max` | 1+ | number (coerced) | Largest operand. |

### Math extensions

Beyond the core operator set, for the tiling and scaling arithmetic descriptor
dispatch formulas need.

| Operator | Operands | Operand types | Description |
| --- | --- | --- | --- |
| `ceil_div` | 2 | number (coerced) | Ceiling division, `ceil(a / b)`. Declines on a zero divisor. |
| `abs` | 1 | number (coerced) | Absolute value. |
| `pow` | 2 | number (coerced) | `a` raised to the power `b`. Declines when the result is not finite — a negative base under a fractional exponent, or an overflow. |
| `log2` | 1 | number (coerced) | Base-2 logarithm. Declines on a non-positive operand. |
| `rsqrt` | 1 | number (coerced) | Reciprocal square root, `1 / sqrt(x)`. Declines on a non-positive operand. |

### Short-hands

| Operator | Operands | Operand types | Description |
| --- | --- | --- | --- |
| `divisible` | 2 | number (coerced) | True when `a` is an exact multiple of `b`. Exactly `{"==": [{"%": [a, b]}, 0]}`, so it declines on a zero divisor like `%` does, and `0` is divisible by everything. |

`value_or_default` is the other short-hand; it is listed under
[Data access](#data-access) because it is how a rule handles a path that does
not resolve.

### Membership

| Operator | Operands | Operand types | Description |
| --- | --- | --- | --- |
| `in` | 2 | needle: any; haystack: array or string | Array haystack: true when it contains `needle` by strict element equality. String haystack: substring test, with a non-string needle rendered via `dump()`. Any other haystack type is false. |

### Presence

These and `value_or_default` are the only operators that do not propagate a
`null` operand — answering "was this supplied?" always yields a real boolean.

| Operator | Operands | Operand types | Description |
| --- | --- | --- | --- |
| `present` | 1+ | any, normally variable references | True when *every* operand resolves to a non-null value. An `and`-fold, so one call decides a whole list. |
| `not_present` | 1+ | any, normally variable references | True when *every* operand resolves to `null`. Also an `and`-fold. |

## Null is unknown, and it propagates

`null` means *unresolved*, not a value. Every operator except `present`,
`not_present`, and `value_or_default` returns `null` when any argument is
`null`, rather than coercing it to `false`/`0`/not-equal. This matters whenever
a data source has optional fields, where an unresolved path means the field is
absent rather than false: if `null` coerced, a narrowing check such as
`{"!=": ["$bias.dtype", "BFLOAT16"]}` would evaluate **true** on input carrying
no `bias` at all, accepting data it never actually examined. Two `null`s are not
equal to each other either — the question is unanswerable, so `==` and `!=` both
decline.

`and` and `or` are three-valued (Kleene): a definite `false` still decides an
`and` and a definite `true` still decides an `or`, even beside a `null`
argument; otherwise the result is `null`. That is what lets
`{"or": [{"not_present": ["$bias"]}, {"and": [{"present": ["$bias"]}, ...]}]}`
accept an absent operand whose field reads cannot run.

Once every argument resolves, value semantics follow JavaScript: `false`, `0`,
`""`, `null` and the empty array are falsy; `Number()`-style coercion drives
arithmetic and ordering. A `null` root is falsy, so an undecided expression
declines.

Malformed rules (unknown operator, wrong argument count, a non-operator object)
raise `JsonExpressionCompileError` at `compile` time, so evaluation stays on the
fast path. Nesting deeper than `MAX_EXPRESSION_DEPTH` is rejected the same way:
compilation and evaluation both recurse per level, and rules are read from
descriptor files on disk, so an over-deep rule must report a bad rule rather
than exhaust the stack.

Not included: the `var` operator (variables are the sigil form only), and the
collection and string operators (`map`, `reduce`, `filter`, `all`, `some`,
`none`, `merge`, `cat`, `substr`, `missing`, `missing_some`).

## Tests

Unit tests live in
[`TestJsonExpression.cpp`](../plugin_sdk/tests/ingestor/TestJsonExpression.cpp) and
[`TestJsonDataSource.cpp`](../plugin_sdk/tests/ingestor/TestJsonDataSource.cpp),
and build into the `hipdnn_plugin_sdk_tests` GTest binary. Like the rest of
`ingestor/`, they are compiled only when `HIPDNN_ENABLE_KERNEL_INGESTOR` is set.
Run them with:

```bash
ctest -R hipdnn_plugin_sdk_tests
# or, filtered directly on the binary:
./hipdnn_plugin_sdk_tests --gtest_filter='TestJsonExpression.*:TestJsonDataSource.*'
```
