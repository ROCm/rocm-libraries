# JsonLogic

Single-header [JsonLogic](https://jsonlogic.com/) **expression compiler**
(`JsonLogic.hpp`). A rule written as an `nlohmann::json` value is compiled once
into a reusable `Expression<Data>`, then evaluated many times against different
data sources. The runtime value type (`Value`) is a small standalone variant
that does **not** depend on nlohmann/json — nlohmann is used only to express the
rule being compiled.

All names live in namespace `hip_kernel_provider_common::jsonlogic`; the examples
below assume `namespace jlogic = hip_kernel_provider_common::jsonlogic;`.

```cpp
#include "hip_kernel_provider_common/JsonLogic.hpp"

namespace jlogic = hip_kernel_provider_common::jsonlogic;

struct MyData {                                  // your data source
    jlogic::Value getData(const std::string& path) const;
};

nlohmann::json rule = {{"+", {"$x", "$y"}}};
auto expr = jlogic::compile<MyData>(rule);       // parse + build tree once
jlogic::Value a = expr(dataA);                    // evaluate - no re-parse
jlogic::Value b = expr(dataB);                    // reuse for other data
```

## Data source

The evaluator is templated on your data type, which must expose:

```cpp
jlogic::Value getData(const std::string& path) const;
```

The compiled expression passes the variable path (e.g. `"a.b.c"`) straight to
this accessor; your type owns path resolution. Conventions:

- `""` is the whole-document request.
- returning `Value()` (null) means "not found" and triggers a `var` default if
  the rule supplies one.

### Sample: `JsonDataSource`

`JsonDataSource.hpp` provides a ready-made data source backed by an
`nlohmann::json` document, so you can evaluate rules against JSON without writing
an accessor:

```cpp
#include "hip_kernel_provider_common/JsonDataSource.hpp"

jlogic::JsonDataSource src(nlohmann::json{{"q", {{"dims", {8, 16}}}}});
auto expr = jlogic::compile<jlogic::JsonDataSource>(rule);
jlogic::Value r = expr(src);
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
`setData` throws `std::invalid_argument` on a malformed path or a non-numeric
index applied to an array. Objects and null in the document read back as `Value`
null, matching `Value`'s scalar/array-only model.

## `Value`

A json-like tagged value with no external dependency. Alternatives: null, bool,
`int64_t`, `double`, `std::string`, and `Array` (`std::vector<Value>`). Numeric
results are stored as integers when exactly integral (so `1 + 1` is `2`, not
`2.0`). Key members: the `is*()` / `as*()` inspectors, `truthy()`, `toNumber()`,
`dump()`, strict `operator==`, and the static `compare`.

There is intentionally no object alternative — nested structure is reached
through the data source's path accessor, not carried in a `Value`.

## Inline variables

Stock JsonLogic references data with `{"var": "path"}` and treats every bare
string as a literal. This implementation additionally lets a variable appear
anywhere a literal can, by prefixing it with a sigil (`$` by default):

| Inline       | Equivalent          | Meaning                 |
| ------------ | ------------------- | ----------------------- |
| `"$x"`       | `{"var": "x"}`      | top-level key           |
| `"$a.b.c"`   | `{"var": "a.b.c"}`  | nested path             |
| `"$arr[0]"`  | `{"var": "arr[0]"}` | array index (subscript) |
| `"$"`        | `{"var": ""}`       | whole document          |
| `"$$text"`   | `"$text"`           | escaped string literal  |

Strings without the sigil stay literals, so stock rules keep working. Pass a
different sigil as the second argument to `compile` / `evaluate` if your keys
begin with `$`.

## Supported operators

Data access: `var` (dotted paths, `[N]` array subscripts, `""` whole-document,
`[path, default]` fallback, and computed `[expr, default]` paths);
`value_or_default` (`{"value_or_default": ["$x", fallback]}`) returns the
variable's value when the path resolves and the fallback otherwise. It keys on
*existence* (an unresolved path reads as `null`), not truthiness, so a present
`0`, `""`, or `false` is returned rather than the fallback. The fallback is any
expression, evaluated lazily, so `["$a", "$b"]` reads "this field, else that
one".

Logic / control: `if` / `?:`, `and`, `or`, `!`, `!!` (short-circuit, lazy
branches).

Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=` (`<`/`<=` accept a third argument
for between-tests). `==`/`!=` are strict: no type coercion, so `1 == "1"` is
false (integers and doubles of equal value still compare equal). Ordering
(`<`/`<=`/`>`/`>=`) still uses numeric coercion.

Arithmetic: `+`, `-` (binary and unary negation), `*`, `/`, `%`, `min`, `max`.
`/`, `%`, and `ceil_div` decline (yield `null`) on a zero divisor rather than
producing `inf`/`NaN`.

Math extensions (value-core, for hipDNN dispatch/constraint formulas): `ceil_div`
(2-arg ceiling division), `abs`, `pow`, `log2`, `rsqrt` (`1/sqrt(x)`). `log2` and
`rsqrt` decline on a non-positive argument.

Membership: `in`. `{"in": [needle, array]}` is true when `array` contains
`needle` (strict element equality); `{"in": [needle, string]}` is a substring
test.

Presence: `present`, `not_present`. `{"present": ["$a", "$b"]}` is true when
every listed path resolves to a non-null value; `{"not_present": [...]}` is true
when every listed path resolves to null. Both are `and`-folds, so one call
decides a whole list. They key on the same *existence* mechanism as
`value_or_default`, and, with `value_or_default`, are the only operators that do
not propagate a null: an unresolved path yields `false`/`true` rather than
`null`, because answering "was this supplied?" is their whole job.

## Null is unknown, and it propagates

`null` means *unresolved*, not a value. Every operator except `present`,
`not_present`, and `value_or_default` returns `null` when any argument is
`null`, rather than coercing it to `false`/`0`/not-equal. This matters because
the consumer is hipDNN's UMD matcher, where an unresolved path is an absent
optional operand: if `null` coerced, a narrowing check such as
`{"!=": ["$bias.dtype", "BFLOAT16"]}` would evaluate **true** on a graph with no
bias and the matcher would accept input it cannot serve. Two `null`s are not
equal to each other either — the question is unanswerable, so `==` and `!=` both
decline.

`and` and `or` are three-valued (Kleene): a definite `false` still decides an
`and` and a definite `true` still decides an `or`, even beside a `null`
argument; otherwise the result is `null`. That is what lets
`{"or": [{"not_present": ["$bias"]}, {"and": [{"present": ["$bias"]}, ...]}]}`
accept an absent operand whose field reads cannot run.

Once every argument resolves, value semantics follow JsonLogic/JS: `false`, `0`,
`""`, `null` and the empty array are falsy; `Number()`-style coercion drives
arithmetic and ordering. A `null` root is falsy, so an undecided expression
declines.

Malformed rules (unknown operator, wrong argument count, a non-operator object)
raise `JsonLogicCompileError` at `compile` time, so evaluation stays on the fast
path.

Not included: collection/string operators (`map`, `reduce`, `filter`, `all`,
`some`, `none`, `merge`, `cat`, `substr`, `missing`, `missing_some`).

## Tests

Unit tests live in `src/tests/core/TestJsonLogic.cpp` and build into the
`hip_kernel_provider_tests` GTest binary. Run them with:

```bash
ctest -R hip_kernel_provider_tests
# or, filtered directly on the binary:
./hip_kernel_provider_tests --gtest_filter=TestJsonLogic.*
```
