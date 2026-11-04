# Relationship Discovery Procedure

This is a shared procedure used by multiple skills to
understand how a file or set of files relates to the rest of the codebase. Run this
procedure before writing, reviewing, or documenting code.

The depth of investigation should match the task. Code review needs all 6 steps. Writing
code typically needs steps 1-4. Documentation needs all 6.

## Execution Strategy

**Parallel execution**: Steps 1-6 are independent searches. Launch all
applicable steps as simultaneous Grep/Glob calls in a single message. Do not
run them sequentially — they all take the target file path as input and do not
depend on each other's output.

**Parallel with file reads**: When a skill says to read target files AND run
relationship discovery, do both in parallel. Reading the target file and
grepping for its includes/callers/tests are independent operations.

**Reuse within a conversation**: If you have already run relationship discovery
for the same target files earlier in this conversation (e.g., during a prior
`/analyze` before a `/code` invocation), reuse the prior results instead of
re-running the same searches. If the target files have been modified since the
prior discovery, re-run only the steps whose results may have changed.

## Step 1: Direct Dependencies (What does this code use?)

Search for `#include` directives in the target files to build the dependency list.

```
grep '#include' <target-files>
```

Classify each dependency:
- **Project headers** (double-quoted includes): these are code we own and can modify
- **Third-party headers** (angle-bracket includes): these are external constraints
- **Standard library headers**: note any that may conflict with GPU kernel constraints

## Step 2: Reverse Dependencies (What uses this code?)

Search for files that include the target headers.

```
grep -r '#include.*<target-header>' --include='*.hpp' --include='*.cpp'
```

This answers: "If I change this file, what else could break?" Categorize results by:
- **Direct includers**: files that `#include` the target
- **Transitive includers**: files that include a direct includer (one level deep is
  usually sufficient)

## Step 3: Callers and Consumers

Search for usage of the public API (functions, classes, type aliases) defined in the
target files.

```
grep -r 'FunctionName\|ClassName\|TypeAlias' --include='*.hpp' --include='*.cpp'
```

For each public symbol, identify:
- **Call sites**: where the function is called
- **Instantiation sites**: where templates are instantiated with concrete types
- **Type usage**: where types/aliases are used in declarations or signatures

## Step 4: Tests

Search for test files that exercise the target code.

```
# By convention, tests often mirror the source path or name
grep -r 'TargetClassName\|target_function' test/ --include='*.cpp' --include='*.hpp'
```

Note:
- Which public APIs have test coverage
- Which public APIs lack test coverage
- Whether tests are unit tests, integration tests, or client examples

## Step 5: Sibling Implementations

Find related implementations that share the same pattern, interface, or directory.

- **Same directory**: list other files in the target's directory — they're likely variants
  or related implementations
- **Same interface**: search for other classes that inherit from the same base or satisfy
  the same concept
- **Same pattern**: search for files with similar naming patterns (e.g., if the target is
  `threadwise_tensor_slice_transfer_v7r2.hpp`, find all `threadwise_tensor_slice_transfer_*.hpp`)

```
ls $(dirname <target-file>)
grep -r 'BaseClassName\|ConceptName' --include='*.hpp' -l
```

## Step 6: Template Instantiation Chain

For template-heavy code, trace the instantiation chain:

1. **Parameters**: What template parameters does this code accept?
2. **Upstream**: Who passes these parameters? Trace back to the point where concrete
   types are chosen (usually a device operation or instance file).
3. **Downstream**: What does this code instantiate with its own parameters? Trace forward
   to the leaf templates.
4. **Instance files**: Check `project_config.md` under **Codebase Knowledge** for
   `instance_file_dirs`. If defined, search those directories for concrete
   instantiations. Otherwise, search the project root for files that reference
   the target class.

```
# Find instance files that reference the target
grep -r 'TargetClassName' --include='*.hpp' --include='*.cpp' -l
```

## Output Summary

After running the relevant steps, produce a brief summary:

```
### Relationship Summary for <target>

**Dependencies**: N project headers, N third-party headers
**Reverse dependencies**: N files include this header
**Callers**: N call sites across M files
**Tests**: [covered | partially covered | no tests found]
**Siblings**: N related implementations in same directory/family
**Instantiation chain**: [described if applicable]
```

This summary should be included in the skill's output (review findings, design doc,
or used to inform code changes).
