You are widening what an existing hipDNN kernel-ingestor engine can serve.

- Engine:           ${inputs.engine_name}
- Anchor graph:     ${inputs.graph}
- Arch:             ${inputs.arch}
- What to widen:    ${inputs.widen}
- Attempt:          ${loop.attempt} of ${loop.max_iterations}
- Operator notes:   ${inputs.notes}

The engine works. It is also narrow: it was written for one graph, so its matcher
admits one layout, one rank and one shape family, and declines everything else. Your
job is to make it serve more, and to prove it with tests rather than with a wider
matcher.

# Two gates, and they pull in opposite directions

1. **The engine still passes everything it passed before.** Loosening a matcher until
   it accepts graphs it then computes wrongly is caught here.
2. **The suite passes strictly more cases than the engine accepted at baseline, and the
   bundle tree gained a case or a sweep grew.** Changing nothing is caught here, and so
   is widening the matcher without adding a test that exercises the new ground.

An admitted shape with no case behind it is a promise nobody checks. That is why the
growth gate counts bundle cases on disk and not claims in your contract.

# Read these first

- **`${vars.authoring_file}`** — the kernel's current admitted envelope and launch ABI.
  This is the thing you are changing; read what it promises today before promising more.
- **`${loop.feedback_path}`** — one section per failed round, naming the failing step and
  its log directory. Open it. The one-line note is not the evidence.

## You are probably not starting from nothing

Attempt ${loop.attempt}; nothing a previous attempt wrote has been reverted:

```
git -C ${vars.repo_root} status --short     # what previous attempts already changed
ls ${vars.ingestor_dir}                     # kernels, pack, descriptors
ls ${vars.provider_tests_dir}               # the matcher and pack tests
```

Read before you widen. A previous round may have widened one axis and failed on
another; redoing its work spends this round's budget on ground already taken.

# Widening is four separate axes — pick deliberately

Do not try to take all of them in one round. Each has its own failure mode and its own
tests, and a round that widens one axis with cases to prove it beats a round that
claims four and demonstrates none.

- **Layout.** Additional strides, most commonly NHWC beside NCHW. Remember that `dims`
  stay canonical `(N, C, H, W)` and the layout lives in the strides — a kernel that
  reads dims to decide layout is reading the wrong thing. Index arithmetic that assumed
  contiguity has to become stride-driven.
- **Rank.** Rank-5 (NCDHW) and rank-3 graphs beside rank-4. Spatial loops and the
  padding/stride/dilation vectors become `rank - 2` long rather than 2.
- **Attributes.** Non-unit dilation, asymmetric pre/post padding, the other convolution
  mode, alternative pointwise modes in a fusion, bias broadcast shapes.
- **Shape family.** Channel counts not divisible by the tile, spatial extents that do
  not divide evenly by the block, batch > 1, group counts other than the one the kernel
  was written for. This is where guard conditions matter: specialising to shapes that
  divide evenly is a correctness defect the moment the matcher admits one that does not.

For each axis you take: widen the matcher, make the kernel actually handle it, and add a
bundle case that exercises it. All three, or it does not count.

# The matcher and the kernel must agree

The failure this flow exists to prevent is a matcher that admits more than the kernel
computes. The matcher is the promise; the kernel is the delivery; the bundle case is the
receipt. When you widen the matcher, ask what the kernel now has to handle that it did
not before — a stride it assumed, a bound it did not guard, an accumulator that was
sized for one dtype — and handle it.

If an axis turns out to be more than this round can do properly, **narrow the matcher
back to what the kernel really serves** and say so in `summary`. Declining honestly is a
correct outcome; admitting and miscomputing is not.

# Scope

You may work under `${vars.ingestor_dir}` and `${vars.provider_tests_dir}`, add bundle
cases under the integration-test bundle tree, and append to a `sweep.json`. You may not
rewrite an existing bundle case, edit another engine, or touch the integration suite or
its category YAMLs. The scope guard checks this rather than trusting it.

# Output contract

Write a JSON object to exactly this path:

    ${step.result_file}

Same integration contract shape the engine already has; the gates re-check it against
the checkout, so it must describe the tree as it now stands. `bundle_case_ids` must
include the cases you added — that list is what the test gate counts against.

In `summary`: which axis you widened, what the kernel had to change to really serve it,
which cases now prove it, and what you deliberately left narrow. Be specific about the
last one — the next run starts from what you say the engine does not yet do.
