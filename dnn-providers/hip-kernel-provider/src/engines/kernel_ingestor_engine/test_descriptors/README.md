# Test descriptors

Authored descriptor sets for the hip-kernel-provider test suites. Production descriptors
live in the sibling `descriptors/` folder. Each sub-folder here is one independent set.

## `embedded_engine/`

Descriptors for kernels compiled into the provider, and the native selectors beside them.

## `packed-fixture-source/`

Descriptors authored in `hip` form, which the build packs into a per-arch archive. Two
authoring constraints, both of which fail silently if broken:

- **Descriptors belong in a child folder, not at that root.** The packer preserves each
  descriptor's authored subpath, so a nested descriptor reaches the archive by climbing
  back out of its own folder. A flat fixture never emits that step and cannot exercise it
  — which is how a containment bug once reached on-device testing with every suite green.

- **The fixture is its own engine, with the tightest matchers that work.** Reusing a
  shipped engine's identity collides on the completed metadata tuple and takes that engine
  down for the whole suite. A loose matcher makes the fixture visible to any test that
  walks the descriptor tree, turning tests red in files it never touched.
