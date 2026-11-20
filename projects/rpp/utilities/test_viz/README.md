# rpp-test-viz

Prototype visual debugger for the GoogleTest suite in `utilities/test/`. It shows the reference, the actual output, `|actual - reference|`, and a pass/fail mask for each comparison in a failing case. The four panes zoom and pan together, and a pixel inspector shows every pane's values under the cursor.

## Usage

Build the capture twin of `rpp_tests`. It isn't built by default, isn't registered with CTest, and isn't installed:

```bash
cmake <utilities/test> -DRPP_TEST_CAPTURE=ON && cmake --build . --target rpp_tests_capture
RPP_TEST_CAPTURE_DIR=/tmp/cap ./rpp_tests_capture --gtest_filter='*ColorCastTest*'
```

Only failing cases are written. Set `RPP_TEST_CAPTURE_ALL=1` to write passing cases too. `RPP_TEST_NO_SKIP_LIST=1` runs the known kernel defects, which is a quick way to get failures to look at.

Then write the viewer. It's one self-contained HTML file with gzipped data, no server, and no network access, about 1 MB for 128 small cases. Open it with VS Code Live Preview (`ms-vscode.live-server`), a browser over a port forward, or a downloaded copy:

```bash
uv run --project utilities/test_viz rpp-test-viz /tmp/cap                   # -> /tmp/cap/viewer.html
uv run --project utilities/test_viz rpp-test-viz /tmp/cap -o out.html --filter '*HIP_I8*'
uv run --project utilities/test_viz rpp-test-viz /tmp/cap --list            # print the cases, write nothing
```

Every selected case is embedded, so narrow a broad capture with `--filter`.

The viewer shows the reference, actual, `|actual - reference|` and status panes with linked wheel zoom and drag-to-pan. There's a slider for each leading axis (batch, depth, ...), a channel selector (all (RGB) or a single channel), slice or global contrast, and a per-channel pixel inspector. It also shows the GTest failure text, and has a case filter and `j`/`k` to step through cases.

## How it hooks in

- `utilities/test/src/framework/capture.hpp` is the seam. In `rpp_tests` its hooks are empty templates. With `RPPTEST_CAPTURE` defined, they resolve to the recorders.
- Each of the four comparators in `compare.hpp` calls one hook. Test files are untouched.
- `utilities/test/src/capture/rpp_capture.hpp` is the RPP adapter. It rewrites a descriptor and ROI into canonical arrays using the comparators' own walks and tolerance rule, so the mask matches the verdict.
- `utilities/test/src/capture/bundle.{hpp,cpp}` has no RPP dependency. It writes the bundle from a GTest listener when a case ends.

Inputs aren't captured yet.

## Bundle format (v1)

One directory per case: `<capture dir>/<Suite>/<Fixture>.<Test>/<Param>/`.

`manifest.json`:

```json
{
  "format": "rpp-test-capture", "version": 1,
  "case": "Image_Color/ColorCastTest.Correctness/HOST_I8toI8_PKD3_FullRoi_2x36x55_...",
  "passed": false,
  "failure": "<GTest failure text>",
  "comparisons": [
    {"kind": "image", "axes": ["n", "h", "w", "c"], "compared": 11880, "failed": 8122,
     "arrays": {"reference": "0_reference.npy", "actual": "0_actual.npy",
                "diff": "0_diff.npy", "status": "0_status.npy"}}
  ]
}
```

Each comparator call adds one entry to `comparisons`. Every array is a dense, row-major `.npy` over the whole logical tensor, so stride padding is dropped but the area outside the ROI is kept:

| array | dtype | meaning |
|---|---|---|
| `reference` | op dtype (`float64` for reductions) | golden |
| `actual` | op dtype | kernel output |
| `diff` | `float32` | `actual - reference`; NaN where not compared |
| `status` | `uint8` | 0 not compared, 1 pass, 2 fail |

Canonical axes by `kind`:

| kind | axes |
|---|---|
| `image` | `n, h, w, c` (NCHW and NHWC both land channels-last) |
| `voxel` | `n, d, h, w, c` |
| `nd` | `n, d1, d2, ...` (logical dims as given) |
| `reduction` | `i` |

The viewer only depends on this format, not on RPP. Anything that writes it can be viewed.
