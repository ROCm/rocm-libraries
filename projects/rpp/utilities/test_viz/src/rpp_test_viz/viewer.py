"""napari viewer: reference | actual | |diff| | status, side by side with a shared camera."""

from __future__ import annotations

from dataclasses import dataclass

import napari
import numpy as np
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtWidgets import QLabel, QListWidget, QListWidgetItem, QSplitter
from qtpy.QtCore import Qt

from .bundle import FAIL, NOT_COMPARED, PASS, Case, Comparison

STATUS_COLORS = {
    None: "transparent",
    NOT_COMPARED: "#30363d",
    PASS: "#2ea04380",
    FAIL: "#f85149",
}
STATUS_NAMES = {NOT_COMPARED: "not compared", PASS: "pass", FAIL: "FAIL"}


@dataclass
class View:
    """How a comparison's canonical arrays map onto napari layers."""

    rgb: bool  # trailing 3-channel axis shown as colour
    labels: list[str]  # napari axis labels for the displayed arrays

    @staticmethod
    def of(cmp: Comparison) -> View:
        axes = list(cmp.axes)
        rgb = axes[-1] == "c" and cmp.reference.shape[-1] == 3
        if axes[-1] == "c" and (rgb or cmp.reference.shape[-1] == 1):
            axes = axes[:-1]
        if len(axes) == 1:
            axes = ["", *axes]
        return View(rgb, axes)

    def image(self, a: np.ndarray) -> np.ndarray:
        return self._shape(a.astype(np.float32), collapse=None)

    def collapsed(self, a: np.ndarray, how) -> np.ndarray:
        return self._shape(a, collapse=how)

    def _shape(self, a: np.ndarray, collapse) -> np.ndarray:
        if a.ndim == len(self.labels) + 1:  # trailing channel axis present
            if collapse is not None:
                a = collapse(a, axis=-1)
            elif not self.rgb:
                a = a[..., 0]
        return a.reshape(1, -1) if a.ndim == 1 else a

    def raw_index(self, coord: tuple[int, ...]) -> tuple[int, ...]:
        """Display coordinate -> index into the canonical arrays (channel axis left whole)."""
        return coord[1:] if self.labels[0] == "" else coord


def _contrast(cmp: Comparison) -> tuple[float, float]:
    dtype = cmp.actual.dtype
    if np.issubdtype(dtype, np.integer) and dtype.itemsize == 1:
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    both = np.concatenate([cmp.reference.ravel(), cmp.actual.ravel()]).astype(
        np.float64
    )
    both = both[np.isfinite(both)]
    if both.size == 0:
        return 0.0, 1.0
    lo, hi = float(both.min()), float(both.max())
    return (lo, hi) if hi > lo else (lo - 0.5, hi + 0.5)


def _fmt(v) -> str:
    v = np.asarray(v)
    if v.ndim:
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    return f"{v.item():.6g}" if np.issubdtype(v.dtype, np.floating) else str(v.item())


class CaseBrowser:
    """Case list + inspector docked next to the four-pane grid."""

    def __init__(self, viewer: napari.Viewer, cases: list[Case]):
        self.viewer = viewer
        self.entries = [
            (case, k) for case in cases for k in range(len(case.comparisons))
        ]
        self.cmp: Comparison | None = None
        self.view: View | None = None

        self.list = QListWidget()
        for case, k in self.entries:
            entry = case.comparisons[k]
            suffix = f" [{k}]" if len(case.comparisons) > 1 else ""
            mark = "PASS" if case.passed else f"{entry['failed']}/{entry['compared']}"
            QListWidgetItem(f"{mark:>13}  {case.name}{suffix}", self.list)
        self.list.currentRowChanged.connect(self.show)

        self.inspector = QLabel()
        self.inspector.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.inspector.setWordWrap(True)
        self.inspector.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.list)
        splitter.addWidget(self.inspector)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        dock = viewer.window.add_dock_widget(
            splitter, name="Cases", area="right", add_vertical_stretch=False
        )
        dock.show()
        viewer.cursor.events.position.connect(self.inspect)

        viewer.grid.enabled = True
        viewer.grid.spacing = 0.05
        if self.entries:
            self.list.setCurrentRow(0)

    def show(self, row: int) -> None:
        if row < 0:
            return
        case, k = self.entries[row]
        cmp = self.cmp = case.load(k)
        view = self.view = View.of(cmp)
        limits = _contrast(cmp)

        # The camera and dims survive the layer swap so stepping through cases keeps the zoom.
        self.viewer.layers.clear()
        self.viewer.add_image(
            view.image(cmp.reference),
            name="reference",
            rgb=view.rgb,
            contrast_limits=limits,
        )
        self.viewer.add_image(
            view.image(cmp.actual), name="actual", rgb=view.rgb, contrast_limits=limits
        )
        absdiff = np.nan_to_num(np.abs(cmp.diff), nan=0.0)
        peak = float(absdiff.max()) or 1.0
        self.viewer.add_image(
            view.collapsed(absdiff, np.max),
            name="|actual - reference|",
            colormap="inferno",
            contrast_limits=(0.0, peak),
        )
        self.viewer.add_labels(
            view.collapsed(cmp.status, np.max).astype(np.uint8),
            name="status",
            colormap=DirectLabelColormap(color_dict=STATUS_COLORS),
        )
        self.viewer.layers.selection.active = self.viewer.layers["actual"]
        self.viewer.dims.axis_labels = view.labels
        self.viewer.title = case.name
        self.inspector.setText(self._summary(case, cmp))

    def _summary(self, case: Case, cmp: Comparison) -> str:
        head = (
            f"<b>{case.name}</b><br>{cmp.kind} {tuple(cmp.reference.shape)} "
            f"axes={','.join(cmp.axes)} ref={cmp.reference.dtype} act={cmp.actual.dtype}<br>"
            f"failed {cmp.failed} of {cmp.compared} compared"
        )
        return head

    def inspect(self, _event=None) -> None:
        if self.cmp is None or not len(self.viewer.layers):
            return
        layer = self.viewer.layers["reference"]
        coord = tuple(
            int(np.round(c)) for c in layer.world_to_data(self.viewer.cursor.position)
        )
        coord = coord[: len(self.view.labels)]
        shape = self.view.image(self.cmp.reference).shape[: len(self.view.labels)]
        if any(c < 0 or c >= s for c, s in zip(coord, shape)):
            return
        idx = self.view.raw_index(coord)
        status = np.atleast_1d(self.cmp.status[idx])
        where = ", ".join(f"{a}={c}" for a, c in zip(self.view.labels, coord) if a)
        lines = [
            f"<b>{where}</b>",
            f"reference: {_fmt(self.cmp.reference[idx])}",
            f"actual: {_fmt(self.cmp.actual[idx])}",
            f"diff: {_fmt(self.cmp.diff[idx])}",
            "status: " + ", ".join(STATUS_NAMES[int(s)] for s in status),
        ]
        case, _ = self.entries[self.list.currentRow()]
        self.inspector.setText(
            self._summary(case, self.cmp) + "<hr>" + "<br>".join(lines)
        )


def launch(cases: list[Case], show: bool = True) -> CaseBrowser:
    """Opens the viewer; hold the returned browser for as long as the window lives."""
    return CaseBrowser(napari.Viewer(title="rpp-test-viz", show=show), cases)
