#!/usr/bin/env python3

################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

import argparse
import re
import string
from typing import Dict, List, Tuple

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output


class Instruction:
    """
    Single assembly instruction
    """

    def __init__(self, inst: str) -> None:
        self._inst = inst

    def is_comment(self) -> bool:
        """
        //-style comments
        """
        return self._inst.lstrip().startswith("//")

    def __str__(self) -> str:
        return self._inst


class RegisterState:
    """
    The liveness state of a register
    """

    def __init__(self, symbol: str, reg_type: str) -> None:
        self._symbol = symbol
        self._reg_type = reg_type

    @property
    def symbol(self) -> str:
        """
        Character representing register state.
        | *space* | "Dead" register |
        | `:` | "Live" register |
        | `^` | This register is written to this instruction. |
        | `v` | This register is read from this instruction. |
        | `x` | This register is written to and read from this instruction. |
        | `_` | This register is allocated but dead. |
        """
        return self._symbol

    @property
    def reg_type(self) -> str:
        return self._reg_type

    def is_live(self) -> bool:
        return self._symbol in [":", "x", "^", "v"]

    def is_allocated(self) -> bool:
        return self._symbol != " "

    def is_written_to(self) -> bool:
        return self._symbol in ["x", "^"]

    def is_read_from(self) -> bool:
        return self._symbol in ["x", "v"]


class RegisterCollection:
    """
    The register states listed at one point in time
    """

    def __init__(self) -> None:
        self._data: List[RegisterState] = []

    def append(self, entry: RegisterState) -> None:
        self._data.append(entry)

    def get_liveness_sum(self, instruction: str) -> List:
        return [
            sum([int(x.is_live()) for x in self._data]),
            sum([int(x.is_allocated()) for x in self._data]),
            self.get_max_allocated_reg(),
            self.get_reg_type(),
            instruction,
        ]

    def get_reg_type(self) -> str:
        if len(self._data) > 0:
            return self._data[0].reg_type
        return ""

    def get_max_allocated_reg(self) -> int:
        """
        Get the largest value of the allocated registers
        """
        # Take advantage of spaces being the non-allocated symbol
        return len("".join([x.symbol for x in self._data]).rstrip())

    @staticmethod
    def get_liveness_columns() -> List[str]:
        """
        Pandas-style column names
        """
        return [
            "Liveness",
            "Allocations",
            "Max Allocated",
            "Register Type",
            "Instruction",
        ]


class RegisterHistory:
    def __init__(self) -> None:
        self._data: List[RegisterCollection] = []

    def append(self, entry: RegisterCollection) -> None:
        self._data.append(entry)

    def get_liveness_df(
        self,
        instructions: List[List[Instruction]],
        line_numbers: List[int],
        filter_comments: bool = True,
    ) -> pd.DataFrame:
        inst_results: List[str] = []
        exclusions: List[int] = []
        for i, line in enumerate(instructions):
            if filter_comments:
                line = list(
                    filter(
                        lambda inst: not (inst.is_comment() or str(inst).strip() == ""),
                        line,
                    )
                )
                if len(line) == 0:
                    exclusions.append(i)
            inst_results.append(";".join([str(inst) for inst in line]))

        df = pd.DataFrame(
            [
                x.get_liveness_sum(result) + [line_number]
                for x, result, line_number in zip(
                    self._data, inst_results, line_numbers
                )
            ]
        )
        df.columns = RegisterCollection.get_liveness_columns() + ["Line Number"]
        df = df.drop(index=exclusions)
        df = df.reset_index()
        return df


def read_liveness(
    filename: str,
) -> Tuple[Dict[str, RegisterHistory], List[List[Instruction]], List[int]]:
    accvgprs = RegisterHistory()
    vgprs = RegisterHistory()
    sgprs = RegisterHistory()
    instructions: List[List[Instruction]] = []
    line_numbers: List[int] = []

    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            entries = line.split("|")

            if len(entries) == 4:
                accvgpr_state = RegisterCollection()
                for symbol in entries[0]:
                    accvgpr_state.append(RegisterState(symbol, "ACCVGPR"))
                accvgprs.append(accvgpr_state)

                vgpr_state = RegisterCollection()
                for symbol in entries[1]:
                    vgpr_state.append(RegisterState(symbol, "VGPR"))
                vgprs.append(vgpr_state)

                sgpr_state = RegisterCollection()
                for symbol in entries[2]:
                    sgpr_state.append(RegisterState(symbol, "SGPR"))
                sgprs.append(sgpr_state)

                # Extract instruction info
                inst_info = (
                    entries[3]
                    .lstrip(string.digits + string.whitespace + ".")
                    .split(";")
                )
                instruction_entries: List[Instruction] = []
                for inst in inst_info:
                    instruction_entries.append(Instruction(inst))
                instructions.append(instruction_entries)

                line_numbers.append(int(re.search(r"\d+", entries[3]).group()))

    return (
        {"ACCVGPR": accvgprs, "VGPR": vgprs, "SGPR": sgprs},
        instructions,
        line_numbers,
    )


def make_figure(reg_type: str, df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df.index.tolist(),
                y=df["Liveness"],
                name="Live " + reg_type,
            ),
            go.Scatter(
                x=df.index.tolist(),
                y=df["Allocations"],
                name="Allocated " + reg_type,
            ),
            go.Scatter(
                x=df.index.tolist(),
                y=df["Max Allocated"],
                name="Max Allocated " + reg_type,
            ),
        ],
    )
    fig.update_layout(template=theme)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RocRoller Register Liveness Visualizer."
    )

    parser.add_argument(
        "filename",
        help="Liveness file generated with ROCROLLER_KERNEL_ANALYSIS=1.",
    )

    parser.add_argument(
        "--filter_comments",
        action="store_true",
        help="Filter comments out of assembly strings",
    )

    parser.add_argument(
        "--port",
        help="Port to run the application on",
        default="8050",
    )

    args = parser.parse_args()

    registerHistories, instructions, line_numbers = read_liveness(args.filename)
    df_vgpr = registerHistories["VGPR"].get_liveness_df(
        instructions, line_numbers, args.filter_comments
    )
    df_sgpr = registerHistories["SGPR"].get_liveness_df(
        instructions, line_numbers, args.filter_comments
    )
    df_accvgpr = registerHistories["ACCVGPR"].get_liveness_df(
        instructions, line_numbers, args.filter_comments
    )

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
    server = app.server

    theme = "plotly_dark"

    fig_vgpr = make_figure("VGPR", df_vgpr)
    fig_sgpr = make_figure("SGPR", df_sgpr)
    fig_accvgpr = make_figure("ACCVGPR", df_accvgpr)

    app.layout = dbc.Container(
        className="p-5",
        children=[
            html.H1(children="Register Liveliness"),
            html.Div(
                children="""
            A RocRoller tool
        """
            ),
            dcc.Graph(id="v reg", figure=fig_vgpr, style={"height": "75vh"}),
            html.Div(
                [
                    "VGPR",
                    html.Pre(id="hover-data-v"),
                ],
            ),
            dcc.Graph(id="s reg", figure=fig_sgpr, style={"height": "75vh"}),
            html.Div(
                [
                    "SGPR",
                    html.Pre(id="hover-data-s"),
                ],
            ),
            dcc.Graph(id="a reg", figure=fig_accvgpr, style={"height": "75vh"}),
            html.Div(
                [
                    "ACCVGPR",
                    html.Pre(id="hover-data-a"),
                ],
            ),
        ],
    )

    def get_hover_str(df: pd.DataFrame, index: int):
        live_reg = df["Liveness"][index]
        alloc_reg = df["Allocations"][index]
        line_number = df["Line Number"][index]
        max_reg = df["Max Allocated"][index]
        return (
            f"Line {line_number} (x={index}): "
            + df["Instruction"][index]
            + f"\nLiving reg: {live_reg}, Allocated reg: {alloc_reg}, "
            + f"Max allocated reg ID: {max_reg - 1}"
        )

    @app.callback(Output("hover-data-v", "children"), Input("v reg", "hoverData"))
    def display_hover_data_v(hoverData):
        if hoverData is None:
            return
        i = hoverData["points"][0]["pointIndex"]
        return get_hover_str(df_vgpr, i)

    @app.callback(Output("hover-data-s", "children"), Input("s reg", "hoverData"))
    def display_hover_data_s(hoverData):
        if hoverData is None:
            return
        i = hoverData["points"][0]["pointIndex"]
        return get_hover_str(df_sgpr, i)

    @app.callback(Output("hover-data-a", "children"), Input("a reg", "hoverData"))
    def display_hover_data_a(hoverData):
        if hoverData is None:
            return
        i = hoverData["points"][0]["pointIndex"]
        return get_hover_str(df_accvgpr, i)

    app.run(port=args.port)
