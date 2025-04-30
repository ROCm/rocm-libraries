# Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Utilities to generate PDF plots (via Asymptote)."""

import logging
import subprocess

import pandas

from dataclasses import dataclass
from pathlib import Path
from perflib.utils import sjoin, cjoin
from typing import List
import tempfile
import numpy as np
import os
import scipy.stats

import perflib.utils

top = Path(__file__).resolve().parent.parent


@dataclass
class BaseFigure:
    tag: str
    title: str
    caption: str
    docdir: Path
    labels: List[str]
    primary: List[Path]
    secondary: List[Path]
    figtype: str
    metadata: dict[str, str]


class PDFFigure(BaseFigure):

    def asycmd(self):
        asycmd = ['asy', '-f', 'pdf']

        ndata = 0
        for filename in self.primary:
            df = pandas.read_csv(filename, sep="\t", comment='#')
            ndata = max(ndata, len(df.index))

        if ndata > 1 and self.figtype == "linegraph":
            asycmd.append(str(top / "datagraphs.asy"))
            scaling = self.metadata.get('scaling')
        elif ndata == 1 or self.figtype == "bargraph":
            asycmd.append(str(top / "bargraph.asy"))

        ivariable = self.metadata.get('ivariable')
        if ivariable == None:
            scaling = self.metadata.get('scaling')
            if scaling != None:
                ivariable = 'ndev'
                asycmd.extend(['-u', f'scaling="{scaling}"'])
        if ivariable != None:
            asycmd.extend(['-u', f'ivariable="{ivariable}"'])

        primary = [x.resolve() for x in self.primary]
        asycmd.extend(['-u', f'filenames="{cjoin(primary)}"'])

        if self.labels is not None:
            asycmd.extend(['-u', f'legendlist="{cjoin(self.labels)}"'])

        if self.secondary is not None:
            secondary = [x for x in self.secondary]
            asycmd.extend(['-u', f'secondary_filenames="{cjoin(secondary)}"'])

        asycmd.extend(['-o', self.filename])

        return [str(x) for x in asycmd]

    def runasy(self):
        asycommand = self.asycmd()
        logging.info('ASY: ' + sjoin(asycommand))

        fout = tempfile.TemporaryFile(mode="w+")
        ferr = tempfile.TemporaryFile(mode="w+")

        proc = subprocess.Popen(asycommand, cwd=top, stdout=fout, stderr=ferr)

        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            logging.info("Asy command killed: " + sjoin(asycommand))
            proc.kill()

            fout.seek(0)
            ferr.seek(0)
            cout = fout.read()
            cerr = ferr.read()

            print(cout)
            print(cerr)

        if proc.returncode != 0:
            logging.warn('ASY command failed: ' + sjoin(asycommand))

            fout.seek(0)
            ferr.seek(0)
            cout = fout.read()
            cerr = ferr.read()

            print(cout)
            print(cerr)

    def make(self, significance):
        self.filename = (Path(self.docdir) / (self.tag + '.pdf')).resolve()


gflopstext = '''\
GFLOP/s are computed based on the Cooley--Tukey operation count \
for a radix-2 transform, and half that for in the case of \
real-complex transforms.  The rocFFT operation count may differ from \
this value: GFLOP/s is provided for the sake of comparison only.'''

efficiencytext = '''\
Efficiency is computed for an idealised FFT which requires exactly \
one read and one write to global memory.  In practice, this \
isn't possible for most problem sizes, as the data does \
not fit into cache, and one must use global memory to store \
intermediary results.  As FFTs are bandwidth-limited on modern hardware, \
the efficiency is measured against the theoretical maximum bandwidth \
for the device.'''


def make_tex(figs,
             docdir,
             outdirs,
             label,
             significance,
             ncompare,
             secondtype=None):
    """Generate PDF containing performance figures."""

    docdir = Path(docdir)

    header = '''\
\\documentclass[12pt]{article}
\\usepackage[margin=1in]{geometry}
\\usepackage{graphicx}
\\usepackage{url}
\\usepackage{hyperref}
\\usepackage{float}
\\usepackage{longtable}
\\begin{document}
\\hypersetup{
  pdfborder={0,0,0},
  linkcolor=blue,
  citecolor=blue,
  urlcolor=blue
}
'''
    globalgeomean = perflib.utils.find_geomean(outdirs, False)
    print("geomean:", globalgeomean)

    tex = header

    tex += "\n\\section{Introduction}\n"

    # tex += "Each data point represents the median of " + str(nsample) + " values, with error bars showing the 95\\% confidence interval for the median.  All transforms are " + precision + "-precision.\n\n"

    if secondtype == "gflops":
        tex += gflopstext + "\n\n"

    tex += "\\vspace{1cm}\n"

    tex += "\n\\section{Device Specification}\n"
    for idx in range(len(outdirs)):
        tex += "\n\\subsection{" + str(label[idx]) + "}\n"
        path = Path(outdirs[idx]) / "specs.txt"
        if path.is_file():
            specs = path.read_text()

            for line in specs.split("\n"):
                if line.startswith("Host info"):
                    tex += "\\noindent " + line
                    tex += "\\begin{itemize}\n"
                elif line.startswith("Device info"):
                    tex += "\\end{itemize}\n"
                    tex += line
                    tex += "\\begin{itemize}\n"
                else:
                    if line.strip() != "":
                        tex += "\\item \\verb|" + line + "|\n"
            tex += "\\end{itemize}\n"
            tex += "\n"

    tex += "\\clearpage\n"
    tex += "\\section{Figures}\n"
    tex += "\\listoffigures\n"
    tex += "\\clearpage\n"

    # Data frames for significant speedups and slowdowns
    df_all_good = pandas.DataFrame()
    df_all_bad = pandas.DataFrame()

    # We need a list of speedups to compute the geometric mean via
    # sicpy.stats; the naive calculation suffers from issues with
    # finite precision.
    speedups = []

    figtex = ""

    for idx, fig in enumerate(figs):
        figtex += '''
\\centering
\\begin{figure}[H]
   \\includegraphics[width=\\textwidth]{'''
        figtex += str(fig.filename.name)
        figtex += '''}
   \\caption{''' + fig.caption + '''}
\\end{figure}
'''
        for p in fig.secondary:
            df = pandas.read_csv(p, sep="\t", comment='#')

            for row in df.itertuples(index=False):
                speedups.append(row.speedup)

            # Significant results:
            df_sig = df.loc[df['speedup_pval'] <= significance]

            # Significant results that are good or bad:
            df_good = df_sig.loc[df_sig['speedup'] > 1]
            df_bad = df_sig.loc[df_sig['speedup'] < 1]

            if not df_good.empty:

                df_all_good = pandas.concat([df_all_good, df_good])

                figtex += "\\begin{longtable}{l|l|l|}\n"
                figtex += "transform & speedup \% & significance\\\\ \n"
                figtex += "\\hline\n"
                for row in df_good.itertuples(index=False):
                    #figtex += str(row.token).replace("_", "\\_")
                    #figtex += "token"
                    transform_type, placeness, length, batch, precision = perflib.utils.parse_token(
                        row.token)
                    figtex += "$" + "\\times{}".join(str(x)
                                                     for x in length) + "$"

                    if np.prod(batch) > 1:
                        figtex += " by $" + "\\times{}".join(
                            str(x) for x in batch) + "$"

                    speedup = '{0:.3f}'.format((row.speedup - 1) * 100)
                    pval = '{0:.3f}'.format(row.speedup_pval)
                    figtex += " & " + str(speedup) + " & " + str(pval) + "\\\\"
                figtex += "\\hline\n"
                figtex += "\\caption{Improvements for " + fig.caption + "}\n"
                figtex += "\\end{longtable}\n"

            if not df_bad.empty:

                df_all_bad = pandas.concat([df_all_bad, df_bad])

                figtex += "\\begin{longtable}{l|l|l|}\n"
                figtex += "transform & slowdown \% & significance\\\\ \n"
                figtex += "\\hline\n"
                for row in df_bad.itertuples(index=False):
                    #figtex += str(row.token).replace("_", "\\_")
                    #figtex += "token"
                    transform_type, placeness, length, batch, precision = perflib.utils.parse_token(
                        row.token)
                    figtex += "$" + "\\times{}".join(str(x)
                                                     for x in length) + "$"

                    if np.prod(batch) > 1:
                        figtex += " by $" + "\\times{}".join(
                            str(x) for x in batch) + "$"

                    speedup = '{0:.3f}'.format((1 - 1 / row.speedup) * 100)

                    pval = '{0:.3f}'.format(row.speedup_pval)
                    figtex += " & " + str(speedup) + " & " + str(pval) + "\\\\"
                figtex += "\\hline\n"
                figtex += "\\caption{Regressions for " + fig.caption + "}\n"
                figtex += "\\end{longtable}\n"

        figtex += "\\clearpage\n"

    nspeedup = len(df_all_good.index)
    nslowdown = len(df_all_bad.index)

    if len(outdirs) > 1:
        print(
            "nspeedup  (" + label[1] + " is faster): " +
            " " * max(len(label[0]) - len(label[1]), 0), nspeedup)
        print(
            "nslowdown (" + label[0] + " is faster): " +
            " " * max(len(label[1]) - len(label[0]), 0), nslowdown)
        tex += "geometric mean overall cases: " + str(globalgeomean) + "\n"

        if ncompare > 0:
            geometric_mean = 1.0
            if len(speedups) > 1:
                geometric_mean = scipy.stats.mstats.gmean(speedups)
            print("geometric mean:", geometric_mean)
            tex += "\\begin{table}[H]\n"
            tex += "\\centering\n"
            tex += "\\begin{tabular}{l|p{5cm}|p{5cm}|l|}\n"
            tex += "ncompare & nspeedup " + "(" + label[
                1] + " faster)" + " & nslowdown (" + label[
                    0] + " faster)" + " & gmean\\\\ \n"
            tex += "\\hline\n"
            tex += str(ncompare) + "&" + str(nspeedup) + "&" + str(
                nslowdown) + "&" + '{0:.3f}'.format(geometric_mean) + "\\\\\n"
            tex += "100\\%" + "&" + '{0:.3f}'.format(
                100 * nspeedup / ncompare) + "\\% " + "&" + '{0:.3f}'.format(
                    100 * nslowdown / ncompare) + "\\% " + "&\\\\\n"
            tex += "\\hline\n"
            tex += "\\end{tabular}\n"
            tex += "\\caption{Overall Performance Changes}\n"
            tex += "\\end{table}\n"

        if nspeedup + nslowdown > 0:
            vals = []
            for row in df_all_good.itertuples(index=False):
                vals.append(100 * (row.speedup - 1))
            for row in df_all_bad.itertuples(index=False):
                vals.append(100 * (1 - (1 / row.speedup)))

            histdatname = os.path.join(docdir, "histogram.dat")

            with open(histdatname, 'w') as f:
                f.write("\t".join(str(x) for x in vals))
                f.write("\n")

            fout = tempfile.TemporaryFile(mode="w+")
            ferr = tempfile.TemporaryFile(mode="w+")

            asycmd = ["asy", "-f", "pdf", "histogram.asy"]
            asycmd.extend(['-u', 'filename="' + histdatname + '"'])
            asycmd.extend(['-o', os.path.join(docdir, "histogram.pdf")])

            asyproc = subprocess.Popen(asycmd,
                                       cwd=top,
                                       stdout=fout,
                                       stderr=ferr)
            try:
                asyproc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                logging.info("asy command killed: " + sjoin(asycmd))
                asyproc.kill()

            if asyproc.returncode != 0:
                logging.warn('ASY command failed: ' + sjoin(asycmd))

                fout.seek(0)
                ferr.seek(0)
                cout = fout.read()
                cerr = ferr.read()

                print(cout)
                print(cerr)

            tex += '''\\centering
    \\begin{figure}[H]
    \\includegraphics[width=\\textwidth]{'''
            tex += "histogram.pdf"
            tex += '''}
    \\caption{''' + "Histogram of performance changes" + '''}\n\\end{figure}'''

        tex += "\\clearpage\n"

    tex += figtex

    if nspeedup > 0:
        tex += "\\clearpage\n"
        tex += "tokens for improved performance\n"
        tex += "\\begin{tiny}"
        tex += "\\begin{verbatim}"
        for row in df_all_good.itertuples(index=False):
            #print(row.token)

            tex += str(row.token) + "\n"
            #tex += "\\small\\texttt{" + str(row.token).replace("_", "\\_") + "}\n"
            #tex += str(row.token).replace("_", "\\_") + "\n"
        tex += "\\end{verbatim}"
        tex += "\\end{tiny}"

    if nslowdown > 0:
        print("There were", nslowdown, "regressions.  The tokens are:")
        tex += "\\clearpage\n"
        tex += "tokens for regressed performance\n"
        tex += "\\begin{tiny}"
        tex += "\\begin{verbatim}"
        for row in df_all_bad.itertuples(index=False):
            print(row.token, row.speedup)
            tex += str(row.token) + "\n"
            #tex += "\\small\\texttt{" + str(row.token).replace("_", "\\_") + "}\n"
        tex += "\\end{verbatim}"
        tex += "\\end{tiny}"

    tex += "\n\\end{document}\n"

    fname = docdir / 'figs.tex'
    fname.write_text(tex)

    log = docdir / 'tex.log'
    with log.open('w') as f:
        #latexcmd = ['latexmk','-pdf', fname.name]
        latexcmd = ['latexmk', '-interaction=batchmode', '-pdf', fname.name]
        texproc = subprocess.Popen(latexcmd,
                                   cwd=fname.parent,
                                   stdout=f,
                                   stderr=f)
        try:
            texproc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            logging.info("tex command killed: " + sjoin(latexcmd))
            texproc.kill()

        if texproc.returncode != 0:
            print("****tex fail****")
