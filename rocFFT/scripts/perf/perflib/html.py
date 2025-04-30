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
"""HTML plotting."""

import os

from .pdf import BaseFigure
from .utils import to_data_frames


def speedup_saturation(speedup):
    diff = abs(1.0 - speedup)
    # scale up a 20% difference into maximum 50% saturation - lower
    # difference should be closer to 255 (white)
    saturation_ratio = min(0.5, diff / 0.2 * 0.5)
    return 255 - int(saturation_ratio * 255.0)


def speedup_colors(speedup):
    ret = []
    for s in speedup:
        saturation = speedup_saturation(s)
        if s < 1.0:
            # slowdown is red
            ret.append('rgb(255,{0},{0})'.format(saturation))
        else:
            # speedup is green
            ret.append('rgb({0},255,{0})'.format(saturation))
    return ret


def significance_colors(significance, threshold):
    ret = []
    for s in significance:
        if s < threshold:
            saturation = 128 + int(128 * (s / threshold))
            ret.append('rgb({0},{0},255)'.format(saturation))
        else:
            ret.append('white')
    return ret


def title_to_html_anchor(title):
    return title.replace(' ', '_')


def token_to_length(tokens):
    length = []
    for token in tokens:
        words = token.split("_")
        for idx in range(len(words)):
            if (words[idx] == "len"):
                lenidx = idx + 1
                thislength = []
                while lenidx < len(words) and words[lenidx].isnumeric():
                    thislength.append(int(words[lenidx]))
                    lenidx += 1
                length.append(thislength)
    return length


def token_to_batch(tokens):
    batch = []
    for token in tokens:
        words = token.split("_")
        for idx in range(len(words)):
            if (words[idx] == "batch"):
                batchidx = idx + 1
                thisbatch = []
                while batchidx < len(words) and words[batchidx].isnumeric():
                    thisbatch.append(int(words[batchidx]))
                    batchidx += 1
                batch.append(thisbatch)
    return batch


def token_to_elements(tokens):
    length = token_to_length(tokens)
    batch = token_to_batch(tokens)
    elements = []
    for i in range(len(length)):
        n = int(1)
        for j in range(len(length[i])):
            n *= int(length[i][j])
        for j in range(len(batch[i])):
            n *= int(batch[i][j])
        elements.append(n)
    return elements


def token_to_size_description(tokens):
    length = token_to_length(tokens)
    batch = token_to_batch(tokens)
    descriptions = []
    for cur_len, cur_batch in zip(length, batch):

        def join_ints(ints):
            return 'x'.join([str(val) for val in ints])

        desc = join_ints(cur_len) + 'b' + join_ints(cur_batch)
        descriptions.append(desc)
    return descriptions


class HTMLFigure(BaseFigure):

    def make(self, sig_threshold):
        from plotly import graph_objs as go
        data_frames = to_data_frames(self.primary, self.secondary)
        for df in data_frames:
            df['elements'] = token_to_elements(df.token)
            df.sort_values(by='elements', inplace=True)
        logscale = False

        traces = []

        traces.append(
            go.Scatter(x=data_frames[0].elements,
                       y=data_frames[0].median_sample,
                       hovertext=token_to_size_description(
                           data_frames[0].token),
                       name=self.labels[0]))

        for i in range(1, len(data_frames)):
            traces.append(
                go.Scatter(x=data_frames[i].elements,
                           y=data_frames[i].median_sample,
                           hovertext=data_frames[i].token,
                           name=self.labels[i]))
            traces.append(
                go.Scatter(x=data_frames[i].elements,
                           y=data_frames[i].speedup,
                           name='Speedup {} over {}'.format(
                               self.labels[i], self.labels[0]),
                           yaxis='y2',
                           error_y=dict(
                               type='data',
                               symmetric=False,
                               array=data_frames[i].speedup_high -
                               data_frames[i].speedup,
                               arrayminus=data_frames[i].speedup -
                               data_frames[i].speedup_low,
                           )))

        if logscale:
            x_title = 'Problem size (elements, logarithmic)'
            axis_type = 'log'
            y_title = 'Time (ms, logarithmic)'
        else:
            x_title = 'Problem size (elements)'
            axis_type = 'linear'
            y_title = 'Time (ms)'

        layout = go.Layout(title=self.title,
                           xaxis=dict(
                               title=x_title,
                               type=axis_type,
                           ),
                           yaxis=dict(title=y_title,
                                      type=axis_type,
                                      rangemode='tozero'),
                           yaxis2=dict(title='Speedup',
                                       overlaying='y',
                                       side='right',
                                       type='linear',
                                       rangemode='tozero'),
                           hovermode='x unified',
                           width=900,
                           height=600,
                           legend=dict(yanchor="top", xanchor="right", x=1.2))

        fig = go.Figure(data=traces, layout=layout)

        # Add speedup=1 reference line
        fig.add_shape(type='line',
                      x0=min(data_frames[0].elements),
                      y0=1,
                      x1=max(data_frames[0].elements),
                      y1=1,
                      line=dict(color='grey', dash='dash'),
                      yref='y2')

        nrows = len(data_frames[0].index)
        headers = ['Problem size', 'Elements']
        values = [
            token_to_size_description(data_frames[0].token),
            data_frames[0].elements
        ]
        fill_colors = [
            ['white'] * nrows,
            ['white'] * nrows,
        ]
        for i in range(len(data_frames)):
            headers.append(self.labels[i] + ' (median)')
            values.append(
                ["{:.4f}".format(x) for x in data_frames[i].median_sample])
            fill_colors.append(['white'] * nrows)

            if i > 0:
                headers.append('Speedup {} over {}'.format(
                    self.labels[i], self.labels[0]))
                values.append(
                    ["{:.4f}".format(x) for x in data_frames[i].speedup])
                fill_colors.append(speedup_colors(data_frames[i].speedup))

                headers.append('Significance of {} over {}'.format(
                    self.labels[i], self.labels[0]))
                values.append(
                    ["{:.4f}".format(x) for x in data_frames[i].speedup_pval])
                fill_colors.append(
                    significance_colors(data_frames[i].speedup_pval,
                                        sig_threshold))

        table = go.Figure(data=[
            go.Table(
                header=dict(values=headers),
                cells=dict(values=values, fill_color=fill_colors),
            ),
        ],
                          layout=go.Layout(title=self.title, ))
        # 900 seems to be enough for 2 dirs
        # widen table by 200px per extra dir
        table_width = 900
        if len(data_frames) > 2:
            table_width += (len(data_frames) - 2) * 200
        table.update_layout(width=table_width, height=600)

        self.plot = fig
        self.table = table


def make_html(figures, title, docdir, outdirs, significance):
    # TODO: this needs to read the output from the post-processing;
    # graphing and post-processing should be separate.

    # # use first dir's dat files as a basis for what to graph.
    # # assumption is that other dirs have the same-named files.
    # dat_files = glob.glob(os.path.join(dirs[0], '*.dat'))
    # # sort files so diagrams show up in consistent order for each run
    # dat_files.sort()

    # construct the output file "figs.html"
    outfile = open(docdir / 'figs.html', 'w')
    outfile.write('''
<html>
  <head>
    <title>{}</title>
  </head>
  <body>
'''.format(title))

    # FIXME: use the find_slower_faster function to get perf changes.

    # make links to each figure at the top of the report
    outfile.write('''<b>Quick links:</b><br/>''')
    for figure in figures:
        plot, table = figure.plot, figure.table
        fig_title = plot.layout.title.text
        anchor = title_to_html_anchor(fig_title)
        outfile.write('''<a href="#{}">{}</a><br/>'''.format(
            anchor, fig_title))

    # include specs in the report
    outfile.write('''<table><tr>''')
    for d in outdirs:
        outfile.write('''<td><b>{} specs:</b><br/><pre>'''.format(
            os.path.basename(os.path.normpath(d))))
        specs_path = os.path.join(d, 'specs.txt')
        try:
            with open(specs_path) as specs:
                for line in specs:
                    outfile.write(line)
        except FileNotFoundError:
            outfile.write("N/A")

        outfile.write('''</pre></td>''')
    outfile.write('''</tr></table><br/><table>''')

    # only the first figure needs js included
    include_js = True
    for figure in figures:
        plot, table = figure.plot, figure.table
        fig_title = plot.layout.title.text
        anchor = title_to_html_anchor(fig_title)
        outfile.write('''<tr><td id="{}">'''.format(anchor))

        outfile.write(
            plot.to_html(full_html=False, include_plotlyjs=include_js))
        include_js = False
        outfile.write('''</td><td>''')
        outfile.write(table.to_html(full_html=False, include_plotlyjs=False))
        outfile.write('''</td></tr>''')

    outfile.write('''
    </table>
    </body>
    </html>
    ''')
