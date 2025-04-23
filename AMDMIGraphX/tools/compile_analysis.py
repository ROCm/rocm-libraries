import pandas as pd
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse


def read_compile_log(file_path):
    # Regex pattern to match lines with "value: __ ms"
    pattern = r'([\w: ]+):\s*([\d.]+(?:e[+-]?\d+)?)ms'

    data = []

    # Read the file and find matches
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                operation = match.group(1).strip()
                try:
                    value = float(match.group(2))
                    data.append({'Operation': operation, 'Value': value})
                except:
                    print(f"Invalid Float: {match.group(2)}")

    return data


def filter_data(data, quantile):

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    df['Line Number'] = df.index
    df['Operation: Line Number'] = df.apply(
        lambda x: x["Operation"] + ": " + str(x['Line Number']), axis=1)
    df['Cumulative Sum'] = df['Value'].cumsum()

    print(
        f"Only included the top {round(1-quantile, 2)}% time-intensive operations"
    )
    top_operation_time = df['Value'].quantile(quantile)
    df = df[df['Value'] >= top_operation_time]

    return df


def plot_compile_analysis(df, output_path):
    df_sorted_by_time = df.sort_values(by='Value', ascending=False)
    df_sorted_by_total_time = df.groupby(
        'Operation')['Value'].sum().sort_values(ascending=False).reset_index()
    df_sorted_by_avg_time = df.groupby('Operation')['Value'].mean(
    ).sort_values(ascending=False).reset_index()
    df_counts = df['Operation'].value_counts().reset_index()

    fig = make_subplots(rows=2,
                        cols=4,
                        row_heights=[0.4, 0.6],
                        specs=[[{
                            'type': 'bar'
                        }, {
                            'type': 'bar'
                        }, {
                            'type': 'bar'
                        }, {
                            'type': 'bar'
                        }],
                               [{
                                   'type': 'scatter',
                                   'colspan': 4
                               }, None, None, None]],
                        subplot_titles=('Time Taken per Line',
                                        'Total Time Taken per Operation',
                                        'Avg Time Taken per Operation',
                                        'Number of times Operation called',
                                        'Compile Time Series Graph'),
                        vertical_spacing=0.25)

    fig.add_trace(go.Bar(
        x=df_sorted_by_time['Operation: Line Number'],
        y=df_sorted_by_time['Value'],
        name='Time Taken per Line',
        marker=dict(color='red'),
        hoverinfo='x+y',
    ),
                  row=1,
                  col=1)

    fig.add_trace(go.Bar(x=df_sorted_by_total_time['Operation'],
                         y=df_sorted_by_total_time['Value'],
                         name='Total Time Taken per Operation',
                         marker=dict(color='green'),
                         hoverinfo='x+y'),
                  row=1,
                  col=2)

    fig.add_trace(go.Bar(x=df_sorted_by_avg_time['Operation'],
                         y=df_sorted_by_avg_time['Value'],
                         name='Avg Time Taken per Operation',
                         marker=dict(color='green'),
                         hoverinfo='x+y'),
                  row=1,
                  col=3)

    fig.add_trace(go.Bar(x=df_counts['Operation'],
                         y=df_counts['count'],
                         name='Number of times Operation called',
                         marker=dict(color='royalblue'),
                         hoverinfo='x+y'),
                  row=1,
                  col=4)

    fig.update_xaxes(
        row=1,
        tickangle=45,
    )

    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df['Cumulative Sum'],
        mode='lines+markers',
        text=df['Operation: Line Number'],
        hoverinfo='text',
        name='Cumulative Time',
        line=dict(width=2),
        marker=dict(size=3),
    ),
                  row=2,
                  col=1)

    fig.update_xaxes(
        title_text='Operation: Line Number',
        tickvals=list(range(len(df))),
        ticktext=df['Operation: Line Number'],
        row=2,
        col=1,
        tickangle=45,
    )

    fig.update_yaxes(title_text='Cumulative Time (ms)', row=2, col=1)

    fig.update_layout(
        title='Compile Time Analysis',
        title_x=0.5,
        title_font=dict(size=24, color='darkblue'),
        # margin=dict(l=0, r=0, t=100, b=50),
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        height=1200,
        template='plotly_white',
        showlegend=False)

    fig.write_html(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', required=True)
    parser.add_argument('--quantile', default=0.95)
    parser.add_argument('--output_path', default="compile_analysis.html")
    args = parser.parse_args()

    data = read_compile_log(args.file_path)
    df = filter_data(data, args.quantile)
    plot_compile_analysis(df, args.output_path)
