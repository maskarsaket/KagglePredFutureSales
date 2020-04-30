from plotly.subplots import make_subplots
import plotly.graph_objects as go

def dualaxislineplot(data, x, primary_y, secondary_y, title):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=data[x], y=data[primary_y], name=primary_y),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=data[x], y=data[secondary_y], name=secondary_y),
        secondary_y=True
    )

    # Add figure title
    fig.update_layout(
        title_text=title
    )

    # Set x-axis title
    # fig.update_xaxes(title_text=x)

    # # Set y-axes titles
    fig.update_yaxes(title_text=primary_y, secondary_y=False)
    fig.update_yaxes(title_text=secondary_y, secondary_y=True)

    return fig