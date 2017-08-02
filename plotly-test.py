import plotly.graph_objs as go
import plotly.plotly as py

import numpy as np

colorscale = [[0, '#FAEE1C'], [0.33, '#F3558E'], [0.66, '#9C1DE7'], [1, '#581B98']]
trace1 = go.Scatter(
    y = np.random.randn(500),
    mode='markers',
    marker=dict(
        size='16',
        color = np.random.randn(500),
        colorscale=colorscale,
        showscale=True
    )
)
data = [trace1]
url_1 = py.plot(data, filename='scatter-for-dashboard', auto_open=False)
py.iplot(data, filename='scatter-for-dashboard')