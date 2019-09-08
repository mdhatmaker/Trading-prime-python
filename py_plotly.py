import plotly
import plotly.plotly as py
import plotly.graph_objs as go

from datetime import datetime
import pandas_datareader.data as web

df = web.DataReader("aapl", 'morningstar',
                    datetime(2015, 1, 1),
                    datetime(2016, 7, 1)).reset_index()

data = [go.Scatter(x=df.Date, y=df.High)]

#py.iplot(data)
plotly.offline.plot(data)
