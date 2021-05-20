import sys
#import plotly
#import plotly.plotly as py
#import plotly.graph_objs as go
import chart_studio.plotly as py
import pandas
import plotly.graph_objects as go
import plotly.io as pio

from datetime import datetime
import pandas_datareader.data as web

"""
trace0 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 15, 13, 17]
)
trace1 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[16, 5, 11, 9]
)
data = [trace0, trace1]

fig = go.Figure(data)
#fig = go.Figure(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1]))
fig.update_layout(title_text='hello world')
pio.write_html(fig, file='hello_world.html', auto_open=True)
#py.plot(data, filename = 'basic-line', auto_open=True)
"""

"""
start = datetime(2018, 5, 1)
end = datetime(2018, 5, 30)

def get_data(ticker):
    try:
        #df = web.DataReader('%s' % (ticker), 'morningstar', start, end, retry_count=0)
        df = web.DataReader('%s' % (ticker), 'yahoo', start, end, retry_count=0)
        print(df.tail(5))
    except ValueError:
     print('Ticker Symbol %s is not available!' % (ticker))

get_data('TSLA') #valid Symbol
#get_data('yyfy') #not a valid Symbol


#symbol = "aapl"
symbol = "btc-usd"
df = web.DataReader(symbol, 'yahoo',
                    datetime(2016, 1, 1),
                    datetime(2022, 7, 1)).reset_index()

data = [go.Scatter(x=df.Date, y=df.High)]

#py.iplot(data)
#plotly.offline.plot(data)
#chart_studio.plotly.plotly.plot(data)

fig = go.Figure(data)
#fig = go.Figure(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1]))
fig.update_layout(title_text=symbol.upper())
pio.write_html(fig, file='hello_world.html', auto_open=True)
"""


copper_data_path = "/Users/michael/data/copper_data/"
filename = "copper_discount_output.2021-05-20-145704.csv"
pathname = copper_data_path + filename

df = pandas.read_csv(pathname)

print(df.head(10))

data = [go.Scatter(x=df.Timestamp, y=df.Discount)]

fig = go.Figure(data)
fig.update_layout(title_text="Copper Discount");
pio.write_html(fig, file='copper_discount.html', auto_open=True)
