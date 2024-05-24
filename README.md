# AI-2024-final-project


## Data source
We get the data from [Binance API](https://github.com/binance/binance-public-data)

Download data with this template command under `./DataDownloader`:
```
python3 download-kline.py -t um -s BTCUSDT -i 5m 
```

Note: Binance does not have labels in their csv files, so the label should be added manually:
```
open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote-volume,ignore
```

`plain_plot.py` can be used to plot candle chart. Use:
```
python3 plain_plot.py ../data/{target}
```