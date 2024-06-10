# AI-2024-final-project

## Requirments

- Python 3.10
  Install the required packages with:

```
pip install -r requirements.txt
```

Note that for the gym-futures-trading package, since it is located at a local path, you may need to install it manually or specify the exact local path.

## Data source

We get the data from [Binance API](https://github.com/binance/binance-public-data)

Run the following command to fetch and set the data up for training:

```
python .\DataDownloader\download-kline.py -t um -s BTCUSDT -i 5m -startDate 2024-01-01 -endDate 2024-04-30 -skip-daily 1
python .\data_utils\process_zipped_data.py

```

`plain_plot.py` can be used to plot candle chart. Use:

```
python3 plain_plot.py ../data/{target}
```
