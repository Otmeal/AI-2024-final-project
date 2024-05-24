import pandas as pd
import mplfinance as mpf
import argparse


def plot_candlestick(file_path):
    # 讀取 CSV 文件
    data = pd.read_csv(file_path)

    # 檢查數據
    print(data.head())

    # 確保時間列為 datetime 類型
    data["time"] = pd.to_datetime(data["open_time"], unit="ms")
    data.set_index("time", inplace=True)

    # 繪製 K 線圖
    mpf.plot(
        data,
        type="candle",
        style="charles",
        title="Candlestick Chart",
        ylabel="Price",
        warn_too_much_data=len(data) + 1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot candlestick chart from a CSV file."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="The path to the CSV file containing the trading data.",
    )

    args = parser.parse_args()
    plot_candlestick(args.file_path)
