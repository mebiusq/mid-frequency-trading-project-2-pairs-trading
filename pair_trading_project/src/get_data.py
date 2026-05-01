import yfinance as yf
import pandas as pd
import os


class DataFetcherMixin:
    def fetch_data(self):  # подгружаем данные
        raw = yf.download(self.cfg['tickers'], start=self.cfg['data_start'], end=self.cfg['test_end'],
                          interval=self.cfg['timeframe'], auto_adjust=False, progress=False, threads=False)
        # пыталась сделать threads=True но вероятно yfinance не дал сделать много запросов сразу(
        self.data = raw['Close'].ffill().dropna(how='any')
        self.open_data = raw['Open'].ffill().dropna(
            how='any')  # рынки FX не торгуются в выходные, а криптовалюты торгуются

        common_idx = self.data.index.intersection(self.open_data.index)
        self.data = self.data.loc[common_idx]
        self.open_data = self.open_data.loc[common_idx]

        os.makedirs('data', exist_ok=True)
        self.data.loc[self.cfg['train_start']:self.cfg['train_end']].to_parquet('data/train_data.parquet')
        self.data.loc[self.cfg['test_start']:self.cfg['test_end']].to_parquet('data/backtest_data.parquet')
