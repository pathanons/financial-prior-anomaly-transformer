import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import re


DEFAULT_STOCK_FEATURES = [
    'log_return_1d', 'return_5d', 'return_20d', 'volume_z', 'abs_return',
    'squared_return', 'rolling_vol_5', 'rolling_vol_20', 'vol_ratio_5_20',
    'gap', 'high_low_range'
]

_STOCK_FEATURE_CACHE = {}


def _split_csv(value, default):
    if value is None or value == '':
        return list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [part.strip() for part in value.split(',') if part.strip()]


def read_stock_ohlcv(data_path, date_col='date', ticker_col='ticker'):
    if os.path.isdir(data_path):
        combined = os.path.join(data_path, 'stock_ohlcv.csv')
        if os.path.exists(combined):
            frame = pd.read_csv(combined)
        else:
            frames = []
            for name in sorted(os.listdir(data_path)):
                if not name.lower().endswith('.csv'):
                    continue
                path = os.path.join(data_path, name)
                part = pd.read_csv(path)
                part[ticker_col] = name.split('_')[0]
                frames.append(part)
            if not frames:
                raise ValueError("No stock CSV files found in {}".format(data_path))
            frame = pd.concat(frames, axis=0, ignore_index=True)
    else:
        frame = pd.read_csv(data_path)

    lower = {col.lower(): col for col in frame.columns}
    rename = {}
    for expected in [date_col, ticker_col, 'open', 'high', 'low', 'close', 'volume']:
        if expected not in frame.columns and expected.lower() in lower:
            rename[lower[expected.lower()]] = expected
    return frame.rename(columns=rename)


def _safe_ticker_name(ticker):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(ticker))


def build_stock_features(frame, date_col='date', ticker_col='ticker', open_col='open',
                         high_col='high', low_col='low', close_col='close',
                         volume_col='volume', volume_window=60, label_window=60,
                         eps=1e-8):
    frame = frame.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values([ticker_col, date_col])
    out = []
    for _, g in frame.groupby(ticker_col, sort=False):
        g = g.sort_values(date_col).copy()
        close = g[close_col].astype(float)
        open_ = g[open_col].astype(float)
        high = g[high_col].astype(float)
        low = g[low_col].astype(float)
        volume = g[volume_col].astype(float)
        prev_close = close.shift(1)
        r = np.log(close / prev_close)
        log_volume = np.log(volume + 1.0)
        past_log_volume = log_volume.shift(1)
        past_r = r.shift(1)

        g['log_return_1d'] = r
        g['return_5d'] = np.log(close / close.shift(5))
        g['return_20d'] = np.log(close / close.shift(20))
        g['volume_z'] = (log_volume - past_log_volume.rolling(volume_window).mean()) / (
            past_log_volume.rolling(volume_window).std(ddof=0) + eps)
        g['abs_return'] = r.abs()
        g['squared_return'] = r ** 2
        g['rolling_vol_5'] = r.rolling(5).std(ddof=0)
        g['rolling_vol_20'] = r.rolling(20).std(ddof=0)
        g['vol_ratio_5_20'] = g['rolling_vol_5'] / (g['rolling_vol_20'] + eps)
        g['gap'] = (open_ - prev_close) / (prev_close + eps)
        g['high_low_range'] = (high - low) / (close + eps)
        g['positive_return'] = r.clip(lower=0.0)
        g['negative_return_magnitude'] = (-r).clip(lower=0.0)
        g['drawdown_20'] = close / close.rolling(20).max() - 1.0

        past_mean = past_r.rolling(label_window).mean()
        past_std = past_r.rolling(label_window).std(ddof=0)
        past_median = past_r.rolling(label_window).median()
        past_mad = past_r.rolling(label_window).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        z_return = (r - past_mean) / (past_std + eps)
        mad_z_return = (r - past_median) / (1.4826 * past_mad + eps)
        g['z_return'] = z_return
        g['mad_z_return'] = mad_z_return
        g['positive_label'] = ((z_return >= 3.0) | (mad_z_return >= 3.5)).astype(int)
        g['negative_label'] = ((z_return <= -3.0) | (mad_z_return <= -3.5)).astype(int)
        g['absolute_label'] = ((z_return.abs() >= 3.0) | (mad_z_return.abs() >= 3.5)).astype(int)
        g['return_volume_3std_label'] = ((z_return.abs() >= 3.0) | (g['volume_z'].abs() >= 3.0)).astype(int)
        g['return_label'] = g['absolute_label']
        g['volume_label'] = (g['volume_z'] >= 2.0).astype(int)
        g['contextual_label'] = ((g['absolute_label'] == 1) & (
            (g['volume_z'] >= 2.0) | (g['vol_ratio_5_20'] >= 1.5))).astype(int)
        out.append(g)

    result = pd.concat(out, axis=0).replace([np.inf, -np.inf], np.nan)
    return result.dropna().reset_index(drop=True)


def _feature_cache_dir(data_path, volume_window, label_window, feature_cache_dir):
    if feature_cache_dir:
        return feature_cache_dir
    base = data_path.rstrip('/\\') if os.path.isdir(data_path) else os.path.splitext(data_path)[0]
    return '{}_features_vw{}_lw{}'.format(base, volume_window, label_window)


def cached_stock_features(data_path, date_col, ticker_col, open_col, high_col, low_col,
                          close_col, volume_col, volume_window, label_window, tickers,
                          feature_cache_dir=None):
    ticker_filter = tuple(_split_csv(tickers, []))
    cache_dir = _feature_cache_dir(data_path, volume_window, label_window, feature_cache_dir)
    key = (
        os.path.abspath(data_path), date_col, ticker_col, open_col, high_col, low_col,
        close_col, volume_col, volume_window, label_window, ticker_filter,
        os.path.abspath(cache_dir)
    )
    if key not in _STOCK_FEATURE_CACHE:
        frames = []
        if os.path.isdir(cache_dir):
            for name in sorted(os.listdir(cache_dir)):
                if name.lower().endswith('_features.csv'):
                    frame = pd.read_csv(os.path.join(cache_dir, name), parse_dates=[date_col])
                    frames.append(frame)
        if not frames:
            print("Building stock features from {}...".format(data_path), flush=True)
            raw = read_stock_ohlcv(data_path, date_col, ticker_col)
            if ticker_filter:
                raw = raw[raw[ticker_col].astype(str).isin(ticker_filter)]
                if raw.empty:
                    raise ValueError("No rows matched --tickers {}".format(','.join(ticker_filter)))
            os.makedirs(cache_dir, exist_ok=True)
            groups = list(raw.groupby(ticker_col, sort=False))
            total = len(groups)
            for i, (ticker, g) in enumerate(groups, 1):
                frame = build_stock_features(
                    g, date_col, ticker_col, open_col, high_col, low_col, close_col,
                    volume_col, volume_window, label_window)
                frame.to_csv(os.path.join(cache_dir, '{}_features.csv'.format(_safe_ticker_name(ticker))),
                             index=False)
                frames.append(frame)
                if i == 1 or i == total or i % 10 == 0:
                    print("Feature cache: {}/{} tickers".format(i, total), flush=True)
            print("Wrote stock feature files to {}".format(cache_dir), flush=True)
        result = pd.concat(frames, axis=0, ignore_index=True)
        if ticker_filter:
            result = result[result[ticker_col].astype(str).isin(ticker_filter)]
            if result.empty:
                raise ValueError("No cached feature rows matched --tickers {}".format(','.join(ticker_filter)))
        if 'absolute_label' in result and 'return_label' not in result:
            result['return_label'] = result['absolute_label']
        if 'volume_z' in result and 'volume_label' not in result:
            result['volume_label'] = (result['volume_z'] >= 2.0).astype(int)
        if 'z_return' in result and 'volume_z' in result and 'return_volume_3std_label' not in result:
            result['return_volume_3std_label'] = (
                (result['z_return'].abs() >= 3.0) | (result['volume_z'].abs() >= 3.0)
            ).astype(int)
        _STOCK_FEATURE_CACHE[key] = result.sort_values([ticker_col, date_col]).reset_index(drop=True)
        print("Built stock feature rows: {}".format(len(_STOCK_FEATURE_CACHE[key])), flush=True)
    else:
        print("Reusing cached stock features from {}".format(data_path), flush=True)
    return _STOCK_FEATURE_CACHE[key]


class StockSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", features=None,
                 label_type='absolute', date_col='date', ticker_col='ticker',
                 open_col='open', high_col='high', low_col='low', close_col='close',
                 volume_col='volume', train_start=None, train_end=None,
                 val_start=None, val_end=None, test_start=None, test_end=None,
                 volume_window=60, label_window=60, tickers=None,
                 feature_cache_dir=None):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.close_col = close_col
        self.feature_cols = _split_csv(features, DEFAULT_STOCK_FEATURES)
        self.label_col = label_type if label_type.endswith('_label') else label_type + '_label'

        features_frame = cached_stock_features(
            data_path, date_col, ticker_col, open_col, high_col, low_col, close_col,
            volume_col, volume_window, label_window, tickers, feature_cache_dir)
        missing = [c for c in self.feature_cols + [self.label_col] if c not in features_frame.columns]
        if missing:
            raise ValueError("Missing stock columns: {}".format(','.join(missing)))

        train_mask, val_mask, test_mask = self._split_masks(
            features_frame[date_col], train_start, train_end, val_start, val_end,
            test_start, test_end)
        self.scaler = StandardScaler()
        self.scaler.fit(features_frame.loc[train_mask, self.feature_cols].values)

        self.frames = {
            'train': self._scaled_frame(features_frame.loc[train_mask].copy()),
            'val': self._scaled_frame(features_frame.loc[val_mask].copy()),
            'test': self._scaled_frame(features_frame.loc[test_mask].copy())
        }
        if mode == 'thre':
            self.frame = self.frames['val']
        else:
            self.frame = self.frames.get(mode, self.frames['test'])
        self.samples, self.labels, self.metadata = self._build_windows(self.frame)
        print("{}: {}".format(mode, (len(self.samples), self.win_size, len(self.feature_cols))))

    def _split_masks(self, dates, train_start, train_end, val_start, val_end, test_start, test_end):
        dates = pd.to_datetime(dates)
        if train_end or val_start or val_end or test_start:
            train_mask = pd.Series(True, index=dates.index)
            val_mask = pd.Series(True, index=dates.index)
            test_mask = pd.Series(True, index=dates.index)
            if train_start:
                train_mask &= dates >= pd.to_datetime(train_start)
            if train_end:
                train_mask &= dates <= pd.to_datetime(train_end)
            if val_start:
                val_mask &= dates >= pd.to_datetime(val_start)
            if val_end:
                val_mask &= dates <= pd.to_datetime(val_end)
            if test_start:
                test_mask &= dates >= pd.to_datetime(test_start)
            if test_end:
                test_mask &= dates <= pd.to_datetime(test_end)
            return train_mask, val_mask, test_mask

        unique_dates = np.array(sorted(dates.drop_duplicates()))
        train_cut = unique_dates[int(len(unique_dates) * 0.6)]
        val_cut = unique_dates[int(len(unique_dates) * 0.8)]
        return dates < train_cut, (dates >= train_cut) & (dates < val_cut), dates >= val_cut

    def _scaled_frame(self, frame):
        frame.loc[:, self.feature_cols] = self.scaler.transform(frame[self.feature_cols].values)
        return frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    def _build_windows(self, frame):
        samples, labels, metadata = [], [], []
        for ticker, g in frame.groupby(self.ticker_col, sort=False):
            g = g.sort_values(self.date_col).reset_index(drop=True)
            values = g[self.feature_cols].values.astype(np.float32)
            label_values = g[self.label_col].values.astype(np.float32)
            dates = g[self.date_col].dt.strftime('%Y-%m-%d').values
            closes = g[self.close_col].values.astype(float) if self.close_col in g else np.full(len(g), np.nan)
            for end in range(self.win_size - 1, len(g), self.step):
                start = end - self.win_size + 1
                x = values[start:end + 1]
                y = label_values[start:end + 1]
                if not np.isfinite(x).all():
                    continue
                samples.append(x)
                labels.append(y)
                metadata.append({
                    'ticker': ticker,
                    'start_date': dates[start],
                    'end_date': dates[end],
                    'dates': dates[start:end + 1].tolist(),
                    'close': closes[start:end + 1].tolist()
                })
        return samples, labels, metadata

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', **kwargs):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'STOCK'):
        dataset = StockSegLoader(data_path, win_size, step, mode, **kwargs)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
