import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment, DEFAULT_STOCK_FEATURES


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def _csv(value, default):
    if value is None or value == '':
        return list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [part.strip() for part in value.split(',') if part.strip()]


def kl_point(p, q, eps=1e-8):
    return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=-1)


def association_point(series_list, prior_list):
    point = 0.0
    for series, prior in zip(series_list, prior_list):
        point = point + (kl_point(prior, series) + kl_point(series, prior)).mean(dim=1)
    return point / len(series_list)


def association_losses(series_list, prior_list):
    series_loss = 0.0
    prior_loss = 0.0
    for series, prior in zip(series_list, prior_list):
        p_detach = prior.detach()
        s_detach = series.detach()
        series_loss = series_loss + (kl_point(p_detach, series) + kl_point(series, p_detach)).mean()
        prior_loss = prior_loss + (kl_point(prior, s_detach) + kl_point(s_detach, prior)).mean()
    return series_loss / len(series_list), prior_loss / len(series_list)


def reconstruction_error(x, x_hat, return_idx):
    feature_error = (x - x_hat) ** 2
    return feature_error.mean(dim=-1), feature_error[:, :, return_idx], feature_error, feature_error.mean()


def return_nll(return_params, r, eps=1e-8):
    mu = return_params['mu']
    sigma = return_params['sigma'] + eps
    nll = 0.5 * (torch.log(sigma ** 2) + ((r - mu) ** 2 / (sigma ** 2)))
    return nll, nll.mean(), mu, sigma


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {
        'stride': 1,
        'prior_type': 'time',
        'features': ','.join(DEFAULT_STOCK_FEATURES),
        'z_state_features': 'log_return_1d,abs_return,volume_z,rolling_vol_5,rolling_vol_20,vol_ratio_5_20',
        'label_type': 'absolute',
        'return_loss_weight': 0.0,
        'use_return_nll': False,
        'nll_weight': 0.0,
        'score_type': 'original',
        'feature_weights': None,
        'score_aggregation': 'mean',
        'threshold_method': 'percentile',
        'threshold_percentile': 99.0,
        'event_tolerance': 1,
        'top_k': None,
        'visualize_ticker': None,
        'event_date': None,
        'output_dir': 'figures',
        'plot_layer': 0,
        'plot_head': 'average'
    }

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.dataset = self.dataset.upper()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_cols = _csv(self.features, DEFAULT_STOCK_FEATURES)
        if self.dataset == 'STOCK':
            self.input_c = len(self.feature_cols)
            self.output_c = len(self.feature_cols)
        self.return_idx = self.feature_cols.index('log_return_1d') if 'log_return_1d' in self.feature_cols else 0
        self.feature_weight_values = None
        if self.feature_weights:
            self.feature_weight_values = [float(x) for x in str(self.feature_weights).split(',') if x.strip()]
            if len(self.feature_weight_values) != len(self.feature_cols):
                raise ValueError("--feature_weights must match the number of --features")
        self.z_state_indices = [self.feature_cols.index(name) for name in _csv(self.z_state_features, [])
                                if name in self.feature_cols]
        self.train_loader = self._loader('train')
        self.vali_loader = self._loader('val')
        self.test_loader = self._loader('test')
        self.thre_loader = self._loader('thre')
        self.build_model()

    def _loader_kwargs(self):
        if self.dataset != 'STOCK':
            return {}
        keys = [
            'features', 'label_type', 'date_col', 'ticker_col', 'open_col', 'high_col',
            'low_col', 'close_col', 'volume_col', 'train_start', 'train_end', 'val_start',
            'val_end', 'test_start', 'test_end', 'volume_window', 'label_window'
        ]
        return {key: getattr(self, key) for key in keys if hasattr(self, key)}

    def _loader(self, mode):
        return get_loader_segment(
            self.data_path, batch_size=self.batch_size, win_size=self.win_size,
            step=self.stride, mode=mode, dataset=self.dataset, **self._loader_kwargs())

    def build_model(self):
        self.model = AnomalyTransformer(
            win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3,
            prior_type=self.prior_type, z_state_indices=self.z_state_indices,
            use_return_nll=self.use_return_nll)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    def _forward_losses(self, input_data):
        x = input_data.float().to(self.device)
        x_hat, series, prior, prior_params, return_params = self.model(x)
        rec_point, return_point, feature_error, rec_loss = reconstruction_error(x, x_hat, self.return_idx)
        l_return = return_point.mean()
        l_nll = torch.tensor(0.0, device=self.device)
        nll_point = None
        if self.use_return_nll:
            nll_point, l_nll, _, _ = return_nll(return_params, x[:, :, self.return_idx])
        l_base = rec_loss + float(self.return_loss_weight) * l_return + float(self.nll_weight) * l_nll
        series_loss, prior_loss = association_losses(series, prior)
        return {
            'x': x, 'x_hat': x_hat, 'series': series, 'prior': prior,
            'prior_params': prior_params, 'return_params': return_params,
            'feature_error': feature_error, 'rec_point': rec_point,
            'return_point': return_point, 'nll_point': nll_point,
            'l_base': l_base, 'series_loss': series_loss, 'prior_loss': prior_loss
        }

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        with torch.no_grad():
            for input_data, _ in vali_loader:
                out = self._forward_losses(input_data)
                loss_1.append((out['l_base'] - self.k * out['series_loss']).item())
                loss_2.append((out['l_base'] + self.k * out['prior_loss']).item())
        return np.average(loss_1), np.average(loss_2)

    def train(self):
        print("======================TRAIN MODE======================")
        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            epoch_time = time.time()
            self.model.train()
            for i, (input_data, _) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                out = self._forward_losses(input_data)
                loss1 = out['l_base'] - self.k * out['series_loss']
                loss2 = out['l_base'] + self.k * out['prior_loss']
                loss1_list.append(loss1.item())
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            vali_loss1, vali_loss2 = self.vali(self.vali_loader)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, np.average(loss1_list), vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def compute_anomaly_score(self, out):
        assdis = association_point(out['series'], out['prior'])
        weight = torch.softmax(-assdis, dim=1)
        if self.score_type == 'original':
            error = out['rec_point']
        elif self.score_type == 'return_recon':
            error = out['return_point']
        elif self.score_type == 'return_nll':
            if out['nll_point'] is None:
                raise ValueError("return_nll score requires --use_return_nll true")
            error = out['nll_point']
        elif self.score_type == 'feature_weighted':
            if self.feature_weight_values is None:
                weights = torch.ones(out['feature_error'].shape[-1], device=self.device)
            else:
                weights = torch.tensor(self.feature_weight_values, device=self.device, dtype=out['feature_error'].dtype)
            error = torch.sum(out['feature_error'] * weights, dim=-1) / weights.sum()
        else:
            raise ValueError("Unknown score_type: {}".format(self.score_type))
        return error * weight

    def _window_outputs(self, loader):
        self.model.eval()
        scores = []
        labels = []
        sample_start = 0
        with torch.no_grad():
            for input_data, batch_labels in loader:
                out = self._forward_losses(input_data)
                batch_score = self.compute_anomaly_score(out).detach().cpu().numpy()
                batch_size = batch_score.shape[0]
                batch_meta = loader.dataset.metadata[sample_start:sample_start + batch_size]
                sample_start += batch_size
                scores.extend(batch_score)
                labels.extend(batch_labels.numpy())
                yield batch_score, batch_labels.numpy(), batch_meta, out

    def aggregate_window_scores(self, loader):
        buckets = defaultdict(list)
        label_buckets = defaultdict(list)
        for batch_scores, batch_labels, batch_meta, _ in self._window_outputs(loader):
            for scores, labels, meta in zip(batch_scores, batch_labels, batch_meta):
                ticker = meta['ticker']
                for date, score, label in zip(meta['dates'], scores, labels):
                    key = (ticker, date)
                    buckets[key].append(float(score))
                    label_buckets[key].append(int(label))
        timeline = {}
        labels = {}
        for key, values in buckets.items():
            if self.score_aggregation == 'max':
                timeline[key] = max(values)
            elif self.score_aggregation == 'mean':
                timeline[key] = float(np.mean(values))
            else:
                raise ValueError("Unknown score_aggregation: {}".format(self.score_aggregation))
            labels[key] = int(max(label_buckets[key]))
        return timeline, labels

    def _threshold(self, val_scores, val_labels):
        scores = np.array(list(val_scores.values()), dtype=float)
        labels = np.array([val_labels[key] for key in val_scores.keys()], dtype=int)
        if self.threshold_method == 'percentile':
            return float(np.percentile(scores, float(self.threshold_percentile)))
        if self.threshold_method == 'best_f1':
            best_threshold = scores[0]
            best_f1 = -1.0
            for threshold in np.unique(scores):
                pred = scores >= threshold
                tp = np.sum((pred == 1) & (labels == 1))
                fp = np.sum((pred == 1) & (labels == 0))
                fn = np.sum((pred == 0) & (labels == 1))
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            return float(best_threshold)
        raise ValueError("Unknown threshold_method: {}".format(self.threshold_method))

    def _point_metrics(self, scores, labels, threshold):
        from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

        keys = list(scores.keys())
        y_score = np.array([scores[key] for key in keys], dtype=float)
        y_true = np.array([labels[key] for key in keys], dtype=int)
        y_pred = (y_score >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0)
        k = int(self.top_k) if self.top_k else max(1, int(y_true.sum()))
        top = np.argsort(-y_score)[:k]
        metrics = {
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'precision_at_k': float(y_true[top].mean()) if len(top) else 0.0,
            'auc_pr': float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
            'auc_roc': float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan
        }
        return metrics, y_pred

    def _events(self, keys, flags):
        by_ticker = defaultdict(list)
        for key, flag in zip(keys, flags):
            by_ticker[key[0]].append((key[1], bool(flag)))
        events = []
        for ticker, rows in by_ticker.items():
            rows = sorted(rows)
            start = prev = None
            for date, flag in rows:
                cur = np.datetime64(date)
                if flag and start is None:
                    start = prev = cur
                elif flag:
                    prev = cur
                elif start is not None:
                    events.append((ticker, start, prev))
                    start = prev = None
            if start is not None:
                events.append((ticker, start, prev))
        return events

    def _event_metrics(self, scores, labels, y_pred):
        keys = list(scores.keys())
        y_true = np.array([labels[key] for key in keys], dtype=int)
        true_events = self._events(keys, y_true)
        pred_events = self._events(keys, y_pred)
        tol = np.timedelta64(int(self.event_tolerance), 'D')

        def overlaps(a, b):
            return a[0] == b[0] and a[1] <= b[2] + tol and a[2] >= b[1] - tol

        matched_pred = sum(any(overlaps(pred, true) for true in true_events) for pred in pred_events)
        matched_true = sum(any(overlaps(pred, true) for pred in pred_events) for true in true_events)
        event_precision = matched_pred / (len(pred_events) + 1e-8)
        event_recall = matched_true / (len(true_events) + 1e-8)
        event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall + 1e-8)
        return {
            'event_precision': float(event_precision),
            'event_recall': float(event_recall),
            'event_f1': float(event_f1),
            'event_hit_rate': float(event_recall)
        }

    def evaluate(self):
        val_scores, val_labels = self.aggregate_window_scores(self.vali_loader)
        test_scores, test_labels = self.aggregate_window_scores(self.test_loader)
        threshold = self._threshold(val_scores, val_labels)
        metrics, y_pred = self._point_metrics(test_scores, test_labels, threshold)
        metrics.update(self._event_metrics(test_scores, test_labels, y_pred))
        for key, value in metrics.items():
            print("{}: {}".format(key, value))
        return metrics

    def test(self):
        self.model.load_state_dict(torch.load(
            os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'),
            map_location=self.device))
        print("======================TEST MODE======================")
        return self.evaluate()

    def visualize_event_case(self):
        import matplotlib.pyplot as plt

        if not self.visualize_ticker or not self.event_date:
            raise ValueError("--visualize_ticker and --event_date are required for visualize mode")
        self.model.load_state_dict(torch.load(
            os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'),
            map_location=self.device))
        event_date = np.datetime64(self.event_date)
        dataset = self.test_loader.dataset
        candidates = [
            (i, meta) for i, meta in enumerate(dataset.metadata)
            if meta['ticker'] == self.visualize_ticker and np.datetime64(meta['dates'][0]) <= event_date <= np.datetime64(meta['dates'][-1])
        ]
        if not candidates:
            raise ValueError("No window found for {} {}".format(self.visualize_ticker, self.event_date))
        index, meta = min(candidates, key=lambda item: abs((np.datetime64(item[1]['end_date']) - event_date).astype(int)))
        x_np, _ = dataset[index]
        x = torch.from_numpy(x_np).unsqueeze(0).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self._forward_losses(x)
            score = self.compute_anomaly_score(out).squeeze(0).cpu().numpy()
        layer = int(self.plot_layer)
        series = out['series'][layer].squeeze(0).cpu().numpy()
        prior = out['prior'][layer].squeeze(0).cpu().numpy()
        if str(self.plot_head) == 'average':
            s_plot = series.mean(axis=0)
            p_plot = prior.mean(axis=0)
        else:
            head = int(self.plot_head)
            s_plot = series[head]
            p_plot = prior[head]
        discrepancy = (kl_point(out['prior'][layer], out['series'][layer]) +
                       kl_point(out['series'][layer], out['prior'][layer])).mean(dim=1).squeeze(0).cpu().numpy()
        feature_error = out['feature_error'].squeeze(0).cpu().numpy()
        event_pos = meta['dates'].index(self.event_date) if self.event_date in meta['dates'] else len(meta['dates']) - 1

        os.makedirs(self.output_dir, exist_ok=True)
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        dates = meta['dates']
        axes[0, 0].plot(dates, meta['close'], label='close')
        axes[0, 0].plot(dates, score / (np.nanmax(score) + 1e-8) * np.nanmax(meta['close']), label='score')
        axes[0, 0].axvline(self.event_date, color='red')
        axes[0, 0].legend()
        for name in ['log_return_1d', 'volume_z', 'rolling_vol_20', 'vol_ratio_5_20']:
            if name in self.feature_cols:
                axes[0, 1].plot(dates, x_np[:, self.feature_cols.index(name)], label=name)
        axes[0, 1].axvline(self.event_date, color='red')
        axes[0, 1].legend()
        axes[1, 0].imshow(s_plot, aspect='auto')
        axes[1, 0].set_title('Series association')
        axes[1, 1].imshow(p_plot, aspect='auto')
        axes[1, 1].set_title('Prior association')
        axes[2, 0].plot(dates, discrepancy)
        axes[2, 0].axvline(self.event_date, color='red')
        axes[2, 0].set_title('Association discrepancy')
        axes[2, 1].bar(self.feature_cols, feature_error[event_pos])
        axes[2, 1].tick_params(axis='x', rotation=90)
        axes[2, 1].set_title('Feature reconstruction error')
        fig.tight_layout()
        output = os.path.join(self.output_dir, 'event_case_{}_{}.png'.format(
            self.visualize_ticker, self.event_date.replace('-', '_')))
        fig.savefig(output, dpi=150)
        print("Saved {}".format(output))
        return output
