import os
import time
import csv
import json
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
        'auto_case_ticker': None,
        'output_dir': 'figures',
        'run_root': None,
        'experiment_name': None,
        'plot_layer': 0,
        'plot_head': 'average'
    }

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.dataset = self.dataset.upper()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.run_dir = None
        if self.run_root:
            self.experiment_name = self.experiment_name or os.path.basename(str(self.model_save_path).rstrip('/\\'))
            self.run_dir = os.path.join(self.run_root, self.experiment_name)
            self.model_save_path = os.path.join(self.run_dir, 'checkpoints')
            self.output_dir = os.path.join(self.run_dir, 'figures')
            os.makedirs(os.path.join(self.run_dir, 'logs'), exist_ok=True)
            os.makedirs(self.model_save_path, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2, sort_keys=True, default=str)
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
            'val_end', 'test_start', 'test_end', 'volume_window', 'label_window', 'tickers',
            'feature_cache_dir'
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
        loss_rows = []

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
            row = {
                'epoch': epoch + 1,
                'train_loss': float(np.average(loss1_list)),
                'vali_loss_series_phase': float(vali_loss1),
                'vali_loss_prior_phase': float(vali_loss2)
            }
            loss_rows.append(row)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, row['train_loss'], vali_loss1))
            self._save_train_losses(loss_rows)
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def _save_train_losses(self, rows):
        if not self.run_dir or not rows:
            return
        path = os.path.join(self.run_dir, 'train_losses.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

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

    def _save_timeline(self, name, scores, labels):
        if not self.run_dir:
            return
        path = os.path.join(self.run_dir, '{}_timeline_scores.csv'.format(name))
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ticker', 'date', 'score', 'label'])
            for (ticker, date), score in sorted(scores.items()):
                writer.writerow([ticker, date, score, labels[(ticker, date)]])

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

    def _save_curves(self, scores, labels):
        if not self.run_dir:
            return
        from sklearn.metrics import precision_recall_curve, roc_curve
        import matplotlib.pyplot as plt

        keys = list(scores.keys())
        y_score = np.array([scores[key] for key in keys], dtype=float)
        y_true = np.array([labels[key] for key in keys], dtype=int)
        if len(np.unique(y_true)) < 2:
            return
        figure_dir = os.path.join(self.run_dir, 'figures')
        os.makedirs(figure_dir, exist_ok=True)

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(recall, precision)
        ax.set_title('Precision-Recall Curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(figure_dir, 'precision_recall_curve.png'), dpi=150)
        plt.close(fig)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle='--', color='0.6')
        ax.set_title('ROC Curve')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(figure_dir, 'roc_curve.png'), dpi=150)
        plt.close(fig)

    def evaluate(self):
        val_scores, val_labels = self.aggregate_window_scores(self.vali_loader)
        test_scores, test_labels = self.aggregate_window_scores(self.test_loader)
        self._save_timeline('val', val_scores, val_labels)
        self._save_timeline('test', test_scores, test_labels)
        threshold = self._threshold(val_scores, val_labels)
        metrics, y_pred = self._point_metrics(test_scores, test_labels, threshold)
        metrics.update(self._event_metrics(test_scores, test_labels, y_pred))
        self._save_curves(test_scores, test_labels)
        for key, value in metrics.items():
            print("{}: {}".format(key, value))
        if self.run_dir:
            with open(os.path.join(self.run_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2, sort_keys=True, allow_nan=True)
            with open(os.path.join(self.run_dir, 'metrics.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['metric', 'value'])
                for key, value in metrics.items():
                    writer.writerow([key, value])
        return metrics

    def test(self):
        self.model.load_state_dict(torch.load(
            os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'),
            map_location=self.device))
        print("======================TEST MODE======================")
        return self.evaluate()

    def _plot_event_case(self, event_date_text, case_name='manual'):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        event_date = np.datetime64(event_date_text)
        dataset = self.test_loader.dataset
        candidates = [
            (i, meta) for i, meta in enumerate(dataset.metadata)
            if meta['ticker'] == self.visualize_ticker and np.datetime64(meta['dates'][0]) <= event_date <= np.datetime64(meta['dates'][-1])
        ]
        if not candidates:
            raise ValueError("No window found for {} {}".format(self.visualize_ticker, event_date_text))
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
        s_plot = s_plot.copy()
        p_plot = p_plot.copy()
        np.fill_diagonal(s_plot, np.nan)
        np.fill_diagonal(p_plot, np.nan)
        diff_plot = np.abs(s_plot - p_plot)
        discrepancy = (kl_point(out['prior'][layer], out['series'][layer]) +
                       kl_point(out['series'][layer], out['prior'][layer])).mean(dim=1).squeeze(0).cpu().numpy()
        feature_error = out['feature_error'].squeeze(0).cpu().numpy()
        event_pos = meta['dates'].index(event_date_text) if event_date_text in meta['dates'] else len(meta['dates']) - 1

        os.makedirs(self.output_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(self.output_dir, 'attention_values_{}_{}.npz'.format(
                self.visualize_ticker, '{}_{}'.format(case_name, event_date_text.replace('-', '_')))),
            series=s_plot,
            prior=p_plot,
            discrepancy=discrepancy,
            score=score,
            feature_error=feature_error,
            dates=np.array(meta['dates']),
            feature_names=np.array(self.feature_cols)
        )
        dates = meta['dates']
        x_axis = np.arange(len(dates))
        tick_step = max(1, len(dates) // 6)
        tick_pos = list(range(0, len(dates), tick_step))
        if tick_pos[-1] != len(dates) - 1:
            tick_pos.append(len(dates) - 1)
        tick_labels = [dates[i] for i in tick_pos]

        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('white')
        fig = plt.figure(figsize=(16, 17))
        fig.suptitle('{} attention diagnostic | {} | {}'.format(self.visualize_ticker, case_name, event_date_text), fontsize=14)
        grid = gridspec.GridSpec(6, 3, figure=fig, height_ratios=[1.0, 1.0, 1.0, 1.55, 1.15, 0.9])

        ax_close = fig.add_subplot(grid[0, :])
        ax_close.plot(x_axis, meta['close'], color='0.2', linewidth=1.2, label='close')
        ax_close.axvline(event_pos, color='red', alpha=0.8)
        ax_close.scatter([event_pos], [meta['close'][event_pos]], facecolors='none', edgecolors='red', s=70, linewidth=1.5)
        ax_close.set_title('Close with selected attention point')
        ax_close.set_ylabel('Close')
        ax_close.grid(True, alpha=0.25)
        ax_close.legend(loc='upper left')

        ax_return = fig.add_subplot(grid[1, :], sharex=ax_close)
        if 'log_return_1d' in self.feature_cols:
            returns = x_np[:, self.feature_cols.index('log_return_1d')]
            std = np.nanstd(returns)
            ax_return.plot(x_axis, returns, color='#1f77b4', linewidth=1.0, label='log_return_1d')
            ax_return.axhline(3 * std, linestyle='--', color='purple', alpha=0.55, label='+3 std')
            ax_return.axhline(-3 * std, linestyle='--', color='purple', alpha=0.55, label='-3 std')
        ax_return.axvline(event_pos, color='red', alpha=0.8)
        ax_return.scatter([event_pos], [returns[event_pos] if 'log_return_1d' in self.feature_cols else 0],
                          facecolors='none', edgecolors='red', s=70, linewidth=1.5)
        ax_return.set_title('Log return')
        ax_return.set_ylabel('log_return')
        ax_return.grid(True, alpha=0.25)
        ax_return.legend(loc='upper left')

        ax_score = fig.add_subplot(grid[2, :], sharex=ax_close)
        ax_score.plot(x_axis, score, color='#8c564b', linewidth=1.1, label='anomaly score')
        ax_score.axvline(event_pos, color='red', alpha=0.8)
        ax_score.scatter([event_pos], [score[event_pos]], facecolors='none', edgecolors='red', s=70, linewidth=1.5,
                         label='selected window')
        ax_score.set_title('Anomaly score')
        ax_score.set_ylabel('score')
        ax_score.grid(True, alpha=0.25)
        ax_score.legend(loc='upper left')

        heatmaps = [
            (fig.add_subplot(grid[3, 0]), s_plot, 'Learned series attention (diag masked)'),
            (fig.add_subplot(grid[3, 1]), p_plot, '{} prior attention (diag masked)'.format(self.prior_type)),
            (fig.add_subplot(grid[3, 2]), diff_plot, '|series - prior| (diag masked)')
        ]
        for ax, values, title in heatmaps:
            image = ax.imshow(values, aspect='auto', vmin=0.0, vmax=np.nanmax(values) if np.nanmax(values) > 0 else 1.0,
                              cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel('Key date')
            ax.set_ylabel('Query date')
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=35, ha='right', fontsize=8)
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_labels, fontsize=8)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)

        ax_row = fig.add_subplot(grid[4, 0])
        ax_row.plot(x_axis, np.nan_to_num(s_plot[event_pos]), label='series selected row')
        ax_row.plot(x_axis, np.nan_to_num(p_plot[event_pos]), label='prior selected row')
        ax_row.set_title('Selected query attention row')
        ax_row.set_xlabel('Key position')
        ax_row.set_ylabel('Weight')
        ax_row.grid(True, alpha=0.25)
        ax_row.legend(loc='upper right')

        ax_diff = fig.add_subplot(grid[4, 1])
        ax_diff.bar(x_axis, np.nan_to_num(diff_plot[event_pos]), color='#8c564b')
        ax_diff.set_title('Selected query absolute difference')
        ax_diff.set_xlabel('Key position')
        ax_diff.set_ylabel('Abs diff')
        ax_diff.grid(True, alpha=0.25)

        ax_text = fig.add_subplot(grid[4:, 2])
        ax_text.axis('off')
        top_key = int(np.nanargmax(np.nan_to_num(s_plot[event_pos])))
        top_lag = int(abs(event_pos - top_key))
        text = [
            'Ticker: {}'.format(self.visualize_ticker),
            'Event date: {}'.format(event_date_text),
            'Window dates: {} to {}'.format(dates[0], dates[-1]),
            'Layer/head: {}/{}'.format(layer, self.plot_head),
            'Prior type: {}'.format(self.prior_type),
            'Score at event: {:.6g}'.format(float(score[event_pos])),
            'Association discrepancy: {:.4f}'.format(float(discrepancy[event_pos])),
            'Top attention key date: {}'.format(dates[top_key]),
            'Top attention lag: {}'.format(top_lag),
            '',
            'Interpretation:',
            'Series is the learned time-to-time attention.',
            'Prior is the configured time/state association.',
            'Large |series-prior| marks where learned attention',
            'departs from the finance prior. Feature bars show',
            'which reconstructed inputs contributed at the event.'
        ]
        ax_text.text(0.0, 1.0, '\n'.join(text), va='top', ha='left', family='monospace', fontsize=10)

        ax_feat = fig.add_subplot(grid[5, 0:2])
        ax_feat.bar(self.feature_cols, feature_error[event_pos])
        ax_feat.tick_params(axis='x', rotation=70)
        ax_feat.set_title('Feature reconstruction error at selected date')
        ax_feat.grid(True, axis='y', alpha=0.25)

        for ax in [ax_close, ax_return, ax_score]:
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=30, ha='right')
        fig.tight_layout()
        output = os.path.join(self.output_dir, 'event_case_{}_{}_{}.png'.format(
            self.visualize_ticker, case_name, event_date_text.replace('-', '_')))
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print("Saved {}".format(output))
        return output

    def _auto_case_dates(self, ticker):
        frame = self.test_loader.dataset.frame
        ticker_frame = frame[frame[self.test_loader.dataset.ticker_col].astype(str) == str(ticker)].copy()
        if ticker_frame.empty:
            raise ValueError("No test rows found for ticker {}".format(ticker))
        ticker_frame = ticker_frame.sort_values(self.test_loader.dataset.date_col).reset_index(drop=True)
        date_col = self.test_loader.dataset.date_col
        label_col = self.test_loader.dataset.label_col
        dates = ticker_frame[date_col].dt.strftime('%Y-%m-%d').tolist()

        max_return_idx = int(ticker_frame['log_return_1d'].astype(float).idxmax())
        normal_frame = ticker_frame[ticker_frame[label_col] == 0]
        if normal_frame.empty:
            normal_frame = ticker_frame
        normal_idx = int(normal_frame['log_return_1d'].abs().astype(float).idxmin())

        scores, _ = self.aggregate_window_scores(self.test_loader)
        ticker_scores = {date: score for (score_ticker, date), score in scores.items() if score_ticker == ticker}
        if not ticker_scores:
            raise ValueError("No scores found for ticker {}".format(ticker))
        max_score_date = max(ticker_scores, key=ticker_scores.get)
        date_to_idx = {date: i for i, date in enumerate(dates)}
        max_score_idx = date_to_idx[max_score_date]

        cases = [
            ('max_log_return', max_return_idx),
            ('normal_low_abs_log_return', normal_idx),
            ('max_score', max_score_idx)
        ]
        selected = []
        for name, idx in cases:
            for day_idx in range(max(0, idx - 5), idx + 1):
                selected.append((name, dates[day_idx]))
        return selected

    def visualize_event_case(self):
        if self.auto_case_ticker:
            self.visualize_ticker = self.auto_case_ticker
        if not self.visualize_ticker:
            raise ValueError("--visualize_ticker or --auto_case_ticker is required for visualize mode")

        self.model.load_state_dict(torch.load(
            os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'),
            map_location=self.device))

        if self.auto_case_ticker:
            outputs = []
            for case_name, event_date in self._auto_case_dates(self.auto_case_ticker):
                outputs.append(self._plot_event_case(event_date, case_name))
            return outputs

        if not self.event_date:
            raise ValueError("--event_date is required unless --auto_case_ticker is used")
        return self._plot_event_case(self.event_date)
